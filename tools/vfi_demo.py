#!/usr/bin/env python3
"""
VFI-DL Demo — bring your own t/t+1 frames, get the t+0.5 interpolated frame.

Two modes:
  --onnx   (default) full nano_v16.onnx FP16 pipeline -> interpolated frame
  --rtl    flow+mask from ONNX, warp+blend through the RTL (vfi_synth) via
           cocotb simulation -> INT8 interpolated frame (bit-exact datapath)

Outputs:
  <out>.png              interpolated frame (FP16 reference, or RTL INT8)
  <out>_strip.png        side-by-side: t | t+0.5 | t+1
  <out>_psnr.txt         PSNR between reference and RTL (--rtl only)

Examples:
  python3 tools/vfi_demo.py --t frame_a.png --t1 frame_b.png --out mid.png
  python3 tools/vfi_demo.py --t a.png --t1 b.png --out mid.png --rtl --res 128
"""

import argparse
import os
import sys
import numpy as np

# ---- optional deps, fail with clear message ----
try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Install: .venv/bin/pip install onnx onnxruntime")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Install: .venv/bin/pip install pillow")
    sys.exit(1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_PATH = os.path.join(ROOT, "nano_v16.onnx")
sys.path.insert(0, os.path.join(ROOT, "golden"))


def load_frames(t_path, t1_path, res=None):
    """Load t and t+1 frames, return as float16 arrays [1,3,H,W] in [0,1]."""
    t = Image.open(t_path).convert("RGB")
    t1 = Image.open(t1_path).convert("RGB")
    if res:
        t = t.resize((res, res), Image.BILINEAR)
        t1 = t1.resize((res, res), Image.BILINEAR)
    a = np.asarray(t, dtype=np.float32) / 255.0
    b = np.asarray(t1, dtype=np.float32) / 255.0
    return (a.transpose(2, 0, 1)[None].astype(np.float16),
            b.transpose(2, 0, 1)[None].astype(np.float16))


def run_onnx(x):
    """Run full model; returns (frame [1,3,H,W], grid [1,H,W,2], mask [1,1,H,W])."""
    model = onnx.load(ONNX_PATH)
    from onnx import helper
    # expose grid (Add_2) and mask (Resize_3) as extra outputs
    grid_vi = helper.make_tensor_value_info("/Add_2_output_0", onnx.TensorProto.FLOAT, None)
    mask_vi = helper.make_tensor_value_info("/Resize_3_output_0", onnx.TensorProto.FLOAT16, None)
    model.graph.output.append(grid_vi)
    model.graph.output.append(mask_vi)
    tmp = os.path.join(ROOT, "weights", "_nano_with_taps.onnx")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    onnx.save(model, tmp)

    sess = ort.InferenceSession(tmp)
    frame, grid, mask = sess.run(None, {sess.get_inputs()[0].name: x})
    return frame, grid, mask


def grid_to_flow_pixels(grid, flow_q=8):
    """Convert normalized GridSample grid to Q<flow_q> pixel displacement.

    align_corners=1:  src_pixel = (grid+1)*(W-1)/2
    flow_pixels = src_pixel - out_pixel, then scaled to fixed point.
    """
    H, W = grid.shape[1], grid.shape[2]
    # grid is [1,H,W,2] in [-1,1]
    gx = grid[0, :, :, 0]
    gy = grid[0, :, :, 1]
    src_x = (gx + 1) * (W - 1) / 2.0
    src_y = (gy + 1) * (H - 1) / 2.0
    ox, oy = np.meshgrid(np.arange(W), np.arange(H))
    fx = (src_x - ox).astype(np.float32)
    fy = (src_y - oy).astype(np.float32)
    scale = (1 << flow_q)
    fx_q = np.round(fx * scale).astype(np.int16)
    fy_q = np.round(fy * scale).astype(np.int16)
    return fx_q, fy_q


def mask_to_q8(mask):
    """Mask float [0,1] -> Q0.8 uint8 [0,255]."""
    m = mask[0, 0]
    return np.round(np.clip(m, 0, 1) * 255).astype(np.uint8)


def psnr(a, b):
    """PSNR between two [H,W,C] float arrays in [0,1]."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(1.0 / mse)


def save_png(arr_01, path):
    """arr [1,3,H,W] float [0,1] -> PNG."""
    a = np.clip(arr_01[0].transpose(1, 2, 0), 0, 1)
    img = Image.fromarray((a * 255).astype(np.uint8))
    img.save(path)


def save_strip(t, mid, t1, path):
    """Side-by-side t | t+0.5 | t+1."""
    a = np.clip(t[0].transpose(1, 2, 0), 0, 1)
    b = np.clip(mid[0].transpose(1, 2, 0), 0, 1)
    c = np.clip(t1[0].transpose(1, 2, 0), 0, 1)
    strip = np.hstack([a, b, c])
    Image.fromarray((strip * 255).astype(np.uint8)).save(path)


def run_rtl_vfi_synth(t_i8, t1_i8, fx_q, fy_q, mask_q, IMG_W, IMG_H, NCH=3):
    """Drive vfi_synth RTL via iverilog+vvp; returns [H,W,C] INT8 output."""
    import subprocess
    weights = os.path.join(ROOT, "weights")
    os.makedirs(weights, exist_ok=True)

    def dump(fname, arr, bits):
        """Write array as 2's-complement hex for $readmemh."""
        a = arr.reshape(-1).astype(np.int64)
        mask = (1 << bits) - 1
        with open(os.path.join(weights, fname), "w") as f:
            for v in a:
                f.write(f"{int(v) & mask:0{bits//4}x}\n")

    # channel-interleaved frames: [1,C,H,W] -> [H,W,C]
    dump("demo_t_i8.txt", t_i8[0].transpose(1, 2, 0), 8)
    dump("demo_t1_i8.txt", t1_i8[0].transpose(1, 2, 0), 8)
    dump("demo_flowx.txt", fx_q, 16)
    dump("demo_flowy.txt", fy_q, 16)
    dump("demo_mask.txt", mask_q, 8)

    # write a version of the tb with the right dims
    tb_src = os.path.join(ROOT, "tb", "demo_rtl_tb.v")
    out_vvp = os.path.join(weights, "demo_sim.vvp")
    rtl = [os.path.join(ROOT, "rtl", f) for f in
           ("vfi_synth.v", "warp_unit.v", "blend_unit.v")]

    subprocess.run(
        ["iverilog", "-g2012", "-o", out_vvp, "-s", "demo_rtl_tb",
         f"-DIMG_W={IMG_W}", f"-DIMG_H={IMG_H}", f"-DNCH={NCH}",
         tb_src] + rtl,
        check=True, capture_output=True)
    # run from ROOT so relative "weights/" paths resolve
    subprocess.run(["vvp", out_vvp], check=True, capture_output=True, cwd=ROOT)

    # read output, channel-interleaved INT8
    vals = np.loadtxt(os.path.join(weights, "demo_out.txt"), dtype=np.int32)
    exp_count = IMG_W * IMG_H * NCH
    if vals.size < exp_count:
        raise RuntimeError(f"RTL produced {vals.size} values, expected {exp_count}")
    vals = vals[:exp_count].reshape(IMG_H, IMG_W, NCH).astype(np.int8)

    # bit-exact check against the fixed-point golden (proves the RTL ran correctly)
    from vfi_synth_ref import vfi_synth_channels  # noqa: E402
    t_c = t_i8[0].transpose(1, 2, 0).astype(np.int8)
    t1_c = t1_i8[0].transpose(1, 2, 0).astype(np.int8)
    golden = vfi_synth_channels(t_c, t1_c, fx_q, fy_q, mask_q, flow_q=8)
    if not np.array_equal(golden, vals):
        raise RuntimeError("RTL output does not match the bit-exact golden reference")
    print("  RTL output verified bit-exact vs fixed-point golden")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", required=True, help="frame at time t")
    ap.add_argument("--t1", required=True, help="frame at time t+1")
    ap.add_argument("--out", required=True, help="output interpolated frame")
    ap.add_argument("--rtl", action="store_true",
                    help="run warp+blend through RTL (cocotb sim) instead of ONNX")
    ap.add_argument("--res", type=int, default=None,
                    help="resize frames to res x res before inference")
    args = ap.parse_args()

    t, t1 = load_frames(args.t, args.t1, args.res)
    x = np.concatenate([t, t1], axis=1).astype(np.float16)

    frame, grid, mask = run_onnx(x)
    H, W = frame.shape[2], frame.shape[3]

    if not args.rtl:
        save_png(frame, args.out)
        save_strip(t, frame, t1, args.out.replace(".png", "_strip.png"))
        print(f"[ONNX FP16] interpolated frame -> {args.out}")
        print(f"  frame {W}x{H}, output range [{float(frame.min()):.3f}, {float(frame.max()):.3f}]")
        print(f"  strip -> {args.out.replace('.png', '_strip.png')}")
        return

    # ---- RTL path: feed flow+mask to vfi_synth via plain-Verilog sim ----
    fx_q, fy_q = grid_to_flow_pixels(grid, flow_q=8)
    mask_q = mask_to_q8(mask)

    # frames: [0,1] -> INT8 signed [-128,127] via round(f*255)-128
    t_i8 = (t * 255).round().astype(np.int16) - 128
    t1_i8 = (t1 * 255).round().astype(np.int16) - 128

    out_i8 = run_rtl_vfi_synth(t_i8, t1_i8, fx_q, fy_q, mask_q, IMG_W=W, IMG_H=H)

    # dequantize back to [0,1] for comparison
    out_01 = (out_i8.astype(np.float32) + 128.0) / 255.0
    out_01 = out_01.transpose(2, 0, 1)[None]
    out_01 = np.clip(out_01, 0, 1)

    save_png(out_01, args.out)
    save_strip(t, out_01, t1, args.out.replace(".png", "_strip.png"))

    # honest comparison: RTL INT8 vs ONNX FP16 reference
    ref_01 = frame  # [1,3,H,W]
    p = psnr(ref_01[0].transpose(1, 2, 0), out_01[0].transpose(1, 2, 0))
    with open(args.out.replace(".png", "_psnr.txt"), "w") as f:
        f.write(f"PSNR (RTL INT8 vs ONNX FP16): {p:.2f} dB\n")
        f.write("Note: RTL datapath is INT8, model reference is FP16 —\n")
        f.write("this measures the INT8 quantization delta on warp+blend.\n")
    print(f"[RTL INT8] interpolated frame -> {args.out}")
    print(f"  PSNR vs FP16 reference: {p:.2f} dB")
    print(f"  PSNR log -> {args.out.replace('.png', '_psnr.txt')}")


if __name__ == "__main__":
    main()