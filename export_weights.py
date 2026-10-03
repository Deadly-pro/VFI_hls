#!/usr/bin/env python3
"""
PyTorch INT8 Quantization & Hex Export for VFI-DL weight_mem

Loads nano_v16.onnx, extracts encoder weights (E1, E2 layers),
applies TFLite-style symmetric INT8 quantization, exports .hex files
matching weight_mem address layout.

Usage:
  python3 export_weights.py [--output-dir weights/]
"""

import os
import sys
import argparse
import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    print("Installing onnx, onnxruntime...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnx", "onnxruntime"])
    import onnx
    import onnxruntime as ort


def requantize_scale(acc_max: int, bits: int = 31) -> tuple[int, int]:
    """
    Compute TFLite M0 and shift for given accumulator range.
    Returns (m0, shift) where m0 is Q0.31 fixed-point multiplier.
    """
    # Target: (acc * m0) >> (31 + shift) fits in INT8 [-128, 127]
    # For symmetric quantization, we want max |acc| * m0 / 2^(31+shift) ≈ 127
    if acc_max == 0:
        return 0x40000000, 0  # 0.5, no shift

    # Find shift so that m0 is in [0.5, 1.0) i.e., [2^30, 2^31)
    shift = 0
    while True:
        # m0 = round(127 * 2^(31+shift) / acc_max)
        m0 = round(127 * (1 << (31 + shift)) / acc_max)
        if m0 >= (1 << 30):  # m0 >= 0.5
            break
        shift += 1
        if shift > 31:
            shift = 31
            break

    m0 = min(m0, (1 << 31) - 1)  # clamp to Q0.31 max
    return m0, shift


def export_weight_mem_hex(weights: np.ndarray, biases: np.ndarray,
                          output_path: str, kernel_first: bool = True):
    """
    Export weights and biases to .hex file matching weight_mem layout.

    If kernel_first=True:  [kernel_weights..., biases...]
    Else:                  [biases..., kernel_weights...]

    Each line = one 8-bit signed value in hex (two's complement).
    """
    with open(output_path, 'w') as f:
        # int() before the mask: numpy 2 rejects `np.int8(-1) & 0xFF` (255 is
        # not representable in int8), which is what made this crash after it
        # had already created a 0-byte file.
        if kernel_first:
            # Kernels first
            for w in weights.flatten():
                f.write(f"{int(w) & 0xFF:02x}\n")
            # Then biases (truncated to 8-bit for demo; real impl needs 32-bit)
            for b in biases.flatten():
                f.write(f"{int(b) & 0xFF:02x}\n")
        else:
            for b in biases.flatten():
                f.write(f"{int(b) & 0xFF:02x}\n")
            for w in weights.flatten():
                f.write(f"{int(w) & 0xFF:02x}\n")

    print(f"Exported {weights.size + biases.size} values to {output_path}")


def extract_onnx_weights(onnx_path: str):
    """Extract convolution weights from ONNX model."""
    model = onnx.load(onnx_path)

    # Build initializer lookup
    init_dict = {}
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        init_dict[init.name] = arr

    # Map layer names to weight tensors based on ONNX node names
    # From earlier inspection: /e1/e1.0/Conv (DW), /e1/e1.3/Conv (PW), etc.
    weights = {}

    for node in model.graph.node:
        if node.op_type == 'Conv':
            w_name = node.input[1]  # weight input
            b_name = node.input[2] if len(node.input) > 2 else None

            if w_name in init_dict:
                w = init_dict[w_name]
                b = init_dict[b_name] if b_name and b_name in init_dict else None

                # Determine layer from node name
                name = node.name.lower()
                if 'e1' in name and 'e1.0' in name:
                    weights['e1_dw'] = (w, b)
                elif 'e1' in name and 'e1.3' in name:
                    weights['e1_pw'] = (w, b)
                elif 'e2' in name and 'e2.0' in name:
                    weights['e2_dw'] = (w, b)
                elif 'e2' in name and 'e2.3' in name:
                    weights['e2_pw'] = (w, b)
                elif 'e3' in name and 'e3.0' in name:
                    weights['e3_dw'] = (w, b)
                elif 'e3' in name and 'e3.3' in name:
                    weights['e3_pw'] = (w, b)
                elif 'btn' in name and 'btn.0' in name:
                    weights['btn_dw'] = (w, b)
                elif 'btn' in name and 'btn.3' in name:
                    weights['btn_pw'] = (w, b)
                elif 'd1' in name and 'd1.0' in name:
                    weights['d1_dw'] = (w, b)
                elif 'd1' in name and 'd1.3' in name:
                    weights['d1_pw'] = (w, b)
                elif 'd2' in name and 'd2.0' in name:
                    weights['d2_dw'] = (w, b)
                elif 'd2' in name and 'd2.3' in name:
                    weights['d2_pw'] = (w, b)
                elif 'd3' in name and 'd3.0' in name:
                    weights['d3_dw'] = (w, b)
                elif 'd3' in name and 'd3.3' in name:
                    weights['d3_pw'] = (w, b)
                elif 'head' in name and 'head.0' in name:
                    weights['head_0'] = (w, b)
                elif 'head' in name and 'head.2' in name:
                    weights['head_2'] = (w, b)

    return weights


def quantize_to_int8(arr: np.ndarray, scale: float = None) -> tuple[np.ndarray, float]:
    """Symmetric INT8 quantization: q = round(x / scale), scale = max(|x|) / 127."""
    if scale is None:
        scale = max(abs(arr.min()), abs(arr.max())) / 127.0
        if scale == 0:
            scale = 1.0
    q = np.round(arr / scale).astype(np.int8)
    # Clamp
    q = np.clip(q, -128, 127)
    return q, scale


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default=os.path.join(here, 'nano_v16.onnx'))
    parser.add_argument('--output-dir', default=os.path.join(here, 'weights'))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading ONNX from {args.onnx}...")
    weights = extract_onnx_weights(args.onnx)

    print("Extracted layers:", list(weights.keys()))

    # Focus on encoder E1 and E2 (our encoder_slice)
    for layer_name in ['e1_dw', 'e1_pw', 'e2_dw', 'e2_pw']:
        if layer_name not in weights:
            print(f"Warning: {layer_name} not found")
            continue

        w, b = weights[layer_name]
        print(f"\n{layer_name}: weight shape={w.shape}, bias shape={b.shape if b is not None else 'None'}")

        # Quantize
        w_q, w_scale = quantize_to_int8(w)
        b_q, b_scale = quantize_to_int8(b) if b is not None else (None, 1.0)

        # Compute requantization params (for reference)
        # Estimate max accumulator: 3x3 kernel * 127 * 127 * C_in for DW
        if 'dw' in layer_name:
            c_in = w.shape[1]  # input channels
            acc_max = 9 * 127 * 127 * c_in
        else:
            c_in = w.shape[1]
            acc_max = c_in * 127 * 127

        m0, shift = requantize_scale(acc_max)
        print(f"  acc_max≈{acc_max}, m0=0x{m0:08x}, shift={shift}")
        print(f"  weight scale={w_scale:.6f}, bias scale={b_scale:.6f}")

        # Export .hex
        # Layout: kernels first, then biases (matches weight_mem in ds_conv_layer_integrated)
        # DW: [C_IN, 3, 3] -> flatten to C_IN*9
        # PW: [C_OUT, C_IN] -> already flat
        if 'dw' in layer_name:
            # ONNX DW weight: [C_out, C_in/groups, kH, kW] with groups=C_in
            # So shape is [C_in, 1, 3, 3] -> reshape to [C_in, 9]
            w_flat = w_q.reshape(w_q.shape[0], -1)  # [C_in, 9]
            out_file = os.path.join(args.output_dir, f"{layer_name}.hex")
            export_weight_mem_hex(w_flat, b_q, out_file)
        else:
            # PW weight: [C_out, C_in, 1, 1] -> [C_out, C_in]
            w_flat = w_q.reshape(w_q.shape[0], -1)
            out_file = os.path.join(args.output_dir, f"{layer_name}.hex")
            export_weight_mem_hex(w_flat, b_q, out_file)

        # Also save scales for reference
        np.savez(os.path.join(args.output_dir, f"{layer_name}_scales.npz"),
                 weight_scale=w_scale, bias_scale=b_scale, m0=m0, shift=shift)

    # Create a master load script
    script_path = os.path.join(args.output_dir, "load_all_weights.py")
    with open(script_path, 'w') as f:
        f.write("""#!/usr/bin/env python3
# Auto-generated weight loading helper for testbenches
# Usage: from load_all_weights import load_weights; load_weights(dut, 'weights/')
#
# NOTE (unverified against the RTL): each .hex holds one stage's kernels
# followed by its biases, but ds_conv_layer_integrated decodes DW and PW into
# separate windows of the same layer: DW at WL_BASE and PW at
# WL_BASE + C_IN*9 + C_IN. Loading every file from address 0 therefore
# overwrites. Pass the right base per file, and give encoder_slice's E2 the
# WL_BASE of E1_WEIGHT_SPACE so the two stages do not collide. Wire this into
# tb/test_encoder_slice.py before trusting it.

import os

def load_weights(dut, weight_dir, bases=None):
    '''Load all .hex files into DUT weight_mem via wl_we/wl_addr/wl_data'''
    layers = ['e1_dw', 'e1_pw', 'e2_dw', 'e2_pw']
    bases = bases or {}
    for layer in layers:
        hex_path = os.path.join(weight_dir, f"{layer}.hex")
        if not os.path.exists(hex_path):
            continue
        base = bases.get(layer, 0)
        with open(hex_path) as hf:
            for addr, line in enumerate(hf):
                val = int(line.strip(), 16)
                # Sign-extend if needed
                if val >= 128:
                    val = val - 256
                dut.wl_we.value = 1
                dut.wl_addr.value = base + addr
                dut.wl_data.value = val
                # Note: testbench must await RisingEdge(dut.clk) after each
    print(f"Loaded {len(layers)} layers from {weight_dir}")
""")

    print(f"\n✓ Export complete. Files in {args.output_dir}/")
    print(f"  Load helper: {script_path}")


if __name__ == '__main__':
    main()