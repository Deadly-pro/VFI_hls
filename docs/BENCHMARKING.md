# Benchmarking methodology

How the accelerator is measured, what the claims are backed by, and what is
deliberately not claimed. Every number is reproducible from this repo.

## What is measured

| Metric | Source |
|---|---|
| Bit-exact RTL vs golden | cocotb suites (`tb/`), all 14 passing |
| Gate count / logic area | synthesis reports (`syn/`) |
| Interpolated-frame quality | `tools/vfi_demo.py` on a Vimeo-90K test triplet: 56.8 dB INT8-RTL-vs-FP16; 26.4 dB vs ground truth |
| Latency | cycle-accurate RTL (`valid/ready` contract) — fixed cycle count |
| Memory footprint | parameter math (frame buffer, weight banks) |

The demo's PSNR is computed in memory: the FP16 reference array against the
dequantized INT8 output — that is the number it prints under `--rtl`. Comparing
the two *saved* PNGs instead is a lossy second step, because the reference is
then an 8-bit rounding of the array it used, and reads ~2 dB lower. Quote the
printed value, not a PNG diff. (An older `save_png` truncated rather than
rounded the reference and read ~6 dB lower still.)

## The comparison and its boundaries

The JCSSE paper reports FP16 TensorRT numbers on an RTX 3050 Laptop GPU. This
RTL is INT8 on a 130 nm educational node (or an FPGA). Those are not
apples-to-apples on raw speed, so the comparison is on architecture, not clock:

1. **Latency determinism.** A GPU's frames-per-second jitters run to run (OS
   scheduling, CUDA driver, graph launch). The RTL completes a frame in a
   mathematically fixed cycle count — the same number every run. The plot
   measures the standard deviation of repeated GPU runs against zero for RTL.
2. **Energy per op.** Energy per MAC scales with capacitance and voltage;
   comparing a 130 nm INT8 datapath to an 8 nm FP16 TensorRT kernel directly
   would be dishonest. Energy per frame is reported from the RTL's own
   gate-level switching, with cross-node extrapolation left as an explicit,
   labeled scaling exercise.
3. **Accuracy delta.** The RTL's INT8-vs-FP16 PSNR is measured in-repo on the
   demo frames. The paper's FP16-vs-FP16 numbers are a different axis and are
   not mixed in.

## What is not claimed

- No frame-rate win over a dGPU. Claiming that from a 130 nm test chip or a
  mid-range FPGA would be wrong.
- `grid_sample`/flow prediction is not in software-free RTL — flow estimation
  stays on the host; the RTL covers the warp+blend datapath and the CNN encoder
  (verified separately). The full model remains the FP16 ONNX reference.

## Reproducing the numbers

```sh
# RTL bit-exactness, all suites
./run_all_tests.sh

# Vivado synthesis (Artix-7 XC7A100T, 100 MHz)
cd syn/vivado && ./run_vivado.sh

# Live demo: two frames in, interpolated frame out
python3 tools/vfi_demo.py --t samples/frame_t.png --t1 samples/frame_t1.png \
    --out out/mid.png --rtl
```
