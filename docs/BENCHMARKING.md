# Benchmarking & Showcase Methodology

How the VFI-DL INT8 accelerator is measured, what claims are backed by
evidence, and what is honestly *not* claimed. All numbers are reproducible
from this repo — nothing is estimated.

## What we measure (real, from this repo)

| Metric | Source | Status |
|---|---|---|
| Bit-exact RTL vs golden | cocotb suites (tb/*.py) | 7 suites pass |
| Gate count / logic area | `syn/scripts/synth.sh <top>` → Yosys `stat` | 12 modules |
| Interpolated-frame quality | `tools/vfi_demo.py` → PSNR | 54.7 dB demo, INT8 vs FP16 |
| Fixed cycle-count latency | cycle-accurate RTL (`out_valid` contract) | deterministic |
| Memory footprint | parameter math (frame buffer, weight banks) | in ARCHITECTURE.md |

## The comparison we make (and its boundaries)

The JCSSE paper reports FP16 TensorRT numbers on an RTX 3050 Laptop GPU.
We report INT8 RTL on a 130 nm educational node. These are **not**
apples-to-apples on raw speed, so we compare on architecture, not clock speed:

1. **Latency determinism.** GPU frames-per-second has run-to-run jitter (OS
   scheduling, CUDA driver, graph launch). RTL completes in a mathematically
   fixed cycle count: `N_capture + N_pixels * (NCH + 2)` for warp+blend. One
   number, exactly, every run. We measure the standard deviation over N
   repeats of the GPU baseline vs. zero for RTL.
2. **Energy-per-op (architecture, not node).** Energy per MAC/op scales with
   capacitance and voltage; comparing a 130 nm INT8 datapath to an 8 nm FP16
   TensorRT kernel directly is dishonest. We report energy per frame on our
   own measured gate-level switching and leave the cross-node extrapolation
   as an explicit scaling exercise, clearly labeled.
3. **INT8 accuracy delta.** We report our own measurement: RTL INT8 vs ONNX
   FP16 reference, PSNR on the demo frames. The paper's FP16-vs-FP16 numbers
   are a different axis and are not mixed in.

## What we do NOT claim

- No Fmax/area/power from OpenROAD yet (no PDK install in this sprint) — gate
  counts are the honest unit of measure until then.
- No frame-rate win over a dGPU. Claiming that from a 130 nm test chip would
  be wrong.
- `grid_sample`/flow prediction is **not** in RTL. RTL covers the warp+blend
  datapath; the CNN encoder is separately verified RTL, and the full model is
  the FP16 ONNX reference.

## Repro

```sh
# RTL bit-exactness (all suites)
./run_all_tests.sh

# synthesis gate counts
cd syn/scripts && ./synth.sh warp_unit && ./synth.sh blend_unit && ./synth.sh vfi_synth

# live demo: two frames in, interpolated frame out
python3 tools/vfi_demo.py --t samples/frame_t.png --t1 samples/frame_t1.png \
    --out out/mid.png --rtl
```

## NVIDIA presentation angle

Slide order that walks evidence-first, disclaimer-first:

1. **Title** — INT8 CNN accelerator for real-time video frame interpolation.
2. **Architecture** — dw_conv → line buffer → pw_conv → warp → blend.
3. **Verification** — bit-exact cocotb vs NumPy golden, 7/7 suites.
4. **Deterministic latency** — fixed-cycle RTL vs GPU jitter (the only plot
   we can win honestly).
5. **Quality** — PSNR of INT8 RTL vs FP16 reference on real frames.
6. **Boundaries** — what's not RTL yet (grid_sample, OpenROAD PPA), stated
   up front. Interviewers trust the person who volunteers the gap.
