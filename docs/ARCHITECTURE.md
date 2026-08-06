# Architecture

What this repo builds, and the constraints it builds under.

## One-line description

A weight-reloadable, fixed-architecture INT8 CNN inference datapath in
hand-written Verilog, verified bit-exact against numpy golden models and
synthesized with Vivado 2026.1 on an Artix-7 XC7A100T.

The datapath implements the compute-heavy parts of the "Open-Frame-gen / Nano"
frame interpolation model (JCSSE 2026): a depthwise-separable CNN encoder plus
a warp+blend stage (bilinear grid_sample, mask blend) that produces an
interpolated frame from two inputs.

## Scope decisions

- **Accelerate the CNN backbone and the warp+blend, not the whole model.**
  Flow prediction stays in software. The warp is a data-dependent gather and is
  implemented as its own RTL unit (`warp_unit`), not fused into the CNN path.
- **INT8 fixed point only.** Activations and weights are signed INT8, symmetric
  per tensor (scale S, zero point 0). Accumulation is INT32. Requantization
  between layers is `round(acc * M0 * 2^-n)` clamped to `[-128, 127]`
  (TFLite-style). Float datapaths are out of scope.
- **Fixed architecture.** The datapath is not reconfigurable; the weight store
  is. This is a CNN accelerator, not an NPU.

## Module hierarchy (build and verify bottom-up, in this order)

| Module | What it does |
|---|---|
| `mac_int8` | INT8 x INT8 -> INT32 multiply-accumulate (the atom) |
| `weight_mem` | on-chip weight store with a load port (`wl_addr/wl_data/wl_we`) |
| `line_buffer` | holds N rows, emits sliding 3x3 windows |
| `requantize` | INT32 -> INT8 (M0, shift, clamp) |
| `dw_conv3x3` | depthwise 3x3 conv = line buffer + 9 MACs + accumulator + requant |
| `pw_conv1x1` | pointwise 1x1 conv (channel mixing), serial MAC array |
| `ds_conv_layer` | depthwise-separable layer: dw -> pw -> requant, with weight store |
| `ds_conv_layer_integrated` | the layer with `weight_mem` wired in for runtime reload |
| `encoder_slice` | two-layer strided encoder (E1 -> E2), stride-2 downsampling |
| `line_buffer_stream` | streaming 3x3 window extractor (1 pixel in / 1 window out) |
| `pw_conv1x1_parallel` | parallel MAC array (PARALLEL_CO=2/4) |
| `warp_unit` | bilinear grid_sample (align_corners=1, border clamp), Q8.8 |
| `blend_unit` | mask blend `(m*a + (255-m)*b + 127) >> 8` |
| `vfi_synth` | top-level: 2x warp + blend per channel, serialized output |

Streaming between stages is a simple `valid/ready` handshake. Weight loading is
a plain `addr/data/write-enable` port.

## Verification

Every module has a numpy golden reference in `golden/` and a cocotb suite in
`tb/`. The golden model is the authority on the exact arithmetic (including the
requant rounding); the testbench compares RTL output against it every cycle and
fails on the first mismatch with the cycle and values. `./run_all_tests.sh`
runs all 14 suites in dependency order.

## Synthesis

Vivado 2026.1 batch flow in `syn/vivado/` (Artix-7 XC7A100T, 100 MHz target).
The earlier Yosys/Sky130 flow is kept in `syn/scripts/` for reference. See
`docs/ROADMAP.md` for the current numbers.

## Repo layout

- `rtl/` — the Verilog; `rtl/synth/` holds flat-port wrappers for cells that
  use unpacked array ports (needed by tools that reject them).
- `golden/` — numpy bit-exact references, one per module.
- `tb/` — cocotb testbenches, one per module.
- `syn/` — synthesis scripts and constraints.
- `tools/` — `vfi_demo.py`, the end-to-end demo.
- `reports/` — the technical writeup (`VFI_DL_Report.md`).
- `docs/` — this file, the roadmap, and the benchmarking methodology.
