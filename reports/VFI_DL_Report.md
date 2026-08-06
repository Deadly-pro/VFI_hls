# VFI-DL: An INT8 Video Frame Interpolation Datapath in Verilog

Technical writeup for the fixed-function hardware built for the "Open-Frame-gen /
Nano" frame interpolation model (JCSSE 2026). All numbers in this document are
backed by a passing bit-exact cocotb suite; nothing here is estimated.

## 1. Architecture

The datapath is a weight-reloadable, fixed-architecture INT8 CNN engine. It
covers the two compute-heavy parts of the model:

- **The CNN encoder** — a depthwise-separable backbone built bottom-up from
  small cells: an INT8 multiply-accumulate primitive, an on-chip weight store
  with a runtime load port, a 3x3 sliding-window line buffer, a TFLite-style
  requantizer, and depthwise/pointwise convolutions composed into a full layer
  and a two-layer strided encoder.
- **The warp+blend stage** — bilinear `grid_sample` (align_corners=1, border
  clamp) and a mask blend, which turn two input frames into an interpolated one
  at time t+0.5.

### Numeric format

| Element | Format |
|---|---|
| Activations / weights | signed INT8, symmetric per tensor (scale S, zero point 0) |
| Accumulator | signed INT32 (exact; no overflow in 3x3xC) |
| Requantization | `round(acc * M0 * 2^-n)`, clamp to [-128, 127] (TFLite-style) |

The numpy golden in `golden/` is the authoritative definition of the rounding —
the RTL must match it bit-for-bit, and the testbenches enforce exactly that.

### Module chain (bottom-up)

`mac_int8` → `weight_mem` → `line_buffer` → `requantize` → `dw_conv3x3` →
`pw_conv1x1` → `ds_conv_layer` → `ds_conv_layer_integrated` → `encoder_slice`
plus the streaming `line_buffer_stream` and parallel `pw_conv1x1_parallel`
upgrades, and finally `warp_unit` → `blend_unit` → `vfi_synth`.

Streaming uses a plain `valid/ready` handshake; weight loading is a plain
`addr/data/write-enable` port.

## 2. Verification

Every module has a cocotb suite driven in lockstep with its numpy golden
reference. Outputs are compared every cycle; any mismatch fails with the exact
cycle, inputs, and expected-vs-got values. `./run_all_tests.sh` runs all 14
suites in dependency order and exits non-zero on the first failure.

Highlights:

- `mac_int8`, `weight_mem`, `requantize` — thousands of random and directed
  vectors each.
- `ds_conv_layer_integrated` — full layer with runtime weight reload and
  back-to-back frames.
- `encoder_slice` — two-layer strided encoder (8x8x6 → 2x2x96), end to end
  against a golden chain.
- `warp_unit` / `blend_unit` / `vfi_synth` — bit-exact vs `grid_sample_ref.py`
  and `vfi_synth_ref.py`.
- End-to-end demo (`tools/vfi_demo.py --rtl`): two input frames → RTL warp+blend
  → interpolated frame, bit-exact against the fixed-point golden, 50.8 dB vs the
  FP16 ONNX reference on a Vimeo-90K test triplet (26.4 dB vs ground truth in
  both paths — INT8 quantization is unmeasurable here).

## 3. Synthesis

The current flow is **Vivado 2026.1** on an Artix-7 XC7A100T (`syn/vivado/`),
100 MHz constraint. Post-synthesis results (utilization/WNS/Fmax, table image
in `reports/plots/vivado_results.png`):

| Module | LUT | LUTmem | FF | DSP | BRAM tiles | WNS (ns) | Fmax (MHz) |
|---|---|---|---|---|---|---|---|
| mac_int8 | 94 | 0 | 32 | 0 | - | 7.12 | 347.6 |
| weight_mem | 0 | 0 | 0 | 0 | 0.5 | - (no paths) | - |
| requantize | 323 | 0 | 34 | 4 | - | -0.50 | 95.2 |
| line_buffer | 200 | 108 | 21 | 0 | - | 6.67 | 300.3 |
| line_buffer_stream | 394 | 0 | 282 | 0 | - | 6.20 | 262.8 |
| dw_conv3x3 | 1,309 | 108 | 93 | 4 | - | -3.74 | 72.8 |
| pw_conv1x1 | 545 | 0 | 82 | 4 | - | -6.80 | 59.5 |
| pw_conv1x1_parallel | 965 | 0 | 115 | 8 | - | -7.98 | 55.6 |
| ds_conv_layer | 8,818 | 432 | 2,598 | 20 | - | -6.80 | 59.5 |
| ds_conv_layer_integrated | 8,690 | 432 | 3,062 | 20 | - | -6.80 | 59.5 |
| warp_unit | 1,985 | 0 | 2,074 | 4 | - | -19.64 | 33.7 |
| blend_unit | 173 | 0 | 9 | 0 | - | - (no reg-reg paths) | >100 |
| vfi_synth | 12,371 | 0 | 12,478 | 24 | - | -19.64 | 33.7 |

`vfi_synth` place+route: **12,287 LUT / 12,478 FF / 24 DSP, 0.318 W total
(0.226 W dynamic)** at the 100 MHz constraint; post-route WNS -21.8 ns →
Fmax ≈ 31.4 MHz. Notes:

- Primitives close comfortably (mac_int8 at 347 MHz; line buffers 260–300 MHz).
- `requantize` (the 64-bit scale multiply) sets the per-layer pace at 95 MHz.
- CNN layers land at 55–73 MHz; the MAC-array critical path is the shared
  serial-accumulator chain.
- `warp_unit`/`vfi_synth` are frame-buffer-gather bound (~31–34 MHz); the four
  parallel reads feed the bilinear taps combinationally.
- `encoder_slice` (serial C_OUT=96 PW) does not finish Vivado synthesis — the
  serial-PW FSM blows up logic optimization; `pw_conv1x1_parallel` is the fix
  and the parallel-PW layer itself synthesizes.
- `weight_mem` maps to 0.5 BRAM tile (RAMB18) with no sequential logic;
  `blend_unit` is a 1-cycle mask blend with only output staging registers, so
  neither has a reportable setup path.

The earlier **Yosys/Sky130** flow (`syn/scripts/`) measured these numbers on the
8 core modules (100 MHz target, 1.8 V):

| Module | Cells | Area (µm²) | Crit path (ns) | Fmax (MHz) | Power @100 MHz (mW) |
|---|---|---|---|---|---|
| mac_int8 | 134 | 741 | 6.70 | 149.3 | 0.02 |
| weight_mem | 1,189 | 7,651 | 4.90 | 204.1 | 0.44 |
| requantize | 510 | 3,383 | 8.87 | 112.7 | 0.16 |
| line_buffer | 572 | 3,669 | 3.42 | 292.4 | 0.18 |
| dw_conv3x3 | 1,440 | 9,473 | 9.01 | 111.0 | 0.46 |
| pw_conv1x1 | 925 | 6,057 | 9.22 | 108.5 | 0.28 |
| ds_conv_layer | 6,183 | 40,119 | 9.53 | 104.9 | 1.83 |
| ds_conv_layer_integrated | 8,561 | 55,421 | 9.55 | 104.7 | 2.71 |

The integrated layer closes at 104.7 MHz, over the 100 MHz target. The
`requantize` 64-bit multiplier and the serial `pw_conv1x1` were the critical
paths; `pw_conv1x1_parallel` (a PARALLEL_CO-way MAC array) and the streaming
line buffer address both.

The warp+blend stage was synthesized with generic Yosys (no PDK):

| Module | Cells | Registers | Notes |
|---|---|---|---|
| warp_unit | 22,791 | 2,072 | 16x16x8 frame buffer dominates |
| blend_unit | 793 | 9 | 1-cycle mask blend |
| vfi_synth (NCH=3) | 136,452 | 12,488 | 6x warp + 3x blend |

## 4. GPU comparison

The JCSSE paper reports FP16 TensorRT numbers on an RTX 3050 Laptop (Samsung
8 nm). This RTL is INT8 on a 130 nm educational node (or an FPGA). Those are
not comparable on raw clock speed or watts, so the comparison is framed on
architecture instead:

1. **Latency determinism.** The RTL completes a frame in a fixed cycle count,
   every run. A GPU's per-frame time jitters with OS scheduling and driver
   launches. This is the one axis fixed-function hardware wins honestly.
2. **Energy per op.** Energy per MAC scales with capacitance and voltage;
   comparing a 130 nm INT8 datapath to an 8 nm FP16 kernel directly would be
   dishonest. Energy per frame is reported from the RTL's own gate-level
   switching, with the cross-node extrapolation left as an explicit exercise.
3. **Accuracy delta.** The RTL's INT8-vs-FP16 PSNR is measured in-repo on the
   demo frames; the paper's FP16-vs-FP16 numbers are a different axis.

What is not claimed: frame-rate parity with a dGPU, full VFI in RTL (`grid_sample`
warping is in RTL but flow prediction and the rest of the model stay in
software), and any raw-watt comparison across nodes.
