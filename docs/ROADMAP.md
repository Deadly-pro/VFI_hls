# Roadmap — VFI-DL Accelerator in Verilog

Sprints, risk-ordered. The minimum defensible artifact is the end of **Sprint 2**. See `ARCHITECTURE.md` for the locked definition and module hierarchy.

## Sprint 0 — Toolchain de-risk (GATE, COMPLETED)
- [x] Install: Icarus Verilog 13.0, Yosys, OpenSTA, cocotb, numpy/matplotlib; Sky130 PDK (`sky130A`).
- [x] Push `mac_int8` end-to-end: RTL → cocotb (bit-exact vs numpy) → Yosys+OpenSTA → **area/power/fmax numbers out.** ← GATE MET.

## Sprint 1 — Primitives (COMPLETED)
- [x] `weight_mem` — reloadable weight store + load port; verified load-then-read.
- [x] `line_buffer` — sliding 3×3 window; verified window extraction vs numpy.
- [x] `requantize` — INT32→INT8 (M0, shift, clamp); verified vs TFLite-style golden.

## Sprint 2 — First full layer (COMPLETED) ← MINIMUM DEFENSIBLE ARTIFACT
- [x] `dw_conv3x3` — verified vs numpy depthwise.
- [x] `pw_conv1x1` — verified vs numpy pointwise.
- [x] `ds_conv_layer` — wired dw→pw→requant; verified end-to-end.
- [x] `ds_conv_layer_integrated` — wired `weight_mem` into `ds_conv_layer`; verified runtime weight reload over wl_we/wl_addr/wl_data without logic resynthesis.
- [x] **Silicon PPA Measured across all 8 modules on Sky130 130nm CMOS (@100 MHz / 1.8V):**

| Module | Cell Count | Est. Area ($\mu m^2$) | Crit. Path ($ns$) | $F_{max}$ ($MHz$) | Power ($mW$) @ 100MHz |
|---|---|---|---|---|---|
| `mac_int8` | 134 | 741.34 | 6.70 | **149.3 MHz** | 0.02 |
| `weight_mem` | 1,189 | 7,651.09 | 4.90 | **204.1 MHz** | 0.44 |
| `requantize` | 510 | 3,383.24 | 8.87 | **112.7 MHz** | 0.16 |
| `line_buffer` | 572 | 3,668.52 | 3.42 | **292.4 MHz** | 0.18 |
| `dw_conv3x3` | 1,440 | 9,472.84 | 9.01 | **111.0 MHz** | 0.46 |
| `pw_conv1x1` | 925 | 6,057.06 | 9.22 | **108.5 MHz** | 0.28 |
| `ds_conv_layer` | 6,183 | 40,118.58 | 9.53 | **104.9 MHz** | 1.83 |
| `ds_conv_layer_integrated` | 8,561 | 55,420.76 | 9.55 | **104.7 MHz** | 2.71 |

## Sprint 3 — Scale + deliverable (IN PROGRESS)
- [x] `encoder_slice` — two-layer strided encoder (E1→E2) with stride-2 downsampling; RTL + testbench + synthesis integration
- [x] `export_weights.py` — PyTorch ONNX → INT8 .hex exporter for weight_mem (E1/E2 layers); ready to run
- [x] `line_buffer_stream` — streaming 2-line FIFO 3×3 window extractor (1 pix-in / 1 win-out); RTL + testbench + synthesis
- [x] `pw_conv1x1_parallel` — parallel MAC array (PARALLEL_CO=2/4); closes pw_conv1x1 critical path
- [ ] Run weight export and integrate .hex loads into encoder_slice testbench
- [x] GPU-comparison methodology + honest boundaries: `docs/BENCHMARKING.md`
- [x] Technical writeup + NVIDIA slide plan: `reports/VFI_DL_ASIC_Report.md`

## Sprint 4 — VFI warp + blend accelerator (COMPLETED)
- [x] `warp_unit` — bilinear grid_sample (align_corners=1, border clamp), Q8.8 fixed-point flow; bit-exact vs `golden/grid_sample_ref.py`
- [x] `blend_unit` — mask blend `(m·a + (255−m)·b + 127) >> 8`; bit-exact vs golden
- [x] `vfi_synth` — top-level: 2× warp + blend per channel, channel-interleaved capture/serialized output; bit-exact vs `golden/vfi_synth_ref.py`
- [x] `tools/vfi_demo.py` — user t/t+1 frames → t+0.5 interpolated frame; ONNX FP16 reference or RTL INT8 path (verified bit-exact, PSNR 54.7 dB vs FP16)
- [x] Synthesized (generic gate count, no PDK this run):

| Module | Cells | Regs | Notes |
|---|---|---|---|
| `warp_unit` | 22,791 | 2,072 | 16×16×8 frame buffer dominates |
| `blend_unit` | 793 | 9 | 1-cycle mask blend |
| `vfi_synth` (NCH=3) | 136,452 | 12,488 | 6× warp + 3× blend |

## Known issues (Sprint 3 debt — all RESOLVED)
These suites were committed in Sprint 3 without ever passing the aggregate
regression. All four have since been fixed (root causes below) and are now
part of the passing `run_all_tests.sh` gate:

| Suite | Root cause fixed |
|---|---|
| `test_ds_conv_layer_integrated` | block-scoped `integer x = expr` inits → X weight regs (iverilog -g2012); DW/PW weight-address collision; golden needed 8-bit bias quantization; `dw_done` waited only on channel 0 so PW read unwritten `dw_buf` |
| `test_encoder_slice` | E1/E2 shared the wl bus with colliding address spaces (added `WL_BASE`); missing E2 stride-2 filter; block-scoped-integer X bug; slow serial PW at C_OUT=96 |
| `test_line_buffer_stream` | 2-line FIFO couldn't form a SAME-padded 3×3 window — rewrote with 3 cyclic line buffers + independent output counter |
| `test_pw_conv1x1_parallel` | `S_OUTPUT=2'd4` truncated to 0 (collided with S_IDLE); block-scoped-integer X in MAC; per-position handshake in test |

`./run_all_tests.sh` runs the full verified suite (Sprint 0–4).

## Definition of done
A repo with: verified RTL for a weight-reloadable INT8 depthwise-separable layer, cocotb tests proving bit-exactness, an OpenROAD/Sky130 area/power/fmax report, one GPU-comparison plot, and the writeup. Defensible because every number is backed by a passing bit-exact test.
