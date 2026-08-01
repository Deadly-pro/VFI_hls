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
- [ ] GPU-comparison plot: energy/frame + latency determinism vs paper's RTX 3050 numbers.
- [ ] 1–2 page writeup: what was built, verification, silicon numbers, the honest GPU comparison, and future work (warp unit, weight-reload as field-update, NPU direction).

## Definition of done
A repo with: verified RTL for a weight-reloadable INT8 depthwise-separable layer, cocotb tests proving bit-exactness, an OpenROAD/Sky130 area/power/fmax report, one GPU-comparison plot, and the writeup. Defensible because every number is backed by a passing bit-exact test.
