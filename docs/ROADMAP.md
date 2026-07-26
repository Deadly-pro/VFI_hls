# Roadmap — VFI-DL Accelerator in Verilog

Sprints, risk-ordered. Tight runway (placement ~now, Nvidia ~4–6 wks after). The
minimum defensible artifact is the end of **Sprint 2**. See `ARCHITECTURE.md` for the
locked definition and module hierarchy.

## Sprint 0 — Toolchain de-risk (GATE, ~1 day)
Prove the WHOLE open flow on something trivial before real RTL rides on it (we lost 2
days to env pain once — OpenROAD setup is the new landmine).
- [ ] Install: Verilator, Icarus, Yosys, cocotb, numpy/matplotlib; **OpenROAD-flow-scripts**.
- [ ] Push `mac_int8` end-to-end: RTL → cocotb (bit-exact vs numpy) → Yosys+OpenROAD →
      **one area/power/fmax number out.** ← this is the gate.

## Sprint 1 — Primitives (~2–3 half-days)
- [ ] `weight_mem` — reloadable weight store + load port; verify load-then-read.
- [ ] `line_buffer` — sliding 3×3 window; verify window extraction vs numpy.
- [ ] `requantize` — INT32→INT8 (M0, shift, clamp); verify vs TFLite-style golden.

## Sprint 2 — First full layer (~2–4 half-days) ← MINIMUM DEFENSIBLE ARTIFACT
- [ ] `dw_conv3x3` — verify vs numpy depthwise.
- [ ] `pw_conv1x1` — verify vs numpy pointwise.
- [ ] `ds_conv_layer` — wire dw→pw→requant + weight_mem; verify end-to-end; **OpenROAD it**
      for area/power/fmax. Demonstrate weight reload (new weights, no resynth).

## Sprint 3 — Scale + deliverable (stretch)
- [ ] Parameterize channels/dims; stack ≥2 layers (mini encoder slice).
- [ ] GPU-comparison plot: energy/frame + latency determinism vs paper's RTX 3050 numbers.
- [ ] 1–2 page writeup: what was built, verification, silicon numbers, the honest GPU
      comparison, and future work (warp unit, weight-reload as field-update, NPU direction).

## Definition of done
A repo with: verified RTL for a weight-reloadable INT8 depthwise-separable layer, cocotb
tests proving bit-exactness, an OpenROAD area/power/fmax report, one GPU-comparison plot,
and the writeup. Defensible because every number is backed by a passing bit-exact test.
