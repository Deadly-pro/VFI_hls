# Roadmap — hls4ml-vfi

Effort is in **focused half-days**, not calendar days — this project runs on APU
fatigue-time until the APU compliance floor is hit, then inherits prime bandwidth
(see `CONTEXT.md` §9). Ordering is by risk, not by the final report's section order.

## Phase 0 — Environment + toolchain de-risk  (GATE, ~1 day)
The whole project is worthless if hls4ml won't talk to Vitis HLS. Prove the pipe first.
- [ ] Build isolated Python **3.10/3.11** env (system python is 3.14 — incompatible).
- [ ] `pip install -r requirements.txt`; verify `import tensorflow, qkeras, hls4ml` clean.
- [ ] Confirm Vitis HLS 2023.x on PATH; check hls4ml↔HLS version compatibility.
- [ ] **De-risk model:** throwaway 2-layer QKeras *dense* net → `convert_from_keras_model`
      → `build(csim=False, synth=True)` → **one csynth report out.** ← this is the gate.

## Phase 1 — The upscaler model  (~2–3 half-days, Prajwal drives)
- [ ] Define tiny ×2 sub-pixel upscaler: ESPCN-style or FSRCNN-small, single-channel.
- [ ] Train on a small SR dataset (grayscale patches; DIV2K/BSD crops or similar).
- [ ] Quantization-aware training in QKeras; record float vs quantized PSNR/SSIM.

## Phase 2 — hls4ml conversion + fit  (~2–4 half-days, the tool-fight risk lives here)
- [ ] Convert QKeras model; `io_stream` for conv; pick initial `ReuseFactor`/precision.
- [ ] csynth; read latency (cycles+ns @ target clock), II, DSP/BRAM/LUT/FF.
- [ ] Tune ReuseFactor/precision until it FITS XC7A100T (~240 DSP, ~135 BRAM). Log tradeoff.

## Phase 3 — Baselines + the one plot  (~1–2 half-days, Claude builds harness)
- [ ] CPU inference latency/throughput (numpy/PyTorch, this machine).
- [ ] GPU inference latency/throughput (PyTorch/CUDA — ties to JCSSE work).
- [ ] FPGA numbers from csynth; estimate throughput & (optionally) perf/W.
- [ ] **One plot:** latency or throughput (ideally perf/W) across CPU/GPU/FPGA.

## Phase 4 — Writeup  (~1 half-day)
- [ ] 1–2 page report: motivation (VFI/DLSS) → what was synthesized (upscaler) →
      numbers → tradeoff analysis → future work (temporal VFI + AXI-peripheral-on-my-SoC).

## Explicit non-goals (runway protection — see `CONTEXT.md` §4)
- No full RIFE/large VFI net through hls4ml.  - No SOTA PSNR chase.
- No multi-week io_stream resource rabbit-hole.  - No live inference demo.

## Definition of done
A tagged repo containing: the model, the hls4ml conversion script, an extracted
numbers table, one CPU/GPU/FPGA plot, and the 1–2 page writeup. Defensible at ~30% built.
