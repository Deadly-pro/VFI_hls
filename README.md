# hls4ml-vfi — A Fixed-Function FPGA Datapath for Neural Upscaling ("Hardware DLSS block")

Mapping the **upsampling stage of video frame generation** to an FPGA datapath with
[hls4ml](https://fastmachinelearning.org/hls4ml/), then reporting real synthesis
numbers and a CPU vs GPU vs FPGA performance comparison.

> **What this is (the one-sentence framing):** I took a frame-generation network from my
> published work, mapped its upsampling core to an FPGA datapath with hls4ml, and
> produced real Vitis HLS synthesis numbers plus a CPU/GPU/FPGA performance comparison.

This is the fixed-function-hardware leg of a three-part frame-generation stack:
- **Algorithm** — "Open-Frame-gen" (JCSSE 2026).
- **SIMT GPU** to run it — the Omni-RISC APU (RV32IM core, separate repo).
- **Fixed-function FPGA datapath** to accelerate the upscaling — *this repo*.

## Scope (deliberately tight — see `CONTEXT.md` §4)
- **Synthesize the UPSCALER, not full temporal VFI.** A tiny ESPCN / FSRCNN-small
  sub-pixel ×2 upscaler (single-channel). Temporal interpolation is the *motivation*;
  the upscaler is what actually goes through hls4ml. Optical-flow/warping chokes hls4ml.
- **Deliverable = a report with real numbers + one comparison plot, not a live demo.**
  Quantized model → hls4ml → Vitis HLS csynth → latency / II / DSP-BRAM-LUT-FF →
  CPU/GPU/FPGA comparison plot. Defensible at ~30% built.

## Target
- **FPGA:** Xilinx Artix-7 **XC7A100T** (~240 DSP48E1, ~135 BRAM) — same board as the APU.
- **Tooling:** Vitis/Vivado HLS 2023.x; Python stack (TF/QKeras/hls4ml) on **Python 3.10/3.11**.

## Repo layout
| Path | Purpose |
|------|---------|
| `models/`     | Keras/QKeras model definitions + training |
| `hls/`        | hls4ml conversion scripts (generated `*_prj/` trees are gitignored) |
| `benchmarks/` | CPU/GPU/FPGA comparison harness + plotting |
| `reports/`    | Extracted csynth numbers, plots, writeup |
| `docs/`       | Roadmap, design notes |
| `CONTEXT.md`  | Full project handoff / working contract |

## Status — minimal deliverable checklist
- [ ] Toolchain de-risk: 2-layer QKeras dense net → hls4ml → csynth report exists.
- [ ] Tiny upsampler (ESPCN/FSRCNN-small, ×2, single-channel) trained + quantized.
- [ ] hls4ml conversion clean; csynth fits XC7A100T (DSP/BRAM within budget).
- [ ] Numbers captured: latency (cycles/ns), II, resource utilization.
- [ ] Baselines: CPU + GPU inference latency/throughput.
- [ ] One comparison plot (latency/throughput, ideally perf/W) across CPU/GPU/FPGA.
- [ ] Short writeup (1–2 pp): motivation, what was synthesized, numbers, tradeoffs, future work.

## Workflow transparency
The model architecture, quantization choices, and design tradeoffs are self-designed.
AI assistance (Claude Code) is used for the benchmarking/verification harness and for
sanity-checking the hls4ml conversion. Assessed as low/no hiring risk; noted here for honesty.
