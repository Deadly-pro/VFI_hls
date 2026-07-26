# VFI-DL — A Deep-Learning Inference Datapath in Verilog

A **weight-reloadable, fixed-architecture INT8 CNN inference datapath**, hand-written in
Verilog RTL, verified bit-exact in simulation, and taken through the open-source
**Yosys + OpenROAD** ASIC flow for real **area / power / fmax** — then benchmarked against
a GPU on **energy-per-frame and latency determinism**.

> The datapath implements the compute-dominant CNN backbone of my published VFI model
> ("Open-Frame-gen / Nano"). No FPGA, no hls4ml — hand-written RTL, open silicon flow.

This is the fixed-function-hardware leg of a frame-generation stack I built end-to-end:
- **Algorithm** — "Open-Frame-gen" (JCSSE 2026).
- **SIMT GPU** in RTL to run it — the Omni-RISC APU (separate repo).
- **Fixed-function DL inference datapath** in RTL — *this repo*.

## Scope (locked — see `docs/ARCHITECTURE.md`)
- **Accelerate the CNN backbone**, not full temporal VFI. The `grid_sample` warp is a
  data-dependent gather and stays out of the RTL.
- **INT8 fixed-point**, symmetric per-tensor. Float is out.
- **Deliverable = verified RTL + OpenROAD area/power/fmax + one GPU-comparison plot + a
  1–2 page writeup.** Every number is backed by a passing bit-exact test.

## Toolchain (all open, runs on Fedora)
| Stage | Tool |
|-------|------|
| Golden model | numpy (`golden/`) |
| RTL | hand-written Verilog (`rtl/`) |
| Verification | cocotb + Verilator/Icarus (`tb/`) |
| Synthesis + P&R | Yosys + OpenROAD-flow-scripts (`syn/`) |
| GPU comparison | Python + matplotlib (`benchmarks/`) |

## Repo layout
| Path | Purpose |
|------|---------|
| `rtl/`        | Verilog RTL (Prajwal writes) |
| `golden/`     | numpy bit-exact reference models (Claude) |
| `tb/`         | cocotb testbenches (Claude) |
| `syn/`        | Yosys/OpenROAD flow config + scripts (Claude) |
| `benchmarks/` | GPU energy/latency comparison + plot (Claude) |
| `reports/`    | Extracted area/power/fmax numbers, plots, writeup |
| `docs/`       | `ARCHITECTURE.md` (locked def), `ROADMAP.md`, `PRIOR_ART.md` |
| `CONTEXT.md`  | Project handoff / working contract |

## Status
See `docs/ROADMAP.md`. Minimum defensible artifact = end of Sprint 2 (a verified,
synthesized, weight-reloadable depthwise-separable INT8 layer).

## Workflow transparency
Microarchitecture, RTL, and design tradeoffs are self-designed. AI assistance (Claude Code)
builds the verification harness (golden models, cocotb tests), the OpenROAD flow scripts,
and the comparison plot, and runs/verifies everything. Assessed low/no hiring risk; noted for honesty.
