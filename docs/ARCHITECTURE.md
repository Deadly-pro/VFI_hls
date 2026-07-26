# Architecture — VFI-DL Accelerator in Verilog (LOCKED PROJECT DEFINITION)

## One-sentence definition
A **weight-reloadable, fixed-architecture INT8 deep-learning inference datapath**,
hand-written in Verilog RTL, functionally verified bit-exact against a Python golden
model in simulation, and pushed through the open-source **Yosys + OpenROAD** ASIC flow
for real **area / power / fmax** — benchmarked against a GPU running the same workload
on **energy-per-frame and latency determinism** (not raw watts). No FPGA. No hls4ml.

The datapath implements the compute-dominant **CNN backbone** of the "Open-Frame-gen /
Nano" VFI model (depthwise-separable encoder-decoder). The `grid_sample` warp is **out
of scope** (data-dependent gather — belongs in software / a separate late unit).

## The five locked conditions
1. **Scope-from-a-core:** first artifact is ONE verified depthwise-separable conv layer,
   not the whole net. Scale only after it is verified + synthesized.
2. **Bit-exact verification before any number is trusted.** Every RTL block matches a
   numpy golden model exactly; failures reported with the real mismatch, reruns mandatory.
3. **INT8 fixed-point** numeric format (per-tensor symmetric to start). Float is out.
4. **GPU comparison framed on energy/op + latency determinism, with process node stated.**
   No silent 130 nm-PDK-vs-8 nm-GPU raw-watt comparison.
5. **Warp stays out of the RTL.** We accelerate the CNN backbone (the FLOPs).

## Numeric format
- Weights & activations: **INT8, symmetric per-tensor** (scale `S`, zero-point 0).
- Accumulator: **INT32**.
- Requantize between layers: `acc32 → round(acc32 * M0 * 2^-n) → clamp[-128,127] → int8`,
  TFLite-style fixed-point multiplier `(M0, n)`. The **numpy golden model defines the exact
  rounding** so RTL must match bit-for-bit.

## Module hierarchy (build + verify bottom-up, in THIS order)
| # | Module | What it does | Interfaces |
|---|--------|--------------|------------|
| 1 | `mac_int8`     | INT8×INT8→INT32 multiply-accumulate (the atom) | data in, acc out |
| 2 | `weight_mem`   | on-chip weight store **with a load port** (the reloadable core) | `wl_addr/wl_data/wl_we` write; read port |
| 3 | `line_buffer`  | holds N rows, emits sliding 3×3 windows | pixel stream in, window out |
| 4 | `requantize`   | INT32 acc → INT8 (M0, shift, clamp) | acc in, int8 out |
| 5 | `dw_conv3x3`   | depthwise 3×3 (per-channel) = line_buffer + 9 MACs + acc + requant | stream in/out |
| 6 | `pw_conv1x1`   | pointwise 1×1 (channel mixing) = MAC array over channels | stream in/out |
| 7 | `ds_conv_layer`| depthwise-separable layer = dw → pw → requant + weight_mem. **First full artifact.** | stream in/out + weight-load + cfg |

Streaming uses simple `valid`/`ready` handshake (AXI-Stream-lite); weight load is a plain
`addr/data/we` port (wrap as AXI-Lite later). Config: `start`, layer dims.

## Repository components
- `rtl/` — the Verilog datapath: microarchitecture, dataflow (weight-stationary vs
  output-stationary), and quantization design.
- `golden/` — numpy bit-exact reference model per module.
- `tb/` — cocotb testbench per module, with test-vector generation.
- `syn/` — Yosys/OpenROAD flow config + scripts.
- `benchmarks/` — GPU-comparison harness + plot.

Every module is verified against its golden model before any synthesis number is trusted.

## Verification approach
- Golden model: numpy, quantized, bit-exact reference (`golden/`).
- Testbench: **cocotb** driving the DUT via Verilator/Icarus, comparing to golden vectors
  in the same Python process. Bit-mismatch = fail, printed with indices + values.

## Synthesis / silicon numbers
- **Yosys** synth → **OpenROAD** (OpenROAD-flow-scripts) place-and-route on an open PDK
  (start Sky130; consider ASAP7 7 nm for a fairer GPU node comparison).
- Extract: cell area (µm²), power (mW, OpenSTA), fmax (MHz). Per module + full layer.

## GPU comparison methodology (honest framing)
- Baseline: Nano GPU numbers from the paper (RTX 3050, Samsung 8 nm).
- Compare on **energy-per-frame / energy-per-MAC (architectural)** and **latency
  determinism** — where fixed-function beats a general-purpose GPU independent of node
  (no instruction fetch / warp scheduler / cache tax). Any absolute-watt claim states the
  process-node gap explicitly.

## Explicitly out of scope
- `grid_sample` warp+blend.  - Float datapaths.  - Full 1080p in one pass (tile/scale).
- Reconfigurable *architecture* (that's an NPU — the "where this goes next" answer).

## The cross-project narrative
"I design DL compute across the whole stack as RTL I wrote myself — the algorithm
(published paper), a SIMT GPU (the APU, separate), and this fixed-function DL inference
datapath — the hardware taken through an open silicon flow to real area/power/fmax."
