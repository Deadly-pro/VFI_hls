# hls4ml + VFI Accelerator — Project Handoff Context

> **Purpose of this file:** seed context for a fresh Claude Code session in the new
> `hls4ml-vfi/` project folder. Move this file into that folder (rename to
> `CONTEXT.md` or fold into a `CLAUDE.md`) and load it at session start. It is
> self-contained — the next session should NOT need the APU repo to understand
> what to build or how to work.

---

## 1. Who this is for

**Prajwal R** — PES University, ECE, Class of 2027. Targeting an **NVIDIA Bangalore
internship** via campus placement. Published author (JCSSE 2026, sole presenter,
frame-generation paper "Open-Frame-gen"). Builds hardware/ML systems solo.

**Career target he stated explicitly:** paid work with room to investigate his own
ideas, prove concepts empirically, and push conclusions forward inside a company —
not executing a fixed spec. Points at **Deep Learning Performance Architect**-type
roles (architecture + benchmarking + emerging trends) over pure verification.
Guidance: chase *teams* with that culture, not job titles.

### How to work with him (working contract — carry this over)
- **Blunt, specific, severity-ordered. No hedged advice, no option-surveys.** Give a
  recommendation, not a menu. He wants a thinking partner, not an answer machine.
- **He writes the core work himself** (on the APU: all RTL; here: the model,
  quantization, and design decisions). Claude's job is to **review, teach the
  generalizable rule behind each issue, build the verification/benchmark harness,
  and VERIFY by running things — never take "it passes" on trust, always rerun.**
- **Trust-but-verify is a hard rule.** Run the code, read the actual output, report
  failures with the real error text.
- Commits: end messages with a `Co-Authored-By: Claude ...` trailer (he wants it).
  Push at the end of every working session (hard-won lesson after a crash lost work).
- Sudo / interactive logins go in HIS terminal (background sessions can't prompt).

---

## 2. Placement urgency (why scope is tight)

- PESU placements for Class of 2027 **may start ~July 27, 2026** (earlier than the
  previously-assumed September). Nvidia specifically may still be **4–6 weeks** after
  season opens — unconfirmed. Watch departmental mailers daily.
- **Mindset that governs every scope call: projects don't need to be complete, they
  need to be deeply understood.** A half-built project explained with total fluency
  beats a finished one that collapses under follow-up questions.

---

## 3. What this project is

A **hardware accelerator for the upsampling stage of video frame generation**,
generated with **hls4ml** and analyzed for FPGA deployment.

**Why it matters (the NVIDIA story):** frame generation + upsampling *is* the
DLSS-style problem. A dedicated hardware datapath for it is the "hardware DLSS block"
concept Prajwal wants to be able to argue. This project is the artifact that sits
exactly on the DL-Performance-Architect narrative.

**The resume/interview value is NOT "I built a working VFI accelerator."** It is:

> "I took a frame-generation network from my published work, mapped it to an FPGA
> datapath with hls4ml, and produced real synthesis numbers plus a CPU/GPU/FPGA
> performance comparison."

That is defensible at 30% built. The deliverable is **a report with real numbers and
one comparison plot**, not a running board.

---

## 4. The two hard scope decisions (NON-NEGOTIABLE given the clock)

1. **Build the UPSAMPLING CNN, not full temporal VFI.** True frame interpolation needs
   optical flow / warping, which hls4ml chokes on — that's a multi-week tool fight.
   hls4ml is happy with small dense nets and modest CNNs (built for low-latency CERN
   inference). Scope the synthesizable core to a **tiny sub-pixel/ESPCN-style or
   FSRCNN-small upscaler** (this is literally the DLSS upscaling operation).
   **VFI/temporal interpolation is the framing/motivation; the upsampler is what you
   actually synthesize.** "Temporal interpolation extends the same datapath pattern"
   is a fine interview sentence.

2. **Deliverable = report with numbers, not a live demo.** No real-time video, no
   SOTA PSNR chase. Quantized model → hls4ml → Vitis HLS synthesis → latency +
   DSP/BRAM/LUT + throughput at a target clock → one roofline/latency plot comparing
   CPU vs GPU (his JCSSE work) vs this FPGA block.

### What NOT to do (time sinks that will eat the runway)
- Don't try to fit a full RIFE/large VFI net through hls4ml.
- Don't chase state-of-the-art image quality.
- Don't rabbit-hole on hls4ml's convolution/`io_stream` resource tuning for weeks.
- Don't build a live inference demo.

---

## 5. Technical plan

### Target
- **FPGA: Xilinx Artix-7 XC7A100T** (same board as the APU — deliberate: keeps a
  coherent "two accelerators on one target" story). Modest part: ~240 DSP48E1,
  ~135 BRAM. A tiny upsampler must be sized to fit — watch DSP/BRAM in csynth.
- Tooling: **Vitis/Vivado HLS 2023.x** (he already runs Vivado 2023.x for the APU).

### Flow
1. **Model** in Keras, ideally **QKeras** (quantization-aware training is the standard
   hls4ml path). Keep it tiny: single-channel/grayscale, small patch, ×2 sub-pixel
   conv upscale (ESPCN) or FSRCNN-small. Reuse the frame-gen intuition from the
   JCSSE paper for the architecture choice.
2. **Convert:** `hls4ml.converters.convert_from_keras_model(...)` → project → run C
   synthesis (`hls_model.build(csim=False, synth=True, ...)`).
3. **Read the csynth report:** latency (cycles + ns at target clock), II, and
   DSP/BRAM/LUT/FF utilization. Tune `ReuseFactor` / precision to fit the part.
4. **Baselines for the comparison:** CPU (numpy/PyTorch on his machine), GPU (a
   PyTorch/CUDA run, ties to the JCSSE work), FPGA (hls4ml csynth latency/throughput,
   optionally estimated energy). Produce **one plot**: latency or throughput (and
   ideally perf/Watt) across CPU/GPU/FPGA.

### DE-RISK FIRST (do this before the real model)
He just lost 2 days to an environment migration; the hls4ml + Vitis HLS install is the
next landmine. **Before touching the VFI model, push a throwaway 2-layer QKeras dense
net all the way through hls4ml → csynth and get one report out.** Half-day de-risk that
proves the toolchain before any real modeling effort rides on it.

### Environment (expect an install tax)
- Python env with **hls4ml, QKeras, TensorFlow/Keras, numpy**. (Note: the APU repo
  pins Python to numpy + pyelftools only — this project needs its own, heavier env.)
- Vitis HLS on PATH for the synthesis step.
- Verify versions early; hls4ml is version-sensitive against the HLS tool.

---

## 6. Division of labor for THIS project
- **Prajwal drives:** model architecture, quantization choices, the design tradeoffs,
  and the interpretation/narrative.
- **Claude builds/verifies:** the benchmarking harness, the CPU/GPU/FPGA comparison
  scaffold and plot, sanity-checks the hls4ml conversion, and runs/reads the reports
  so the "numbers + plot" deliverable is fast to produce. Same trust-but-verify rule.

---

## 7. The cross-project story (make it ONE narrative, not three résumé lines)

Fuse the three artifacts:

> "I work the full frame-generation stack — the **algorithm** (JCSSE paper), a **SIMT
> GPU** to run it (the Omni-RISC APU), and a **fixed-function FPGA datapath** to
> accelerate the upsampling (this hls4ml project)."

**North-star closing line (do NOT commit to building before placement):** drop the
hls4ml-generated block onto the RISC-V SoC's AXI bus as a peripheral — his custom core
driving his custom DL accelerator, both on the same Artix-7. Use it as the "here's
where this goes" answer to a follow-up, and as the reason both projects share a board.

---

## 8. Minimal deliverable checklist (the whole project, scoped)
- [ ] Toolchain de-risk: 2-layer QKeras dense net → hls4ml → csynth report exists.
- [ ] Tiny upsampler model (ESPCN/FSRCNN-small, ×2, single-channel) trained + quantized.
- [ ] hls4ml conversion clean; csynth fits XC7A100T (DSP/BRAM within budget).
- [ ] Numbers captured: latency (cycles/ns), II, resource utilization.
- [ ] Baselines run: CPU + GPU inference latency/throughput.
- [ ] One comparison plot (latency/throughput, ideally perf/W) across CPU/GPU/FPGA.
- [ ] Short writeup (1–2 pages): motivation (VFI/DLSS), what was synthesized (the
      upsampler), the numbers, the tradeoff analysis, and the "extends to temporal
      interpolation + AXI-peripheral-on-my-SoC" future-work paragraph.
- [ ] README note on workflow transparency (model self-designed, AI-assisted
      harness/verification) — assessed low/no hiring risk, include for honesty.

---

## 9. Sequencing relative to the APU (context, not this project's work)
The APU (RV32IM 5-stage core, `Omni-RISC/`) is ~50% done and is the **#1
interview-cold priority**. Plan agreed: **drive the APU to the compliance floor
(finish M-extension → CSR/trap → riscv-arch-test passing), then FREEZE it** — GPU and
coherence are stretch tiers with steep diminishing returns before placement. The
*low-focus* half of this hls4ml project (toolchain de-risk, model picking) can overlap
APU fatigue-time now; the *deep* half inherits prime bandwidth once the APU floor is
hit. Different muscles (Verilog/timing vs Python/quantization), so partial overlap is
fine — but don't run both deep-focus efforts at once.

*(APU status at handoff: mid-M-extension — multiplier + divider built and unit-verified,
divider integration into the pipeline in progress. Not relevant to this project beyond
the shared-board story.)*
