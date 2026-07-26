# Prior Art — hls4ml-vfi

Compiled to position the project and to seed the writeup's references. Bottom line:
**no identical project exists** — the combination (SR CNN + hls4ml-generated datapath +
Artix-7 + synthesis-report deliverable) appears novel. All academic SR-on-FPGA is
hand-written RTL/HLS; all portfolio-level hls4ml work is tiny classifiers.

## Closest analog (STUDY THIS)
- **AMIQ Consulting — FSRCNN on Vivado HLS.** Hand-written HLS (not hls4ml), Zynq/PYNQ-Z1.
  FSRCNN ~4036 params, `ap_fixed<24,9>`, 64×90 mono. Fit: **BRAM 98%, DSP 68%, LUT 86%,
  FF 34%**; ~41× vs FSRCNN-CPU; "small ΔPSNR" fixed vs float. Our contrast line:
  *hls4ml-generated vs hand HLS.*
  - https://www.consulting.amiq.com/2021/08/23/how-to-accelerate-an-image-upscaling-cnn-on-fpga-using-hls/
  - https://github.com/amiq-consulting/image-upscaling-CNN

## hls4ml capability + the key risk
- **Fast CNNs on FPGAs with hls4ml** (Aarrestad et al. 2021) — introduces `io_stream` conv;
  large CNNs *require* it; conv layers dominate resources; fit small parts via pruning +
  quantization + high `ReuseFactor`. arXiv:2101.05108 ;
  https://iopscience.iop.org/article/10.1088/2632-2153/ac0ea1
- **RISK: no native pixel-shuffle / sub-pixel / depth-to-space layer in hls4ml.**
  `Conv2DTranspose` and `UpSampling2D` ARE supported. ⇒ prefer **FSRCNN transpose-conv**
  over ESPCN pixel-shuffle, OR prove pixel-shuffle in the de-risk phase first.
  hls4ml docs https://fastmachinelearning.org/hls4ml/ ; UpSampling2D issue #517 ;
  layer-support issue #38 ; hls4ml tutorial Part 6 (CNNs)
  https://fastmachinelearning.org/hls4ml-tutorial/part6_cnns.html

## Other FPGA SR accelerators (all hand-RTL/HLS; DSP is the usual bottleneck)
- **Light-FSRCNN deconv accelerator** — Kintex-7 410T, DSPs fully used, deconv = 82–95% of
  compute. Confirms upsample stage dominates. arXiv:1801.05997
- **PKU real-time SR (UHD)** — ZC706, DSP 95%, BRAM 30%. Big-device reference table.
  https://ceca.pku.edu.cn/docs/20181030012447954751.pdf
- **Kim/Choi TCSVT** — 2K→4K@60fps dedicated HW (upper bound). doi 8429522
- **CUHK TODAES 2024** — edge-FPGA SR, resource alloc as constraint problem (most rigorous
  recent RTL prior art). https://dl.acm.org/doi/10.1145/3652855
- ASIC point: **SRNPU** (first CNN-SR ASIC, real-time Full-HD).

## "Hardware DLSS" landscape (frame as ANALOGY, not equivalence)
- DLSS = NN on Tensor Cores (DLSS 4 = Transformer + Multi-Frame Gen); XeSS = XMX/DP4a;
  FSR = algorithmic→AI (FSR4). All proprietary, GPU-resident. **Nobody ships DLSS on FPGA.**
  Our framing: "an open FPGA datapath for the neural-upscaling step GPUs do in fixed silicon."
- Temporal VFI needs dedicated optical-flow HW even on GPUs (NVIDIA Optical Flow Accelerator)
  ⇒ validates scoping to the feed-forward spatial upsampler for hls4ml.

## Model + flow reference repos
- Model: `suxrobGM/fsrcnn` (SW FSRCNN) ; `leftthomas/ESPCN`.
- hls4ml→Vitis flow/report presentation: `17jlee/fpga-enabled_ml-accelerator` ;
  hls4ml tutorial Part 6.

*Accuracy notes: PSNR figures largely unverified across these sources; pixel-shuffle
support must be verified hands-on against our exact hls4ml version.*
