#!/usr/bin/env python3
"""
GPU Comparison Plot Generator for VFI-DL ASIC vs RTX 3050 Laptop

Generates two plots for the technical report:
1. latency_determinism.png - ASIC fixed cycles vs GPU jitter
2. energy_per_frame.png - ASIC energy vs GPU power across resolutions

Uses JCSSE paper data (Table IV) and our ASIC measurements.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ---- ASIC Measured Data (Sky130 @ 100 MHz) ----
ASIC_FREQ_MHZ = 104.7  # Measured Fmax
ASIC_POWER_MW = 2.71   # ds_conv_layer_integrated @ 100 MHz
ASIC_CYCLES_PER_FRAME_8x8 = 8 * 8 * 4 * 4  # Approximate: positions * C_OUT
# For 8x8x4->4: DW phase ~64*4 + PW phase ~64*4*4 = 1280 cycles
# For encoder slice (E1: 8x8x6->48, E2: 4x4x48->96):
# E1: 64*48 + 64*48*6 = 3072 + 18432 = 21504 cycles
# E2: 16*96 + 16*96*48 = 1536 + 73728 = 75264 cycles
# Total encoder: ~96,768 cycles @ 104.7 MHz = 0.924 ms

ASIC_LATENCY_MS_8x8 = ASIC_CYCLES_PER_FRAME_8x8 / (ASIC_FREQ_MHZ * 1e6) * 1000
ASIC_LATENCY_MS_ENCODER = 96768 / (ASIC_FREQ_MHZ * 1e6) * 1000
ASIC_ENERGY_UJ_PER_FRAME = ASIC_POWER_MW * ASIC_LATENCY_MS_ENCODER  # mW * ms = uJ

# ---- RTX 3050 Laptop GPU Data (from JCSSE Table IV) ----
# Model-only latency, FP16 TensorRT, CUDA Graphs
GPU_DATA = {
    '720p':    {'latency_ms': 4.48, 'fps': 223.2, 'vram_mb': 81},
    '900p':    {'latency_ms': 7.38, 'fps': 135.5, 'vram_mb': 121},
    '1080p80': {'latency_ms': 6.58, 'fps': 152.0, 'vram_mb': 116},
    '1080p':   {'latency_ms': 9.89, 'fps': 101.1, 'vram_mb': 177},
    '1440p':   {'latency_ms': 17.84, 'fps': 56.0,  'vram_mb': 308},
}

# RTX 3050 Laptop TGP = 35-50W (assume 40W typical for GPU compute)
GPU_POWER_W = 40.0  # Watts, GPU-only during inference
GPU_ENERGY_UJ = {k: GPU_POWER_W * v['latency_ms'] * 1000 for k, v in GPU_DATA.items()}  # uJ

# GPU latency jitter (from paper: mean ± std)
GPU_JITTER_MS = {
    '720p': 0.12,
    '900p': 0.25,
    '1080p80': 0.21,
    '1080p': 0.45,
    '1440p': 0.82,
}

# ASIC deterministic latency (zero jitter)
ASIC_JITTER_MS = 0.0

# Resolution mapping for encoder slice (our accelerator does encoder backbone only)
# Our encoder: 8x8 input -> 2x2 output (2 stride-2 layers)
# Paper encoder: 1920x1080 -> ... (multiple layers)
# We'll compare at equivalent "encoder" workload scale

def generate_latency_determinism_plot():
    """Plot: ASIC zero-jitter vs GPU jitter across resolutions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    resolutions = list(GPU_DATA.keys())
    gpu_lat = [GPU_DATA[r]['latency_ms'] for r in resolutions]
    gpu_jitter = [GPU_JITTER_MS[r] for r in resolutions]

    # Plot 1: Latency with error bars
    ax1.errorbar(range(len(resolutions)), gpu_lat, yerr=gpu_jitter,
                 fmt='o-', capsize=5, label='RTX 3050 (FP16 TensorRT)', color='#E67E22')
    ax1.axhline(y=ASIC_LATENCY_MS_ENCODER, color='#27AE60', linestyle='--', linewidth=2,
                label=f'VFI-DL ASIC (Sky130, INT8)\n{ASIC_LATENCY_MS_ENCODER:.3f} ms (deterministic)')
    ax1.set_xticks(range(len(resolutions)))
    ax1.set_xticklabels(resolutions, rotation=15)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Inference Latency: ASIC vs GPU\n(Encoder Backbone Only)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Jitter comparison (log scale)
    ax2.bar(range(len(resolutions)), gpu_jitter, width=0.6, color='#E67E22', alpha=0.7,
            label='RTX 3050 (±1σ)')
    ax2.bar([len(resolutions)], [ASIC_JITTER_MS], width=0.6, color='#27AE60', alpha=0.7,
            label='VFI-DL ASIC (deterministic)')
    ax2.set_xticks(list(range(len(resolutions))) + [len(resolutions)])
    ax2.set_xticklabels(resolutions + ['ASIC\n(INT8)'], rotation=15)
    ax2.set_ylabel('Latency Std Dev (ms)')
    ax2.set_yscale('log')
    ax2.set_title('Latency Determinism\n(Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('/home/deadly-pro/VFI_hls/reports/plots/latency_determinism.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved latency_determinism.png")


def generate_energy_per_frame_plot():
    """Plot: Energy per frame comparison across resolutions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    resolutions = list(GPU_DATA.keys())
    gpu_energy = [GPU_ENERGY_UJ[r] / 1000 for r in resolutions]  # mJ
    asic_energy = ASIC_ENERGY_UJ_PER_FRAME / 1000  # mJ (constant for our fixed encoder)

    x = np.arange(len(resolutions))
    width = 0.35

    bars1 = ax.bar(x - width/2, gpu_energy, width, label='RTX 3050 (FP16, full pipeline)',
                   color='#E67E22', alpha=0.8)
    bars2 = ax.bar(x + width/2, [asic_energy]*len(resolutions), width,
                   label=f'VFI-DL ASIC (INT8, encoder only)\n{asic_energy:.2f} mJ/frame',
                   color='#27AE60', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(resolutions)
    ax.set_ylabel('Energy per Frame (mJ)')
    ax.set_title('Energy per Frame: ASIC vs GPU\n(Encoder: 8x8x6→2x2x96 @ 104.7 MHz Sky130)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('/home/deadly-pro/VFI_hls/reports/plots/energy_per_frame.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved energy_per_frame.png")


def generate_area_breakdown_plot():
    """Plot: ASIC area breakdown by module."""
    fig, ax = plt.subplots(figsize=(10, 6))

    modules = ['mac_int8', 'weight_mem', 'requantize', 'line_buffer',
               'dw_conv3x3', 'pw_conv1x1', 'ds_conv_layer', 'ds_conv_layer_integrated']
    areas = [741, 7651, 3383, 3669, 9473, 6057, 40119, 55421]  # µm²

    # For integrated, show sub-breakdown
    colors = ['#3498DB', '#3498DB', '#3498DB', '#3498DB',
              '#E67E22', '#E67E22', '#27AE60', '#27AE60']

    bars = ax.barh(modules, areas, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Area (µm²)')
    ax.set_title('Sky130 ASIC Area Breakdown (ds_conv_layer_integrated = 55,421 µm²)')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, area in zip(bars, areas):
        ax.annotate(f'{area:,}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0), textcoords='offset points', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('/home/deadly-pro/VFI_hls/reports/plots/area_breakdown.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved area_breakdown.png")


if __name__ == '__main__':
    os.makedirs('/home/deadly-pro/VFI_hls/reports/plots', exist_ok=True)
    generate_latency_determinism_plot()
    generate_energy_per_frame_plot()
    generate_area_breakdown_plot()
    print("\nAll plots generated in reports/plots/")