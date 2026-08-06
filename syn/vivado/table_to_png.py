#!/usr/bin/env python3
"""table_to_png.py — render the Vivado results summary as a PNG table.

Usage: python3 syn/vivado/table_to_png.py [out.png]
Output defaults to reports/plots/vivado_results.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_vivado import parse_util, parse_wns, RPT_DIR  # noqa: E402

import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(RPT_DIR), "..", "..", "reports", "plots", "vivado_results.png")

MODULE_ORDER = ["mac_int8", "weight_mem", "requantize", "line_buffer",
                "line_buffer_stream", "dw_conv3x3", "pw_conv1x1",
                "pw_conv1x1_parallel", "ds_conv_layer", "ds_conv_layer_integrated",
                "encoder_slice", "warp_unit", "blend_unit", "vfi_synth"]


def main():
    rows = []
    for top in MODULE_ORDER:
        util_path = os.path.join(RPT_DIR, f"{top}_util.rpt")
        if not os.path.exists(util_path):
            rows.append([top, "-", "-", "-", "-", "-", "-", "-"])
            continue
        util = parse_util(util_path)
        wns, fmax = parse_wns(os.path.join(RPT_DIR, f"{top}_vivado.log"))
        rows.append([
            top,
            util.get("lut", "-"),
            util.get("lut_mem", "-"),
            util.get("ff", "-"),
            util.get("dsp", "-"),
            util.get("bram_tiles", "-"),
            f"{wns:.2f}" if wns is not None else "-",
            f"{fmax:.1f}" if fmax else "-",
        ])

    col_labels = ["Module", "LUT", "LUT mem", "FF", "DSP", "BRAM tiles",
                  "WNS (ns)", "Fmax (MHz)"]
    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.42 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f4e79")
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("VFI-DL on Artix-7 XC7A100T — Vivado 2026.1, 100 MHz target",
                 fontsize=13, pad=12)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
