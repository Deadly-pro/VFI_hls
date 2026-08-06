#!/usr/bin/env python3
"""summarize_vivado.py — Parse syn/vivado/reports/*_util.rpt + *_vivado.log
into a LUT/FF/DSP/BRAM + WNS/Fmax table for all VFI-DL modules.

Usage:  python3 syn/vivado/summarize_vivado.py
"""
import os
import re
import sys
import glob

RPT_DIR = os.path.join(os.path.dirname(__file__), "reports")

SITE_ROWS = {
    "Slice LUTs": "lut",
    "LUT as Memory": "lut_mem",
    "Slice Registers": "ff",
    "DSPs": "dsp",
    "RAMB36/FIFO": "ramb36",
    "RAMB18": "ramb18",
}


def parse_util(filepath):
    """Return dict of site -> used count from a report_utilization rpt."""
    out = {}
    with open(filepath) as f:
        for line in f:
            m = re.match(r"\|\s*([^|]+?)\s+\|\s*(\d+)\s+\|", line)
            if not m:
                continue
            site = m.group(1).strip().rstrip("*")
            if site in SITE_ROWS:
                out[SITE_ROWS[site]] = int(m.group(2))
    out["bram_tiles"] = out.get("ramb36", 0) + 0.5 * out.get("ramb18", 0)
    return out


def parse_wns(log_path):
    """Return (wns_ns, fmax_mhz) from the WNS_RESULT line in the vivado log."""
    if not os.path.exists(log_path):
        return None, None
    for line in open(log_path):
        m = re.search(r"WNS_RESULT\s+\S+\s+([+-]?[\d.]+)", line)
        if m:
            wns = float(m.group(1))
            fmax = 1000.0 / (10.0 - wns)  # 100 MHz target (10 ns period)
            return wns, fmax
    return None, None


def main():
    util_files = sorted(glob.glob(os.path.join(RPT_DIR, "*_util.rpt")))
    if not util_files:
        print(f"No utilization reports in {RPT_DIR}")
        sys.exit(1)

    rows = []
    for uf in util_files:
        top = os.path.basename(uf).replace("_util.rpt", "")
        util = parse_util(uf)
        wns, fmax = parse_wns(os.path.join(RPT_DIR, f"{top}_vivado.log"))
        rows.append((top, util, wns, fmax))

    hdr = f"{'module':28s} {'LUT':>6s} {'LUTmem':>7s} {'FF':>6s} {'DSP':>4s} {'BRAMt':>6s} {'WNS(ns)':>9s} {'Fmax(MHz)':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for top, util, wns, fmax in rows:
        bram = util.get("bram_tiles")
        bram_s = "-" if not bram else f"{bram:.1f}"
        print(f"{top:28s} {util.get('lut', '-'):>6} {util.get('lut_mem', '-'):>7} "
              f"{util.get('ff', '-'):>6} {util.get('dsp', '-'):>4} {bram_s:>6} "
              f"{wns if wns is not None else '-':>9} {f'{fmax:.1f}' if fmax else '-':>9}")


if __name__ == "__main__":
    main()
