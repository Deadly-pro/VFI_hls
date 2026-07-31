#!/usr/bin/env python3
# syn/scripts/summarize_ppa.py - Parse Yosys stat reports and print PPA table

import os
import re
import glob

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

def parse_stat_file(filepath):
    """Parse Yosys stat output for cell count, area, etc."""
    with open(filepath, 'r') as f:
        content = f.read()

    result = {
        'module': os.path.basename(filepath).replace('_stat.txt', ''),
        'cells': 0,
        'area_um2': 0.0,
        'wire_bits': 0,
        'public_wires': 0,
        'memories': 0,
        'memory_bits': 0,
        'processes': 0,
    }

    # Parse cell counts
    cell_matches = re.findall(r'(\w+)\s+(\d+)', content)
    for cell_type, count in cell_matches:
        if cell_type in ('sky130_fd_sc_hd__', '_', 'cells'):  # skip headers
            continue
        try:
            result['cells'] += int(count)
        except ValueError:
            pass

    # Try to find total cells line
    total_match = re.search(r'Number of cells:\s+(\d+)', content)
    if total_match:
        result['cells'] = int(total_match.group(1))

    # Try to find area (Sky130 reports area in µm²)
    area_match = re.search(r'Chip area for module.*?:\s+([\d.]+)', content)
    if area_match:
        result['area_um2'] = float(area_match.group(1))
    else:
        # Estimate area from cell count (rough Sky130 std cell ~6.3 µm² avg)
        result['area_um2'] = result['cells'] * 6.3

    return result


def main():
    stat_files = glob.glob(os.path.join(REPORTS_DIR, "*_stat.txt"))
    if not stat_files:
        print("No stat reports found in", REPORTS_DIR)
        return

    modules = []
    for f in stat_files:
        modules.append(parse_stat_file(f))

    # Sort by module name
    modules.sort(key=lambda x: x['module'])

    # Print table
    print(f"{'Module':<30} {'Cells':>8} {'Area (µm²)':>14} {'Est. Fmax (MHz)':>16} {'Est. Power (mW)':>16}")
    print("-" * 90)

    for m in modules:
        # Rough Fmax estimate based on module complexity
        if m['cells'] < 200:
            fmax = 250
        elif m['cells'] < 1000:
            fmax = 180
        elif m['cells'] < 3000:
            fmax = 130
        else:
            fmax = 100

        # Rough power estimate: ~0.1 µW/MHz/cell at 1.8V Sky130
        power_mw = m['cells'] * 0.1 * 100 / 1000  # at 100 MHz

        print(f"{m['module']:<30} {m['cells']:>8} {m['area_um2']:>14.2f} {fmax:>16} {power_mw:>16.2f}")

    # Total row
    total_cells = sum(m['cells'] for m in modules)
    total_area = sum(m['area_um2'] for m in modules)
    print("-" * 90)
    print(f"{'TOTAL':<30} {total_cells:>8} {total_area:>14.2f}")

if __name__ == "__main__":
    main()