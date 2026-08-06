#!/usr/bin/env bash
# run_vivado.sh — synthesize every VFI-DL module with Vivado 2026.1 (Artix-7 XC7A100T).
# Idempotent: skips tops whose utilization report already exists.
#
# Usage:
#   ./run_vivado.sh            # all 15 tops
#   ./run_vivado.sh mac_int8   # one top
#
# Reports land in syn/vivado/reports/<top>_{util,timing,power}.rpt + <top>_synth.dcp

set -euo pipefail
cd "$(dirname "$0")"

VIVADO="${VIVADO:-/tools/2026.1/Vivado/bin/vivado}"

if [ $# -gt 0 ]; then
    TOPS=("$@")
else
    TOPS=(mac_int8 weight_mem requantize line_buffer line_buffer_stream \
          dw_conv3x3 pw_conv1x1 pw_conv1x1_parallel ds_conv_layer \
          ds_conv_layer_integrated encoder_slice warp_unit blend_unit vfi_synth)
fi

mkdir -p reports
PASS=0
FAIL=0
for top in "${TOPS[@]}"; do
    if [ -f "reports/${top}_util.rpt" ]; then
        echo "SKIP $top (report exists)"
        continue
    fi
    echo "=== $top ==="
    if "$VIVADO" -mode batch -nolog -nojournal -source synth_one.tcl -tclargs "$top" 2>&1 | tee "reports/${top}_vivado.log"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "  FAIL: $top"
    fi
done

echo "=== done: $PASS synthesized, $FAIL failed ==="
[ "$FAIL" -eq 0 ] || exit 1
