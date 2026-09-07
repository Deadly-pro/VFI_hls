#!/usr/bin/env bash
# Run all cocotb testbenches in sequence.
# Prerequisites: iverilog, cocotb, numpy installed in .venv
#
# Usage:
#   ./run_all_tests.sh

set -euo pipefail
cd "$(dirname "$0")"

export PATH="$PWD/.venv/bin:$PATH"

PASS=0
FAIL=0

run_test() {
    local module="$1"
    local toplevel="$2"
    local sources="$3"
    echo ""
    echo "============================================"
    echo "  $module ($toplevel)"
    echo "============================================"
    cd tb
    if make clean 2>/dev/null; true; then
        if make MODULE="$module" TOPLEVEL="$toplevel" VERILOG_SOURCES="$sources"; then
            PASS=$((PASS + 1))
            echo "  PASS"
        else
            FAIL=$((FAIL + 1))
            echo "  FAIL"
        fi
    fi
    cd ..
}

run_test test_mac_int8      mac_int8      "../rtl/mac_int8.v"
run_test test_weight_mem    weight_mem    "../rtl/weight_mem.v"
run_test test_requantize    requantize    "../rtl/requantize.v"
run_test test_line_buffer   line_buffer   "../rtl/line_buffer.v"
run_test test_dw_conv3x3    dw_conv3x3    "../rtl/dw_conv3x3.v ../rtl/line_buffer.v ../rtl/requantize.v"
run_test test_pw_conv1x1    pw_conv1x1    "../rtl/pw_conv1x1.v"
run_test test_ds_conv_layer ds_conv_layer "../rtl/ds_conv_layer.v ../rtl/dw_conv3x3.v ../rtl/line_buffer.v ../rtl/requantize.v ../rtl/pw_conv1x1.v"
run_test test_ds_conv_layer_integrated ds_conv_layer_integrated "../rtl/ds_conv_layer_integrated.v ../rtl/dw_conv3x3.v ../rtl/line_buffer.v ../rtl/requantize.v ../rtl/pw_conv1x1.v ../rtl/weight_mem.v"
run_test test_line_buffer_stream line_buffer_stream "../rtl/line_buffer_stream.v"
run_test test_pw_conv1x1_parallel pw_conv1x1_parallel "../rtl/pw_conv1x1_parallel.v"
run_test test_encoder_slice    encoder_slice    "../rtl/encoder_slice.v ../rtl/ds_conv_layer_integrated.v ../rtl/dw_conv3x3.v ../rtl/line_buffer.v ../rtl/requantize.v ../rtl/pw_conv1x1.v ../rtl/pw_conv1x1_parallel.v ../rtl/weight_mem.v"
run_test test_warp_unit      warp_unit       "../rtl/warp_unit.v"
run_test test_blend_unit     blend_unit      "../rtl/blend_unit.v"
run_test test_vfi_synth      vfi_synth       "../rtl/vfi_synth.v ../rtl/warp_unit.v ../rtl/blend_unit.v"

echo ""
echo "============================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "============================================"

[ "$FAIL" -eq 0 ] || exit 1
