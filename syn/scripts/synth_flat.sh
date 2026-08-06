#!/bin/bash
# synthesize a module with the flat synth leaves if needed
set -e
TOP="$1"
EXTRA="${2:-}"
RTL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/rtl"
LEAF=" "
case $TOP in
  dw_conv3x3_flat) EXTRA="$RTL/synth/leaf_flat.v $RTL/line_buffer.v $RTL/requantize.v" ;;
  pw_conv1x1_flat) EXTRA="$RTL/synth/leaf_flat.v" ;;
  ds_conv_layer_integrated_flat) EXTRA="$RTL/synth/leaf_flat.v $RTL/line_buffer.v $RTL/requantize.v" ;;
esac
yosys -p "read_verilog -sv $RTL/synth/ds_conv_layer_integrated_flat.v" -p "read_verilog -sv $EXTRA" -q
