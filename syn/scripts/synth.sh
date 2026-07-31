#!/bin/bash
# syn/scripts/synth.sh - Run Yosys synthesis for a given TOP module
# Usage: ./synth.sh <TOP_MODULE>

set -e

TOP=${1:-mac_int8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Synthesizing $TOP ==="

# Determine source files based on module
case $TOP in
    mac_int8)
        SRC="$PROJECT_ROOT/rtl/mac_int8.v"
        ;;
    weight_mem)
        SRC="$PROJECT_ROOT/rtl/weight_mem.v"
        ;;
    requantize)
        SRC="$PROJECT_ROOT/rtl/requantize.v"
        ;;
    line_buffer)
        SRC="$PROJECT_ROOT/rtl/line_buffer.v"
        ;;
    dw_conv3x3)
        SRC="$PROJECT_ROOT/rtl/dw_conv3x3.v $PROJECT_ROOT/rtl/line_buffer.v $PROJECT_ROOT/rtl/requantize.v"
        ;;
    pw_conv1x1)
        SRC="$PROJECT_ROOT/rtl/pw_conv1x1.v"
        ;;
    ds_conv_layer)
        SRC="$PROJECT_ROOT/rtl/ds_conv_layer.v $PROJECT_ROOT/rtl/dw_conv3x3.v $PROJECT_ROOT/rtl/line_buffer.v $PROJECT_ROOT/rtl/requantize.v $PROJECT_ROOT/rtl/pw_conv1x1.v"
        ;;
    ds_conv_layer_integrated)
        SRC="$PROJECT_ROOT/rtl/ds_conv_layer_integrated.v $PROJECT_ROOT/rtl/dw_conv3x3.v $PROJECT_ROOT/rtl/line_buffer.v $PROJECT_ROOT/rtl/requantize.v $PROJECT_ROOT/rtl/pw_conv1x1.v $PROJECT_ROOT/rtl/weight_mem.v"
        ;;
    *)
        echo "Unknown TOP: $TOP"
        exit 1
        ;;
esac

mkdir -p "$SCRIPT_DIR/../reports" "$SCRIPT_DIR/../output"

# Run Yosys with SystemVerilog support for array ports
# Sky130 liberty detection
SKY130_LIB=""
for path in \
    "/usr/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib" \
    "$PDK_ROOT/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib" \
    "$HOME/.volare/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"; do
    if [ -f "$path" ]; then
        SKY130_LIB="$path"
        break
    fi
done

# Build Yosys script
YOSYS_SCRIPT="
read_verilog -sv $SRC
hierarchy -check -top $TOP
"

if [ -n "$SKY130_LIB" ]; then
    echo "Using Sky130 liberty: $SKY130_LIB"
    YOSYS_SCRIPT="$YOSYS_SCRIPT
synth -top $TOP
dfflibmap -liberty $SKY130_LIB
abc -liberty $SKY130_LIB
stat -liberty $SKY130_LIB > $SCRIPT_DIR/../reports/${TOP}_stat.txt
write_verilog $SCRIPT_DIR/../output/${TOP}.v
"
else
    echo "Sky130 PDK not found — using generic synthesis"
    YOSYS_SCRIPT="$YOSYS_SCRIPT
synth -top $TOP
stat > $SCRIPT_DIR/../reports/${TOP}_stat.txt
write_verilog $SCRIPT_DIR/../output/${TOP}.v
"
fi

echo "$YOSYS_SCRIPT" | yosys -q

echo "=== $TOP synthesis complete ==="
echo "Reports: $SCRIPT_DIR/../reports/${TOP}_stat.txt"
echo "Netlist: $SCRIPT_DIR/../output/${TOP}.v"