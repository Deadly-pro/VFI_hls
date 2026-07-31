# Yosys synthesis for VFI-DL accelerator modules
#
# Usage:
#   yosys -c synth.tcl -D TOP=mac_int8
#   yosys -c synth.tcl -D TOP=ds_conv_layer_integrated
#
# Outputs: syn/reports/<TOP>_stat.txt, syn/output/<TOP>.v (gate-level netlist)
#
# Uses Sky130 PDK if available, otherwise generic gate-count synthesis.

# --- configuration ---
if {![info exists TOP]} {
    set TOP "mac_int8"
}

set RTL_DIR  "../rtl"
set RPT_DIR  "../reports"
set OUT_DIR  "../output"

file mkdir $RPT_DIR
file mkdir $OUT_DIR

# --- source files by module ---
set src_map(mac_int8)                    [list $RTL_DIR/mac_int8.v]
set src_map(weight_mem)                  [list $RTL_DIR/weight_mem.v]
set src_map(requantize)                  [list $RTL_DIR/requantize.v]
set src_map(line_buffer)                 [list $RTL_DIR/line_buffer.v]
set src_map(dw_conv3x3)                  [list $RTL_DIR/dw_conv3x3.v $RTL_DIR/line_buffer.v $RTL_DIR/requantize.v]
set src_map(pw_conv1x1)                  [list $RTL_DIR/pw_conv1x1.v]
set src_map(ds_conv_layer)               [list $RTL_DIR/ds_conv_layer.v $RTL_DIR/dw_conv3x3.v \
                                                 $RTL_DIR/line_buffer.v $RTL_DIR/requantize.v \
                                                 $RTL_DIR/pw_conv1x1.v]
set src_map(ds_conv_layer_integrated)    [list $RTL_DIR/ds_conv_layer_integrated.v $RTL_DIR/dw_conv3x3.v \
                                                 $RTL_DIR/line_buffer.v $RTL_DIR/requantize.v \
                                                 $RTL_DIR/pw_conv1x1.v $RTL_DIR/weight_mem.v]

if {![info exists src_map($TOP)]} {
    puts "ERROR: unknown TOP=$TOP"
    exit 1
}

# --- read RTL with SystemVerilog support for array ports ---
foreach f $src_map($TOP) {
    read_verilog -sv $f
}

# --- synthesize ---
# Try Sky130 liberty if available
set sky130_lib ""
foreach path [list \
    "/usr/share/pdk/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib" \
    "$::env(PDK_ROOT)/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib" \
    "$::env(HOME)/.volare/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib" \
] {
    if {[file exists $path]} {
        set sky130_lib $path
        break
    }
}

if {$sky130_lib ne ""} {
    puts "Using Sky130 liberty: $sky130_lib"
    synth -top $TOP
    dfflibmap -liberty $sky130_lib
    abc -liberty $sky130_lib
} else {
    puts "Sky130 PDK not found — using generic synthesis"
    synth -top $TOP
}

# --- reports ---
stat -top $TOP
tee -o $RPT_DIR/${TOP}_stat.txt stat -top $TOP

# --- write netlist ---
write_verilog $OUT_DIR/${TOP}.v

puts ""
puts "=== Synthesis complete for $TOP ==="
puts "Report: $RPT_DIR/${TOP}_stat.txt"
puts "Netlist: $OUT_DIR/${TOP}.v"