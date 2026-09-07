# synth_one.tcl — Vivado batch synthesis for one VFI-DL module
# Usage:
#   vivado -mode batch -nolog -nojournal -source synth_one.tcl -tclargs <TOP>
#
# Mirrors the module/dependency lists in ../scripts/synth.tcl (the old Yosys
# flow). Runs synth_design for TOP on the Artix-7 XC7A100T, 100 MHz clock,
# writes utilization + timing reports and a netlist checkpoint.

set top [lindex $argv 0]
if {$top eq ""} { set top "mac_int8" }

set script_dir [file dirname [file normalize [info script]]]
set rtl_dir    [file join $script_dir .. .. rtl]
set rpt_dir    [file join $script_dir reports]
file mkdir $rpt_dir

set src_map(mac_int8)                  [list $rtl_dir/mac_int8.v]
set src_map(weight_mem)                [list $rtl_dir/weight_mem.v]
set src_map(requantize)                [list $rtl_dir/requantize.v]
set src_map(line_buffer)               [list $rtl_dir/line_buffer.v]
set src_map(line_buffer_stream)        [list $rtl_dir/line_buffer_stream.v]
set src_map(dw_conv3x3)                [list $rtl_dir/dw_conv3x3.v $rtl_dir/line_buffer.v $rtl_dir/requantize.v]
set src_map(pw_conv1x1)                [list $rtl_dir/pw_conv1x1.v]
set src_map(pw_conv1x1_parallel)       [list $rtl_dir/pw_conv1x1_parallel.v]
set src_map(ds_conv_layer)             [list $rtl_dir/ds_conv_layer.v $rtl_dir/dw_conv3x3.v \
                                              $rtl_dir/line_buffer.v $rtl_dir/requantize.v \
                                              $rtl_dir/pw_conv1x1.v]
set src_map(ds_conv_layer_integrated)  [list $rtl_dir/ds_conv_layer_integrated.v $rtl_dir/dw_conv3x3.v \
                                              $rtl_dir/line_buffer.v $rtl_dir/requantize.v \
                                              $rtl_dir/pw_conv1x1.v $rtl_dir/weight_mem.v]
set src_map(encoder_slice)             [list $rtl_dir/encoder_slice.v $rtl_dir/ds_conv_layer_integrated.v \
                                              $rtl_dir/dw_conv3x3.v $rtl_dir/line_buffer.v \
                                              $rtl_dir/requantize.v $rtl_dir/pw_conv1x1.v \
                                              $rtl_dir/pw_conv1x1_parallel.v $rtl_dir/weight_mem.v]
set src_map(warp_unit)                 [list $rtl_dir/warp_unit.v]
set src_map(blend_unit)                [list $rtl_dir/blend_unit.v]
set src_map(vfi_synth)                 [list $rtl_dir/vfi_synth.v $rtl_dir/warp_unit.v $rtl_dir/blend_unit.v]

if {![info exists src_map($top)]} {
    puts "ERROR: unknown TOP=$top"
    exit 1
}

create_project -in_memory -part xc7a100tcsg324-1
foreach f $src_map($top) { read_verilog -sv $f }
synth_design -top $top -part xc7a100tcsg324-1

create_clock -period 10.000 -name clk [get_ports clk]

report_utilization    -file [file join $rpt_dir ${top}_util.rpt]
report_timing_summary -file [file join $rpt_dir ${top}_timing.rpt]
report_power          -file [file join $rpt_dir ${top}_power.rpt]
write_checkpoint -force [file join $rpt_dir ${top}_synth.dcp]

set paths [get_timing_paths -max_paths 1 -setup]
if {[llength $paths] > 0} {
    puts "WNS_RESULT $top [get_property SLACK [lindex $paths 0]]"
} else {
    puts "WNS_RESULT $top none"
}
puts "SYNTH_DONE $top"
