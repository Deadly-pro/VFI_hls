# impl_one.tcl — Vivado implementation (place + route) for one VFI-DL top.
# Consumes the synth checkpoint produced by synth_one.tcl.
# Usage:
#   vivado -mode batch -nolog -nojournal -source impl_one.tcl -tclargs <TOP>
#
# Full place+route is only run for vfi_synth (the whole-datapath flagship);
# post-route WNS is the real Fmax evidence. No bitstream: no board pins yet.

set top [lindex $argv 0]
if {$top eq ""} { set top "vfi_synth" }

set script_dir [file dirname [file normalize [info script]]]
set rpt_dir    [file join $script_dir reports]

open_checkpoint [file join $rpt_dir ${top}_synth.dcp]

place_design
route_design

report_utilization    -file [file join $rpt_dir ${top}_postroute_util.rpt]
report_timing_summary -file [file join $rpt_dir ${top}_postroute_timing.rpt]
report_power          -file [file join $rpt_dir ${top}_postroute_power.rpt]

set wns [get_property SLACK [lindex [get_timing_paths -max_paths 1 -setup] 0]]
puts "IMPL_WNS $top $wns"
puts "IMPL_DONE $top"
