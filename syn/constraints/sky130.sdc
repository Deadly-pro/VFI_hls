# syn/constraints/sky130.sdc - Synopsys Design Constraints for Sky130
# Target: 100 MHz (10 ns period) for all VFI-DL modules
# Generated for OpenSTA timing analysis

###############################################################################
# CLOCK DEFINITIONS
###############################################################################

# Primary clock: 100 MHz (10 ns period)
create_clock -name clk -period 10.00 [get_ports clk]

# Clock uncertainty (jitter + skew margin)
set_clock_uncertainty 0.15 [get_clocks clk]

# Clock latency (source + network)
set_clock_latency -source 0.05 [get_clocks clk]
set_clock_latency -early -rise 0.02 [get_clocks clk]
set_clock_latency -early -fall 0.02 [get_clocks clk]
set_clock_latency -late  -rise 0.10 [get_clocks clk]
set_clock_latency -late  -fall 0.10 [get_clocks clk]

###############################################################################
# INPUT/OUTPUT DELAYS
###############################################################################

# Input delays relative to clk (data arrives 2 ns after clock edge)
set_input_delay -clock clk -max 2.0 [all_inputs]
set_input_delay -clock clk -min 0.5 [all_inputs]

# Output delays relative to clk (data must be stable 2 ns before next clock)
set_output_delay -clock clk -max 2.0 [all_outputs]
set_output_delay -clock clk -min 0.5 [all_outputs]

# Remove input/output delays from clock and reset ports
set_input_delay -clock clk 0.0 [get_ports clk]
set_input_delay -clock clk 0.0 [get_ports rst_n]

###############################################################################
# DRIVING CELL / LOAD CONSTRAINTS
###############################################################################

# Driving cell for inputs (typical buffer)
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_1 [all_inputs]
set_driving_cell -lib_cell sky130_fd_sc_hd__buf_1 -clock [get_clocks clk]

# Load capacitance for outputs (typical fanout of 4)
set_load 0.05 [all_outputs]

###############################################################################
# RESET RECOVERY / REMOVAL (async active-low reset)
###############################################################################

# Async reset recovery/removal checks
# These are handled by the standard cell library timing arcs

###############################################################################
# MAX TRANSITION / CAPACITANCE
###############################################################################

# Maximum transition time (Sky130 typical)
set_max_transition 0.50 [current_design]

# Maximum fanout load
set_max_fanout 16 [current_design]

###############################################################################
# CASE ANALYSIS / FALSE PATHS
###############################################################################

# No false paths needed for this synchronous design
# All paths are synchronous to clk

###############################################################################
# GENERATE REPORTS
###############################################################################

# Report timing summary
report_checks -path_delay min_max -format full_clock_expanded -digits 4

# Report clock characteristics
report_clocks

# Report unconstrained paths
report_unconstrained -check_pins

# Report worst violating paths
report_timing -max_paths 10 -format full_clock_expanded -nworst 10

###############################################################################
# END OF SDC
###############################################################################