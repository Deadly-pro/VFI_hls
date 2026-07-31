# Synthesis Makefile
#
# Usage:
#   make                    # synthesize mac_int8 (default)
#   make TOP=ds_conv_layer  # synthesize ds_conv_layer
#   make all                # synthesize all modules

TOP ?= mac_int8

SCRIPT = scripts/synth.tcl

MODULES = mac_int8 weight_mem requantize line_buffer dw_conv3x3 pw_conv1x1 ds_conv_layer

.PHONY: $(MODULES) all clean

$(TOP):
	cd scripts && yosys -c synth.tcl -D TOP=$(TOP)

all:
	@for m in $(MODULES); do \
		echo "=== Synthesizing $$m ==="; \
		cd scripts && yosys -c synth.tcl -D TOP=$$m && cd ..; \
	done

clean:
	rm -rf reports output
