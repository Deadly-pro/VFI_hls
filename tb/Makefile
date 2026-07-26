# cocotb test runner (Icarus Verilog). Run a single module's tests, e.g.:
#   make MODULE=test_mac_int8 TOPLEVEL=mac_int8 VERILOG_SOURCES=../rtl/mac_int8.v
# Defaults below target mac_int8 (Sprint 0). `make clean` between modules.

TOPLEVEL_LANG ?= verilog
SIM           ?= icarus

VERILOG_SOURCES ?= $(PWD)/../rtl/mac_int8.v
TOPLEVEL        ?= mac_int8
MODULE          ?= test_mac_int8

include $(shell cocotb-config --makefiles)/Makefile.sim
