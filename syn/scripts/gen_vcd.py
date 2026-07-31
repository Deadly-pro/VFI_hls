#!/usr/bin/env python3
# syn/scripts/gen_vcd.py - Generate VCD waveforms from cocotb for power analysis
# Usage: python3 gen_vcd.py <test_module> <top_module> [output.vcd]

import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TB_DIR = os.path.join(PROJECT_ROOT, "tb")

def run_cocotb_with_vcd(test_module, top_module, verilog_sources, vcd_file):
    """Run cocotb test with VCD dumping enabled."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(PROJECT_ROOT, "golden") + ":" + env.get("PYTHONPATH", "")
    env["COCOTB_SCHEDULER_DEBUG"] = "0"

    # Build sim
    sim_build = os.path.join(TB_DIR, "sim_build")
    os.makedirs(sim_build, exist_ok=True)

    # Determine iverilog command
    src_files = " ".join(verilog_sources)
    iverilog_cmd = f"iverilog -g2012 -o {sim_build}/sim.vvp -s {top_module} {src_files}"

    print(f"Building simulation: {iverilog_cmd}")
    result = subprocess.run(iverilog_cmd, shell=True, cwd=TB_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"iverilog failed:\n{result.stderr}")
        return False

    # Run with VCD
    vcd_path = os.path.join(TB_DIR, vcd_file)
    cocotb_cmd = (
        f"MODULE={test_module} TOPLEVEL={top_module} "
        f"COCOTB_TEST_MODULES={test_module} "
        f"vvp -M {os.path.join(os.environ.get('VIRTUAL_ENV', ''), 'lib/python3.12/site-packages/cocotb/libs')} "
        f"-m libcocotbvpi_icarus {sim_build}/sim.vvp "
        f"+vcd={vcd_path} +vcdfile={vcd_path}"
    )

    print(f"Running simulation with VCD: {vcd_path}")
    result = subprocess.run(cocotb_cmd, shell=True, cwd=TB_DIR, env=env, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    return result.returncode == 0


def main():
    if len(sys.argv) < 3:
        print("Usage: gen_vcd.py <test_module> <top_module> [output.vcd]")
        print("Example: gen_vcd.py test_ds_conv_layer_integrated ds_conv_layer_integrated ds_conv_integrated.vcd")
        sys.exit(1)

    test_module = sys.argv[1]
    top_module = sys.argv[2]
    vcd_file = sys.argv[3] if len(sys.argv) > 3 else f"{top_module}.vcd"

    # Module -> source files mapping
    src_map = {
        "mac_int8": ["../rtl/mac_int8.v"],
        "weight_mem": ["../rtl/weight_mem.v"],
        "requantize": ["../rtl/requantize.v"],
        "line_buffer": ["../rtl/line_buffer.v"],
        "dw_conv3x3": ["../rtl/dw_conv3x3.v", "../rtl/line_buffer.v", "../rtl/requantize.v"],
        "pw_conv1x1": ["../rtl/pw_conv1x1.v"],
        "ds_conv_layer": ["../rtl/ds_conv_layer.v", "../rtl/dw_conv3x3.v", "../rtl/line_buffer.v", "../rtl/requantize.v", "../rtl/pw_conv1x1.v"],
        "ds_conv_layer_integrated": ["../rtl/ds_conv_layer_integrated.v", "../rtl/dw_conv3x3.v", "../rtl/line_buffer.v", "../rtl/requantize.v", "../rtl/pw_conv1x1.v", "../rtl/weight_mem.v"],
    }

    if top_module not in src_map:
        print(f"Unknown top_module: {top_module}")
        sys.exit(1)

    verilog_sources = src_map[top_module]

    success = run_cocotb_with_vcd(test_module, top_module, verilog_sources, vcd_file)
    if success:
        print(f"\n✓ VCD generated: {os.path.join(TB_DIR, vcd_file)}")
        print(f"  Use with OpenSTA for power analysis")
    else:
        print(f"\n✗ VCD generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()