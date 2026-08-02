"""cocotb testbench for pw_conv1x1_parallel — bit-exact vs serial reference."""

import os
import sys
import random
import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from pw_conv1x1_ref import pw_conv1x1_golden


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


@cocotb.test()
async def test_pw_conv1x1_parallel(dut):
    """Test parallel PW against golden reference."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    C_IN = int(dut.C_IN.value)
    C_OUT = int(dut.C_OUT.value)
    PARALLEL_CO = int(dut.PARALLEL_CO.value)

    random.seed(0x5035A0)
    np.random.seed(0x5035A0)

    # Test image: 4x4 spatial, C_IN channels
    H, W = 4, 4
    image = np.random.randint(-128, 127, size=(H, W, C_IN), dtype=np.int8)

    weight = np.random.randint(-128, 127, size=(C_OUT, C_IN), dtype=np.int8)
    bias = np.random.randint(-2000, 2000, size=(C_OUT,), dtype=np.int32)
    rq_m0 = 0x40000000
    rq_shift = 0

    # Golden reference
    golden = pw_conv1x1_golden(image, weight, bias, rq_m0, rq_shift)  # [H, W, C_OUT]
    golden_flat = golden.reshape(-1).tolist()

    # Reset
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.in_last.value = 0
    dut.rq_m0.value = rq_m0
    dut.rq_shift.value = rq_shift
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Load weights into DUT registers
    for co in range(C_OUT):
        for ci in range(C_IN):
            dut.weight[co * C_IN + ci].value = int(weight[co, ci])
    for co in range(C_OUT):
        dut.bias[co].value = int(bias[co])

    await RisingEdge(dut.clk)

    # Stream input: spatial position by position, channel-interleaved
    outputs = []
    for py in range(H):
        for px in range(W):
            for ci in range(C_IN):
                dut.in_valid.value = 1
                dut.in_data.value = int(image[py, px, ci])
                dut.in_last.value = 1 if ci == C_IN - 1 else 0
                await RisingEdge(dut.clk)

    dut.in_valid.value = 0
    dut.in_last.value = 0

    # Collect outputs
    timeout = H * W * C_OUT + 50
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            outputs.append(read_signed(dut.out_data, 8))
            if dut.out_last.value == 1:
                break

    assert outputs == golden_flat, f"Mismatch: got {outputs}, expected {golden_flat}"
    print(f"✓ test_pw_conv1x1_parallel passed: C_IN={C_IN}, C_OUT={C_OUT}, PARALLEL_CO={PARALLEL_CO}")


@cocotb.test()
async def test_pw_conv1x1_parallel_vs_serial(dut):
    """Verify parallel output matches serial pw_conv1x1 golden."""
    # Same test as above - the golden is the serial reference
    await test_pw_conv1x1_parallel(dut)