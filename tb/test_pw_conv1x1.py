"""cocotb testbench for pw_conv1x1 — bit-exact vs golden pointwise conv.

Streams channel values for each spatial position, collects output channels,
and compares against numpy golden.

Run from tb/:
  make clean
  make MODULE=test_pw_conv1x1 TOPLEVEL=pw_conv1x1 \
       VERILOG_SOURCES=../rtl/pw_conv1x1.v
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from pw_conv1x1_ref import pw_conv1x1_golden  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_data.value = 0
    dut.in_last.value = 0
    dut.rq_m0.value = 0
    dut.rq_shift.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


@cocotb.test()
async def test_pw_conv(dut):
    """Small 4x4 image, C_IN=4, C_OUT=4."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    C_IN = int(dut.C_IN.value)
    C_OUT = int(dut.C_OUT.value)
    H, W = 4, 4
    random.seed(0xF00D)

    # random input
    image = np.array(
        [[[random.randint(-64, 63) for _ in range(C_IN)]
          for _ in range(W)] for _ in range(H)],
        dtype=np.int8,
    )

    # random weights
    weight = np.array(
        [[random.randint(-16, 15) for _ in range(C_IN)] for _ in range(C_OUT)],
        dtype=np.int8,
    )

    # random biases
    bias = np.array(
        [random.randint(-500, 500) for _ in range(C_OUT)],
        dtype=np.int32,
    )

    m0 = random.randint(1 << 30, (1 << 31) - 1)
    shift = random.randint(2, 8)

    # golden reference
    expected = pw_conv1x1_golden(image, weight, bias, m0, shift)

    # load weights into DUT
    for co in range(C_OUT):
        for ci in range(C_IN):
            dut.weight[co * C_IN + ci].value = int(weight[co, ci]) & 0xFF
    for co in range(C_OUT):
        dut.bias[co].value = int(bias[co]) & 0xFFFFFFFF
    dut.rq_m0.value = m0 & 0xFFFFFFFF
    dut.rq_shift.value = shift

    await RisingEdge(dut.clk)
    await Timer(1, "ns")

    # stream input and collect output
    got = []

    for r in range(H):
        for c in range(W):
            # feed C_IN channel values
            for ci in range(C_IN):
                dut.in_valid.value = 1
                dut.in_data.value = int(image[r, c, ci]) & 0xFF
                dut.in_last.value = 1 if ci == C_IN - 1 else 0
                await RisingEdge(dut.clk)
                await Timer(1, "ns")

                if int(dut.out_valid.value) == 1:
                    got.append(read_signed(dut.out_data, 8))

            dut.in_valid.value = 0
            dut.in_last.value = 0

            # wait for compute + output cycles
            for _ in range(C_OUT * C_IN + C_OUT + 5):
                await RisingEdge(dut.clk)
                await Timer(1, "ns")
                if int(dut.out_valid.value) == 1:
                    got.append(read_signed(dut.out_data, 8))

    # drain
    for _ in range(20):
        await RisingEdge(dut.clk)
        await Timer(1, "ns")
        if int(dut.out_valid.value) == 1:
            got.append(read_signed(dut.out_data, 8))

    expected_flat = expected.flatten().tolist()
    assert len(got) == len(expected_flat), (
        f"output count mismatch: got {len(got)}, exp {len(expected_flat)}"
    )
    for i, (g, e) in enumerate(zip(got, expected_flat)):
        pos = i // C_OUT
        co = i % C_OUT
        r, c = divmod(pos, W)
        assert g == e, (
            f"pos ({r},{c}) co={co}: got {g}, exp {e}"
        )
