"""cocotb testbench for requantize — bit-exact vs golden model.

2-cycle pipeline: feed input, wait 2 rising edges, compare output.

Run from tb/:
  make clean
  make MODULE=test_requantize TOPLEVEL=requantize VERILOG_SOURCES=../rtl/requantize.v
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from requantize_ref import requantize_ref  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.acc.value = 0
    dut.m0.value = 0
    dut.shift.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


@cocotb.test()
async def test_directed(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    vectors = [
        (1000, 1 << 30, 0),     # M0=0.5, shift=0
        (-1000, 1 << 30, 0),    # negative acc
        (100000, 1 << 30, 5),   # larger shift
        (127 * 2, 1 << 30, 0),  # near clamp boundary
        (300 * 2, 1 << 30, 0),  # should clamp to 127
        (-300 * 2, 1 << 30, 0), # should clamp to -128
        (0, 1 << 30, 0),        # zero
    ]

    results = []
    # feed all vectors
    for acc, m0, sh in vectors:
        dut.in_valid.value = 1
        dut.acc.value = acc & 0xFFFFFFFF
        dut.m0.value = m0 & 0xFFFFFFFF
        dut.shift.value = sh
        await RisingEdge(dut.clk)
        results.append((acc, m0, sh))

    dut.in_valid.value = 0

    # wait for pipeline to flush (2 cycle latency)
    await RisingEdge(dut.clk)
    await Timer(1, "ns")

    # now read outputs one at a time, they come out pipelined
    # we already waited 1 extra edge after the last input, let's re-drive
    # Actually: pipeline is 2 stages. We need to collect outputs as they emerge.
    pass  # directed test below uses a simpler approach


@cocotb.test()
async def test_pipeline(dut):
    """Feed vectors, collect outputs after 2-cycle latency, compare."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    random.seed(0xDEAD)

    vectors = []
    # directed cases
    vectors.append((1000, 1 << 30, 0))
    vectors.append((-1000, 1 << 30, 0))
    vectors.append((100000, 1 << 30, 5))
    vectors.append((300, (1 << 31) - 1, 0))  # M0 near 1.0
    vectors.append((-300, (1 << 31) - 1, 0))
    vectors.append((0, 1 << 30, 10))
    # random cases
    for _ in range(2000):
        acc = random.randint(-(1 << 24), (1 << 24) - 1)
        m0 = random.randint(1 << 30, (1 << 31) - 1)  # Q0.31 in [0.5, 1.0)
        sh = random.randint(0, 20)
        vectors.append((acc, m0, sh))

    expected = [requantize_ref(a, m, s) for a, m, s in vectors]

    # feed inputs, collect outputs after 2-cycle latency
    got_outputs = []
    for i, (acc, m0, sh) in enumerate(vectors):
        dut.in_valid.value = 1
        dut.acc.value = acc & 0xFFFFFFFF
        dut.m0.value = m0 & 0xFFFFFFFF
        dut.shift.value = sh
        await RisingEdge(dut.clk)
        await Timer(1, "ns")

        if int(dut.out_valid.value) == 1:
            got_outputs.append(read_signed(dut.out, 8))

    # drain pipeline
    dut.in_valid.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
        await Timer(1, "ns")
        if int(dut.out_valid.value) == 1:
            got_outputs.append(read_signed(dut.out, 8))

    assert len(got_outputs) == len(expected), (
        f"output count mismatch: got {len(got_outputs)}, exp {len(expected)}"
    )
    for i, (got, exp) in enumerate(zip(got_outputs, expected)):
        a, m, s = vectors[i]
        assert got == exp, (
            f"vec {i}: acc={a} m0=0x{m:08x} shift={s} -> got {got}, exp {exp}"
        )
