"""cocotb testbench for blend_unit — bit-exact vs fixed-point blend golden."""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from vfi_synth_ref import blend_fixed  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_a.value = 0
    dut.in_b.value = 0
    dut.mask.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


@cocotb.test()
async def test_blend_extremes(dut):
    """mask=0 => b, mask=255 => ~a."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    for mask_val, label in [(0, "mask=0"), (255, "mask=255")]:
        for a_val, b_val in [(100, -50), (-100, 50), (127, -128), (0, 0)]:
            dut.in_valid.value = 1
            dut.in_a.value = a_val & 0xFF
            dut.in_b.value = b_val & 0xFF
            dut.mask.value = mask_val
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
            got = read_signed(dut.pixel_out, 8)

            a = np.array([[a_val]], dtype=np.int8)
            b = np.array([[b_val]], dtype=np.int8)
            m = np.array([[mask_val]], dtype=np.uint8)
            exp = int(blend_fixed(a, b, m)[0, 0])
            assert got == exp, f"{label} a={a_val} b={b_val}: got {got}, exp {exp}"


@cocotb.test()
async def test_blend_random(dut):
    """Random masks/frames match golden bit-exact."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    random.seed(0xB1E0)
    N = 2000
    a = np.random.randint(-128, 128, size=N, dtype=np.int8)
    b = np.random.randint(-128, 128, size=N, dtype=np.int8)
    m = np.random.randint(0, 256, size=N, dtype=np.uint8)

    expected = blend_fixed(a.reshape(1, -1), b.reshape(1, -1),
                           m.reshape(1, -1)).flatten().tolist()

    got = []
    for i in range(N):
        dut.in_valid.value = 1
        dut.in_a.value = int(a[i]) & 0xFF
        dut.in_b.value = int(b[i]) & 0xFF
        dut.mask.value = int(m[i])
        await RisingEdge(dut.clk)
        await Timer(1, "ns")
        if int(dut.out_valid.value) == 1:
            got.append(read_signed(dut.pixel_out, 8))

    assert got == expected, f"blend mismatch at some index: got {len(got)} vals"