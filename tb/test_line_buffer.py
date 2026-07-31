"""cocotb testbench for line_buffer — 3x3 window extraction vs numpy golden.

Streams a small image through the DUT one pixel per clock, collects the
output windows (after the pipeline latency), and compares against the
numpy zero-padded reference.

Run from tb/:
  make clean
  make MODULE=test_line_buffer TOPLEVEL=line_buffer VERILOG_SOURCES=../rtl/line_buffer.v
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from line_buffer_ref import line_buffer_golden  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.pixel_in.value = 0
    dut.frame_start.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


def read_window(dut):
    """Read the 9 window outputs from the DUT."""
    return [
        read_signed(dut.win_0, 8), read_signed(dut.win_1, 8), read_signed(dut.win_2, 8),
        read_signed(dut.win_3, 8), read_signed(dut.win_4, 8), read_signed(dut.win_5, 8),
        read_signed(dut.win_6, 8), read_signed(dut.win_7, 8), read_signed(dut.win_8, 8),
    ]


@cocotb.test()
async def test_small_image(dut):
    """Stream an 8x8 image, compare all windows to golden."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    W, H = 8, 8
    random.seed(42)
    image = np.array(
        [[random.randint(-128, 127) for _ in range(W)] for _ in range(H)],
        dtype=np.int8,
    )

    expected = line_buffer_golden(image)

    # pulse frame_start
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    dut.frame_start.value = 0
    await Timer(1, "ns")

    # stream pixels and collect outputs
    got_windows = []
    for r in range(H):
        for c in range(W):
            dut.in_valid.value = 1
            dut.pixel_in.value = int(image[r, c]) & 0xFF
            await RisingEdge(dut.clk)
            await Timer(1, "ns")

            if int(dut.out_valid.value) == 1:
                got_windows.append(read_window(dut))

    # Drain the complete-frame output phase.
    dut.in_valid.value = 0
    for _ in range(W * H + 2):
        await RisingEdge(dut.clk)
        await Timer(1, "ns")
        if int(dut.out_valid.value) == 1:
            got_windows.append(read_window(dut))

    assert len(got_windows) == len(expected), (
        f"window count mismatch: got {len(got_windows)}, exp {len(expected)}"
    )
    for i, (got, exp) in enumerate(zip(got_windows, expected)):
        r, c = divmod(i, W)
        assert got == exp, (
            f"pixel ({r},{c}): got {got}, exp {exp}"
        )
