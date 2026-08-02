"""cocotb testbench for line_buffer_stream — streaming 3x3 window extractor."""

import os
import sys
import random
import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from line_buffer_ref import line_buffer_golden


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def read_win(dut):
    return [read_signed(sig, 8) for sig in
            (dut.win_0, dut.win_1, dut.win_2, dut.win_3, dut.win_4,
             dut.win_5, dut.win_6, dut.win_7, dut.win_8)]


async def run_stream_test(dut):
    """Stream an image through streaming line_buffer, compare all windows to golden."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    IMG_W = int(dut.IMG_W.value)
    IMG_H = int(dut.IMG_H.value)
    NUM_PX = IMG_W * IMG_H

    # Generate test image
    random.seed(0x51E4E4)
    np.random.seed(0x51E4E4)
    image = np.random.randint(-128, 127, size=(IMG_H, IMG_W), dtype=np.int8)

    # Golden reference (whole-frame, same as before)
    golden_windows = line_buffer_golden(image)  # list of 9-element lists

    # Reset
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.pixel_in.value = 0
    dut.frame_start.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Stream input, collecting windows as they stream out (the RTL emits one
    # window per cycle once the first corner is captured, plus a tail after).
    outputs = []
    for py in range(IMG_H):
        for px in range(IMG_W):
            dut.in_valid.value = 1
            dut.pixel_in.value = int(image[py, px])
            await RisingEdge(dut.clk)
            if dut.out_valid.value == 1:
                outputs.append(read_win(dut))

    dut.in_valid.value = 0

    # drain the tail windows emitted after the last input pixel
    for _ in range(IMG_W + IMG_H + 10):
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            outputs.append(read_win(dut))

    # Verify count
    assert len(outputs) == NUM_PX, f"Window count mismatch: got {len(outputs)}, expected {NUM_PX}"

    # Verify each window
    for i, (got, exp) in enumerate(zip(outputs, golden_windows)):
        assert got == exp, f"Window {i} mismatch: got {got}, expected {exp}"

    print(f"✓ test_line_buffer_stream passed: {len(outputs)} windows match golden")


@cocotb.test()
async def test_line_buffer_stream(dut):
    await run_stream_test(dut)


@cocotb.test()
async def test_line_buffer_stream_8x8(dut):
    """Same test with 8x8 for exact match with original line_buffer test."""
    await run_stream_test(dut)