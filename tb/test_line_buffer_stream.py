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


@cocotb.test()
async def test_line_buffer_stream(dut):
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

    # Stream input
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    dut.frame_start.value = 0

    for py in range(IMG_H):
        for px in range(IMG_W):
            dut.in_valid.value = 1
            dut.pixel_in.value = int(image[py, px])
            await RisingEdge(dut.clk)

    dut.in_valid.value = 0

    # Collect outputs (streaming produces IMG_W * IMG_H windows)
    outputs = []
    # Streaming: first output after ~2 rows + 1 col latency, then one per cycle
    timeout = NUM_PX + IMG_W + IMG_H + 10
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            win = [
                read_signed(dut.win_0, 8), read_signed(dut.win_1, 8), read_signed(dut.win_2, 8),
                read_signed(dut.win_3, 8), read_signed(dut.win_4, 8), read_signed(dut.win_5, 8),
                read_signed(dut.win_6, 8), read_signed(dut.win_7, 8), read_signed(dut.win_8, 8)
            ]
            outputs.append(win)

    # Verify count
    assert len(outputs) == NUM_PX, f"Window count mismatch: got {len(outputs)}, expected {NUM_PX}"

    # Verify each window
    for i, (got, exp) in enumerate(zip(outputs, golden_windows)):
        assert got == exp, f"Window {i} mismatch: got {got}, expected {exp}"

    print(f"✓ test_line_buffer_stream passed: {len(outputs)} windows match golden")


@cocotb.test()
async def test_line_buffer_stream_8x8(dut):
    """Same test with 8x8 for exact match with original line_buffer test."""
    await test_line_buffer_stream(dut)