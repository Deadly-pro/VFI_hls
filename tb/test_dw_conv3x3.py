"""cocotb testbench for dw_conv3x3 — bit-exact vs golden depthwise 3x3.

Streams a small image through the DUT, collects output pixels after pipeline
latency, and compares to the numpy golden model.

Run from tb/:
  make clean
  make MODULE=test_dw_conv3x3 TOPLEVEL=dw_conv3x3 \
       VERILOG_SOURCES="../rtl/dw_conv3x3.v ../rtl/line_buffer.v ../rtl/requantize.v"
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from dw_conv3x3_ref import dw_conv3x3_golden  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.pixel_in.value = 0
    dut.frame_start.value = 0
    dut.bias.value = 0
    dut.rq_m0.value = 0
    dut.rq_shift.value = 0
    for i in range(9):
        dut.kw[i].value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


@cocotb.test()
async def test_dw_conv(dut):
    """Stream 8x8 image through depthwise conv, compare to golden."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    W, H = 8, 8
    random.seed(0xCAFE)

    # random image
    image = np.array(
        [[random.randint(-128, 127) for _ in range(W)] for _ in range(H)],
        dtype=np.int8,
    )

    # random 3x3 kernel (small values to avoid overflow before requant)
    kernel = np.array(
        [random.randint(-16, 15) for _ in range(9)],
        dtype=np.int8,
    ).reshape(3, 3)

    bias = random.randint(-1000, 1000)
    m0 = random.randint(1 << 30, (1 << 31) - 1)  # Q0.31 in [0.5, 1.0)
    shift = random.randint(2, 10)

    # golden reference
    expected_img = dw_conv3x3_golden(image, kernel, bias, m0, shift)
    expected = expected_img.flatten().tolist()

    # load kernel weights
    kernel_flat = kernel.flatten().tolist()
    for i in range(9):
        dut.kw[i].value = int(kernel_flat[i]) & 0xFF

    dut.bias.value = bias & 0xFFFFFFFF
    dut.rq_m0.value = m0 & 0xFFFFFFFF
    dut.rq_shift.value = shift

    # pulse frame_start
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    dut.frame_start.value = 0
    await Timer(1, "ns")

    # stream image
    got = []
    for r in range(H):
        for c in range(W):
            dut.in_valid.value = 1
            dut.pixel_in.value = int(image[r, c]) & 0xFF
            await RisingEdge(dut.clk)
            await Timer(1, "ns")

            if int(dut.out_valid.value) == 1:
                got.append(read_signed(dut.pixel_out, 8))

    # Drain the complete-frame window extractor and convolution pipeline.
    dut.in_valid.value = 0
    for _ in range(W * H + 5):
        await RisingEdge(dut.clk)
        await Timer(1, "ns")
        if int(dut.out_valid.value) == 1:
            got.append(read_signed(dut.pixel_out, 8))

    assert len(got) == len(expected), (
        f"output count mismatch: got {len(got)}, exp {len(expected)}"
    )
    for i, (g, e) in enumerate(zip(got, expected)):
        r, c = divmod(i, W)
        assert g == e, (
            f"pixel ({r},{c}): got {g}, exp {e}"
        )
