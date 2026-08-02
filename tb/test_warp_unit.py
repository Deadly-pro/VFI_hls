"""cocotb testbench for warp_unit — bit-exact vs fixed-point grid_sample golden."""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from grid_sample_ref import warp_bilinear_fixed  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.pixel_in.value = 0
    dut.flow_valid.value = 0
    dut.flow_x.value = 0
    dut.flow_y.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


@cocotb.test()
async def test_warp_identity(dut):
    """Zero flow must reproduce the input frame exactly."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    W, H = int(dut.IMG_W.value), int(dut.IMG_H.value)
    random.seed(0xA1A0)

    image = np.array(
        [[random.randint(-128, 127) for _ in range(W)] for _ in range(H)],
        dtype=np.int8,
    )

    # capture phase
    for r in range(H):
        for c in range(W):
            dut.in_valid.value = 1
            dut.pixel_in.value = int(image[r, c]) & 0xFF
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, "ns")

    # warp phase: zero flow
    got = []
    for r in range(H):
        for c in range(W):
            dut.flow_valid.value = 1
            dut.flow_x.value = 0
            dut.flow_y.value = 0
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
            if int(dut.out_valid.value) == 1:
                got.append(read_signed(dut.pixel_out, 8))

    expected = image.flatten().tolist()
    assert len(got) == len(expected), f"count mismatch: got {len(got)}, exp {len(expected)}"
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g == e, f"pixel {i}: got {g}, exp {e}"


@cocotb.test()
async def test_warp_translate(dut):
    """Constant displacement matches golden bit-exact."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    W, H = int(dut.IMG_W.value), int(dut.IMG_H.value)
    FLOW_Q = int(dut.FLOW_Q.value)
    random.seed(0xA1A1)

    image = np.array(
        [[random.randint(-128, 127) for _ in range(W)] for _ in range(H)],
        dtype=np.int8,
    )

    # fractional-pixel displacement, fixed-point
    dx = round(1.25 * (1 << FLOW_Q))
    dy = round(-0.5 * (1 << FLOW_Q))

    flow_x = np.full((H, W), dx, dtype=np.int16)
    flow_y = np.full((H, W), dy, dtype=np.int16)

    expected = warp_bilinear_fixed(image, flow_x, flow_y, FLOW_Q).flatten().tolist()

    # capture
    for r in range(H):
        for c in range(W):
            dut.in_valid.value = 1
            dut.pixel_in.value = int(image[r, c]) & 0xFF
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, "ns")

    # warp
    got = []
    for r in range(H):
        for c in range(W):
            dut.flow_valid.value = 1
            dut.flow_x.value = int(flow_x[r, c]) & 0xFFFF
            dut.flow_y.value = int(flow_y[r, c]) & 0xFFFF
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
            if int(dut.out_valid.value) == 1:
                got.append(read_signed(dut.pixel_out, 8))

    assert len(got) == len(expected), f"count mismatch: got {len(got)}, exp {len(expected)}"
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g == e, f"pixel {i}: got {g}, exp {e}"


@cocotb.test()
async def test_warp_random_flow(dut):
    """Random flow field matches golden bit-exact."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    W, H = int(dut.IMG_W.value), int(dut.IMG_H.value)
    FLOW_Q = int(dut.FLOW_Q.value)
    random.seed(0xA1A2)

    image = np.array(
        [[random.randint(-128, 127) for _ in range(W)] for _ in range(H)],
        dtype=np.int8,
    )

    # random flow in range [-3, 3] pixels (fits FLOW_W=16 with Q8 easily)
    flow_x = np.random.randint(-3 * (1 << FLOW_Q), 3 * (1 << FLOW_Q) + 1,
                               size=(H, W)).astype(np.int16)
    flow_y = np.random.randint(-3 * (1 << FLOW_Q), 3 * (1 << FLOW_Q) + 1,
                               size=(H, W)).astype(np.int16)

    expected = warp_bilinear_fixed(image, flow_x, flow_y, FLOW_Q).flatten().tolist()

    # capture
    for r in range(H):
        for c in range(W):
            dut.in_valid.value = 1
            dut.pixel_in.value = int(image[r, c]) & 0xFF
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
    dut.in_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, "ns")

    # warp
    got = []
    for r in range(H):
        for c in range(W):
            dut.flow_valid.value = 1
            dut.flow_x.value = int(flow_x[r, c]) & 0xFFFF
            dut.flow_y.value = int(flow_y[r, c]) & 0xFFFF
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
            if int(dut.out_valid.value) == 1:
                got.append(read_signed(dut.pixel_out, 8))

    assert len(got) == len(expected), f"count mismatch: got {len(got)}, exp {len(expected)}"
    for i, (g, e) in enumerate(zip(got, expected)):
        assert g == e, f"pixel {i}: got {g}, exp {e}"