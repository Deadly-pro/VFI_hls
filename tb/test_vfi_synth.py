"""cocotb testbench for vfi_synth — end-to-end warp+blend vs golden."""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from vfi_synth_ref import vfi_synth_fixed  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.capture_valid.value = 0
    dut.pixel_t_in.value = 0
    dut.pixel_t1_in.value = 0
    dut.wb_valid.value = 0
    dut.flow_x.value = 0
    dut.flow_y.value = 0
    dut.mask.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


@cocotb.test()
async def test_vfi_synth(dut):
    """Full warp+blend on an 8x8 RGB frame pair, bit-exact vs golden."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    W, H = int(dut.IMG_W.value), int(dut.IMG_H.value)
    NCH = int(dut.NCH.value)
    FLOW_Q = int(dut.FLOW_Q.value)
    random.seed(0x5F15)

    # channel-interleaved t and t+1 frames
    t = np.random.randint(-128, 128, size=(H, W, NCH), dtype=np.int8)
    t1 = np.random.randint(-128, 128, size=(H, W, NCH), dtype=np.int8)

    # flow + mask per pixel (shared across channels)
    flow_x = np.random.randint(-2 * (1 << FLOW_Q), 2 * (1 << FLOW_Q) + 1,
                               size=(H, W)).astype(np.int16)
    flow_y = np.random.randint(-2 * (1 << FLOW_Q), 2 * (1 << FLOW_Q) + 1,
                               size=(H, W)).astype(np.int16)
    mask = np.random.randint(0, 256, size=(H, W), dtype=np.uint8)

    # golden per channel
    expected = []
    for c in range(NCH):
        exp = vfi_synth_fixed(t[:, :, c], t1[:, :, c],
                              flow_x, flow_y, mask, FLOW_Q).flatten().tolist()
        expected.append(exp)

    # capture: channel-interleaved pixels
    for r in range(H):
        for c in range(W):
            for ch in range(NCH):
                dut.capture_valid.value = 1
                dut.pixel_t_in.value = int(t[r, c, ch]) & 0xFF
                dut.pixel_t1_in.value = int(t1[r, c, ch]) & 0xFF
                await RisingEdge(dut.clk)
                await Timer(1, "ns")
    dut.capture_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, "ns")

    # warp+blend: flow/mask per pixel, one position per NCH cycles;
    # collect emissions (pipelined NCH cycles after each flow position)
    got = []
    for r in range(H):
        for c in range(W):
            dut.wb_valid.value = 1
            dut.flow_x.value = int(flow_x[r, c]) & 0xFFFF
            dut.flow_y.value = int(flow_y[r, c]) & 0xFFFF
            dut.mask.value = int(mask[r, c])
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
            dut.wb_valid.value = 0
            for _ in range(NCH + 2):
                await RisingEdge(dut.clk)
                await Timer(1, "ns")
                if int(dut.out_valid.value) == 1:
                    got.append(read_signed(dut.pixel_out, 8))
    dut.wb_valid.value = 0

    # drain any remaining emissions
    for _ in range(H * W * NCH + 20):
        await RisingEdge(dut.clk)
        await Timer(1, "ns")
        if int(dut.out_valid.value) == 1:
            got.append(read_signed(dut.pixel_out, 8))

    # got should be interleaved: pos0ch0,pos0ch1,pos0ch2,pos1ch0,...
    # rebuild per-channel streams
    expected_flat = []
    for pos in range(H * W):
        for c in range(NCH):
            expected_flat.append(expected[c][pos])

    assert len(got) == len(expected_flat), \
        f"count mismatch: got {len(got)}, exp {len(expected_flat)}"
    for i, (g, e) in enumerate(zip(got, expected_flat)):
        assert g == e, f"output {i}: got {g}, exp {e}"

    print(f"✓ test_vfi_synth passed: {len(got)} outputs, NCH={NCH}, {W}x{H}")


@cocotb.test()
async def test_vfi_synth_identity(dut):
    """mask=255, zero flow => interpolated frame == t frame."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    W, H = int(dut.IMG_W.value), int(dut.IMG_H.value)
    NCH = int(dut.NCH.value)
    FLOW_Q = int(dut.FLOW_Q.value)
    random.seed(0x5F22)

    t = np.random.randint(-128, 128, size=(H, W, NCH), dtype=np.int8)
    t1 = np.random.randint(-128, 128, size=(H, W, NCH), dtype=np.int8)

    flow_x = np.zeros((H, W), dtype=np.int16)
    flow_y = np.zeros((H, W), dtype=np.int16)
    mask = np.full((H, W), 255, dtype=np.uint8)  # fully t

    # golden: identity warp of t, mask=255 => ~t
    expected_flat = []
    for pos in range(H * W):
        for c in range(NCH):
            expected_flat.append(int(t.flatten()[pos * NCH + c]))

    for r in range(H):
        for c in range(W):
            for ch in range(NCH):
                dut.capture_valid.value = 1
                dut.pixel_t_in.value = int(t[r, c, ch]) & 0xFF
                dut.pixel_t1_in.value = int(t1[r, c, ch]) & 0xFF
                await RisingEdge(dut.clk)
                await Timer(1, "ns")
    dut.capture_valid.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, "ns")

    got = []
    for r in range(H):
        for c in range(W):
            dut.wb_valid.value = 1
            dut.flow_x.value = 0
            dut.flow_y.value = 0
            dut.mask.value = 255
            await RisingEdge(dut.clk)
            await Timer(1, "ns")
            dut.wb_valid.value = 0
            for _ in range(NCH + 2):
                await RisingEdge(dut.clk)
                await Timer(1, "ns")
                if int(dut.out_valid.value) == 1:
                    got.append(read_signed(dut.pixel_out, 8))
    dut.wb_valid.value = 0

    for _ in range(H * W * NCH + 20):
        await RisingEdge(dut.clk)
        await Timer(1, "ns")
        if int(dut.out_valid.value) == 1:
            got.append(read_signed(dut.pixel_out, 8))

    assert len(got) == len(expected_flat), \
        f"count mismatch: got {len(got)}, exp {len(expected_flat)}"
    for i, (g, e) in enumerate(zip(got, expected_flat)):
        assert g == e, f"output {i}: got {g}, exp {e}"

    print(f"✓ test_vfi_synth_identity passed: mask=255 reproduces t")