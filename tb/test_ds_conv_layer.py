"""cocotb testbench for ds_conv_layer — bit-exact depthwise-separable conv.

Streams a small multi-channel image (channel-interleaved), collects output,
and compares to the numpy golden model.

Run from tb/:
  make clean
  make MODULE=test_ds_conv_layer TOPLEVEL=ds_conv_layer \
       VERILOG_SOURCES="../rtl/ds_conv_layer.v ../rtl/dw_conv3x3.v \
       ../rtl/line_buffer.v ../rtl/requantize.v ../rtl/pw_conv1x1.v"
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from ds_conv_layer_ref import ds_conv_layer_golden  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.pixel_in.value = 0
    dut.frame_start.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


@cocotb.test()
async def test_ds_conv(dut):
    """4x4 image, C_IN=4, C_OUT=4 depthwise-separable conv."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)

    C_IN = int(dut.C_IN.value)
    C_OUT = int(dut.C_OUT.value)
    IMG_W = int(dut.IMG_W.value)
    IMG_H = int(dut.IMG_H.value)
    random.seed(0xBEEF)

    # random multi-channel image
    image = np.array(
        [[[random.randint(-64, 63) for _ in range(C_IN)]
          for _ in range(IMG_W)] for _ in range(IMG_H)],
        dtype=np.int8,
    )

    # random DW kernels (one 3x3 per input channel)
    dw_kernel = np.array(
        [[[random.randint(-8, 7) for _ in range(3)] for _ in range(3)]
         for _ in range(C_IN)],
        dtype=np.int8,
    )

    dw_bias = np.array(
        [random.randint(-200, 200) for _ in range(C_IN)],
        dtype=np.int32,
    )

    dw_m0 = random.randint(1 << 30, (1 << 31) - 1)
    dw_shift = random.randint(2, 6)

    # random PW weights
    pw_kernel = np.array(
        [[random.randint(-16, 15) for _ in range(C_IN)] for _ in range(C_OUT)],
        dtype=np.int8,
    )

    pw_bias = np.array(
        [random.randint(-300, 300) for _ in range(C_OUT)],
        dtype=np.int32,
    )

    pw_m0 = random.randint(1 << 30, (1 << 31) - 1)
    pw_shift = random.randint(2, 6)

    # golden reference
    expected = ds_conv_layer_golden(
        image, dw_kernel, dw_bias, dw_m0, dw_shift,
        pw_kernel, pw_bias, pw_m0, pw_shift,
    )

    # load DW weights
    for ci in range(C_IN):
        for t in range(9):
            r, c = divmod(t, 3)
            dut.dw_weight[ci * 9 + t].value = int(dw_kernel[ci, r, c]) & 0xFF
    for ci in range(C_IN):
        dut.dw_bias[ci].value = int(dw_bias[ci]) & 0xFFFFFFFF
    dut.dw_m0.value = dw_m0 & 0xFFFFFFFF
    dut.dw_shift.value = dw_shift

    # load PW weights
    for co in range(C_OUT):
        for ci in range(C_IN):
            dut.pw_weight[co * C_IN + ci].value = int(pw_kernel[co, ci]) & 0xFF
    for co in range(C_OUT):
        dut.pw_bias[co].value = int(pw_bias[co]) & 0xFFFFFFFF
    dut.pw_m0.value = pw_m0 & 0xFFFFFFFF
    dut.pw_shift.value = pw_shift

    await RisingEdge(dut.clk)
    await Timer(1, "ns")

    # pulse frame_start
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    dut.frame_start.value = 0
    await Timer(1, "ns")

    # stream input: channel-interleaved
    got = []
    for r in range(IMG_H):
        for c in range(IMG_W):
            for ci in range(C_IN):
                dut.in_valid.value = 1
                dut.pixel_in.value = int(image[r, c, ci]) & 0xFF
                await RisingEdge(dut.clk)
                await Timer(1, "ns")
                if int(dut.out_valid.value) == 1:
                    got.append(read_signed(dut.pixel_out, 8))

    dut.in_valid.value = 0

    # drain: DW pipeline + PW processing takes many cycles
    # Upper bound: DW latency + NUM_PX * (C_IN + C_OUT*C_IN + C_OUT) + margin
    drain_cycles = 20 + IMG_W * IMG_H * (C_IN + C_OUT * C_IN + C_OUT + 10)
    for _ in range(drain_cycles):
        await RisingEdge(dut.clk)
        await Timer(1, "ns")
        if int(dut.out_valid.value) == 1:
            got.append(read_signed(dut.pixel_out, 8))

    # expected output: (H, W, C_OUT) flattened row-major, channel last
    expected_flat = expected.flatten().tolist()
    assert len(got) == len(expected_flat), (
        f"output count mismatch: got {len(got)}, exp {len(expected_flat)}"
    )
    for i, (g, e) in enumerate(zip(got, expected_flat)):
        pos = i // C_OUT
        co = i % C_OUT
        r, c = divmod(pos, IMG_W)
        assert g == e, (
            f"pos ({r},{c}) co={co}: got {g}, exp {e}"
        )
