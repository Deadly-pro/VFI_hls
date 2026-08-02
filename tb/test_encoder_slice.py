"""cocotb testbench for encoder_slice — two-layer strided encoder.

Chains E1 (ds_conv_layer_integrated, stride-2) -> E2 (stride-2) and compares
the 2×2×C_OUT result against a chained NumPy golden. Weight-load addresses
mirror encoder_slice.v: E1 owns [0, E1_WEIGHT_SPACE), E2 owns
[E1_WEIGHT_SPACE, E1_WEIGHT_SPACE + E2_WEIGHT_SPACE).
"""

import os
import sys
import random
import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from ds_conv_layer_ref import ds_conv_layer_golden


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def q8(v):
    """Quantize a 32-bit bias through the RTL's 8-bit weight_mem port
    (truncate to 8 bits, then sign-extend). Golden must match."""
    x = int(v) & 0xFF
    return x - 256 if x & 0x80 else x


def q8_arr(a):
    return np.array([q8(x) for x in a], dtype=np.int32)


def stride2_downsample(x):
    """Keep even (row, col) positions: [H, W, C] -> [H/2, W/2, C]."""
    return x[0::2, 0::2, :]


@cocotb.test()
async def test_encoder_slice_two_layer(dut):
    """Two-layer encoder with stride-2 downsampling."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    IMG_W = int(dut.IMG_W.value)
    IMG_H = int(dut.IMG_H.value)
    C_IN  = int(dut.C_IN.value)
    C_MID = int(dut.C_MID.value)
    C_OUT = int(dut.C_OUT.value)

    E1_OUT_W = IMG_W // 2
    E2_OUT_W = E1_OUT_W // 2
    E2_OUT_H = E1_OUT_W // 2
    TOTAL_OUT_PX = E2_OUT_W * E2_OUT_H

    random.seed(0xE0C0)
    np.random.seed(0xE0C0)

    # ---- Generate test data ----
    image = np.random.randint(-128, 127, size=(IMG_H, IMG_W, C_IN), dtype=np.int8)
    e1_dw_kernel = np.random.randint(-128, 127, size=(C_IN, 3, 3), dtype=np.int8)
    e1_dw_bias = np.random.randint(-2000, 2000, size=(C_IN,), dtype=np.int32)
    e1_pw_weight = np.random.randint(-128, 127, size=(C_MID, C_IN), dtype=np.int8)
    e1_pw_bias = np.random.randint(-2000, 2000, size=(C_MID,), dtype=np.int32)
    e2_dw_kernel = np.random.randint(-128, 127, size=(C_MID, 3, 3), dtype=np.int8)
    e2_dw_bias = np.random.randint(-2000, 2000, size=(C_MID,), dtype=np.int32)
    e2_pw_weight = np.random.randint(-128, 127, size=(C_OUT, C_MID), dtype=np.int8)
    e2_pw_bias = np.random.randint(-2000, 2000, size=(C_OUT,), dtype=np.int32)

    e1_dw_m0 = 0x40000000; e1_dw_shift = 0
    e1_pw_m0 = 0x40000000; e1_pw_shift = 0
    e2_dw_m0 = 0x40000000; e2_dw_shift = 0
    e2_pw_m0 = 0x40000000; e2_pw_shift = 0

    # ---- Golden reference (chained), with 8-bit-quantized biases ----
    e1_full = ds_conv_layer_golden(
        image, e1_dw_kernel, q8_arr(e1_dw_bias), e1_dw_m0, e1_dw_shift,
        e1_pw_weight, q8_arr(e1_pw_bias), e1_pw_m0, e1_pw_shift
    )  # [8, 8, C_MID]
    e1_strided = stride2_downsample(e1_full)  # [4, 4, C_MID]
    e2_full = ds_conv_layer_golden(
        e1_strided, e2_dw_kernel, q8_arr(e2_dw_bias), e2_dw_m0, e2_dw_shift,
        e2_pw_weight, q8_arr(e2_pw_bias), e2_pw_m0, e2_pw_shift
    )  # [4, 4, C_OUT]
    e2_strided = stride2_downsample(e2_full)  # [2, 2, C_OUT]
    golden_flat = e2_strided.reshape(-1).tolist()

    # ---- Weight-load address map (mirrors encoder_slice.v) ----
    E1_DW_SPACE = C_IN * 9 + C_IN
    E1_PW_SPACE = C_MID * C_IN + C_MID
    E1_WEIGHT_SPACE = E1_DW_SPACE + E1_PW_SPACE
    E2_DW_SPACE = C_MID * 9 + C_MID
    E2_PW_SPACE = C_OUT * C_MID + C_OUT

    # ---- Reset ----
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.pixel_in.value = 0
    dut.frame_start.value = 0
    dut.wl_we.value = 0
    dut.wl_addr.value = 0
    dut.wl_data.value = 0
    dut.e1_dw_m0.value = e1_dw_m0
    dut.e1_dw_shift.value = e1_dw_shift
    dut.e1_pw_m0.value = e1_pw_m0
    dut.e1_pw_shift.value = e1_pw_shift
    dut.e2_dw_m0.value = e2_dw_m0
    dut.e2_dw_shift.value = e2_dw_shift
    dut.e2_pw_m0.value = e2_pw_m0
    dut.e2_pw_shift.value = e2_pw_shift

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    async def load_e1_weights():
        dut.wl_we.value = 1
        for ch in range(C_IN):
            for t in range(9):
                dut.wl_addr.value = ch * 9 + t
                dut.wl_data.value = int(e1_dw_kernel[ch, t // 3, t % 3])
                await RisingEdge(dut.clk)
        for ch in range(C_IN):
            dut.wl_addr.value = C_IN * 9 + ch
            dut.wl_data.value = int(e1_dw_bias[ch]) & 0xFF
            await RisingEdge(dut.clk)
        for co in range(C_MID):
            for ci in range(C_IN):
                dut.wl_addr.value = E1_DW_SPACE + co * C_IN + ci
                dut.wl_data.value = int(e1_pw_weight[co, ci])
                await RisingEdge(dut.clk)
        for co in range(C_MID):
            dut.wl_addr.value = E1_DW_SPACE + C_MID * C_IN + co
            dut.wl_data.value = int(e1_pw_bias[co]) & 0xFF
            await RisingEdge(dut.clk)
        dut.wl_we.value = 0
        await RisingEdge(dut.clk)

    async def load_e2_weights():
        dut.wl_we.value = 1
        for ch in range(C_MID):
            for t in range(9):
                dut.wl_addr.value = E1_WEIGHT_SPACE + ch * 9 + t
                dut.wl_data.value = int(e2_dw_kernel[ch, t // 3, t % 3])
                await RisingEdge(dut.clk)
        for ch in range(C_MID):
            dut.wl_addr.value = E1_WEIGHT_SPACE + C_MID * 9 + ch
            dut.wl_data.value = int(e2_dw_bias[ch]) & 0xFF
            await RisingEdge(dut.clk)
        for co in range(C_OUT):
            for ci in range(C_MID):
                dut.wl_addr.value = E1_WEIGHT_SPACE + E2_DW_SPACE + co * C_MID + ci
                dut.wl_data.value = int(e2_pw_weight[co, ci])
                await RisingEdge(dut.clk)
        for co in range(C_OUT):
            dut.wl_addr.value = E1_WEIGHT_SPACE + E2_DW_SPACE + C_OUT * C_MID + co
            dut.wl_data.value = int(e2_pw_bias[co]) & 0xFF
            await RisingEdge(dut.clk)
        dut.wl_we.value = 0
        await RisingEdge(dut.clk)

    await load_e1_weights()
    await load_e2_weights()

    # ---- Stream input ----
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    dut.frame_start.value = 0

    for py in range(IMG_H):
        for px in range(IMG_W):
            for ci in range(C_IN):
                dut.in_valid.value = 1
                dut.pixel_in.value = int(image[py, px, ci])
                await RisingEdge(dut.clk)
    dut.in_valid.value = 0

    # ---- Collect E2 output (out_last pulses per C_OUT group, so collect
    # until the expected count, not until out_last) ----
    outputs = []
    # Serial pw_conv1x1 is O(C_OUT*C_IN) per position: E1 PW ~22k cycles,
    # E2 PW ~76k cycles at the real Nano dims. Give it generous headroom.
    timeout = TOTAL_OUT_PX * C_OUT + 130000
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            outputs.append(read_signed(dut.pixel_out, 8))
            if len(outputs) == len(golden_flat):
                break

    assert len(outputs) == len(golden_flat), \
        f"Output count mismatch: got {len(outputs)}, expected {len(golden_flat)}"
    for i, (got, exp) in enumerate(zip(outputs, golden_flat)):
        assert got == exp, f"Mismatch at output[{i}]: got {got}, expected {exp}"

    print(f"✓ test_encoder_slice_two_layer passed: {len(outputs)} outputs match golden")
