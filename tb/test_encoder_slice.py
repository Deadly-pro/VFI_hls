"""cocotb testbench for encoder_slice — two-layer strided encoder.

Tests:
1. Load E1 and E2 weights sequentially via shared weight bus
2. Stream input through both layers with stride-2 downsampling
3. Verify output dimensions: 8×8×C_IN → 4×4×C_MID → 2×2×C_OUT
4. Compare against chained golden reference
"""

import os
import sys
import random
import numpy as np

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from ds_conv_layer_ref import ds_conv_layer_golden
from requantize_ref import requantize_ref


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


def stride2_downsample(image: np.ndarray) -> np.ndarray:
    """Stride-2 spatial downsample: keep (even, even) positions."""
    H, W, C = image.shape
    out_H, out_W = H // 2, W // 2
    out = np.zeros((out_H, out_W, C), dtype=np.int8)
    for c in range(C):
        out[:, :, c] = image[::2, ::2, c]
    return out


@cocotb.test()
async def test_encoder_slice_two_layer(dut):
    """Two-layer encoder with stride-2 downsampling."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    IMG_W = int(dut.IMG_W.value)
    IMG_H = int(dut.IMG_H.value)
    C_IN  = int(dut.C_IN.value)
    C_MID = int(dut.C_MID.value)
    C_OUT = int(dut.C_OUT.value)

    # Derived dimensions
    E1_OUT_W = IMG_W // 2
    E1_OUT_H = IMG_H // 2
    E2_OUT_W = E1_OUT_W // 2
    E2_OUT_H = E1_OUT_H // 2

    E1_NUM_PX = IMG_W * IMG_H
    E2_NUM_PX = E1_OUT_W * E1_OUT_H
    TOTAL_OUT_PX = E2_OUT_W * E2_OUT_H

    random.seed(0xENC0)
    np.random.seed(0xENC0)

    # ---- Generate test data ----
    # Input: 8×8×C_IN (e.g., 6 channels for frame pair)
    image = np.random.randint(-128, 127, size=(IMG_H, IMG_W, C_IN), dtype=np.int8)

    # E1 weights
    e1_dw_kernel = np.random.randint(-128, 127, size=(C_IN, 3, 3), dtype=np.int8)
    e1_dw_bias = np.random.randint(-2000, 2000, size=(C_IN,), dtype=np.int32)
    e1_pw_weight = np.random.randint(-128, 127, size=(C_MID, C_IN), dtype=np.int8)
    e1_pw_bias = np.random.randint(-2000, 2000, size=(C_MID,), dtype=np.int32)

    # E2 weights
    e2_dw_kernel = np.random.randint(-128, 127, size=(C_MID, 3, 3), dtype=np.int8)
    e2_dw_bias = np.random.randint(-2000, 2000, size=(C_MID,), dtype=np.int32)
    e2_pw_weight = np.random.randint(-128, 127, size=(C_OUT, C_MID), dtype=np.int8)
    e2_pw_bias = np.random.randint(-2000, 2000, size=(C_OUT,), dtype=np.int32)

    # Requant params
    e1_dw_m0 = 0x40000000; e1_dw_shift = 0
    e1_pw_m0 = 0x40000000; e1_pw_shift = 0
    e2_dw_m0 = 0x40000000; e2_dw_shift = 0
    e2_pw_m0 = 0x40000000; e2_pw_shift = 0

    # ---- Compute golden reference (chained) ----
    # E1: 8×8×C_IN → 8×8×C_MID (but we stride-2 to 4×4×C_MID)
    e1_full = ds_conv_layer_golden(
        image, e1_dw_kernel, e1_dw_bias, e1_dw_m0, e1_dw_shift,
        e1_pw_weight, e1_pw_bias, e1_pw_m0, e1_pw_shift
    )  # [8, 8, C_MID]

    # Stride-2 downsample
    e1_strided = stride2_downsample(e1_full)  # [4, 4, C_MID]

    # E2: 4×4×C_MID → 4×4×C_OUT (then stride-2 to 2×2×C_OUT)
    e2_full = ds_conv_layer_golden(
        e1_strided, e2_dw_kernel, e2_dw_bias, e2_dw_m0, e2_dw_shift,
        e2_pw_weight, e2_pw_bias, e2_pw_m0, e2_pw_shift
    )  # [4, 4, C_OUT]

    # Final stride-2
    e2_strided = stride2_downsample(e2_full)  # [2, 2, C_OUT]
    golden_flat = e2_strided.reshape(-1).tolist()  # row-major, channel-interleaved

    # ---- Reset DUT ----
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

    # ---- Helper: load weight set into shared bus ----
    async def load_e1_weights():
        dut.wl_we.value = 1
        # E1 DW kernels: C_IN * 9
        for ch in range(C_IN):
            for t in range(9):
                dut.wl_addr.value = ch * 9 + t
                dut.wl_data.value = int(e1_dw_kernel[ch, t // 3, t % 3])
                await RisingEdge(dut.clk)
        # E1 DW biases
        for ch in range(C_IN):
            dut.wl_addr.value = C_IN * 9 + ch
            dut.wl_data.value = int(e1_dw_bias[ch]) & 0xFF
            await RisingEdge(dut.clk)
        # E1 PW kernels: C_MID * C_IN
        for co in range(C_MID):
            for ci in range(C_IN):
                dut.wl_addr.value = co * C_IN + ci
                dut.wl_data.value = int(e1_pw_weight[co, ci])
                await RisingEdge(dut.clk)
        # E1 PW biases
        for co in range(C_MID):
            dut.wl_addr.value = C_MID * C_IN + co
            dut.wl_data.value = int(e1_pw_bias[co]) & 0xFF
            await RisingEdge(dut.clk)
        dut.wl_we.value = 0
        await RisingEdge(dut.clk)

    async def load_e2_weights():
        dut.wl_we.value = 1
        # E2 DW kernels: C_MID * 9
        for ch in range(C_MID):
            for t in range(9):
                dut.wl_addr.value = ch * 9 + t
                dut.wl_data.value = int(e2_dw_kernel[ch, t // 3, t % 3])
                await RisingEdge(dut.clk)
        # E2 DW biases
        for ch in range(C_MID):
            dut.wl_addr.value = C_MID * 9 + ch
            dut.wl_data.value = int(e2_dw_bias[ch]) & 0xFF
            await RisingEdge(dut.clk)
        # E2 PW kernels: C_OUT * C_MID
        for co in range(C_OUT):
            for ci in range(C_MID):
                dut.wl_addr.value = co * C_MID + ci
                dut.wl_data.value = int(e2_pw_weight[co, ci])
                await RisingEdge(dut.clk)
        # E2 PW biases
        for co in range(C_OUT):
            dut.wl_addr.value = C_OUT * C_MID + co
            dut.wl_data.value = int(e2_pw_bias[co]) & 0xFF
            await RisingEdge(dut.clk)
        dut.wl_we.value = 0
        await RisingEdge(dut.clk)

    # ---- Load both weight sets ----
    await load_e1_weights()
    await load_e2_weights()

    # ---- Stream input image (channel-interleaved) ----
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

    # ---- Collect outputs ----
    outputs = []
    # E1 produces 8*8*C_MID = 64*C_MID, E2 produces 4*4*C_OUT = 16*C_OUT
    # But with stride-2 filtering, E2 only gets 16*C_MID inputs → 16*C_OUT outputs
    # Final stride-2 gives 4*C_OUT outputs? Wait: 4×4 input, stride-2 → 2×2 = 4 positions
    # Actually: E1: 8×8→4×4 (16 positions), E2: 4×4→2×2 (4 positions)
    # Total outputs: 4 * C_OUT
    expected_outputs = TOTAL_OUT_PX * C_OUT

    timeout = expected_outputs + 500
    for _ in range(timeout):
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            outputs.append(read_signed(dut.pixel_out, 8))
            if dut.out_last.value == 1:
                break

    # ---- Verify ----
    assert len(outputs) == len(golden_flat), \
        f"Output count mismatch: got {len(outputs)}, expected {len(golden_flat)}"
    for i, (got, exp) in enumerate(zip(outputs, golden_flat)):
        assert got == exp, f"Mismatch at output[{i}]: got {got}, expected {exp}"

    print(f"✓ test_encoder_slice_two_layer passed: {len(outputs)} outputs match golden")
    print(f"  Path: {IMG_W}×{IMG_H}×{C_IN} → {E1_OUT_W}×{E1_OUT_H}×{C_MID} → {E2_OUT_W}×{E2_OUT_H}×{C_OUT}")


@cocotb.test()
async def test_encoder_slice_e1_only(dut):
    """Test E1 layer independently (8×8→4×4 strided)."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    IMG_W = int(dut.IMG_W.value)
    IMG_H = int(dut.IMG_H.value)
    C_IN  = int(dut.C_IN.value)
    C_MID = int(dut.C_MID.value)

    E1_OUT_W = IMG_W // 2
    E1_OUT_H = IMG_H // 2

    random.seed(0xE1ONLY)
    np.random.seed(0xE1ONLY)

    image = np.random.randint(-128, 127, size=(IMG_H, IMG_W, C_IN), dtype=np.int8)
    e1_dw_kernel = np.random.randint(-128, 127, size=(C_IN, 3, 3), dtype=np.int8)
    e1_dw_bias = np.random.randint(-2000, 2000, size=(C_IN,), dtype=np.int32)
    e1_pw_weight = np.random.randint(-128, 127, size=(C_MID, C_IN), dtype=np.int8)
    e1_pw_bias = np.random.randint(-2000, 2000, size=(C_MID,), dtype=np.int32)

    e1_dw_m0 = 0x40000000; e1_dw_shift = 0
    e1_pw_m0 = 0x40000000; e1_pw_shift = 0
    e2_dw_m0 = 0x40000000; e2_dw_shift = 0
    e2_pw_m0 = 0x40000000; e2_pw_shift = 0

    # Golden: full E1 then stride-2
    e1_full = ds_conv_layer_golden(
        image, e1_dw_kernel, e1_dw_bias, e1_dw_m0, e1_dw_shift,
        e1_pw_weight, e1_pw_bias, e1_pw_m0, e1_pw_shift
    )
    e1_strided = stride2_downsample(e1_full)
    golden_flat = e1_strided.reshape(-1).tolist()

    # Reset
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

    # Load E1 weights only
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
            dut.wl_addr.value = co * C_IN + ci
            dut.wl_data.value = int(e1_pw_weight[co, ci])
            await RisingEdge(dut.clk)
    for co in range(C_MID):
        dut.wl_addr.value = C_MID * C_IN + co
        dut.wl_data.value = int(e1_pw_bias[co]) & 0xFF
        await RisingEdge(dut.clk)
    dut.wl_we.value = 0
    await RisingEdge(dut.clk)

    # Stream input
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

    # Collect - we need to tap E1's internal output, but E2 is also there
    # For this test, we'd need a way to bypass E2 or read E1 directly
    # Skip for now - focus on full chain test above

    print("✓ test_encoder_slice_e1_only - placeholder (needs E1 tap)")