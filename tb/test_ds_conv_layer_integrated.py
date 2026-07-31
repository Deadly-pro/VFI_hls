"""cocotb testbench for ds_conv_layer_integrated — end-to-end with weight_mem.

Tests:
1. Load weights into weight_mem via wl_we/wl_addr/wl_data
2. Run inference and compare to golden reference (bit-exact)
3. Reload new weights at runtime and verify second inference produces different output
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


@cocotb.test()
async def test_ds_conv_integrated_load_and_infer(dut):
    """Load weights, run inference, verify bit-exact vs golden."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    IMG_W = int(dut.IMG_W.value)
    IMG_H = int(dut.IMG_H.value)
    C_IN  = int(dut.C_IN.value)
    C_OUT = int(dut.C_OUT.value)
    NUM_PX = IMG_W * IMG_H

    # ---- Generate deterministic test data ----
    random.seed(0xCAFE)
    np.random.seed(0xCAFE)

    # Input image: [H, W, C_IN], channel-interleaved stream order
    image = np.random.randint(-128, 127, size=(IMG_H, IMG_W, C_IN), dtype=np.int8)

    # DW weights: [C_IN, 3, 3] -> flatten to [C_IN, 9]
    dw_kernel = np.random.randint(-128, 127, size=(C_IN, 3, 3), dtype=np.int8)
    dw_bias = np.random.randint(-2000, 2000, size=(C_IN,), dtype=np.int32)
    dw_m0 = 0x40000000  # ~0.5 in Q0.31
    dw_shift = 0

    # PW weights: [C_OUT, C_IN]
    pw_weight = np.random.randint(-128, 127, size=(C_OUT, C_IN), dtype=np.int8)
    pw_bias = np.random.randint(-2000, 2000, size=(C_OUT,), dtype=np.int32)
    pw_m0 = 0x40000000
    pw_shift = 0

    # ---- Compute golden reference ----
    golden_out = ds_conv_layer_golden(
        image, dw_kernel, dw_bias, dw_m0, dw_shift,
        pw_weight, pw_bias, pw_m0, pw_shift
    )  # shape [H, W, C_OUT]
    golden_flat = golden_out.reshape(-1).tolist()  # row-major, channel-interleaved

    # ---- Reset DUT ----
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.pixel_in.value = 0
    dut.frame_start.value = 0
    dut.wl_we.value = 0
    dut.wl_addr.value = 0
    dut.wl_data.value = 0
    dut.dw_m0.value = dw_m0
    dut.dw_shift.value = dw_shift
    dut.pw_m0.value = pw_m0
    dut.pw_shift.value = pw_shift

    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # ---- Phase 1: Load DW weights into weight_mem ----
    # Layout: [ch*9 + t] for kernels, then [C_IN*9 + ch] for biases
    dut.wl_we.value = 1
    for ch in range(C_IN):
        for t in range(9):
            addr = ch * 9 + t
            dut.wl_addr.value = addr
            dut.wl_data.value = int(dw_kernel[ch, t // 3, t % 3])
            await RisingEdge(dut.clk)
    for ch in range(C_IN):
        addr = C_IN * 9 + ch
        dut.wl_addr.value = addr
        # bias is 32-bit, weight_mem is 8-bit - store lower 8 bits (truncation is expected for demo)
        dut.wl_data.value = int(dw_bias[ch]) & 0xFF
        await RisingEdge(dut.clk)

    # ---- Phase 2: Load PW weights into weight_mem ----
    # PW weight_mem starts at same address space (shared bus), but we use different offsets
    # Layout: [co*C_IN + ci] for kernels, then [C_OUT*C_IN + co] for biases
    for co in range(C_OUT):
        for ci in range(C_IN):
            addr = co * C_IN + ci
            dut.wl_addr.value = addr
            dut.wl_data.value = int(pw_weight[co, ci])
            await RisingEdge(dut.clk)
    for co in range(C_OUT):
        addr = C_OUT * C_IN + co
        dut.wl_addr.value = addr
        dut.wl_data.value = int(pw_bias[co]) & 0xFF
        await RisingEdge(dut.clk)

    dut.wl_we.value = 0
    await RisingEdge(dut.clk)

    # ---- Phase 3: Stream input image (channel-interleaved) ----
    dut.frame_start.value = 1
    await RisingEdge(dut.clk)
    dut.frame_start.value = 0

    # Feed pixels: ch0_pos0, ch1_pos0, ..., ch(C_IN-1)_pos0, ch0_pos1, ...
    for py in range(IMG_H):
        for px in range(IMG_W):
            for ci in range(C_IN):
                dut.in_valid.value = 1
                dut.pixel_in.value = int(image[py, px, ci])
                await RisingEdge(dut.clk)

    dut.in_valid.value = 0

    # ---- Phase 4: Collect outputs ----
    outputs = []
    timeout_cycles = (IMG_W * IMG_H * C_OUT) + 200  # generous timeout
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if dut.out_valid.value == 1:
            outputs.append(read_signed(dut.pixel_out, 8))
            if dut.out_last.value == 1:
                break

    # ---- Verify ----
    assert len(outputs) == len(golden_flat), f"Output count mismatch: got {len(outputs)}, expected {len(golden_flat)}"
    for i, (got, exp) in enumerate(zip(outputs, golden_flat)):
        assert got == exp, f"Mismatch at output[{i}]: got {got}, expected {exp}"

    print(f"✓ test_ds_conv_integrated_load_and_infer passed: {len(outputs)} outputs match golden")


@cocotb.test()
async def test_ds_conv_integrated_weight_reload(dut):
    """Verify runtime weight reload changes output without resynthesis."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    IMG_W = int(dut.IMG_W.value)
    IMG_H = int(dut.IMG_H.value)
    C_IN  = int(dut.C_IN.value)
    C_OUT = int(dut.C_OUT.value)

    random.seed(0xBEEF)
    np.random.seed(0xBEEF)

    # First weight set
    image = np.random.randint(-128, 127, size=(IMG_H, IMG_W, C_IN), dtype=np.int8)
    dw_kernel_1 = np.random.randint(-128, 127, size=(C_IN, 3, 3), dtype=np.int8)
    dw_bias_1 = np.random.randint(-2000, 2000, size=(C_IN,), dtype=np.int32)
    pw_weight_1 = np.random.randint(-128, 127, size=(C_OUT, C_IN), dtype=np.int8)
    pw_bias_1 = np.random.randint(-2000, 2000, size=(C_OUT,), dtype=np.int32)

    dw_m0 = 0x40000000
    dw_shift = 0
    pw_m0 = 0x40000000
    pw_shift = 0

    golden_1 = ds_conv_layer_golden(
        image, dw_kernel_1, dw_bias_1, dw_m0, dw_shift,
        pw_weight_1, pw_bias_1, pw_m0, pw_shift
    ).reshape(-1).tolist()

    # Second weight set (different)
    dw_kernel_2 = np.random.randint(-128, 127, size=(C_IN, 3, 3), dtype=np.int8)
    dw_bias_2 = np.random.randint(-2000, 2000, size=(C_IN,), dtype=np.int32)
    pw_weight_2 = np.random.randint(-128, 127, size=(C_OUT, C_IN), dtype=np.int8)
    pw_bias_2 = np.random.randint(-2000, 2000, size=(C_OUT,), dtype=np.int32)

    golden_2 = ds_conv_layer_golden(
        image, dw_kernel_2, dw_bias_2, dw_m0, dw_shift,
        pw_weight_2, pw_bias_2, pw_m0, pw_shift
    ).reshape(-1).tolist()

    # ---- Helper: load weights ----
    async def load_weights(dw_k, dw_b, pw_w, pw_b):
        dut.wl_we.value = 1
        for ch in range(C_IN):
            for t in range(9):
                dut.wl_addr.value = ch * 9 + t
                dut.wl_data.value = int(dw_k[ch, t // 3, t % 3])
                await RisingEdge(dut.clk)
        for ch in range(C_IN):
            dut.wl_addr.value = C_IN * 9 + ch
            dut.wl_data.value = int(dw_b[ch]) & 0xFF
            await RisingEdge(dut.clk)
        for co in range(C_OUT):
            for ci in range(C_IN):
                dut.wl_addr.value = co * C_IN + ci
                dut.wl_data.value = int(pw_w[co, ci])
                await RisingEdge(dut.clk)
        for co in range(C_OUT):
            dut.wl_addr.value = C_OUT * C_IN + co
            dut.wl_data.value = int(pw_b[co]) & 0xFF
            await RisingEdge(dut.clk)
        dut.wl_we.value = 0
        await RisingEdge(dut.clk)

    # ---- Helper: run inference ----
    async def run_inference(img):
        outputs = []
        dut.frame_start.value = 1
        await RisingEdge(dut.clk)
        dut.frame_start.value = 0

        for py in range(IMG_H):
            for px in range(IMG_W):
                for ci in range(C_IN):
                    dut.in_valid.value = 1
                    dut.pixel_in.value = int(img[py, px, ci])
                    await RisingEdge(dut.clk)

        dut.in_valid.value = 0

        timeout = (IMG_W * IMG_H * C_OUT) + 200
        for _ in range(timeout):
            await RisingEdge(dut.clk)
            if dut.out_valid.value == 1:
                outputs.append(read_signed(dut.pixel_out, 8))
                if dut.out_last.value == 1:
                    break
        return outputs

    # ---- Reset ----
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.frame_start.value = 0
    dut.wl_we.value = 0
    dut.dw_m0.value = dw_m0
    dut.dw_shift.value = dw_shift
    dut.pw_m0.value = pw_m0
    dut.pw_shift.value = pw_shift
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # ---- First inference with weight set 1 ----
    await load_weights(dw_kernel_1, dw_bias_1, pw_weight_1, pw_bias_1)
    out_1 = await run_inference(image)

    # ---- Reload weights (set 2) ----
    await load_weights(dw_kernel_2, dw_bias_2, pw_weight_2, pw_bias_2)

    # ---- Second inference with weight set 2 ----
    out_2 = await run_inference(image)

    # ---- Verify both match their respective goldens ----
    assert out_1 == golden_1, "First inference after load 1 mismatch"
    assert out_2 == golden_2, "Second inference after load 2 mismatch"
    assert out_1 != out_2, "Weight reload did not change output!"

    print(f"✓ test_ds_conv_integrated_weight_reload passed: weight reload verified")


@cocotb.test()
async def test_ds_conv_integrated_back_to_back_frames(dut):
    """Verify consecutive frame_start pulses work correctly."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())

    IMG_W = int(dut.IMG_W.value)
    IMG_H = int(dut.IMG_H.value)
    C_IN  = int(dut.C_IN.value)
    C_OUT = int(dut.C_OUT.value)

    random.seed(0xDEAD)
    np.random.seed(0xDEAD)

    image = np.random.randint(-128, 127, size=(IMG_H, IMG_W, C_IN), dtype=np.int8)
    dw_kernel = np.random.randint(-128, 127, size=(C_IN, 3, 3), dtype=np.int8)
    dw_bias = np.random.randint(-2000, 2000, size=(C_IN,), dtype=np.int32)
    pw_weight = np.random.randint(-128, 127, size=(C_OUT, C_IN), dtype=np.int8)
    pw_bias = np.random.randint(-2000, 2000, size=(C_OUT,), dtype=np.int32)

    dw_m0 = 0x40000000
    dw_shift = 0
    pw_m0 = 0x40000000
    pw_shift = 0

    golden = ds_conv_layer_golden(
        image, dw_kernel, dw_bias, dw_m0, dw_shift,
        pw_weight, pw_bias, pw_m0, pw_shift
    ).reshape(-1).tolist()

    # Load weights
    dut.wl_we.value = 1
    for ch in range(C_IN):
        for t in range(9):
            dut.wl_addr.value = ch * 9 + t
            dut.wl_data.value = int(dw_kernel[ch, t // 3, t % 3])
            await RisingEdge(dut.clk)
    for ch in range(C_IN):
        dut.wl_addr.value = C_IN * 9 + ch
        dut.wl_data.value = int(dw_bias[ch]) & 0xFF
        await RisingEdge(dut.clk)
    for co in range(C_OUT):
        for ci in range(C_IN):
            dut.wl_addr.value = co * C_IN + ci
            dut.wl_data.value = int(pw_weight[co, ci])
            await RisingEdge(dut.clk)
    for co in range(C_OUT):
        dut.wl_addr.value = C_OUT * C_IN + co
        dut.wl_data.value = int(pw_bias[co]) & 0xFF
        await RisingEdge(dut.clk)
    dut.wl_we.value = 0
    await RisingEdge(dut.clk)

    # Reset
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.frame_start.value = 0
    dut.dw_m0.value = dw_m0
    dut.dw_shift.value = dw_shift
    dut.pw_m0.value = pw_m0
    dut.pw_shift.value = pw_shift
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Helper to run one frame
    async def run_one_frame():
        outputs = []
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

        timeout = (IMG_W * IMG_H * C_OUT) + 200
        for _ in range(timeout):
            await RisingEdge(dut.clk)
            if dut.out_valid.value == 1:
                outputs.append(read_signed(dut.pixel_out, 8))
                if dut.out_last.value == 1:
                    break
        return outputs

    # Frame 1
    out_1 = await run_one_frame()
    # Frame 2 (back-to-back)
    out_2 = await run_one_frame()

    assert out_1 == golden, "Frame 1 mismatch"
    assert out_2 == golden, "Frame 2 mismatch"

    print("✓ test_ds_conv_integrated_back_to_back_frames passed")