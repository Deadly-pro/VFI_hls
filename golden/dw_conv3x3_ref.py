"""Bit-exact golden model for dw_conv3x3 — depthwise 3x3 convolution.

For a single channel: convolve the 3x3 kernel over an HxW input with SAME
padding (zero-padded borders), add bias (INT32), then requantize to INT8.

Uses the same requantize logic as requantize_ref.
"""

import numpy as np
from requantize_ref import requantize_ref


def dw_conv3x3_golden(
    image: np.ndarray,  # (H, W) int8
    kernel: np.ndarray,  # (3, 3) int8
    bias: int,  # int32
    m0: int,  # Q0.31 requant multiplier
    shift: int,  # requant shift
) -> np.ndarray:
    """Return (H, W) int8 output."""
    H, W = image.shape
    padded = np.pad(image, 1, mode="constant", constant_values=0).astype(np.int32)
    kernel_i32 = kernel.astype(np.int32)
    out = np.zeros((H, W), dtype=np.int8)

    for r in range(H):
        for c in range(W):
            acc = bias
            for kr in range(3):
                for kc in range(3):
                    acc += int(padded[r + kr, c + kc]) * int(kernel_i32[kr, kc])
            out[r, c] = requantize_ref(int(acc), m0, shift)

    return out
