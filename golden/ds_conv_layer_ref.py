"""Bit-exact golden model for ds_conv_layer — depthwise-separable convolution.

Chains dw_conv3x3 (per-channel) → pw_conv1x1 (channel mixing).

Input:  (H, W, C_in)  int8
DW kernel: (C_in, 3, 3) int8   — one 3x3 per input channel
DW bias:   (C_in,) int32
PW kernel: (C_out, C_in) int8
PW bias:   (C_out,) int32
Output: (H, W, C_out) int8
"""

import numpy as np
from dw_conv3x3_ref import dw_conv3x3_golden
from pw_conv1x1_ref import pw_conv1x1_golden


def ds_conv_layer_golden(
    image: np.ndarray,       # (H, W, C_in) int8
    dw_kernel: np.ndarray,   # (C_in, 3, 3) int8
    dw_bias: np.ndarray,     # (C_in,) int32
    dw_m0: int,
    dw_shift: int,
    pw_kernel: np.ndarray,   # (C_out, C_in) int8
    pw_bias: np.ndarray,     # (C_out,) int32
    pw_m0: int,
    pw_shift: int,
) -> np.ndarray:
    H, W, C_in = image.shape
    C_out = pw_kernel.shape[0]

    # Phase 1: depthwise — process each channel independently
    dw_out = np.zeros((H, W, C_in), dtype=np.int8)
    for ci in range(C_in):
        dw_out[:, :, ci] = dw_conv3x3_golden(
            image[:, :, ci], dw_kernel[ci], int(dw_bias[ci]), dw_m0, dw_shift
        )

    # Phase 2: pointwise — channel mixing
    pw_out = pw_conv1x1_golden(dw_out, pw_kernel, pw_bias, pw_m0, pw_shift)

    return pw_out
