"""Bit-exact golden model for pw_conv1x1 — pointwise 1x1 convolution.

Channel mixing: for each spatial position, compute
    out[co] = requant( sum_ci( in[ci] * weight[co, ci] ) + bias[co] )

Input:  (H, W, C_in)  int8
Kernel: (C_out, C_in)  int8
Bias:   (C_out,)       int32
Output: (H, W, C_out)  int8
"""

import numpy as np
from requantize_ref import requantize_ref


def pw_conv1x1_golden(
    image: np.ndarray,   # (H, W, C_in) int8
    weight: np.ndarray,  # (C_out, C_in) int8
    bias: np.ndarray,    # (C_out,) int32
    m0: int,
    shift: int,
) -> np.ndarray:
    H, W, C_in = image.shape
    C_out = weight.shape[0]
    out = np.zeros((H, W, C_out), dtype=np.int8)

    for r in range(H):
        for c in range(W):
            for co in range(C_out):
                acc = int(bias[co])
                for ci in range(C_in):
                    acc += int(image[r, c, ci]) * int(weight[co, ci])
                out[r, c, co] = requantize_ref(acc, m0, shift)

    return out
