"""Full VFI synthesis reference — warp(t,flow), warp(t+1,flow), mask-blend.

Matches the ONNX tail of nano_v16.onnx:
  out = clip( mask * warp(t, flow) + (1-mask) * warp(t+1, flow), 0, 1 )
but in INT8 fixed-point to match the RTL (vfi_synth.v).

Mask convention: Q0.8 unsigned, mask=1.0 <=> 255.
  out = (mask * a + (255-mask) * b + 127) >> 8  (round-half-up)
"""

import numpy as np
from grid_sample_ref import warp_bilinear_fixed


def blend_fixed(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mask-blend two INT8 frames. mask is Q0.8 unsigned [0,255].

    out = (mask*a + (255-mask)*b + 127) >> 8, clamped to INT8.
    """
    a_i = a.astype(np.int32)
    b_i = b.astype(np.int32)
    m = mask.astype(np.int32)
    acc = m * a_i + (255 - m) * b_i
    acc = (acc + 127) >> 8
    acc = np.clip(acc, -128, 127)
    return acc.astype(np.int8)


def vfi_synth_fixed(t: np.ndarray, t1: np.ndarray,
                    flow_x: np.ndarray, flow_y: np.ndarray,
                    mask: np.ndarray, flow_q: int = 8) -> np.ndarray:
    """Full single-channel VFI synthesis (fixed-point, bit-exact vs RTL).

    Args:
        t, t1: HxW int8 frames (t and t+1).
        flow_x, flow_y: HxW int16 fixed-point flow.
        mask: HxW uint8 Q0.8 blend mask (from flow-head sigmoid).
        flow_q: flow fractional bits.

    Returns:
        HxW int8 interpolated frame at t+0.5.
    """
    warp_t = warp_bilinear_fixed(t, flow_x, flow_y, flow_q)
    warp_t1 = warp_bilinear_fixed(t1, flow_x, flow_y, flow_q)
    return blend_fixed(warp_t, warp_t1, mask)


def vfi_synth_channels(frames_t: np.ndarray, frames_t1: np.ndarray,
                       flow_x: np.ndarray, flow_y: np.ndarray,
                       mask: np.ndarray, flow_q: int = 8) -> np.ndarray:
    """Multi-channel version. Frames are HxWxC, flow/mask are HxW."""
    C = frames_t.shape[2]
    outs = []
    for c in range(C):
        outs.append(vfi_synth_fixed(frames_t[:, :, c], frames_t1[:, :, c],
                                    flow_x, flow_y, mask, flow_q))
    return np.stack(outs, axis=2)