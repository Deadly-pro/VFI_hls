"""Bilinear grid_sample reference — matches ONNX GridSample (align_corners=1,
bilinear, padding_mode=border/clamp) but in INT8 fixed-point to match the RTL.

Flow convention: displacement in pixels, fixed-point Q<FLOW_Q>.
  flow_x[y,x] = int(round(disp_px * 2**FLOW_Q))
  src = out_pixel + flow  (so flow=0  =>  identity warp)

The RTL warp_unit.v implements the exact same arithmetic; this is the
bit-exactness contract.
"""

import numpy as np


def warp_bilinear_fixed(frame: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray,
                        flow_q: int = 8) -> np.ndarray:
    """Bilinear warp of a single INT8 frame by fixed-point flow.

    Args:
        frame: HxW int8 (or float in [-128,127]) source image.
        flow_x, flow_y: HxW int16 fixed-point (flow_q fractional bits).
        flow_q: number of fractional bits in flow.

    Returns:
        HxW int8 warped frame (border-clamped, round-half-up).
    """
    H, W = frame.shape
    out = np.zeros((H, W), dtype=np.int8)
    scale = 1 << flow_q
    mask_f = scale - 1
    round_bit = 1 << (2 * flow_q - 1)

    for y in range(H):
        for x in range(W):
            sx = (x << flow_q) + int(flow_x[y, x])
            sy = (y << flow_q) + int(flow_y[y, x])
            x0 = sx >> flow_q
            y0 = sy >> flow_q
            wx = sx & mask_f
            wy = sy & mask_f
            x1 = x0 + 1
            y1 = y0 + 1

            # border clamp
            x0c = max(0, min(x0, W - 1))
            x1c = max(0, min(x1, W - 1))
            y0c = max(0, min(y0, H - 1))
            y1c = max(0, min(y1, H - 1))

            p00 = int(frame[y0c, x0c])
            p01 = int(frame[y0c, x1c])
            p10 = int(frame[y1c, x0c])
            p11 = int(frame[y1c, x1c])

            inv = scale
            top = p00 * (inv - wx) + p01 * wx
            bot = p10 * (inv - wx) + p11 * wx
            acc = top * (inv - wy) + bot * wy

            r = (acc + round_bit) >> (2 * flow_q)
            out[y, x] = max(-128, min(127, r))

    return out


def make_flow_identity(H: int, W: int, flow_q: int = 8) -> tuple:
    """Zero flow = identity warp (src == out)."""
    return (np.zeros((H, W), dtype=np.int16),
            np.zeros((H, W), dtype=np.int16))


def make_flow_translate(H: int, W: int, dx: float, dy: float, flow_q: int = 8) -> tuple:
    """Constant displacement (in pixels) flow field."""
    fx = np.full((H, W), int(round(dx * (1 << flow_q))), dtype=np.int16)
    fy = np.full((H, W), int(round(dy * (1 << flow_q))), dtype=np.int16)
    return fx, fy