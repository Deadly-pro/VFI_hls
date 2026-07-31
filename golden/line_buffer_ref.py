"""Bit-exact golden model for line_buffer — 3x3 sliding window extractor.

Two modes:
  1. line_buffer_golden(image) — whole-image reference: pad with zeros, extract
     every 3x3 window.  Used to generate expected outputs for the testbench.
  2. LineBufferRef — cycle-accurate streaming model that mirrors the RTL's
     shift-register behavior.  Used when you need to match the RTL's exact
     output timing (pixel-by-pixel).

The RTL streams pixels row-major, one per clock, and starts emitting valid
windows after 2*W+2 pixels (enough to fill 2 lines + reach column 1 of
the 3rd line — the first position with a full 3x3 neighborhood).

For positions at the image border, the window is zero-padded (SAME conv).
"""

import numpy as np


def line_buffer_golden(image: np.ndarray) -> list:
    """Whole-image: return list of H*W 9-element windows (row-major)."""
    H, W = image.shape
    padded = np.pad(image, 1, mode="constant", constant_values=0)
    windows = []
    for r in range(H):
        for c in range(W):
            win = padded[r:r+3, c:c+3].flatten().tolist()
            windows.append(win)
    return windows
