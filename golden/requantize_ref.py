"""Bit-exact golden model for requantize (INT32 → INT8).

TFLite-style fixed-point rescale:
    out = clamp( round( acc * M0 * 2^(-shift) ), -128, 127 )

M0 is a signed 32-bit fixed-point multiplier in Q0.31 format (value in
[0.5, 1.0), so its INT32 representation is in [2^30, 2^31-1]).
shift is the additional right-shift amount (non-negative).

The multiply is 64-bit (acc * M0), then we arithmetic-right-shift by
(31 + shift).  The 31 accounts for M0 being Q0.31.

Rounding: round-half-up (add the rounding bit before shifting).
"""


def requantize_ref(acc: int, m0: int, shift: int) -> int:
    wide = acc * m0
    total_shift = 31 + shift
    round_bit = 1 << (total_shift - 1)
    shifted = (wide + round_bit) >> total_shift
    return max(-128, min(127, shifted))
