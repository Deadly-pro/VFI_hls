"""Bit-exact golden model for mac_int8.

The RTL accumulator is a registered 32-bit signed value. This model mirrors it
cycle-for-cycle: step() applies one clock's worth of inputs and returns acc AFTER
that edge, exactly as the hardware would present it.
"""


def wrap_int32(x: int) -> int:
    """Two's-complement wrap into signed 32-bit range, matching HW overflow."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x & (1 << 31) else x


class MacInt8Ref:
    def __init__(self):
        self.acc = 0

    def reset(self):
        self.acc = 0

    def step(self, en: int, clear: int, a: int, b: int) -> int:
        """One clock edge. clear has priority over en (matches RTL)."""
        if clear:
            self.acc = 0
        elif en:
            self.acc = wrap_int32(self.acc + a * b)
        return self.acc
