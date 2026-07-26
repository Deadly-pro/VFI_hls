"""Bit-exact golden model for weight_mem.

Synchronous (registered) read with 1-cycle latency, matching an SRAM. step()
applies one clock edge and returns rd_data AS PRESENTED AFTER that edge:
the registered read samples memory BEFORE the same-cycle write (read-before-write),
so a simultaneous read+write to the same address returns the OLD value.
"""


class WeightMemRef:
    def __init__(self, depth: int):
        self.depth = depth
        self.mem = [0] * depth
        self.rd_data = 0

    def step(self, wl_we: int, wl_addr: int, wl_data: int, rd_addr: int) -> int:
        captured = self.mem[rd_addr]        # registered read of pre-write memory
        if wl_we:
            self.mem[wl_addr] = wl_data     # weight update
        self.rd_data = captured
        return self.rd_data
