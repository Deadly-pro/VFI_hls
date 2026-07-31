"""cocotb testbench for weight_mem — bit-exact vs the numpy golden model.

Phase 1: load every address with a known value (avoids reading uninitialized x).
Phase 2: random reads interleaved with runtime weight updates, verifying the
1-cycle registered-read latency and read-before-write semantics.

Run from tb/:
  make clean
  make MODULE=test_weight_mem TOPLEVEL=weight_mem VERILOG_SOURCES=../rtl/weight_mem.v
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from weight_mem_ref import WeightMemRef  # noqa: E402


def read_signed(sig, bits: int) -> int:
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def step(dut, ref, we, waddr, wdata, raddr, i):
    dut.wl_we.value = we
    dut.wl_addr.value = waddr
    dut.wl_data.value = wdata
    dut.rd_addr.value = raddr
    await RisingEdge(dut.clk)
    await Timer(1, "ns")
    exp = ref.step(we, waddr, wdata, raddr)
    got = read_signed(dut.rd_data, 8)
    assert got == exp, (
        f"step {i}: we={we} waddr={waddr} wdata={wdata} raddr={raddr} "
        f"-> rd_data got {got}, exp {exp}"
    )


@cocotb.test()
async def test_load_then_read(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    depth = int(dut.DEPTH.value)
    ref = WeightMemRef(depth)
    random.seed(0xBEEF)

    # Prime the registered read before checking it. Memory contents are
    # intentionally unknown until every address has been loaded below.
    # First write address 0. The following clock is the first defined read;
    # both the golden model and DUT see address 0's pre-write value (unknown in
    # hardware), so begin comparisons only after this seeding write.
    dut.wl_we.value = 1
    dut.wl_addr.value = 0
    dut.wl_data.value = -128
    dut.rd_addr.value = 0
    await RisingEdge(dut.clk)
    ref.step(1, 0, -128, 0)
    await Timer(1, "ns")

    # Phase 1 — load every remaining address with a known signed value.
    for a in range(1, depth):
        w = ((a * 7) % 256) - 128          # deterministic spread over [-128,127]
        await step(dut, ref, 1, a, w, 0, i=a)

    # Commit the final write and sample known address 0 before random traffic.
    await step(dut, ref, 0, 0, 0, 0, i=depth)

    # Phase 2 — random reads + occasional runtime updates.
    for i in range(3000):
        raddr = random.randrange(depth)
        we = 1 if random.random() < 0.3 else 0
        waddr = random.randrange(depth)
        wdata = random.randint(-128, 127)
        await step(dut, ref, we, waddr, wdata, raddr, i=depth + i)
