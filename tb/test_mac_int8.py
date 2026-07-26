"""cocotb testbench for mac_int8 — bit-exact vs the numpy golden model.

Convention (reused by every later module):
  1. drive inputs, 2. await RisingEdge, 3. settle 1 ns, 4. read + compare to golden.
Any mismatch fails immediately with the exact cycle, inputs, got, and expected.

Run from tb/:  make            (uses Icarus via the Makefile)
"""

import os
import sys
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from mac_int8_ref import MacInt8Ref  # noqa: E402


def read_signed(sig, bits: int) -> int:
    """Read a signal as a signed integer, width-independent (API-robust)."""
    v = int(sig.value)
    return v - (1 << bits) if v & (1 << (bits - 1)) else v


async def reset(dut):
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.clear.value = 0
    dut.a.value = 0
    dut.b.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    await Timer(1, "ns")


async def apply_cycle(dut, ref, en, clear, a, b, i):
    dut.en.value = en
    dut.clear.value = clear
    dut.a.value = a
    dut.b.value = b
    await RisingEdge(dut.clk)
    await Timer(1, "ns")  # let the NBA settle before reading
    exp = ref.step(en, clear, a, b)
    got = read_signed(dut.acc, 32)
    assert got == exp, (
        f"cycle {i}: en={en} clear={clear} a={a} b={b} -> got {got}, exp {exp}"
    )


@cocotb.test()
async def test_directed(dut):
    """A few hand-checked vectors so a failure is easy to read."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    ref = MacInt8Ref()
    await reset(dut)
    ref.reset()
    # 5*5=25, then +(-3*10)=-30 -> -5, then clear -> 0, then 127*127
    vectors = [
        (1, 0, 5, 5),
        (1, 0, -3, 10),
        (1, 1, 7, 7),   # clear wins over en
        (1, 0, 127, 127),
        (0, 0, 100, 100),  # en=0: hold
    ]
    for i, (en, clr, a, b) in enumerate(vectors):
        await apply_cycle(dut, ref, en, clr, a, b, i)


@cocotb.test()
async def test_random(dut):
    """2000 random cycles with occasional clears."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    ref = MacInt8Ref()
    await reset(dut)
    ref.reset()
    random.seed(0xC0FFEE)
    for i in range(2000):
        en = random.randint(0, 1)
        clear = 1 if random.random() < 0.05 else 0
        a = random.randint(-128, 127)
        b = random.randint(-128, 127)
        await apply_cycle(dut, ref, en, clear, a, b, i)
