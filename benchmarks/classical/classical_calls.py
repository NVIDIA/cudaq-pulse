#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Classical-call benchmarks: what calls cost to compile, schedule, report,
simulate and emulate, and the memory they take.

- compile: trace, module build and schedule time per call, from 10^3 calls
  up to ``--max-calls`` (default 10^4; 10^5 takes about a minute), on 1 and 4
  links, and the classical report's time per call.
- memory: the peak Python allocation of compiling, per call.
- cache: the time of ``lower()`` on a cached kernel that makes calls.
- simulate: active reset with a decoder call against plain active reset,
  and an adaptive T1 tracker per path.
- emulate: calls per second in virtual time, in process and over UDP.

Usage:
  python benchmarks/classical/classical_calls.py [--max-calls N] [--repeats R]
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
import tracemalloc

import cudaq_pulse as pulse
from cudaq_pulse.classical import emulation
from cudaq_pulse.quantum_systems import PhysicalQudit

PI_AMP = math.pi / 20


def _device(t1_us: float = 0.0) -> pulse.QuantumSystem:
    qudit = PhysicalQudit(
        index=0, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=t1_us, t2_star_us=0.0
    )
    return pulse.quantum_system(name="bench", qudits={0: qudit})


def many_calls(n: int, links: int):
    """A kernel of *n* calls, round-robin over *links* links, each answer
    setting the next drive's amplitude."""

    def same(x: float) -> float:
        return x

    functions = [
        pulse.classical(same, name=f"f{k}", max_latency_ns=20, link=k) for k in range(links)
    ]

    @pulse.kernel
    def kernel(q):
        d = get_drive_line(q)
        for i in range(n):
            a = functions[i % links].submit(0.1).result()
            drive(d, square(8, a))

    return kernel


@pulse.classical(max_latency_ns=200)
def decode(bits: pulse.Length[pulse.Bits, 1]) -> pulse.Length[pulse.Bits, 1]:
    return [bits[0]]


@pulse.kernel
def active_reset(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(1000, 0.05))
    sync(d, r)
    if m:
        drive(d, square(40, PI_AMP))


@pulse.kernel
def waiting_reset(q):
    # Active reset that idles as long as the decoder's budget, for a
    # comparison on the same timeline.
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(1000, 0.05))
    sync(d, r)
    wait(d, 401)
    if m:
        drive(d, square(40, PI_AMP))


@pulse.kernel
def decoded_reset(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(1000, 0.05))
    sync(d, r)
    if decode.submit([m]).result()[0]:
        drive(d, square(40, PI_AMP))


@pulse.kernel
def feedback(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(300, 0.05))
    ok, fix = decode.submit([m]).result(timeout=500 * pulse.ns)
    if ok:
        if fix[0]:
            drive(d, square(40, 0.1))


def _median(fn, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def bench_compile(max_calls: int, repeats: int) -> None:
    print("compile (asap): per call, by number of calls and links")
    print(
        f"  {'calls':>7} {'links':>5} {'trace us':>9} {'build us':>9} {'sched us':>9} {'report us':>10}"
    )
    n = 1000
    while n <= max_calls:
        for links in (1, 4):
            kernel = many_calls(n, links)
            runs = []
            for _ in range(repeats if n < 50_000 else 1):
                lowered = pulse.compile(kernel, schedule="asap").lower(_device().qudits[0])
                runs.append(lowered.metrics)
            trace = statistics.median(m.trace_ms for m in runs)
            build = statistics.median(m.ffi_ms for m in runs)
            sched = statistics.median(m.schedule_ms for m in runs)
            report = _median(lowered.classical_report, repeats if n < 50_000 else 1)
            print(
                f"  {n:>7} {links:>5} {trace * 1e3 / n:>9.2f} {build * 1e3 / n:>9.2f} "
                f"{sched * 1e3 / n:>9.2f} {report * 1e6 / n:>10.2f}"
            )
        n *= 10


def bench_memory() -> None:
    q = _device().qudits[0]
    sizes = (500, 4000)
    peaks = {}
    for n in sizes:
        kernel = many_calls(n, 1)
        tracemalloc.start()
        pulse.compile(kernel, schedule="asap").lower(q)
        peaks[n] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    per_call = (peaks[sizes[1]] - peaks[sizes[0]]) / (sizes[1] - sizes[0])
    print(
        f"memory: compile peak {peaks[sizes[1]] / 1024:.0f} KiB at {sizes[1]} calls, "
        f"{per_call:.0f} B per call"
    )


def bench_cache(repeats: int) -> None:
    dev = _device()
    q = dev.qudits[0]
    compiled = pulse.compile(decoded_reset, quantum_system=dev)
    compiled.lower(q)
    t = _median(lambda: [compiled.lower(q) for _ in range(1000)], repeats) / 1000
    print(f"cache: lower() of a cached kernel that calls: {t * 1e6:.1f} us")


def bench_simulate(repeats: int) -> None:
    dev = _device(t1_us=20.0)
    q = dev.qudits[0]
    plain = pulse.compile(waiting_reset, quantum_system=dev).lower(q)
    called = pulse.compile(decoded_reset, quantum_system=dev).lower(q)
    plus = [2**-0.5, 2**-0.5]
    a = _median(lambda: pulse.simulate(plain, initial_state=plus, levels=2), repeats)
    b = _median(lambda: pulse.simulate(called, initial_state=plus, levels=2), repeats)
    print(
        f"simulate: active reset idling for the decoder's budget {a * 1e3:.1f} ms, "
        f"with the decoder call {b * 1e3:.1f} ms ({(b / a - 1) * 100:+.1f}%)"
    )


def bench_emulate() -> None:
    lowered = pulse.compile(feedback).lower(_device().qudits[0])
    local = emulation.emulate(lowered, shots_count=2000, seed=1)
    functions = lowered.classical
    handlers = {fn.name: fn.implementation for fn in functions}
    _, address, stop = emulation.serve_udp(functions, handlers)
    try:
        udp = emulation.emulate(
            lowered, shots_count=500, seed=1, channel=emulation.UdpChannel(address)
        )
    finally:
        stop()
    print(
        f"emulate: {local.calls_per_second:,.0f} calls/s in process, "
        f"{udp.calls_per_second:,.0f} calls/s over UDP loopback"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-calls", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    bench_compile(args.max_calls, args.repeats)
    bench_memory()
    bench_cache(args.repeats)
    bench_simulate(args.repeats)
    bench_emulate()


if __name__ == "__main__":
    main()
