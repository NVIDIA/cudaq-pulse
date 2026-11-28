# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The requests a link holds at once, counted for one loop of many calls.

The classical check takes a long loop from one copy of its body: a run per
call. For the count of requests in flight it asks, per run, from which
iteration on the loop repeats itself, and, per time, how many of the
loop's requests are in flight. Both are answered from the runs in order,
not by comparing each run with every other. These tests hold the answers
to a count of every execution (``summarize-trips=0``), on loops whose
calls overlap, are held for several periods, start in any order and lie
within one period or not."""

from __future__ import annotations

import random

from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline


def _loop(trips, period, calls, before=(), slots=64):
    """A loop of *trips* iterations *period* cycles apart; *calls* are
    (start, latency in ns) of the calls in its body, *before* those of
    single calls ahead of it. A call is held for 1 + latency cycles, on a
    link of *slots* slots."""
    latencies = sorted({latency for _, latency in (*calls, *before)})
    lines = [
        "module attributes {",
        '    classical.links = {"0" = {bytes_per_ns = 0.0 : f64, feedback_ns = 0.0 : f64, '
        f"fixed_ns = 0.0 : f64, slot_bytes = 1024 : i64, slots = {slots} : i64}}}},",
        "    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>} {",
    ]
    for latency in latencies:
        lines.append(
            f"  classical.callable @f{latency} {{dispatch_key = {latency + 1} : i64, "
            f'device = 0 : i64, args = [], results = ["f64"], latency_min_ns = 0.0, '
            f"latency_nominal_ns = {float(latency)}, latency_max_ns = {float(latency)}, "
            'mode = "bounded", submit = "eager", dispatch_key_override}'
        )
    lines += [
        "  func.func @test() {",
        "    %l = classical.get_link {device = 0 : i64} : !classical.link",
        "    %lb = arith.constant 0 : index",
        "    %step = arith.constant 1 : index",
        f"    %ub = arith.constant {trips} : index",
    ]
    for i, (start, latency) in enumerate(before):
        lines.append(
            f"    %s{i}, %sh{i} = classical.submit %l, @f{latency}() "
            f"<{{duration_cycles = 1 : i64, start_cycles = {start} : i64}}> "
            ": () -> (!classical.link, !classical.future)"
        )
    lines.append("    scf.for %i = %lb to %ub step %step {")
    for i, (start, latency) in enumerate(calls):
        lines.append(
            f"      %c{i}, %h{i} = classical.submit %l, @f{latency}() "
            f"<{{duration_cycles = 1 : i64, period_cycles = array<i64: {period}>, "
            f"start_cycles = {start} : i64}}> : () -> (!classical.link, !classical.future)"
        )
    lines += ["    }", "    return", "  }", "}", ""]
    return "\n".join(lines)


def _held(text, options=""):
    """The lines of the check's errors that count requests in flight."""
    module = MLIRPipeline().parse_to_module(text)
    try:
        module.run_passes([f"pulse-classical-check{options}"])
    except RuntimeError as error:
        return sorted(
            line.split("): ", 1)[-1] for line in str(error).splitlines() if " holds " in line
        )
    return []


def _random_loop(rng):
    period = rng.choice([50, 200, 1000])
    trips = rng.choice([64, 70, 150])
    count = rng.randrange(1, 40)
    # Starts within one period or over a few; held for less than a period
    # or for several.
    spread = period * rng.choice([1, 1, 3])
    latencies = [rng.choice([1, period // 4, period, 3 * period, 7 * period]) for _ in range(4)]
    calls = [(1000 + rng.randrange(spread), rng.choice(latencies)) for _ in range(count)]
    before = [(rng.randrange(1500), rng.choice(latencies)) for _ in range(rng.randrange(0, 4))]
    # One slot: whatever is held at once is reported, with its time.
    return _loop(trips, period, calls, before, slots=1)


def test_a_loop_of_calls_is_counted_as_every_execution_is():
    rng = random.Random(5)
    over = 0
    for case in range(300):
        text = _random_loop(rng)
        summarized, every = _held(text), _held(text, "{summarize-trips=0}")
        assert summarized == every, case
        over += bool(every)
    # Nearly every loop holds two requests at some time, and says how many.
    assert over > 250


def test_loops_whose_calls_lie_in_one_period_and_in_several_are_counted_alike():
    # 40 calls 5 cycles apart in a period of 1,000: each held for 12 cycles
    # (starts and ends within one period); half of them held for three
    # periods (the ends lie in several); and 160 calls 25 apart, whose
    # starts and ends lie in four periods.
    near = [(1000 + 5 * i, 11) for i in range(40)]
    mixed = [(start, 11 if i % 2 else 3000) for i, (start, _) in enumerate(near)]
    apart = [(1000 + 25 * i, 11) for i in range(160)]
    for calls in (near, mixed, apart):
        text = _loop(70, 1000, calls, slots=1)
        every = _held(text, "{summarize-trips=0}")
        assert every and _held(text) == every, calls[:2]


def test_calls_held_for_several_periods_reach_their_count_after_as_many():
    # One call an iteration, held for 70 periods: 70 are in flight from the
    # 70th start on, six more than the link's slots.
    text = _loop(100, 10, [(0, 699)])
    expected = ["link 0 holds 70 requests at once at 690, more than its 64 slots"]
    assert _held(text) == expected
    assert _held(text, "{summarize-trips=0}") == expected
    # Held for 64 periods, the slots are enough.
    assert _held(_loop(100, 10, [(0, 639)])) == []
