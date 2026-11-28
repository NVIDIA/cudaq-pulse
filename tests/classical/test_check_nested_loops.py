# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The classical check of loops of calls inside loops.

The check takes every loop of many trips from one copy of its body: a call
inside several such loops is one run, with a period and a count per loop.
How many requests a link holds at once, and whether they go out in program
order, are worked out from those runs. These tests hold both to a count of
every execution (``summarize-trips=0``) on random nests: loops laid out as
a scheduler lays them out, calls held long against the periods, and nests
with periods of zero, negative, shorter than the loop inside, or not the
same for the calls of one loop."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline


@dataclass
class Call:
    start: int
    duration: int
    latency: int
    periods: list[int]


@dataclass
class Loop:
    trips: int
    body: list


@dataclass
class Branch:
    then: list
    other: list


@dataclass
class _Writer:
    lines: list[str] = field(default_factory=list)
    names: int = 0

    def name(self) -> int:
        self.names += 1
        return self.names

    def body(self, items: list, indent: str) -> None:
        for item in items:
            if isinstance(item, Call):
                n = self.name()
                periods = ""
                if item.periods:
                    periods = f"period_cycles = array<i64: {', '.join(map(str, item.periods))}>, "
                self.lines.append(
                    f"{indent}%c{n}, %h{n} = classical.submit %l, @f{item.latency}() "
                    f"<{{duration_cycles = {item.duration} : i64, {periods}"
                    f"start_cycles = {item.start} : i64}}> "
                    ": () -> (!classical.link, !classical.future)"
                )
            elif isinstance(item, Loop):
                n = self.name()
                self.lines.append(f"{indent}%ub{n} = arith.constant {item.trips} : index")
                self.lines.append(f"{indent}scf.for %i{n} = %lb to %ub{n} step %step {{")
                self.body(item.body, indent + "  ")
                self.lines.append(f"{indent}}}")
            else:
                self.lines.append(f"{indent}scf.if %yes {{")
                self.body(item.then, indent + "  ")
                self.lines.append(f"{indent}}} else {{")
                self.body(item.other, indent + "  ")
                self.lines.append(f"{indent}}}")


def _calls(items: list):
    for item in items:
        if isinstance(item, Call):
            yield item
        elif isinstance(item, Loop):
            yield from _calls(item.body)
        else:
            yield from _calls(item.then)
            yield from _calls(item.other)


def _module(items: list, slots: int = 1) -> str:
    """A scheduled module of the nest *items* on one link of *slots* slots.
    A call is held for its duration and its latency (1 ns a cycle)."""
    lines = [
        "module attributes {",
        '    classical.links = {"0" = {bytes_per_ns = 0.0 : f64, feedback_ns = 0.0 : f64, '
        f"fixed_ns = 0.0 : f64, slot_bytes = 1024 : i64, slots = {slots} : i64}}}},",
        "    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>} {",
    ]
    for latency in sorted({call.latency for call in _calls(items)}):
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
        "    %yes = arith.constant true",
    ]
    writer = _Writer(lines)
    writer.body(items, "    ")
    lines += ["    return", "  }", "}", ""]
    return "\n".join(lines)


def _errors(text: str, options: str) -> list[str]:
    module = MLIRPipeline().parse_to_module(text)
    try:
        module.run_passes([f"pulse-classical-check{options}"])
    except RuntimeError as error:
        return sorted(
            line.split("): ", 1)[1]
            for line in str(error).splitlines()
            if "): " in line and "note:" not in line
        )
    return []


def _random_nest(rng: random.Random, kind: int, trips: list[int], branches: bool) -> list:
    """*kind* 0: loops as a scheduler lays them out, one period per loop, the
    length of its body or more; 1: calls held long against the periods; 2:
    periods and starts of any kind."""

    def set_period(items: list, depth: int, period: int, mixed: bool) -> None:
        for call in _calls(items):
            call.periods[depth] = period + (rng.randint(-2, 2) if mixed else 0)

    def body(depth: int, at: int) -> tuple[list, int]:
        items: list = []
        begin = at
        for _ in range(rng.randint(1, 3 if depth else 4)):
            roll = rng.random()
            if depth < 3 and roll < 0.4:
                count = rng.choice(trips)
                inner, length = body(depth + 1, at)
                period = length + rng.choice([0, 0, 1, 5])
                if kind == 2:
                    period = rng.choice(
                        [period, period, 0, -period, max(1, length // 2), rng.randint(-5, 30)]
                    )
                set_period(inner, depth, period, kind == 2 and rng.random() < 0.15)
                items.append(Loop(count, inner))
                at += period * count if period > 0 else length
            elif branches and roll < 0.5:
                then, used = body_of_calls(depth, at)
                other, also = body_of_calls(depth, at)
                items.append(Branch(then, other))
                at += max(used, also)
            else:
                call, used = one_call(depth, at)
                items.append(call)
                at += used
        return items, at - begin

    def one_call(depth: int, at: int) -> tuple[Call, int]:
        duration = rng.choice([1, 1, 2, 5])
        latency = rng.choice([0, 1, 3, 8]) if kind == 0 else rng.choice([0, 2, 10, 40, 150, 600])
        if kind == 2 and rng.random() < 0.1:
            at -= rng.randint(0, 20)
        return Call(at, duration, latency, [0] * depth), duration + rng.choice([0, 0, 1, 4])

    def body_of_calls(depth: int, at: int) -> tuple[list, int]:
        items: list = []
        begin = at
        for _ in range(rng.randint(0, 2)):
            call, used = one_call(depth, at)
            items.append(call)
            at += used
        return items, at - begin

    return body(0, rng.randint(0, 20))[0]


def _executions(items: list) -> int:
    total = 0
    for item in items:
        if isinstance(item, Call):
            total += 1
        elif isinstance(item, Loop):
            total += item.trips * _executions(item.body)
        else:
            total += _executions(item.then) + _executions(item.other)
    return total


def _compare(seed: int, cases: int, trips: list[int], options: str, branches: bool) -> int:
    rng = random.Random(seed)
    reported = 0
    for case in range(cases):
        items = _random_nest(rng, rng.choice([0, 0, 1, 1, 2]), trips, branches)
        if _executions(items) > 40000:
            continue
        text = _module(items)
        every = _errors(text, "{summarize-trips=0}")
        assert _errors(text, options) == every, (seed, case)
        reported += bool(every)
    return reported


def test_loops_inside_loops_are_counted_and_ordered_as_every_execution_is():
    # Every loop of two or more trips kept: up to three deep.
    trips = [1, 2, 2, 3, 3, 4, 5, 7, 12]
    assert _compare(11, 500, trips, "{summarize-trips=2}", branches=False) > 250


def test_loops_inside_loops_with_branches_are_checked_as_every_execution_is():
    trips = [1, 2, 2, 3, 3, 4, 5, 7, 12]
    assert _compare(12, 300, trips, "{summarize-trips=2}", branches=True) > 150


def test_long_loops_inside_long_loops_are_checked_as_every_execution_is():
    # As the check runs: loops of 64 trips or more kept, shorter ones written out.
    assert _compare(13, 60, [3, 64, 64, 70, 100], "", branches=False) > 20


def _random_wide(rng: random.Random) -> list:
    """Rounds of repeats of many calls: within one repeat or over a few,
    held for less than a repeat or for several, some of them in a loop of
    their own in the repeat."""
    period = rng.choice([50, 200, 1000])
    repeats, rounds = rng.choice([2, 3, 5, 9]), rng.choice([2, 3, 4, 7])
    spread = period * rng.choice([1, 1, 3])
    latencies = [rng.choice([1, period // 4, period, 3 * period, 7 * period]) for _ in range(4)]
    around = repeats * period + spread + rng.choice([0, 10, 5 * period])
    if rng.random() < 0.15:
        around = rng.choice([period, around // 2])
    calls: list = [
        Call(1000 + rng.randrange(spread), 1, rng.choice(latencies), [around, period])
        for _ in range(rng.randrange(8, 40))
    ]
    if rng.random() < 0.4:
        # A loop of calls inside the repeat, a few cycles apart.
        step = rng.choice([1, 3, period // 10])
        inner = [
            Call(
                1000 + rng.randrange(max(1, step)), 1, rng.choice(latencies), [around, period, step]
            )
            for _ in range(rng.randrange(8, 20))
        ]
        calls.insert(rng.randrange(len(calls)), Loop(rng.choice([2, 4]), inner))
    body: list = [Loop(repeats, calls)]
    if rng.random() < 0.5:
        body.insert(0, Call(900, 1, rng.choice(latencies), [around]))
    before = [
        Call(rng.randrange(1500), 1, rng.choice(latencies), []) for _ in range(rng.randrange(3))
    ]
    return [*before, Loop(rounds, body)]


def test_loops_of_many_calls_inside_loops_are_counted_as_every_execution_is():
    rng = random.Random(14)
    reported = 0
    for case in range(200):
        text = _module(_random_wide(rng))
        every = _errors(text, "{summarize-trips=0}")
        assert _errors(text, "{summarize-trips=2}") == every, case
        reported += any(" holds " in line for line in every)
    assert reported > 150


def test_rounds_of_repeats_hold_what_one_round_holds():
    # 1,000 rounds of 1,000 repeats, a call a repeat held for three of them: four in flight from
    # the fourth repeat of the first round, and no more a million calls later.
    repeats = Loop(1000, [Call(0, 1, 30, [10000, 10])])
    text = _module([Loop(1000, [repeats])], slots=3)
    assert _errors(text, "") == ["link 0 holds 4 requests at once at 30, more than its 3 slots"]
    assert _errors(_module([Loop(1000, [repeats])], slots=4), "") == []


def test_a_call_whose_loops_carry_it_past_the_cycles_counted_is_reported():
    # 4,096 rounds 2^51 cycles apart: the last is past 2^61 cycles.
    repeats = Loop(100, [Call(0, 1, 1, [2**51, 10])])
    assert _errors(_module([Loop(4096, [repeats])]), "") == [
        "the call of 'f1' starts at 0 and moves with each of the 2 loops around it: in their "
        "last iterations it is past the cycles a schedule counts in"
    ]
