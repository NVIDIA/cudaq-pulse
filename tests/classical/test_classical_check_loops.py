# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The classical check of long rolled loops of calls.

The check takes a loop of many trips from one copy of its body: how many
requests a link holds at once and whether they go out in program order,
counted where the loop's iterations do not repeat the one before. With
``summarize-trips=0`` it counts every iteration. The two report the same
on the scheduled programs of loops branching on answers, and on those
programs with fewer slots, longer or earlier requests and periods off by a
few cycles, which break the rules from some iteration on.

The machine "two_slots" gives ``max_concurrent_drives=2``, which only the
rcp schedules keep: in a test parametrized by schedule, its programs are
scheduled ``rcp`` where the test id says ``asap`` and ``alap_rcp`` where it
says ``alap`` (``_policy``); the default machine's programs run as the id
says.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterator
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._native._cudaq_pulse_native import PulseModuleBuilder

from tests._handles import Q
from tests.classical.test_answer_loops import (
    NAMES,
    POLICIES,
    _kernels,
    _lowers_as_extrapolated,
    _machines,
    _policy,
    _random_kernel,
    _saturating,
    decide,
)


def _check(text: str, options: str = "") -> str:
    module = PulseModuleBuilder().load(text.encode(), b"", False, [])
    try:
        module.run_passes([f"pulse-classical-check{options}"])
    except RuntimeError as error:
        return "\n".join(line for line in str(error).splitlines() if "note:" not in line)
    return "ok"


def _requests(text: str, rng: random.Random, edit: Callable[[str], str]) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if "classical.submit" in line and "start_cycles" in line and rng.random() < 0.6:
            lines[i] = edit(line)
    return "\n".join(lines) + "\n"


def _number(name: str, change: Callable[[int], int]) -> Callable[[str], str]:
    return lambda line: re.sub(
        rf"{name} = (\d+)", lambda m: f"{name} = {change(int(m.group(1)))}", line
    )


def _period(delta: int) -> Callable[[str], str]:
    def edit(line: str) -> str:
        def last(m: re.Match[str]) -> str:
            values = [int(v) for v in m.group(1).split(",")]
            values[-1] = max(1, values[-1] + delta)
            return f"period_cycles = array<i64: {', '.join(map(str, values))}>"

        return re.sub(r"period_cycles = array<i64: ([-\d, ]+)>", last, line)

    return edit


def _variants(text: str, seed: int) -> Iterator[str]:
    rng = random.Random(seed)
    yield text
    for slots in (1, 2, 5):
        yield re.sub(r"slots = \d+", f"slots = {slots}", text)
    for extra in (3, 40, 500):
        yield _requests(text, rng, _number("duration_cycles", lambda d, e=extra: d + e))
    yield _requests(text, rng, _number("start_cycles", lambda s: max(0, s - 30)))
    for delta in (-1, -7, -60, 5):
        yield _requests(text, rng, _period(delta))


def _scheduled(kernel: Any, schedule: str, options: dict[str, Any]) -> str | None:
    program = pulse.compile(
        kernel, schedule=_policy(schedule, options), unroll_threshold=0, passes=(), **options
    )
    try:
        return program.lower(Q[0], Q[1], Q[2]).mlir("pulse")
    except RuntimeError:
        return None


def _differ(kernels: dict[str, Any], schedule: str) -> tuple[list[str], int]:
    differ, failing = [], 0
    for name, kernel in kernels.items():
        for machine, options in _machines().items():
            text = _scheduled(kernel, schedule, options)
            if text is None:
                continue
            for index, variant in enumerate(_variants(text, sum(map(ord, name)))):
                summarized, every = _check(variant), _check(variant, "{summarize-trips=0}")
                failing += every != "ok"
                if summarized != every or _check(variant, "{summarize-trips=2}") != every:
                    differ.append(f"{name}/{machine}/{index}")
    return differ, failing


@pytest.mark.parametrize("schedule", POLICIES)
def test_a_long_loop_of_calls_is_checked_from_one_copy_as_every_iteration(schedule):
    kernels = {f"{name}/{trips}": _kernels(trips)[name] for name in NAMES for trips in (65, 130)}
    kernels.update({f"saturating/{trips}": _saturating(trips) for trips in (70, 130)})
    differ, failing = _differ(kernels, schedule)
    assert differ == []
    assert failing > 50


@pytest.mark.parametrize("schedule", POLICIES)
def test_random_long_loops_of_calls_are_checked_from_one_copy_as_every_iteration(schedule):
    kernels: dict[str, Any] = {}
    for seed in range(12):
        trips = random.Random(seed).choice((64, 97, 200))
        kernels[f"random{seed}/{trips}"] = _random_kernel(seed, trips, seed % 3 == 0)
    differ, failing = _differ(kernels, schedule)
    assert differ == []
    assert failing > 50


def _calls_before(n: int) -> dict[str, Any]:
    """A call before a loop that makes none: the check counts the loop's
    iterations only where requests are."""

    @pulse.kernel
    def readouts(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        other = get_readout_line(r)
        m = readout(ro, square(100, 0.05))
        if decide.submit([m]).result()[0]:
            drive(d, square(40, 0.1))
        for _ in range(n):
            drive(d, square(8, 0.1))
            readout(other, square(12, 0.05))

    @pulse.kernel
    def drives(q, r, t):
        d = get_drive_line(q)
        e = get_drive_line(r)
        ro = get_readout_line(q)
        m = readout(ro, square(100, 0.05))
        if decide.submit([m]).result()[0]:
            drive(d, square(40, 0.1))
        for _ in range(n):
            drive(d, square(8, 0.1))
            drive(e, square(12, 0.05))

    return {"readouts": readouts, "drives": drives}


def test_a_call_before_a_loop_of_10_7_iterations_lowers_in_bounded_time_and_memory():
    _lowers_as_extrapolated(_calls_before, 10**7)


def _nested(trips: tuple[int, ...]) -> Any:
    """Rounds around repeats (around repeats): a call and a branch on its
    answer inside loops of *trips*, outermost first, with a call of its own
    at each level but the innermost."""
    outer, *rest = trips

    @pulse.kernel
    def two(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(outer):
            first = decide.submit([readout(ro, square(60, 0.05))]).result()
            sync(d, ro)
            if first[0]:
                drive(d, square(12, 0.2))
            for _ in range(rest[0]):
                m = readout(ro, square(100, 0.05))
                fix = decide.submit([m]).result()
                sync(d, ro)
                if fix[0]:
                    drive(d, square(20, 0.3))

    @pulse.kernel
    def three(q, r, t):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(outer):
            zeroth = decide.submit([readout(ro, square(40, 0.05))]).result()
            sync(d, ro)
            if zeroth[0]:
                drive(d, square(8, 0.2))
            for _ in range(rest[0]):
                first = decide.submit([readout(ro, square(60, 0.05))]).result()
                sync(d, ro)
                if first[0]:
                    drive(d, square(12, 0.2))
                for _ in range(rest[1]):
                    m = readout(ro, square(100, 0.05))
                    fix = decide.submit([m]).result()
                    sync(d, ro)
                    if fix[0]:
                        drive(d, square(20, 0.3))

    return two if len(trips) == 2 else three


# The loop kept as one run is the outermost of 64 or more trips unless
# walking it and keeping a longer loop inside it is much less work: the
# longest loop outside, inside and in the middle, loops of equal trips, an
# outer loop of 64 or more around one long enough to be kept in its place
# (the nests with a loop of 800 or more) and around one that is not, and
# none of 64 or more. Each level but the innermost has a call of its own.
_NESTS = [
    (130, 3),
    (3, 130),
    (70, 65),
    (65, 70),
    (64, 130),
    (64, 1200),
    (70, 1500),
    (64, 64),
    (5, 6),
    (70, 2, 3),
    (2, 70, 3),
    (2, 3, 70),
    (65, 2, 70),
    (64, 2, 800),
    (66, 70, 2),
    (64, 2, 64),
]


@pytest.mark.parametrize("schedule", POLICIES)
def test_nested_loops_of_calls_are_checked_from_the_longest_as_every_iteration(schedule):
    kernels = {"x".join(map(str, trips)): _nested(trips) for trips in _NESTS}
    # Every nest is scheduled on the default machine, so none is passed over.
    for name, kernel in kernels.items():
        assert _scheduled(kernel, schedule, {}) is not None, name
    differ, failing = _differ(kernels, schedule)
    assert differ == []
    assert failing > 50
