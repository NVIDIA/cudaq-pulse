# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Lowering loops whose lines move by different periods.

The scheduler simulates such a loop's iterations until each group of lines
repeats, then extrapolates, so a loop of 10**9 or 10**15 iterations lowers
and checks in the time and memory of a few; those tests run in a child
process with a CPU, wall and memory limit. What it schedules is what scheduling every
iteration does: the same kernel with its loops unrolled before scheduling,
compared by each line's busy intervals (pulses the unrolled program fuses
leave them alike) and the duration.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from fractions import Fraction
from typing import Any

import cudaq_pulse as pulse
import pytest

from tests._handles import Q

# The child's limits: lowering and checking every kernel below at 10**15
# trips takes about a second and under 100 MB.
CPU_S = 60
RSS_MB = 600


def _kernels(n: int) -> dict[str, Any]:
    @pulse.kernel
    def two_lines(q, r, t):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, gaussian(8, 0.1, 2.0))

    @pulse.kernel
    def three_lines(q, r, t):
        a = get_drive_line(q)
        b = get_drive_line(r)
        p = get_readout_line(q)
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, square(7, 0.1))
            wait(b, 2)
            readout(p, square(40, 0.05))

    @pulse.kernel
    def late_start(q, r, t):
        a = get_drive_line(q)
        b = get_drive_line(r)
        drive(b, square(100, 0.1))
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, square(8, 0.1))

    @pulse.kernel
    def then(q, r, t):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, gaussian(8, 0.1, 2.0))
        sync(a, b)
        drive(a, square(6, 0.2))
        drive(b, square(2, 0.1))

    @pulse.kernel
    def feedback(q, r, t):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        for _ in range(n):
            m = readout(probe, square(40, 0.05))
            if m:
                drive(line, square(8, 0.1))
            else:
                drive(line, square(4, 0.2))

    @pulse.kernel
    def feedback_and_another_line(q, r, t):
        a = get_drive_line(q)
        p = get_readout_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            m = readout(p, square(40, 0.05))
            if m:
                drive(a, square(8, 0.1))
            drive(b, square(3, 0.1))

    @pulse.kernel
    def held_back(q, r, t):
        a = get_drive_line(q)
        c = get_drive_line(r)
        p = get_readout_line(q)
        for _ in range(n):
            sync(p, c)
            m = readout(p, square(10, 0.05))
            drive(c, square(50, 0.1))
            if m:
                drive(a, square(45, 0.1))

    @pulse.kernel
    def holding_back(q, r, t):
        a = get_drive_line(q)
        c = get_drive_line(r)
        p = get_readout_line(q)
        for _ in range(n):
            m = readout(p, square(45, 0.05))
            if m:
                drive(a, square(10, 0.1))
                drive(c, square(50, 0.1))
            sync(a, c)

    @pulse.kernel
    def partly_synced(q, r, t):
        a = get_drive_line(q)
        p = get_readout_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(16, 0.1))
            sync(a, p)
            readout(p, square(40, 0.05))
            drive(b, square(4, 0.1))
            wait(b, 2)

    @pulse.kernel
    def only_syncs_on_two(q, r, t):
        c = get_drive_line(t)
        p = get_readout_line(q)
        s = get_readout_line(r)
        for _ in range(n):
            wait(s, 26)
            sync(p, c)
        drive(c, square(26, 0.1))

    @pulse.kernel
    def in_a_branch(q, r, t):
        a = get_drive_line(q)
        b = get_drive_line(r)
        p = get_readout_line(q)
        m = readout(p, square(40, 0.05))
        if m:
            for _ in range(n):
                drive(a, square(5, 0.1))
                drive(b, square(8, 0.1))
        drive(a, square(4, 0.1))

    @pulse.kernel
    def on_a_measurement_before(q, r, t):
        line = get_drive_line(q)
        other = get_drive_line(r)
        probe = get_readout_line(q)
        m = readout(probe, square(40, 0.05))
        sync(probe, line)
        for _ in range(n):
            if m:
                drive(line, square(8, 0.1))
            wait(line, 3)
            drive(other, square(5, 0.1))

    @pulse.kernel
    def on_two_measurements_before(q, r, t):
        a = get_drive_line(q)
        b = get_drive_line(r)
        c = get_drive_line(t)
        p = get_readout_line(q)
        s = get_readout_line(r)
        drive(a, square(15, 0.1))
        m0 = readout(s, square(65, 0.05))
        m1 = readout(s, square(58, 0.05))
        for _ in range(n):
            drive(b, square(60, 0.1))
            if m0:
                drive(c, gaussian(36, 0.1, 2.0))
                wait(c, 29)
            wait(p, 6)
            sync(p, b)
            if m1:
                drive(c, gaussian(20, 0.1, 2.0))

    @pulse.kernel
    def peeled_on_a_measurement_before(q, r, t):
        a = get_drive_line(q)
        c = get_drive_line(t)
        p = get_readout_line(q)
        s = get_readout_line(r)
        m0 = readout(p, square(41, 0.05))
        sync(a, p)
        sync(s, p)
        for _ in range(n):
            if m0:
                wait(s, 4)
                drive(c, square(18, 0.1))
            m1 = readout(s, square(35, 0.05))
            sync(c, s)
            if m0:
                drive(a, square(38, 0.1))
            if m1:
                drive(c, square(13, 0.1))
            wait(a, 28)

    @pulse.kernel
    def nested(q, r, t):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            for _ in range(3):
                drive(a, square(5, 0.1))
                wait(a, 2)
            drive(b, gaussian(8, 0.1, 2.0))

    return {
        "two_lines": two_lines,
        "three_lines": three_lines,
        "late_start": late_start,
        "then": then,
        "feedback": feedback,
        "feedback_and_another_line": feedback_and_another_line,
        "held_back": held_back,
        "holding_back": holding_back,
        "partly_synced": partly_synced,
        "only_syncs_on_two": only_syncs_on_two,
        "in_a_branch": in_a_branch,
        "on_a_measurement_before": on_a_measurement_before,
        "on_two_measurements_before": on_two_measurements_before,
        "peeled_on_a_measurement_before": peeled_on_a_measurement_before,
        "nested": nested,
    }


NAMES = list(_kernels(1))

# Where a loop's schedule differs over more than its last iteration, the
# loop is unrolled: its program grows with its trip count.
UNROLLED = {("held_back", "alap")}


def _occupancy(schedule: Any) -> dict[tuple[str, int], list[list[int]]]:
    """Each line's busy intervals by kind, touching ones merged."""
    spans: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for event in schedule:
        start = event.start_cycles
        spans.setdefault((event.kind, event.line), []).append(
            (start, start + event.duration_cycles)
        )
    merged: dict[tuple[str, int], list[list[int]]] = {}
    for key, intervals in spans.items():
        out: list[list[int]] = []
        for start, end in sorted(intervals):
            if out and start <= out[-1][1]:
                out[-1][1] = max(out[-1][1], end)
            else:
                out.append([start, end])
        merged[key] = out
    return merged


def _differs(kernel: Any, schedule: str) -> bool:
    """Whether *kernel*'s rolled schedule differs from its unrolled one's."""
    rolled = pulse.compile(kernel, schedule=schedule, unroll_threshold=0).lower(Q[0], Q[1], Q[2])
    flat = pulse.compile(kernel, schedule=schedule, unroll_threshold=10**6).lower(Q[0], Q[1], Q[2])
    return (rolled.duration, _occupancy(rolled.schedule)) != (
        flat.duration,
        _occupancy(flat.schedule),
    )


@pytest.mark.parametrize("trips", [1, 2, 3, 5, 8, 13, 40, 97])
@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_a_loop_whose_lines_move_by_different_periods_schedules_as_every_iteration_does(
    name, schedule, trips
):
    assert not _differs(_kernels(trips)[name], schedule)


_DRIVES = ["a", "b", "c"]
_READOUTS = ["p", "s"]


def _statement(rng: random.Random, depth: int, measured: list[str], indent: int) -> list[str]:
    pad = "    " * indent
    roll = rng.random()
    if roll < 0.35:
        return [f"{pad}drive({rng.choice(_DRIVES)}, square({rng.randint(1, 60)}, 0.1))"]
    if roll < 0.5:
        return [f"{pad}wait({rng.choice(_DRIVES + _READOUTS)}, {rng.randint(1, 30)})"]
    if roll < 0.65:
        name = f"m{len(measured)}"
        measured.append(name)
        width = rng.randint(10, 80)
        return [f"{pad}{name} = readout({rng.choice(_READOUTS)}, square({width}, 0.05))"]
    if roll < 0.85 and measured and depth < 2:
        lines = [f"{pad}if {rng.choice(measured)}:"]
        for _ in range(rng.randint(1, 2)):
            lines += _statement(rng, depth + 1, list(measured), indent + 1)
        if rng.random() < 0.5:
            lines.append(f"{pad}else:")
            for _ in range(rng.randint(1, 2)):
                lines += _statement(rng, depth + 1, list(measured), indent + 1)
        return lines
    if roll < 0.9:
        first, second = rng.sample(_DRIVES + _READOUTS, 2)
        return [f"{pad}sync({first}, {second})"]
    return [f"{pad}drive({rng.choice(_DRIVES)}, gaussian({rng.randint(4, 40)}, 0.1, 2.0))"]


def _random_kernel(seed: int, trips: int, branched: bool) -> Any:
    """A random loop of *trips* iterations between a random prologue and
    epilogue; *branched*, in a branch on a measurement."""
    rng = random.Random(seed)
    lines = [
        "def k(q, r, t):",
        "    a = get_drive_line(q)",
        "    b = get_drive_line(r)",
        "    c = get_drive_line(t)",
        "    p = get_readout_line(q)",
        "    s = get_readout_line(r)",
    ]
    before: list[str] = []
    for _ in range(rng.randint(0, 3)):
        lines += _statement(rng, 2, before, 1)
    indent = 1
    if branched:
        lines += [f"    w = readout(p, square({rng.randint(10, 80)}, 0.05))", "    if w:"]
        indent = 2
    lines.append("    " * indent + f"for _ in range({trips}):")
    measured = list(before) if rng.random() < 0.2 else []
    for _ in range(rng.randint(2, 6)):
        lines += _statement(rng, 0, measured, indent + 1)
    if branched:
        lines.append(f"        drive({rng.choice(_DRIVES)}, square({rng.randint(1, 60)}, 0.1))")
    for _ in range(rng.randint(0, 2)):
        lines += _statement(rng, 2, [], 1)
    namespace: dict[str, Any] = {}
    exec(compile("\n".join(lines) + "\n", f"<random {seed}>", "exec"), namespace)  # noqa: S102 - generated kernels
    return pulse.kernel(namespace["k"])


# A loop in a branch is not compared under alap: there a measurement the
# branch tests waits only for the iterations the scheduler simulates, which
# for a loop all of whose lines move by one period is not its first.
@pytest.mark.parametrize(("branched", "schedules"), [(False, ("asap", "alap")), (True, ("asap",))])
def test_random_loops_schedule_as_every_iteration_does(branched, schedules):
    differ = [
        (seed, trips, schedule)
        for seed in range(80)
        for trips in (3, 13, 40)
        for schedule in schedules
        if _differs(_random_kernel(seed, trips, branched), schedule)
    ]
    assert differ == []


_CHILD = """
import json, os, resource, signal, sys, threading, time
signal.alarm({wall})
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
_scale = 1 if sys.platform == "darwin" else 1024
def _watch():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _scale > {mb} << 20:
            os._exit(3)
        time.sleep(0.005)
threading.Thread(target=_watch, daemon=True).start()

import cudaq_pulse as pulse
from cudaq_pulse.controller import ControllerLimits, check
from tests._handles import Q
from tests.test_unsynced_lowering import UNROLLED, _kernels

limits = ControllerLimits(issue_cycles=4, min_wait_cycles=4, max_amplitude=0.05, max_events=7)
out = {{}}
for name, kernel in _kernels({trips}).items():
    for schedule in ("asap", "alap"):
        if (name, schedule) in UNROLLED:
            continue
        program = pulse.compile(kernel, schedule=schedule, unroll_threshold=0)
        lowered = program.lower(Q[0], Q[1], Q[2])
        found = [str(lowered.duration.exact_ns), len(check(lowered, limits))]
        out[f"{{name}}/{{schedule}}"] = found
print(json.dumps(out))
"""


def _in_a_child(trips: int) -> dict[str, list[Any]]:
    """The duration and limit violations of every kernel of *trips*
    iterations, lowered in a child process that is stopped past CPU_S
    seconds or RSS_MB MB."""
    code = _CHILD.format(wall=2 * CPU_S, cpu=CPU_S, mb=RSS_MB, trips=trips)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=4 * CPU_S,
    )
    assert out.returncode == 0, f"the child exited {out.returncode}: {out.stderr[-2000:]}"
    result: dict[str, list[Any]] = json.loads(out.stdout.splitlines()[-1])
    return result


@pytest.mark.parametrize("many", [10**9, 10**15])
def test_a_loop_of_10_9_or_10_15_iterations_lowers_and_checks_in_bounded_time_and_memory(many):
    from cudaq_pulse.controller import ControllerLimits, check

    limits = ControllerLimits(issue_cycles=4, min_wait_cycles=4, max_amplitude=0.05, max_events=7)
    found = _in_a_child(many)
    # Past its first iterations each loop adds the same to both.
    small: dict[tuple[str, int], tuple[Fraction, int]] = {}
    for trips in (100, 101):
        for name, kernel in _kernels(trips).items():
            for schedule in ("asap", "alap"):
                lowered = pulse.compile(kernel, schedule=schedule, unroll_threshold=0)
                lowered = lowered.lower(Q[0], Q[1], Q[2])
                small[f"{name}/{schedule}", trips] = (
                    lowered.duration.exact_ns,
                    len(check(lowered, limits)),
                )
    assert len(found) == 2 * len(NAMES) - len(UNROLLED)
    for key, (duration, violations) in found.items():
        (d100, v100), (d101, v101) = small[key, 100], small[key, 101]
        assert Fraction(duration) == d100 + (many - 100) * (d101 - d100), key
        assert violations == v100 + (many - 100) * (v101 - v100), key
