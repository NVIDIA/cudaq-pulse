# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-constrained schedules (rcp, alap_rcp) of rolled loops whose
lanes do not move as each iteration does: beside a long pulse on another
line, which the loop reaches only after many iterations, or taken in turn
by the iterations of a loop in an arm of a branch. Such a loop is scheduled
in the time and memory of a few iterations, whatever its trip count; those
tests run in a child process with a CPU, wall and memory limit. At a few
hundred iterations, a rolled loop's schedule is that of the loop unrolled,
or, beside a long pulse with lanes to spare, that of a schedule without
lanes."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import cudaq_pulse as pulse
import pytest

from tests import _child
from tests._handles import Q

# The child's limits: scheduling a few iterations takes well under a second
# and a few hundred MB.
CPU_S = 20
RSS_MB = 600

_KERNELS = """
TRIPS = {trips}


@pulse.kernel
def beside_long(q, r):
    a = get_drive_line(q)
    b = get_drive_line(r)
    drive(b, square(4000, 0.1))
    for _ in range(TRIPS):
        drive(a, square(5, 0.1))


@pulse.kernel
def beside_late(q, r):
    a = get_drive_line(q)
    b = get_drive_line(r)
    wait(b, 1000000000)
    drive(b, square(40, 0.1))
    for _ in range(TRIPS):
        drive(a, square(5, 0.1))


@pulse.kernel
def in_an_arm(q, r, s, t):
    m = readout(get_readout_line(q), square(100, 0.05))
    if m:
        a = get_drive_line(r)
        b = get_drive_line(s)
        for _ in range(TRIPS):
            drive(a, gaussian(40, 0.1, 10))
            drive(b, gaussian(40, 0.1, 10))
            sync(a, b)
    else:
        drive(get_drive_line(t), square(200, 0.1))


@pulse.kernel
def branch_loop(q, r, s):
    ro = get_readout_line(q)
    a = get_drive_line(r)
    b = get_drive_line(s)
    for _ in range(TRIPS):
        m = readout(ro, square(50, 0.05))
        if m:
            drive(a, square(20, 0.1))
            drive(b, square(20, 0.1))
        else:
            drive(a, gaussian(20, 0.1, 5))
        sync(ro, a, b)


@pulse.kernel
def long_in_middle(q, r, s):
    a = get_drive_line(q)
    b = get_drive_line(r)
    c = get_drive_line(s)
    for _ in range(TRIPS):
        drive(a, square(10, 0.1))
        wait(a, 10)
    wait(b, 10)
    for _ in range(TRIPS):
        drive(b, square(10, 0.1))
        wait(b, 10)
    wait(c, 100005)
    drive(c, square(300, 0.1))


@pulse.kernel
def arm_long(q, r, s, t):
    m = readout(get_readout_line(q), square(100, 0.05))
    a = get_drive_line(r)
    b = get_drive_line(s)
    c = get_drive_line(t)
    if m:
        drive(c, square(3001, 0.1))
        for _ in range(TRIPS):
            drive(a, square(40, 0.1))
            drive(b, square(40, 0.1))
            sync(a, b)
    else:
        drive(c, square(5003, 0.1))


@pulse.kernel
def in_a_loop(q, r, s):
    ro = get_readout_line(q)
    a = get_drive_line(r)
    b = get_drive_line(s)
    for _ in range(3):
        for _ in range(TRIPS):
            m = readout(ro, square(50, 0.05))
            if m:
                drive(a, square(20, 0.1))
                drive(b, square(20, 0.1))
            else:
                drive(a, gaussian(20, 0.1, 5))
            sync(ro, a, b)


@pulse.kernel
def holds_a_loop(q, r, s):
    a = get_drive_line(q)
    b = get_drive_line(r)
    c = get_drive_line(s)
    drive(c, square(4000, 0.1))
    for _ in range(TRIPS):
        for _ in range(3):
            drive(a, square(5, 0.1))
        drive(b, square(8, 0.1))
        sync(a, b)


@pulse.kernel
def in_a_loop_beside(q, r):
    a = get_drive_line(q)
    b = get_drive_line(r)
    drive(b, square(4000, 0.1))
    for _ in range(3):
        for _ in range(TRIPS):
            drive(a, square(5, 0.1))


@pulse.kernel
def own_periods_beside(q, r, s):
    ro = get_readout_line(q)
    a = get_drive_line(r)
    b = get_drive_line(s)
    drive(b, square(4000, 0.1))
    wait(b, 1000)
    for _ in range(TRIPS):
        drive(a, square(4, 0.1))
        m = readout(ro, square(50, 0.05))
        if m:
            drive(b, square(64, 0.1))
            wait(a, 30)
        else:
            readout(ro, square(50, 0.05))
    for _ in range(300):
        drive(b, square(40, 0.1))


QUDITS = dict(beside_long=2, beside_late=2, in_an_arm=4, branch_loop=3, long_in_middle=3,
              arm_long=4, in_a_loop=3, holds_a_loop=3, in_a_loop_beside=2,
              own_periods_beside=3)
"""

SHAPES = ["beside_long", "beside_late", "in_an_arm", "branch_loop", "long_in_middle"]

_CHILD = """
import json
import cudaq_pulse as pulse
from tests._handles import Q
{kernels}
kernel = {shape}
lanes = pulse.ControlSystem(max_concurrent_drives={drives})
lowered = pulse.compile(kernel, schedule="{schedule}", unroll_threshold=0, control_system=lanes)
lowered = lowered.lower(*Q[: QUDITS["{shape}"]])
schedule = lowered.schedule
print(json.dumps({{"events": len(schedule), "last": repr(schedule[-1]),
                  "duration": float(lowered.duration)}}))
"""


def _kernels(trips: int) -> dict[str, object]:
    found: dict[str, object] = {"pulse": pulse}
    exec(_KERNELS.format(trips=trips), found)
    return found


def _in_a_child(shape: str, schedule: str, trips: int, drives: int = 4) -> dict[str, object]:
    """What scheduling *shape* of *trips* iterations under *schedule* with
    *drives* drive lanes gives, in a child process that is stopped past
    CPU_S seconds or RSS_MB MB."""
    code = _child.limits(CPU_S, RSS_MB) + _CHILD.format(
        kernels=_KERNELS.format(trips=trips), shape=shape, schedule=schedule, drives=drives
    )
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
    result: dict[str, object] = json.loads(out.stdout.splitlines()[-1])
    return result


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
@pytest.mark.parametrize("shape", SHAPES)
def test_a_loop_of_10_15_iterations_is_scheduled_in_bounded_time_and_memory(shape, schedule):
    found = _in_a_child(shape, schedule, 10**15)
    assert found["events"] >= 10**15


# Loops in loops, loops that hold loops, loops whose lines move by periods of
# their own, and lanes that repeat only over more iterations than eight.
NESTED = ["in_a_loop", "holds_a_loop", "in_a_loop_beside", "own_periods_beside"]


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
@pytest.mark.parametrize("shape", NESTED)
def test_a_nested_loop_of_10_15_iterations_is_scheduled_in_bounded_time_and_memory(shape, schedule):
    found = _in_a_child(shape, schedule, 10**15)
    assert found["events"] >= 10**15


@pytest.mark.parametrize("drives", [9, 16, 32])
@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
@pytest.mark.parametrize("shape", ["arm_long", "branch_loop"])
def test_lanes_that_repeat_over_many_iterations_are_scheduled_in_bounded_time(
    shape, schedule, drives
):
    found = _in_a_child(shape, schedule, 10**15, drives)
    assert found["events"] >= 10**15


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
@pytest.mark.parametrize(
    "shape", ["in_an_arm", "branch_loop", "long_in_middle", "in_a_loop", "arm_long"]
)
def test_a_rolled_loop_is_scheduled_as_the_loop_unrolled(shape, schedule):
    kernels = _kernels(300)
    handles = Q[: kernels["QUDITS"][shape]]
    rolled = pulse.compile(kernels[shape], schedule=schedule, unroll_threshold=0).lower(*handles)
    unrolled = pulse.compile(kernels[shape], schedule=schedule, unroll_threshold=10**6).lower(
        *handles
    )
    assert "scf.for" in rolled.mlir() and "scf.for" not in unrolled.mlir()
    assert sorted(map(repr, rolled.schedule)) == sorted(map(repr, unrolled.schedule))


@pytest.mark.parametrize("trips", [300, 3000])
@pytest.mark.parametrize("shape", ["beside_long", "beside_late"])
def test_beside_a_long_pulse_with_lanes_to_spare_rcp_places_each_pulse_as_asap(shape, trips):
    kernels = _kernels(trips)
    handles = Q[: kernels["QUDITS"][shape]]
    lanes = pulse.compile(kernels[shape], schedule="rcp", unroll_threshold=0).lower(*handles)
    free = pulse.compile(kernels[shape], schedule="asap", unroll_threshold=0).lower(*handles)
    assert list(lanes.schedule) == list(free.schedule)


@pytest.mark.parametrize("trips", [300, 3000])
@pytest.mark.parametrize("shape", ["beside_long", "beside_late"])
def test_beside_a_long_pulse_alap_rcp_plays_the_loop_back_to_back(shape, trips):
    kernels = _kernels(trips)
    handles = Q[: kernels["QUDITS"][shape]]
    lowered = pulse.compile(kernels[shape], schedule="alap_rcp", unroll_threshold=0).lower(*handles)
    starts = [e.start_cycles for e in lowered.schedule if e.qudit == 0]
    assert len(starts) == trips
    assert starts == list(range(starts[0], starts[0] + 5 * trips, 5))
