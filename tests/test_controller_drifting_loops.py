# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller checks of rolled loops whose lines move by periods of their own.

A loop whose lines drift apart (each line repeats with its own period, a
branch's arms take different times) is checked from one copy of its body
and each line's period, under a control system too, so its check takes the
time and memory of a few iterations whatever its trip count; those tests
run in a child process with a CPU, wall and memory limit. What a check
reports is what it reports with every iteration played.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, controller
from cudaq_pulse.controller import check

from tests import _child
from tests._handles import Q

# The child's limits: a check of one body takes well under a second and a
# few hundred MB.
CPU_S = 30
RSS_MB = 600

_KERNELS = """
import json
import cudaq_pulse as pulse
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.controller import ControllerLimitError

TRIPS = {trips}

@pulse.kernel
def groups(q0, q1, q2):
    a = get_drive_line(q0)
    b = get_drive_line(q1)
    c = get_drive_line(q2)
    r = get_readout_line(q0)
    for _ in range(TRIPS):
        m = readout(r, square(50, 0.05))
        if m:
            drive(a, square(8, 0.1))
        else:
            drive(b, square(20, 0.1))
        drive(c, square(40, 0.1))
        drive(c, square(40, 0.1))

@pulse.kernel
def inner(q0, q1, q2):
    a = get_drive_line(q0)
    b = get_drive_line(q1)
    c = get_drive_line(q2)
    r = get_readout_line(q0)
    for _ in range(TRIPS):
        m = readout(r, square(50, 0.05))
        if m:
            drive(a, square(8, 0.1))
        else:
            drive(b, square(20, 0.1))
        for _ in range(2):
            drive(c, square(40, 0.1))

@pulse.kernel
def in_an_arm(q0, q1, q2):
    a = get_drive_line(q0)
    b = get_drive_line(q1)
    r = get_readout_line(q0)
    m = readout(r, square(100, 0.05))
    if m:
        for _ in range(TRIPS):
            drive(a, square(7, 0.1))
            drive(b, square(11, 0.1))
    else:
        drive(get_drive_line(q2), square(200, 0.1))

qpu = pulse.quantum_system(
    "transmon_surface17",
    control_system=ControlSystem(
        controllers=[Controller(drive_lines=range(17), readout_lines=range(17){limits})]
    ),
)
try:
    lowered = pulse.compile(
        {kernel}, quantum_system=qpu, schedule="asap", unroll_threshold=0
    ).lower(
        *qpu.qudits[:3]
    )
except ControllerLimitError as error:
    print(json.dumps({{"count": len(error.violations), "message": str(error)}}))
else:
    print(json.dumps({{"ops": lowered.metrics.op_count}}))
"""


def _in_a_child(kernel: str, trips: int, limits: str = "") -> dict[str, object]:
    """What compiling *kernel* of *trips* iterations under a control system
    of one controller of *limits* gives, in a child process that is stopped
    past CPU_S seconds or RSS_MB MB."""
    code = _child.limits(CPU_S, RSS_MB) + _KERNELS.format(
        trips=trips, kernel=kernel, limits=f", {limits}" if limits else ""
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


@pytest.mark.parametrize("kernel", ["groups", "inner", "in_an_arm"])
def test_a_drifting_loop_of_10_15_iterations_is_checked_by_a_control_system_in_bounds(kernel):
    assert _in_a_child(kernel, 10**15) == _in_a_child(kernel, 100)


@pytest.mark.parametrize("kernel", ["groups", "in_an_arm"])
def test_a_violation_in_every_drifting_iteration_of_10_15_is_counted_in_bounds(kernel):
    limits = "instruction_cycles=64"
    few = _in_a_child(kernel, 100, limits)
    more = _in_a_child(kernel, 101, limits)
    many = _in_a_child(kernel, 10**15, limits)
    count = int(str(few["count"]))
    each = int(str(more["count"])) - count
    assert each and many["count"] == count + each * (10**15 - 100)
    assert many["message"] == str(few["message"]).replace(
        f"... and {count - 10} more", f"... and {count + each * (10**15 - 100) - 10} more"
    )


def _every_iteration(monkeypatch: pytest.MonkeyPatch, lowered: Any, limits: Any) -> list[str]:
    """What ``check`` reports with every iteration of every loop played."""
    walk = controller._walk

    def played(*args: Any, summary: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", played)
        return list(check(lowered, limits))


def _drifting(n: int) -> dict[str, Any]:
    """Loops whose lines move by periods of their own."""

    @pulse.kernel
    def groups(q, r, s):
        a = get_drive_line(q)
        b = get_drive_line(r)
        c = get_drive_line(s)
        p = get_readout_line(q)
        for _ in range(n):
            m = readout(p, square(50, 0.05))
            if m:
                drive(a, square(8, 0.1))
            else:
                drive(b, square(20, 0.1))
            drive(c, square(40, 0.1))
            drive(c, gaussian(40, 0.1, 10.0))

    @pulse.kernel
    def late_readout(q, r, s):
        # The readout waits for the pulse that depends on it.
        a = get_drive_line(r)
        p = get_readout_line(q)
        for _ in range(n):
            m = readout(p, square(20, 0.05))
            if m:
                drive(a, square(24, 0.1))
            wait(a, 4)

    @pulse.kernel
    def slower_arms(q, r, s):
        a = get_drive_line(r)
        b = get_drive_line(s)
        p = get_readout_line(q)
        drive(b, square(300, 0.1))
        for _ in range(n):
            m = readout(p, square(30, 0.05))
            if m:
                drive(a, square(40, 0.1))
            else:
                drive(b, square(36, 0.1))
                drive(b, square(6, 0.2))

    @pulse.kernel
    def in_an_arm(q, r, s):
        a = get_drive_line(q)
        b = get_drive_line(r)
        p = get_readout_line(q)
        m = readout(p, square(100, 0.05))
        if m:
            for _ in range(n):
                drive(a, square(7, 0.1))
                drive(b, square(11, 0.1))
        else:
            drive(get_drive_line(s), square(200, 0.1))

    return {
        "groups": groups,
        "late_readout": late_readout,
        "slower_arms": slower_arms,
        "in_an_arm": in_an_arm,
    }


LIMITS = [
    Controller(drive_lines=range(17), readout_lines=range(17), instruction_cycles=16),
    Controller(
        drive_lines=range(17),
        readout_lines=range(17),
        instruction_cycles=2,
        feedback_latency_cycles=60,
        max_amplitude=0.15,
    ),
    Controller(drive_lines=range(17), readout_lines=range(17), feedback_latency_cycles=500),
    Controller(
        drive_lines=range(17),
        readout_lines=range(17),
        feedback_latency_cycles=30,
        max_pulses=40,
        supports=[c for c in pulse.Capability if c is not pulse.Capability.LOOP],
    ),
]


def _system(limits: Controller) -> Any:
    return pulse.quantum_system(
        "transmon_surface17", control_system=ControlSystem(controllers=[limits])
    )


@pytest.mark.parametrize("trips", [65, 300])
@pytest.mark.parametrize("name", ["groups", "late_readout", "slower_arms", "in_an_arm"])
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_drifting_loop_reports_what_every_iteration_does(monkeypatch, trips, name, schedule):
    qpu = pulse.quantum_system("transmon_surface17")
    lowered = pulse.compile(
        _drifting(trips)[name], quantum_system=qpu, schedule=schedule, unroll_threshold=0
    ).lower(*qpu.qudits[:3])
    for limits in LIMITS:
        found = check(lowered, limits)
        assert list(found) == _every_iteration(monkeypatch, lowered, limits)


def _on_system(kernel: Any, schedule: str, limits: Controller) -> str:
    """*kernel* compiled for a control system of *limits*: what it exceeds."""
    qpu = _system(limits)
    try:
        pulse.compile(kernel, quantum_system=qpu, schedule=schedule, unroll_threshold=0).lower(
            *qpu.qudits[:3]
        )
    except controller.ControllerLimitError as error:
        return str(error)
    return "fits"


@pytest.mark.parametrize("name", ["groups", "slower_arms", "in_an_arm"])
@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize("limits", LIMITS[:2])
def test_a_control_system_checks_a_drifting_loop_from_one_copy(monkeypatch, name, schedule, limits):
    # Compiled for a control system, the loop is checked from its summary:
    # no walk plays its iterations, and the result is every iteration's.
    kernel = _drifting(300)[name]
    walked: list[int] = []
    walk = controller._walk

    def counted(*args: Any, **kwargs: Any) -> Any:
        found = walk(*args, **kwargs)
        walked.append(len(found[1]))
        return found

    def played(*args: Any, summary: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", counted)
        summarized = _on_system(kernel, schedule, limits)
    assert max(walked) < 300
    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", played)
        assert summarized == _on_system(kernel, schedule, limits)


def test_a_pulse_drifting_from_its_readout_reports_each_copy_too_early(monkeypatch):
    # The pulse in the first arm moves by 40 cycles a copy and its readout
    # by 42: the first copies are too early for 500 cycles of latency, until
    # the pulse falls behind far enough; the check finds them from one copy.
    lowered = pulse.compile(
        _drifting(400)["slower_arms"], schedule="asap", unroll_threshold=0
    ).lower(Q[0], Q[1], Q[2])
    limits = Controller(feedback_latency_cycles=500)
    walked: list[int] = []
    walk = controller._walk

    def counted(*args: Any, **kwargs: Any) -> Any:
        found = walk(*args, **kwargs)
        walked.append(len(found[1]))
        return found

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", counted)
        found = list(check(lowered, limits))
    assert max(walked) < 100
    assert 2 < len(found) < 100
    assert found == _every_iteration(monkeypatch, lowered, limits)
