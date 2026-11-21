# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller checks of pulses in the two arms of one measured ``if``.

Only one arm of a branch plays, so the controller's spacing and detector
checks never pair a pulse with one in the other arm, a rolled loop's pulses
in an arm too: the loop's iterations, played or summarized, keep the arm
they are in.
"""

from __future__ import annotations

from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, controller
from cudaq_pulse.controller import check
from cudaq_pulse.quantum_systems import PhysicalQudit

SCHEDULES = ["asap", "alap", "rcp", "alap_rcp"]


def _target(*limits: Controller) -> Any:
    qudits = {
        k: PhysicalQudit(k, 4.5e9 + 0.1e9 * k, 37.7, 9.2, anharmonicity_hz=-215e6) for k in range(2)
    }
    if not limits:
        return pulse.quantum_system(name="two", qudits=qudits)
    return pulse.quantum_system(
        name="two", qudits=qudits, control_system=ControlSystem(controllers=list(limits))
    )


def _controller(**limits: Any) -> Controller:
    return Controller(clock_ghz=2.0, drive_lines=[0, 1], readout_lines=[0, 1], **limits)


def _kernels(n: int) -> dict[str, Any]:
    @pulse.kernel
    def loop_in_else(q, s):
        d = get_drive_line(q)
        r = get_readout_line(q)
        w = drag(24 * pulse.ns, 0.25, 6.0 * pulse.ns, 0.5)
        m = readout(r, square(1000 * pulse.ns, 0.05))
        sync(d, r)
        if m:
            drive(d, w)
        else:
            for _ in range(n):
                drive(d, w)

    @pulse.kernel
    def loop_in_then(q, s):
        d = get_drive_line(q)
        r = get_readout_line(q)
        w = drag(24 * pulse.ns, 0.25, 6.0 * pulse.ns, 0.5)
        m = readout(r, square(1000 * pulse.ns, 0.05))
        sync(d, r)
        if m:
            for _ in range(n):
                drive(d, w)
        else:
            drive(d, w)
            drive(d, w)

    @pulse.kernel
    def nested_arms(q, s):
        d = get_drive_line(q)
        r = get_readout_line(q)
        w = drag(24 * pulse.ns, 0.25, 6.0 * pulse.ns, 0.5)
        p = get_readout_line(s)
        m = readout(r, square(1000 * pulse.ns, 0.05))
        k = readout(p, square(1000 * pulse.ns, 0.05))
        sync(d, r, p)
        if m:
            if k:
                for _ in range(n):
                    drive(d, w)
            else:
                drive(d, w)
        else:
            for _ in range(n):
                drive(d, w)

    @pulse.kernel
    def loops_in_both(q, s):
        d = get_drive_line(q)
        r = get_readout_line(q)
        w = drag(24 * pulse.ns, 0.25, 6.0 * pulse.ns, 0.5)
        m = readout(r, square(1000 * pulse.ns, 0.05))
        sync(d, r)
        if m:
            for _ in range(n):
                drive(d, w)
        else:
            for _ in range(n + 3):
                drive(d, w)
                wait(d, 8)

    @pulse.kernel
    def in_a_loop(q, s):
        d = get_drive_line(q)
        r = get_readout_line(q)
        w = drag(24 * pulse.ns, 0.25, 6.0 * pulse.ns, 0.5)
        for _ in range(n):
            m = readout(r, square(500 * pulse.ns, 0.05))
            sync(d, r)
            if m:
                drive(d, w)
            else:
                for _ in range(3):
                    drive(d, w)
            sync(d, r)

    return {
        "loop_in_else": loop_in_else,
        "loop_in_then": loop_in_then,
        "nested_arms": nested_arms,
        "loops_in_both": loops_in_both,
        "in_a_loop": in_a_loop,
    }


def _compiled(kernel: Any, schedule: str, threshold: int, limits: Controller) -> str:
    qpu = _target(limits)
    try:
        pulse.compile(
            kernel, quantum_system=qpu, schedule=schedule, unroll_threshold=threshold
        ).lower(*qpu.qudits)
    except controller.ControllerLimitError as error:
        return str(error)
    return "fits"


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("name", ["loop_in_else", "loop_in_then", "nested_arms", "loops_in_both"])
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_a_rolled_loop_in_one_arm_is_not_paired_with_the_other_arm(n, name, schedule):
    # Played (10 iterations) or summarized (100), as with the loop unrolled.
    limits = _controller(feedback_latency_cycles=400)
    kernel = _kernels(n)[name]
    assert _compiled(kernel, schedule, 1, limits) == "fits"
    assert _compiled(kernel, schedule, 1000, limits) == "fits"


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_a_loop_in_an_arm_in_a_loop_is_not_paired_with_the_other_arm(schedule):
    limits = _controller(feedback_latency_cycles=400)
    kernel = _kernels(100)["in_a_loop"]
    assert _compiled(kernel, schedule, 1, limits) == "fits"


def _every_iteration(monkeypatch: pytest.MonkeyPatch, lowered: Any, limits: Any) -> list[str]:
    """What ``check`` reports with every iteration of every loop played."""
    walk = controller._walk

    def played(*args: Any, summary: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", played)
        return list(check(lowered, limits))


@pytest.mark.parametrize("n", [10, 100])
@pytest.mark.parametrize("name", list(_kernels(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap"])
def test_pulses_of_one_arm_are_still_paired_with_each_other(monkeypatch, n, name, schedule):
    # A spacing the pulses in one arm break is reported, once per pulse,
    # summarized as with every iteration played.
    qpu = _target()
    lowered = pulse.compile(
        _kernels(n)[name], quantum_system=qpu, schedule=schedule, unroll_threshold=1
    ).lower(*qpu.qudits)
    limits = _controller(instruction_cycles=64, detectors={0: 0, 1: 0})
    found = list(check(lowered, limits))
    assert found == _every_iteration(monkeypatch, lowered, limits)
    assert found and not any("starts 0 cycles after" in v for v in found)
