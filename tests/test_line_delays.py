# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calibrated line delays, declared with each line (``DriveLine(5,
delay_ns=4.0)``): whole grid steps move a line's operations, the rest is its
fine delay, and a change of delays alone schedules again."""

from __future__ import annotations

import json

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.controller import ControllerLimitError, DriveLine, ReadoutLine

from tests._handles import Q


@pulse.kernel
def two_lines(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    drive(d, square(8, 0.1))
    drive(e, square(8, 0.1))
    sync(d, e)
    drive(d, square(8, 0.1))


@pulse.kernel
def feedback(q, r):
    ro = get_readout_line(q)
    d = get_drive_line(r)
    m = readout(ro, square(100, 0.05))
    sync(ro, d)
    if m:
        drive(d, square(20, 0.3))


@pulse.kernel
def rabi(q, amplitude: float):
    d = get_drive_line(q)
    r = get_readout_line(q)
    for _ in range(3):
        drive(d, gaussian(16, amplitude, 4.0))
        wait(d, 4)
    sync(d, r)
    readout(r, square(40, 0.05))


def _delayed(delays=None, links=None, **controller):
    """A system of one controller (of the fields *controller*) playing the
    lines of qudits 0 to 3, each delayed as *delays* gives, by (role, qudit)."""
    delays = delays or {}
    box = pulse.Controller(
        drive_lines=[DriveLine(q, delay_ns=delays.get(("drive", q), 0.0)) for q in range(4)],
        readout_lines=[ReadoutLine(q, delay_ns=delays.get(("readout", q), 0.0)) for q in range(4)],
        **controller,
    )
    return pulse.ControlSystem(controllers=[box], links=links or {})


def _starts(lowered, kind="drive"):
    return [(e.qudit, e.start) for e in lowered.schedule if e.kind == kind]


def _lines(lowered):
    return {(line["role"], line["qudit"]): line for line in lowered.export_to_controller()["lines"]}


def test_whole_cycles_of_a_delay_move_the_lines_operations():
    plain = pulse.compile(two_lines, schedule="asap").lower(Q[0], Q[1])
    machine = _delayed({("drive", 0): 1.5})
    delayed = pulse.compile(two_lines, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    # 1.5 ns is 3 cycles of the 2 GHz clock; only qudit 0's line moves.
    assert _starts(plain) == [(0, 0), (1, 0), (0, 8)]
    assert _starts(delayed) == [(0, 3), (1, 0), (0, 11)]
    lines = _lines(delayed)
    assert lines[("drive", 0)]["fine_delay_ns"] == 0.0
    assert lines[("drive", 1)]["fine_delay_ns"] == 0.0
    assert "pulse.line_delays" in delayed.mlir()


def test_without_delays_nothing_changes():
    lowered = pulse.compile(two_lines).lower(Q[0], Q[1])
    assert "line_delays" not in lowered.mlir()
    assert all("fine_delay_ns" not in line for line in lowered.export_to_controller()["lines"])
    # A delay of no whole cycle moves nothing and adds no attribute.
    machine = _delayed({("drive", 0): 0.25}, max_fine_delay_ns=0.5)
    lowered = pulse.compile(two_lines, control_system=machine).lower(Q[0], Q[1])
    assert "line_delays" not in lowered.mlir()
    assert _lines(lowered)[("drive", 0)]["fine_delay_ns"] == pytest.approx(0.25)


def test_the_rest_of_a_delay_is_the_lines_fine_delay():
    # A 1 GHz controller starts every 2 cycles of the 2 GHz program clock:
    # steps of 1 ns.
    machine = _delayed({("drive", 1): 2.3}, clock_ghz=1.0, max_fine_delay_ns=1.0)
    lowered = pulse.compile(two_lines, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    assert _starts(lowered) == [(0, 0), (1, 4), (0, 8)]
    lines = _lines(lowered)
    assert lines[("drive", 1)]["fine_delay_ns"] == pytest.approx(0.3)
    assert lines[("drive", 0)]["fine_delay_ns"] == 0.0
    # The export counts the controller's own cycles.
    program = lowered.export_to_controller()
    starts = [e["start_cycles"] for e in program["events"] if e["kind"] == "drive"]
    assert starts == [0, 2, 4]
    json.dumps(program)


def test_a_rest_the_controller_cannot_add_fails_to_compile():
    machine = _delayed({("drive", 1): 2.3}, clock_ghz=1.0)
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.compile(two_lines, control_system=machine).lower(Q[0], Q[1])
    message = str(caught.value)
    assert message.startswith("the drive line of qudit 1 is delayed 2.3 ns (its delay_ns)")
    assert (
        "0.3 ns (0.3 of its controller's cycles) from a whole number of its 1 ns timesteps"
        in message
    )
    assert "give it max_fine_delay_ns, raise its alignment_tolerance_cycles to 0.3" in message
    assert "calibrate the delay to 2 or 3 ns" in message
    # Within the tolerance, the delay rounds to the nearest step.
    machine = _delayed({("drive", 1): 2.7}, clock_ghz=1.0, alignment_tolerance_cycles=1)
    lowered = pulse.compile(two_lines, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    assert _starts(lowered) == [(0, 0), (1, 6), (0, 8)]
    assert _lines(lowered)[("drive", 1)]["fine_delay_ns"] == 0.0
    # Past the fine delay a controller has.
    machine = _delayed({("drive", 1): 2.3}, clock_ghz=1.0, max_fine_delay_ns=0.25)
    with pytest.raises(pulse.CompilationError, match="adds at most 0.25 ns of fine delay"):
        pulse.compile(two_lines, control_system=machine).lower(Q[0], Q[1])


@pytest.mark.parametrize(
    "make, error, match",
    [
        (lambda: DriveLine(0, delay_ns=-1.0), pulse.InvalidValueError, "cannot be negative"),
        (lambda: DriveLine(0, delay_ns=-1 * pulse.ns), pulse.InvalidValueError, "-1.0 ns"),
        (lambda: DriveLine(0, delay_ns=float("nan")), pulse.InvalidValueError, "a number of ns"),
        (lambda: ReadoutLine(-1), pulse.InvalidValueError, "negative carrier index"),
        (lambda: DriveLine("d0"), pulse.InvalidTypeError, "a qudit or a list of qudits"),
        (
            lambda: pulse.Controller(drive_lines=[ReadoutLine(0)]),
            pulse.InvalidTypeError,
            "holds DriveLines or carrier indices",
        ),
        (
            lambda: pulse.Controller(drive_lines=[0, DriveLine(0, delay_ns=1.0)]),
            pulse.InvalidValueError,
            "lists a carrier twice",
        ),
    ],
)
def test_line_delays_are_checked(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_a_line_is_its_qudit_and_its_delay():
    line = DriveLine(5, delay_ns=4 * pulse.ns)
    assert (line.qudit, line.delay_ns) == (5, 4.0) and line == DriveLine(5, 4.0)
    assert line != DriveLine(5) and line != ReadoutLine(5, 4.0) and DriveLine(5) != 5
    assert repr(line) == "DriveLine(5, delay_ns=4.0)" and repr(ReadoutLine(5)) == "ReadoutLine(5)"
    with pytest.raises(AttributeError):
        line.delay_ns = 1.0  # type: ignore[misc]
    box = pulse.Controller(drive_lines=[line, 6], readout_lines=[ReadoutLine(5, delay_ns=12.0)])
    assert box.drive_lines == (line, DriveLine(6))
    assert box.as_dict()["drive_lines"] == [{"qudit": 5, "delay_ns": 4.0}, 6]
    assert not hasattr(pulse.ControlSystem(controllers=[box]), "line_delays_ns")


def test_delays_are_part_of_the_system():
    a = _delayed({("drive", 0): 1.0, ("readout", 1): 2.0})
    b = _delayed({("readout", 1): 2.0, ("drive", 0): 1.0})
    assert a == b and hash(a) == hash(b)
    assert a != _delayed() and a != _delayed({("drive", 0): 1.5})


def test_moved_lines_are_checked_like_any_others():
    machine = _delayed(feedback_latency_cycles=20)
    assert pulse.compile(feedback, schedule="rcp", control_system=machine).lower(Q[0], Q[1])
    later = _delayed({("drive", 1): 5.0}, feedback_latency_cycles=20)
    assert pulse.compile(feedback, schedule="rcp", control_system=later).lower(Q[0], Q[1])
    # The readout ends 10 cycles later, too close to the branch on it.
    late_readout = _delayed({("readout", 0): 5.0}, feedback_latency_cycles=20)
    with pytest.raises(ControllerLimitError, match="feedback"):
        pulse.compile(feedback, schedule="rcp", control_system=late_readout).lower(Q[0], Q[1])


# ── Scheduling again ─────────────────────────────────────────────────────────


def _fresh(kernel):
    """The same kernel, without the specializations *kernel* traced."""
    return pulse.kernel(schedule=kernel.schedule, unroll_threshold=kernel.unroll_threshold)(
        kernel.__pulse_kernel_fn__
    )


def _outputs(lowered):
    return (
        lowered.mlir(),
        json.dumps(lowered.export_to_controller(), sort_keys=True),
        json.dumps(lowered.export_to_emulator(), sort_keys=True),
    )


def _retimed(lowered):
    return lowered.generic.metrics.trace_ms == 0.0


MIXED = {"clock_ghz": 1.0, "max_fine_delay_ns": 1.0}
# Traced with the first, then scheduled again with each of the others.
DELAYS = (
    {("drive", 0): 1.0},
    {("drive", 0): 2.5, ("readout", 0): 1.5, ("drive", 1): 4.0},
    {},
    {("readout", 0): 3.0},
)


@pytest.mark.parametrize("controller", ({}, MIXED), ids=("program_clock", "mixed"))
@pytest.mark.parametrize("schedule", ("asap", "alap", "rcp", "alap_rcp"))
@pytest.mark.parametrize(
    "kernel, args",
    [(two_lines, (Q[0], Q[1])), (feedback, (Q[0], Q[1])), (rabi, (Q[0], 0.3))],
    ids=("two_lines", "feedback", "rabi"),
)
def test_new_delays_alone_schedule_again_to_the_same_bytes(kernel, args, schedule, controller):
    kernel = _fresh(kernel)
    for k, delays in enumerate(DELAYS):
        machine = _delayed(delays, **controller)
        again = pulse.compile(kernel, schedule=schedule, control_system=machine).lower(*args)
        assert _retimed(again) == (k > 0)
        full = pulse.compile(_fresh(kernel), schedule=schedule, control_system=machine).lower(*args)
        assert not _retimed(full)
        assert _outputs(again) == _outputs(full)


def test_a_program_without_delays_is_traced_as_before():
    kernel = _fresh(two_lines)
    pulse.compile(kernel).lower(Q[0], Q[1])
    assert not kernel._traced
    machine = _delayed({("drive", 0): 1.0})
    pulse.compile(kernel, control_system=machine).lower(Q[0], Q[1])
    # The same delays again trace again, as without them.
    assert not _retimed(pulse.compile(kernel, control_system=machine).lower(Q[0], Q[1]))


@pytest.mark.slow
def test_new_delays_schedule_the_matrix_again_to_the_same_bytes():
    from tests.classical import _schedule_matrix as matrix

    first = _delayed({("drive", 1): 1.0}, links=matrix.MACHINE.links)
    then = _delayed({("drive", 0): 2.5, ("readout", 1): 1.5}, links=matrix.MACHINE.links)
    checked = 0
    for case in matrix.all_cases()[::5]:
        if matrix.expected_error(case) is not None:
            continue
        kernel = matrix.program(case)
        matrix.compile_case(case, kernel=kernel, control_system=first).lower(
            Q[0], Q[1], matrix.LOOP, matrix.FLOAT_TAU
        )
        again = matrix.compile_case(case, kernel=kernel, control_system=then).lower(
            Q[0], Q[1], matrix.LOOP, matrix.FLOAT_TAU
        )
        full = matrix.compile_case(case, kernel=_fresh(kernel), control_system=then).lower(
            Q[0], Q[1], matrix.LOOP, matrix.FLOAT_TAU
        )
        if not _retimed(again):
            continue
        assert _outputs(again) == _outputs(full), case
        checked += 1
    assert checked > 20


def test_a_call_with_new_values_keeps_the_delays():
    machine = _delayed({("drive", 0): 1.0})
    compiled = pulse.compile(rabi, schedule="asap", control_system=machine)
    assert [s for _, s in _starts(compiled.lower(Q[0], 0.2))][0] == 2
    assert [s for _, s in _starts(compiled.lower(Q[0], 0.4))][0] == 2
