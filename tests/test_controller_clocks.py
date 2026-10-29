# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Clocks of a control system: the program clock its controllers give a
device, and the reference clock they lock to."""

from __future__ import annotations

import json

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import DeviceDefinitionError
from cudaq_pulse.classical import Bits, Float, Link
from cudaq_pulse.controller import ControllerLimitError

from tests._handles import ladder_target


def _system(*clocks, **kwargs):
    """A control system of one controller per clock, each playing the drive
    and readout lines of two qudits of four."""
    return pulse.ControlSystem(
        controllers=[
            pulse.Controller(
                drive_lines=[2 * k, 2 * k + 1], readout_lines=[2 * k, 2 * k + 1], clock_ghz=c
            )
            for k, c in enumerate(clocks)
        ],
        **kwargs,
    )


def _device(*clocks, system=None, **config):
    control = system or _system(*clocks)
    return pulse.device({"target": ladder_target(4), "control_system": control, **config})


# ── The program clock ────────────────────────────────────────────────────────


def test_the_program_clock_is_the_fastest_controller_clock_the_others_divide():
    assert _device(1.0, 0.5).clock_ghz == 1.0
    assert _device(0.25, 1.0).clock_ghz == 1.0
    # A controller without a clock runs on the program clock.
    assert _device(None, 0.5).clock_ghz == 0.5
    # Without any controller clock, the quantum system's.
    assert _device(None, None).clock_ghz == ladder_target(4).clock_ghz


def test_uneven_controller_clocks_need_a_declared_program_clock():
    with pytest.raises(DeviceDefinitionError) as caught:
        _device(2.4, 1.0)
    [problem] = caught.value.problems
    assert problem.path == "clock_ghz"
    # Every clock that works, never one picked for the user.
    assert "set 'clock_ghz' in its config to one of 12 GHz" in problem.message
    assert "2.4 GHz (controllers[0]'s; then controllers[1] every 12 program cycles)" in (
        problem.message
    )
    assert _device(2.4, 1.0, clock_ghz=12.0).clock_ghz == 12.0
    assert _device(2.4, 1.0, clock_ghz=1.0).clock_ghz == 1.0


def test_a_declared_program_clock_has_a_whole_cycle_ratio_to_every_controller():
    with pytest.raises(DeviceDefinitionError, match="no whole-cycle ratio") as caught:
        _device(1.0, 0.5, clock_ghz=3.1415926)
    assert [p.path for p in caught.value.problems] == ["clock_ghz"]
    assert _device(1.0, 0.5, clock_ghz=2.0).clock_ghz == 2.0


def test_a_device_built_directly_derives_its_program_clock_too():
    target = ladder_target(4)
    assert pulse.Device(target, name="t", control_system=_system(1.0, 0.5)).clock_ghz == 1.0
    with pytest.raises(DeviceDefinitionError, match="do not divide evenly"):
        pulse.Device(target, name="t", control_system=_system(2.4, 1.0))


def test_a_derived_program_clock_round_trips_through_the_system_file():
    device = _device(1.0, 0.5)
    again = pulse.device(json.loads(json.dumps(device.as_dict())))
    assert again.clock_ghz == 1.0 and again == device


# ── The reference clock ──────────────────────────────────────────────────────


def test_every_controller_clock_is_a_whole_multiple_of_the_reference_clock():
    assert _device(1.0, 0.5, system=_system(1.0, 0.5, reference_clock_ghz=0.25)).clock_ghz == 1.0
    with pytest.raises(DeviceDefinitionError) as caught:
        _device(system=_system(1.0, None, reference_clock_ghz=0.3))
    paths = [p.path for p in caught.value.problems]
    # The second runs on the program clock, 1 GHz: neither is a multiple.
    assert paths == [
        "control_system.controllers[0].clock_ghz",
        "control_system.controllers[1].clock_ghz",
    ]
    assert "its 1 GHz clock (the program clock) is not a whole multiple of the 0.3 GHz" in str(
        caught.value
    )


# ── Scheduling on each controller's grid ─────────────────────────────────────
#
# Two controllers at 1 and 0.5 GHz on a 2 GHz program clock: qudits 0 and 1
# move in steps of 2 program cycles, qudits 2 and 3 in steps of 4.

SCHEDULES = ("asap", "alap", "rcp", "alap_rcp")
LINKS = {"controller": Link(fixed_ns=10)}


def _mixed(**kwargs):
    return _device(system=_system(1.0, 0.5, **kwargs), clock_ghz=2.0)


def _grid(qudit):
    return 2 if qudit < 2 else 4


# The loop stays rolled, so its body is placed on the grid once for every
# iteration.
@pulse.kernel(unroll_threshold=0)
def mixed(a, b):
    da, ta = get_drive_line(a)
    db, tb = get_drive_line(b)
    ra, rta = get_readout_line(a)
    drive(da, gaussian(40, 0.1, 10.0), ta)
    wait(db, 8)
    drive(db, square(pulse.round_duration(db, 15), 0.1), tb)
    sync(da, db, ra)
    drive(da, gaussian(40, 0.1, 10.0), ta)
    readout(ra, square(pulse.round_duration(ra, 99), 0.05), rta)
    for _ in range(4):
        drive(db, square(12, 0.1), tb)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_every_operation_starts_and_ends_on_its_own_controllers_grid(schedule):
    qpu = _mixed()
    lowered = pulse.compile(mixed, schedule=schedule, device=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    events = [e for e in lowered.schedule if e.kind in ("drive", "readout", "wait")]
    assert {e.qubit for e in events} == {0, 2}
    assert all(e.start % _grid(e.qubit) == 0 and e.end % _grid(e.qubit) == 0 for e in events)
    text = lowered.mlir()
    assert "pulse.drive_line_grid = array<i64: 2, 2, 4, 4>" in text
    assert "pulse.line_grid = 4 : i64" in text


@pulse.kernel
def back_to_back(a, b):
    da, ta = get_drive_line(a)
    db, tb = get_drive_line(b)
    drive(da, square(2, 0.1), ta)
    drive(da, square(2, 0.1), ta)
    drive(db, square(4, 0.1), tb)
    drive(db, square(4, 0.1), tb)


def test_each_line_keeps_its_own_controllers_finer_grid():
    qpu = _mixed()
    lowered = pulse.compile(back_to_back, schedule="asap", passes=[], device=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    starts = {(e.qubit, e.start) for e in lowered.schedule if e.kind == "drive"}
    assert starts == {(0, 0), (0, 2), (2, 0), (2, 4)}


@pulse.kernel
def twice(q: pulse.Qudit):
    d, t = get_drive_line(q)
    drive(d, square(4, 0.1), t)
    drive(d, square(4, 0.2), t)


@pulse.kernel
def short_twice(q: pulse.Qudit):
    d, t = get_drive_line(q)
    drive(d, square(2, 0.1), t)
    drive(d, square(2, 0.2), t)


@pulse.kernel
def shaped_anywhere(q: pulse.Qudit):
    d, t = get_drive_line(q)
    drive(d, custom(8, ramp), t)


def test_a_runtime_qudits_one_schedule_is_on_every_controllers_grid():
    # Compiled once for any qudit, its lines start where every grid meets...
    qpu = _mixed()
    compiled = pulse.compile(twice, schedule="asap", passes=[], device=qpu)
    for qudit in (0, 2):
        lowered = compiled.lower(qpu.qudits[qudit])
        assert [e.start for e in lowered.schedule if e.kind == "drive"] == [0, 4]
    # ...and each call checks its lengths on the grid of the controller it binds.
    short = pulse.compile(short_twice, schedule="asap", passes=[], device=qpu)
    assert [e.start for e in short.lower(qpu.qudits[0]).schedule if e.kind == "drive"] == [0, 4]
    with pytest.raises(ControllerLimitError, match="line d2 moves in timesteps of 4 cycles"):
        short(qpu.qudits[2])
    # Which rate the samples need depends on the qudit.
    with pytest.raises(pulse.CompilationError, match="runtime qudit's line"):
        pulse.compile(shaped_anywhere, device=qpu).lower(qpu.qudits[0])


def test_a_sync_meets_on_every_lines_grid():
    qpu = _mixed()
    lowered = pulse.compile(mixed, schedule="asap", device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    drives = [e for e in lowered.schedule if e.kind == "drive" and e.qubit == 0]
    # Qudit 2's line is busy until 8 + 16 = 24 and qudit 0's until 40; the
    # sync meets at 40, a step of both grids.
    assert drives[1].start == 40


@pulse.kernel
def callee_body(q):
    d, t = get_drive_line(q)
    drive(d, square(8, 0.1), t)
    drive(d, square(4, 0.1), t)


callee = pulse.kernel(schedule="asap")(callee_body.__pulse_kernel_fn__)


@pulse.kernel
def calls_a_block(a, b):
    da, ta = get_drive_line(a)
    drive(da, square(2, 0.1), ta)
    callee(a)
    callee(b)


@pytest.mark.parametrize("schedule", ("asap", "alap"))
def test_a_kernel_with_its_own_schedule_is_placed_on_its_lines_grid(schedule):
    qpu = _mixed()
    lowered = pulse.compile(calls_a_block, schedule=schedule, device=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    drives = [e for e in lowered.schedule if e.kind == "drive"]
    assert all(e.start % _grid(e.qubit) == 0 for e in drives)


@pulse.classical(args=(Bits(1),), results=(Float(),), max_latency_ns=60, link="controller")
def correction(bits):
    return 0.1 * bits[0]


@pulse.kernel
def with_a_call(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    m = readout(r, square(100, 0.05), rt)
    amplitude = correction.submit([m]).result()
    drive(d, square(8, amplitude), t)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_classical_calls_keep_each_controllers_lines_on_its_grid(schedule):
    qpu = _mixed(links=LINKS)
    for qudit in (0, 2):
        lowered = pulse.compile(with_a_call, schedule=schedule, device=qpu).lower(qpu.qudits[qudit])
        assert all(
            e.start % _grid(qudit) == 0 for e in lowered.schedule if e.kind in ("drive", "readout")
        )


# ── Off the grid ─────────────────────────────────────────────────────────────


@pulse.kernel
def off_grid(q):
    d, t = get_drive_line(q)
    drive(d, gaussian(41, 0.1, 10.0), t)
    wait(d, 5)


def test_a_length_off_its_controllers_grid_fails_to_compile_naming_the_line_and_the_fix():
    qpu = _mixed()
    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(off_grid, device=qpu).lower(qpu.qudits[0])
    drive, wait = caught.value.violations
    assert drive == (
        "line d0 moves in timesteps of 2 cycles; 41 cycles isn't a whole number of "
        "timesteps (the drive at cycle 0; pulse.round_duration(line, 41) gives 42, or "
        "40 rounding down)"
    )
    assert wait.startswith("line d0 moves in timesteps of 2 cycles; 5 cycles isn't")
    with pytest.raises(ControllerLimitError, match="line d2 moves in timesteps of 4 cycles"):
        pulse.compile(off_grid, device=qpu).lower(qpu.qudits[2])


@pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=60, link="controller")
def wait_length(x):
    return x


@pulse.kernel
def answer_wait_off_grid(q):
    d, t = get_drive_line(q)
    drive(d, square(8, 0.1), t)
    wait(d, wait_length.submit(1.0).result(), bounds=(6, 402))
    drive(d, square(8, 0.1), t)


def test_a_wait_from_an_answer_has_its_bounds_on_every_controllers_grid():
    qpu = _mixed(links=LINKS)
    # Qudit 0's own grid is 2, but every later op on every line moves by it.
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.compile(answer_wait_off_grid, device=qpu).lower(qpu.qudits[0])
    assert "move in timesteps of 4 cycles; its bounds (6, 402) aren't whole" in str(caught.value)


@pulse.kernel
def variable_wait(q, tau: float):
    d, t = get_drive_line(q)
    drive(d, square(8, 0.1), t)
    wait(d, tau * 2.0, bounds=(0, 1000))
    drive(d, square(8, 0.1), t)


def test_a_variable_wait_off_its_controllers_grid_fails_at_the_call():
    qpu = _mixed()
    compiled = pulse.compile(variable_wait, device=qpu)
    assert [e.start for e in compiled.lower(qpu.qudits[2], 6.0).schedule if e.kind == "drive"] == [
        0,
        20,
    ]
    with pytest.raises(ValueError, match="nearest lengths are 4 and 8"):
        compiled.lower(qpu.qudits[2], 3.0)
    # Qudit 0's grid takes a length of 6.
    assert compiled.lower(qpu.qudits[0], 3.0).schedule


# ── Each controller's limits in its own cycles ───────────────────────────────


@pulse.kernel
def feedback(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    m = readout(r, square(100, 0.05), rt)
    if m:
        drive(d, square(8, 0.1), t)


def test_rcp_waits_each_readers_feedback_latency_in_program_cycles():
    controllers = [
        pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], clock_ghz=1.0),
        pulse.Controller(
            drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=0.5, feedback_latency_cycles=10
        ),
    ]
    qpu = _device(system=pulse.ControlSystem(controllers=controllers), clock_ghz=2.0)
    lowered = pulse.compile(feedback, schedule="rcp", device=qpu).lower(qpu.qudits[2])
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    # 10 cycles at 0.5 GHz are 40 of the program's.
    assert drive.start >= readout.end + 40


def test_each_lines_delay_is_split_by_its_own_controllers_fine_delay():
    controllers = [
        pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], max_fine_delay_ns=1.0),
        pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3]),
    ]

    def compiled(delays):
        system = pulse.ControlSystem(controllers=controllers, line_delays_ns=delays)
        qpu = _device(system=system)
        return pulse.compile(feedback, device=qpu).lower(qpu.qudits[0])

    assert compiled({("drive", 0): 2.3}).fine_delays[("drive", 0)] == pytest.approx(0.3)
    with pytest.raises(pulse.CompilationError, match="the controller has no fine delay"):
        compiled({("drive", 2): 2.3})


# ── Samples at each line's rate ──────────────────────────────────────────────


def ramp(t):
    return 0.1 + 0.01 * t


@pulse.kernel
def shaped_on_both(a, b):
    da, ta = get_drive_line(a)
    db, tb = get_drive_line(b)
    envelope = custom(8, ramp)
    drive(da, envelope, ta)
    drive(db, envelope, tb)


def _samples(lowered):
    return [
        node.split("samples = ")[1]
        for node in lowered.mlir().splitlines()
        if "custom_samples" in node and "samples = " in node
    ]


def test_custom_samples_the_envelope_at_the_rate_of_the_line_that_plays_it():
    qpu = _mixed()
    lowered = pulse.compile(shaped_on_both, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    drives = [e for e in lowered.schedule if e.kind == "drive"]
    assert [e.duration_cycles for e in drives] == [8, 8]
    text = lowered.mlir()
    assert "rate = array<i64: 1, 2>" in text and "rate = array<i64: 1, 4>" in text


@pulse.kernel
def shaped(q):
    d, t = get_drive_line(q)
    drive(d, custom(8, ramp), t)


@pulse.kernel
def held_for_two_cycles(q):
    d, t = get_drive_line(q)
    drive(d, custom_samples([ramp(t) for t in (1, 1, 3, 3, 5, 5, 7, 7)]), t)


def test_a_custom_envelope_is_simulated_at_its_own_controllers_rate():
    # On qudit 0's controller at 1 GHz, each sample is held for two cycles.
    qpu = _mixed()
    lowered = pulse.compile(shaped, device=qpu).lower(qpu.qudits[0])
    plain = _device(None, None)
    held = pulse.compile(held_for_two_cycles, device=plain).lower(plain.qudits[0])
    a = pulse.simulate(lowered).density_matrix
    b = pulse.simulate(held).density_matrix
    np.testing.assert_allclose(a, b, atol=1e-12)


@pulse.kernel
def made_for_one_ghz(q):
    d, t = get_drive_line(q)
    drive(d, custom_samples([0.1, 0.2, 0.3, 0.4], 1.0), t)


@pulse.kernel
def scaled_custom(q):
    d, t = get_drive_line(q)
    drive(d, wf_scale(custom(8, ramp), 0.5), t)


def test_sample_arrays_play_only_on_lines_of_the_rate_they_were_made_for():
    qpu = _mixed()
    assert pulse.compile(made_for_one_ghz, device=qpu).lower(qpu.qudits[0]).schedule
    with pytest.raises(pulse.CompilationError) as caught:
        pulse.compile(made_for_one_ghz, device=qpu).lower(qpu.qudits[2])
    message = str(caught.value)
    assert message.startswith(
        "drive(): the waveform on the drive line of qudit 2 holds samples made for 1 GHz, "
        "and the line's controller plays 0.5 GHz (one sample every 4 clock cycles)"
    )
    assert "sample_rate_ghz=0.5" in message and "custom(duration, f)" in message


def test_a_waveform_made_of_custom_is_made_again_at_its_lines_rate():
    qpu = _mixed()
    lowered = pulse.compile(scaled_custom, device=qpu).lower(qpu.qudits[2])
    assert "rate = array<i64: 1, 4>" in lowered.mlir()
    assert [e.duration_cycles for e in lowered.schedule if e.kind == "drive"] == [8]
