# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Clocks: the device's program clock, a controller's own clock and the grid
it puts its lines on, ``cycles_on``, the reference clock and the exports in a
controller's own cycles."""

from __future__ import annotations

import dataclasses
import json
import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import _clocks
from cudaq_pulse.classical import Bits, Float, Link
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import ControllerLimitError, ControllerLimits, check, sample_waveform

from tests._handles import LADDER, Q, ladder_target

SCHEDULES = ("asap", "alap", "rcp", "alap_rcp")


def _machine(clock_ghz: float, **kwargs):
    return MachineModel(controller=ControllerLimits(clock_ghz=clock_ghz), **kwargs)


# ── The program clock ────────────────────────────────────────────────────────


def test_a_config_sets_the_program_clock_and_the_fingerprint_follows():
    default = pulse.device({"target": ladder_target(4)})
    same = pulse.device({"target": ladder_target(4), "clock_ghz": 2.0})
    slow = pulse.device({"target": ladder_target(4), "clock_ghz": 1.0})
    assert default.clock_ghz == same.clock_ghz == 2.0 and default == same
    assert default.fingerprint == same.fingerprint
    assert slow.clock_ghz == 1.0 and slow.target.clock_ghz == 1.0
    assert slow.fingerprint != default.fingerprint


@pytest.mark.parametrize("bad", [0, -1.0, "2", True, math.inf])
def test_a_config_clock_must_be_a_positive_number(bad):
    with pytest.raises(ValueError, match="clock_ghz"):
        pulse.device({"target": ladder_target(4), "clock_ghz": bad})


def test_the_program_clock_comes_from_controllers_that_divide_evenly():
    assert _clocks.program_clock_ghz(None, {}, 2.0) == 2.0
    assert _clocks.program_clock_ghz(None, {"a": 2.0, "b": 1.0, "c": 0.5}, 9.0) == 2.0
    assert _clocks.program_clock_ghz(1.5, {"a": 3.0, "b": 0.5}, 9.0) == 1.5


def test_uneven_controller_clocks_need_a_declared_program_clock():
    with pytest.raises(ValueError) as caught:
        _clocks.program_clock_ghz(None, {"awg": 2.4, "readout": 1.0}, 2.0)
    message = str(caught.value)
    assert "set 'clock_ghz'" in message
    # Every clock that works, never one picked for the user.
    assert "12 GHz, which every controller's clock divides" in message
    assert "2.4 GHz (awg's; then readout every 12 program cycles)" in message
    assert "1 GHz (readout's; then awg every 5 program cycles)" in message


def test_a_clock_without_a_whole_cycle_ratio_is_an_error():
    with pytest.raises(ValueError, match="no whole-cycle ratio"):
        _clocks.clock_ratio(math.pi, 2.0)
    assert _clocks.clock_ratio(0.5, 2.0) == _clocks.Fraction(1, 4)
    assert _clocks.clock_ratio(2.4, 1.0).denominator == 5


# ── cycles_on ────────────────────────────────────────────────────────────────


@pulse.kernel
def rounded(q):
    d, t = get_drive_line(q)
    drive(d, square(pulse.cycles_on(d, 37), 0.1), t)
    drive(d, square(pulse.cycles_on(d, 37, "down"), 0.1), t)
    drive(d, square(pulse.cycles_on(d, 37, rounding=pulse.Rounding.NEAREST), 0.1), t)
    drive(d, square(pulse.cycles_on(d, 2.5), 0.1), t)


def _durations(lowered):
    return [e.duration_cycles for e in lowered.schedule if e.kind == "drive"]


def test_cycles_on_rounds_onto_the_line_grid():
    def durations(machine=None):
        return _durations(pulse.compile(rounded, passes=[], machine=machine).lower(Q[0]))

    assert durations() == [37, 37, 37, 3]
    # A 0.5 GHz controller starts every 4 cycles of the 2 GHz program clock.
    assert durations(_machine(0.5)) == [40, 36, 36, 4]
    # A faster controller times every program cycle.
    assert durations(_machine(4.0)) == [37, 37, 37, 3]


def test_cycles_on_needs_a_line_and_a_known_length():
    @pulse.kernel
    def specialized(q, n: float):
        d, t = get_drive_line(q)
        drive(d, square(pulse.cycles_on(d, n), 0.1), t)

    @pulse.kernel
    def not_a_line(q):
        d, t = get_drive_line(q)
        drive(d, square(pulse.cycles_on(t, 4), 0.1), t)

    @pulse.kernel
    def a_number(q):
        d, t = get_drive_line(q)
        drive(d, square(pulse.cycles_on(3, 4), 0.1), t)

    @pulse.kernel
    def bad_mode(q):
        d, t = get_drive_line(q)
        drive(d, square(pulse.cycles_on(d, 4, "sideways"), 0.1), t)

    # Like a duration, a float that reaches it is specialized on.
    with pytest.warns(RuntimeWarning, match="cycles_on"):
        lowered = pulse.compile(specialized, machine=_machine(0.5)).lower(Q[0], 13.0)
    assert _durations(lowered) == [16]
    with pytest.raises(pulse.CompilationError, match="takes a line"):
        pulse.compile(not_a_line).lower(Q[0])
    with pytest.raises(pulse.CompilationError, match="takes a line"):
        pulse.compile(a_number).lower(Q[0])
    with pytest.raises(pulse.CompilationError, match="up, down, nearest"):
        pulse.compile(bad_mode).lower(Q[0])
    with pytest.raises(TypeError, match="inside a @cudaq_pulse.kernel"):
        pulse.cycles_on(None, 4)


# ── Scheduling on the grid ───────────────────────────────────────────────────


@pulse.kernel
def mixed(a, b):
    da, ta = get_drive_line(a)
    db, tb = get_drive_line(b)
    ra, rta = get_readout_line(a)
    drive(da, gaussian(40, 0.1, 10.0), ta)
    wait(db, 8)
    drive(db, square(pulse.cycles_on(db, 15), 0.1), tb)
    sync(da, db, ra)
    drive(da, gaussian(40, 0.1, 10.0), ta)
    readout(ra, square(pulse.cycles_on(ra, 99), 0.05), rta)
    for _ in range(4):
        drive(db, square(12, 0.1), tb)


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_every_operation_starts_and_ends_on_its_controllers_grid(schedule):
    lowered = pulse.compile(mixed, schedule=schedule, machine=_machine(0.5)).lower(Q[0], Q[1])
    events = [e for e in lowered.schedule if e.kind in ("drive", "readout", "wait")]
    assert events and all(e.start % 4 == 0 and e.end % 4 == 0 for e in events)
    assert "pulse.line_grid = 4 : i64" in lowered.mlir()
    # The program clock runs on; only the starts moved.
    plain = pulse.compile(mixed, schedule=schedule).lower(Q[0], Q[1])
    assert len(plain.schedule) == len(lowered.schedule)


def test_a_sync_meets_on_the_grid():
    lowered = pulse.compile(mixed, schedule="asap", machine=_machine(0.5)).lower(Q[0], Q[1])
    drives = [e for e in lowered.schedule if e.kind == "drive" and e.line == 0]
    # Line 0 is busy until 40; the sync meets there, on the grid.
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
    drive(da, square(4, 0.1), ta)
    callee(a)
    callee(b)


@pytest.mark.parametrize("schedule", ("asap", "alap"))
def test_a_kernel_with_its_own_schedule_is_placed_on_the_grid(schedule):
    lowered = pulse.compile(calls_a_block, schedule=schedule, machine=_machine(0.5)).lower(
        Q[0], Q[1]
    )
    assert all(e.start % 4 == 0 for e in lowered.schedule if e.kind == "drive")


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
def test_classical_calls_keep_lines_on_the_grid(schedule):
    machine = _machine(0.5, links={"controller": Link(fixed_ns=10)})
    lowered = pulse.compile(with_a_call, schedule=schedule, machine=machine).lower(Q[0])
    assert all(e.start % 4 == 0 for e in lowered.schedule if e.kind in ("drive", "readout"))
    program = lowered.export_controller()
    assert program["clock_ghz"] == 0.5 and program["program_clock_ghz"] == 2.0
    report = lowered.classical_report()
    call = program["calls"][0]
    assert call["submit_cycles"] == -(-report[0].submit_cycles // 4)
    assert call["due_cycles"] == -(-report[0].due_cycles // 4)


# ── Off the grid ─────────────────────────────────────────────────────────────


@pulse.kernel
def off_grid(q):
    d, t = get_drive_line(q)
    drive(d, gaussian(41, 0.1, 10.0), t)
    wait(d, 5)


def test_a_length_off_the_grid_fails_to_compile_naming_the_line_and_the_fix():
    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(off_grid, machine=_machine(1.0)).lower(Q[3])
    drive, wait = caught.value.violations
    assert drive.startswith("the drive line of qudit 3 (line 0), on the controller at 1 GHz")
    assert "every 2 program cycles" in drive
    assert "the drive at 0 lasts 41 cycles; the nearest valid lengths are 40 and 42" in drive
    assert "pulse.cycles_on(line, 41)" in drive
    assert "the wait at 42 lasts 5 cycles; the nearest valid lengths are 4 and 6" in wait
    # A controller that starts every cycle takes it.
    assert pulse.compile(off_grid, machine=_machine(2.0)).lower(Q[3]).schedule


def test_check_against_a_slower_controller_reports_what_is_off_its_grid():
    lowered = pulse.compile(off_grid).lower(Q[0])
    found = check(lowered, ControllerLimits(clock_ghz=1.0))
    assert len(found) == 2 and all("every 2 program cycles" in f for f in found)
    assert "the wait at 41 lasts 5 cycles" in found[1]
    assert "and starts off the grid; the nearest valid starts are 40 and 42" in found[1]
    assert check(lowered, ControllerLimits(clock_ghz=4.0)) == []


@pulse.kernel
def variable_wait(q, tau: float):
    d, t = get_drive_line(q)
    drive(d, square(8, 0.1), t)
    wait(d, tau * 2.0, bounds=(0, 1000))
    drive(d, square(8, 0.1), t)


def test_a_variable_wait_off_the_grid_fails_at_the_call():
    compiled = pulse.compile(variable_wait, machine=_machine(0.5))
    lowered = compiled.lower(Q[0], 6.0)
    assert [e.start for e in lowered.schedule if e.kind == "drive"] == [0, 20]
    with pytest.raises(ValueError, match="nearest lengths are 4 and 8"):
        compiled.lower(Q[0], 3.0)


# ── Exports in the controller's own cycles ───────────────────────────────────


def test_the_export_counts_the_controllers_own_cycles():
    lowered = pulse.compile(mixed, schedule="asap", machine=_machine(0.5)).lower(Q[0], Q[1])
    program = lowered.export_controller()
    assert program["clock_ghz"] == 0.5 and program["program_clock_ghz"] == 2.0
    assert "reference_clock_ghz" not in program
    timed = ("drive", "readout", "wait")
    played = [e for e in program["events"] if e["kind"] in timed]
    assert [(e["start_cycles"], e["duration_cycles"], e["end_cycles"]) for e in played] == [
        (e.start // 4, e.duration_cycles // 4, e.end // 4)
        for e in lowered.schedule
        if e.kind in timed
    ]
    params = played[0]["waveform"]["params"]
    assert params == {"duration": 10.0, "amplitude": 0.1, "sigma": 2.5}
    json.dumps(program)


def test_samples_at_a_rate_of_the_controllers_clock():
    lowered = pulse.compile(mixed, schedule="asap", machine=_machine(0.5)).lower(Q[0], Q[1])
    program = lowered.export_controller(sample_rate_ghz=1.0)
    first = next(e for e in program["events"] if e["kind"] == "drive")
    # Two samples per own cycle: the envelope at every other program cycle.
    samples = np.array(first["samples"]["re"])
    assert len(samples) == 2 * first["duration_cycles"] == 20
    node = {"shape": "gaussian", "params": {"duration": 40.0, "amplitude": 0.1, "sigma": 10.0}}
    np.testing.assert_allclose(samples, sample_waveform(node, 40).real[::2], rtol=1e-12)
    with pytest.raises(ValueError, match="whole multiple"):
        lowered.export_controller(sample_rate_ghz=0.75)


def test_the_emulation_export_counts_the_controllers_own_cycles():
    lowered = pulse.compile(mixed, schedule="asap", machine=_machine(0.5)).lower(Q[0], Q[1])
    data = lowered.export_emulation()
    assert data["clock_ghz"] == 0.5 and data["program_clock_ghz"] == 2.0
    starts = [e["start_cycles"] for e in data["events"]]
    assert starts == [e.start // 4 for e in lowered.schedule]


def test_every_waveform_time_is_scaled():
    from cudaq_pulse.controller import _PARAMETERS

    times = {"duration", "sigma", "beta", "risefall"}
    shapes = {s for s, names in _PARAMETERS.items() if "duration" in names}
    assert set(_clocks._TIMES_OF) == shapes
    for shape in shapes:
        assert _clocks._TIMES_OF[shape] == tuple(k for k in _PARAMETERS[shape] if k in times)


def test_every_time_in_the_exports_is_converted():
    from tests.classical import _schedule_matrix as matrix

    known = {
        *_clocks.EVENT_TIMES,
        *_clocks.EVENT_BOUNDS,
        *_clocks.EVENT_LOWER_BOUNDS,
        *_clocks.EMULATION_EVENT_TIMES,
        *_clocks.EPOCH_TIMES,
        *_clocks.EPOCH_LOWER_BOUNDS,
        *_clocks.CALL_TIMES,
    }
    found = set()

    def walk(data):
        if isinstance(data, dict):
            found.update(k for k in data if k.endswith("_cycles"))
            for key, value in data.items():
                if key != "limits":  # the controller's own, as given
                    walk(value)
        elif isinstance(data, list):
            for value in data:
                walk(value)

    machine = dataclasses.replace(matrix.MACHINE, controller=ControllerLimits(clock_ghz=1.0))
    for case in matrix.all_cases()[::7]:
        if matrix.expected_error(case) is not None:
            continue
        try:
            lowered = matrix.lower_case(case, Q, machine=machine)
        except (ControllerLimitError, ValueError):
            continue
        walk(lowered.export_controller())
        walk(lowered.export_emulation())
    assert "submit_cycles" in found and "min_shift_cycles" in found
    assert found <= known


def test_a_single_clock_export_is_unchanged():
    lowered = pulse.compile(mixed, machine=_machine(2.0)).lower(Q[0], Q[1])
    program = lowered.export_controller()
    assert program["clock_ghz"] == 2.0
    assert "program_clock_ghz" not in program and "reference_clock_ghz" not in program
    assert "line_grid" not in lowered.mlir()


def test_a_faster_controller_counts_more_cycles():
    lowered = pulse.compile(off_grid, machine=_machine(4.0)).lower(Q[0])
    program = lowered.export_controller()
    assert program["clock_ghz"] == 4.0 and program["program_clock_ghz"] == 2.0
    drive = next(e for e in program["events"] if e["kind"] == "drive")
    assert drive["duration_cycles"] == 82 and drive["waveform"]["params"]["duration"] == 82.0
    assert "line_grid" not in lowered.mlir()


def test_a_run_time_wait_has_its_bounds_in_the_controllers_cycles():
    from tests.test_controller import answered

    for clock_ghz, bounds in ((None, (8, 400)), (1.0, (4, 200)), (4.0, (16, 800))):
        machine = MachineModel() if clock_ghz is None else _machine(clock_ghz)
        program = pulse.compile(answered, machine=machine).lower(Q[0]).export_controller()
        wait = next(e for e in program["events"] if e["kind"] == "wait")
        assert (wait["min_cycles"], wait["max_cycles"]) == bounds


# ── The reference clock ──────────────────────────────────────────────────────


def test_a_reference_clock_divides_every_controller_clock_and_is_exported():
    machine = _machine(0.5, reference_clock_ghz=0.25)
    program = pulse.compile(mixed, machine=machine).lower(Q[0], Q[1]).export_controller()
    assert program["reference_clock_ghz"] == 0.25
    machine = MachineModel(controller=ControllerLimits(), reference_clock_ghz=0.5)
    program = pulse.compile(mixed, machine=machine).lower(Q[0], Q[1]).export_controller()
    assert program["reference_clock_ghz"] == 0.5 and "program_clock_ghz" not in program
    with pytest.raises(ValueError, match="not a whole multiple of the 0.3 GHz reference"):
        pulse.compile(mixed, machine=_machine(0.5, reference_clock_ghz=0.3)).lower(Q[0], Q[1])
    with pytest.raises(ValueError, match="must be positive"):
        MachineModel(reference_clock_ghz=0.0)
    assert MachineModel(reference_clock_ghz=0.5) != MachineModel()


def test_without_a_reference_each_controller_start_is_known_to_one_own_cycle():
    plan = _clocks.ClockPlan(
        2.0,
        (
            _clocks.ControllerClock("awg", 2.0, _clocks.Fraction(1)),
            _clocks.ControllerClock("readout", 0.5, _clocks.Fraction(1, 4)),
        ),
    )
    assert plan.start_uncertainty_ns() == {"awg": 0.5, "readout": 2.0}
    assert plan.budget_uncertainty_ns("awg", "readout") == 2.5
    assert plan.budget_uncertainty_ns("awg", "awg") == 0.0
    assert plan.start_uncertainty_ns({"readout": 1}) == {"awg": 0.5, "readout": 4.0}
    shared = _clocks.ClockPlan(2.0, plan.controllers, reference_clock_ghz=0.5)
    assert shared.budget_uncertainty_ns("awg", "readout") == 0.0
    assert shared.start_uncertainty_ns({"readout": 1}) == {"awg": 0.0, "readout": 2.0}


# ── Simulation ───────────────────────────────────────────────────────────────


@pulse.kernel
def product(q):
    d, t = get_drive_line(q)
    drive(d, wf_mul(gaussian(40, 0.2, 10.0), square(40, 0.5)), t)


def test_a_product_is_simulated_at_its_controllers_clock():
    fast = pulse.compile(product).lower(Q[0])
    slow = pulse.compile(product, machine=_machine(0.5)).lower(Q[0])
    llvm_fast, llvm_slow = fast.mlir("llvm"), slow.mlir("llvm")
    assert llvm_fast != llvm_slow
    a = pulse.simulate(fast).density_matrix
    b = pulse.simulate(slow).density_matrix
    # Ten samples instead of forty of one smooth envelope: close, not equal.
    assert not np.array_equal(a, b) and np.allclose(a, b, atol=5e-3)


def test_the_ladder_device_is_untouched():
    assert LADDER.clock_ghz == 2.0
