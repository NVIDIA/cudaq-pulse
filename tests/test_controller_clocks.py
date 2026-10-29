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


# ── One export per controller, in its own cycles ─────────────────────────────


def _timed(program):
    return [e for e in program["events"] if e["kind"] in ("drive", "readout", "wait")]


def test_each_controller_exports_its_part_of_the_program_in_its_own_cycles():
    qpu = _mixed()
    lowered = pulse.compile(mixed, schedule="asap", device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    with pytest.raises(ValueError) as caught:
        lowered.export_controller()
    assert str(caught.value) == (
        "the program plays lines of 2 controllers (controllers[0], controllers[1]); "
        "pass controller= (0 to 1) to export one controller's program"
    )
    for index, qudit, cycles in ((0, 0, 2), (1, 2, 4)):
        program = lowered.export_controller(controller=index)
        assert program["controller"] == index
        assert program["clock_ghz"] == (1.0, 0.5)[index] and program["program_clock_ghz"] == 2.0
        assert program["limits"] == qpu.control_system.controllers[index].as_dict()
        assert program["qudits"] == [qudit] and {line["qudit"] for line in program["lines"]} == {
            qudit
        }
        mine = [e for e in lowered.schedule if e.kind in ("drive", "readout", "wait")]
        mine = [e for e in mine if e.qubit == qudit]
        assert [(e["start_cycles"], e["duration_cycles"]) for e in _timed(program)] == [
            (e.start // cycles, e.duration_cycles // cycles) for e in mine
        ]
        # The sync lists only the lines this controller plays.
        [sync] = [e for e in program["events"] if e["kind"] == "sync"]
        ids = {line["line"] for line in program["lines"]}
        assert sync["lines"] and set(sync["lines"]) <= ids
        json.dumps(program)
    with pytest.raises(ValueError, match=r"controller=2, and the program's control system has 2"):
        lowered.export_controller(controller=2)
    with pytest.raises(TypeError, match="index of a controller"):
        lowered.export_controller(controller="1")  # type: ignore[arg-type]


def test_a_program_on_one_controller_exports_without_naming_it():
    qpu = _mixed(links=LINKS)
    lowered = pulse.compile(with_a_call, device=qpu).lower(qpu.qudits[2])
    program = lowered.export_controller()
    assert program["controller"] == 1 and program["clock_ghz"] == 0.5
    assert program == lowered.export_controller(controller=1)
    drive = next(e for e in program["events"] if e["kind"] == "drive")
    assert drive["duration_cycles"] == 2 and drive["waveform"]["params"]["duration"] == 2.0
    # The other controller takes no part: no lines, events or calls.
    other = lowered.export_controller(controller=0)
    assert other["lines"] == [] and other["events"] == [] and other["calls"] == []


def test_samples_at_a_rate_of_each_controllers_clock():
    qpu = _mixed()
    lowered = pulse.compile(mixed, schedule="asap", device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    program = lowered.export_controller(sample_rate_ghz=1.0, controller=1)
    first = next(e for e in program["events"] if e["kind"] == "drive")
    assert len(first["samples"]["re"]) == 2 * first["duration_cycles"]
    with pytest.raises(ValueError, match="whole multiple of the 0.5 GHz clock"):
        lowered.export_controller(sample_rate_ghz=0.75, controller=1)
    assert lowered.export_controller(sample_rate_ghz=1.0, controller=0)["events"]


@pulse.kernel
def read_then_correct(a, b):
    ra, rta = get_readout_line(a)
    db, tb = get_drive_line(b)
    m = readout(ra, square(100, 0.05), rta)
    amplitude = correction.submit([m]).result()
    drive(db, square(8, amplitude), tb)


def test_a_call_is_submitted_by_its_measuring_controller_and_answered_to_its_taker():
    qpu = _mixed(links=LINKS)
    lowered = pulse.compile(read_then_correct, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    first, second = (lowered.export_controller(controller=k) for k in (0, 1))
    assert [e["kind"] for e in first["events"]] == ["readout", "submit"]
    assert [e["kind"] for e in second["events"]] == ["answer", "drive"]
    answer, drive = second["events"]
    # Event ids are the program's, so the answer's value keeps its name.
    assert drive["waveform"]["runtime"]["amplitude"]["answer"] == answer["index"]
    for program in (first, second):
        assert [c["function"] for c in program["calls"]] == ["correction"]
        assert [f["name"] for f in program["manifest"]["functions"]] == ["correction"]
    starts = {e.kind: e.start for e in lowered.schedule}
    assert first["events"][1]["start_cycles"] == starts["submit"] // 2
    assert answer["start_cycles"] == starts["answer"] // 4


def test_each_controllers_emulation_export_is_its_part_in_its_cycles():
    qpu = _mixed(links=LINKS)
    lowered = pulse.compile(read_then_correct, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    with pytest.raises(ValueError, match="pass controller="):
        lowered.export_emulation()
    first, second = (lowered.export_emulation(controller=k) for k in (0, 1))
    assert first["controller"] == 0 and first["clock_ghz"] == 1.0
    assert second["controller"] == 1 and second["clock_ghz"] == 0.5
    assert [e["kind"] for e in first["events"]] == ["readout", "submit"]
    assert [e["kind"] for e in second["events"]] == ["answer", "drive"]
    starts = {e.kind: e.start for e in lowered.schedule}
    assert first["events"][0]["start_cycles"] == starts["readout"] // 2
    assert second["events"][1]["start_cycles"] == starts["drive"] // 4
    assert list(first["links"]) == list(second["links"]) == ["0"]


@pulse.kernel
def answer_wait(q):
    d, t = get_drive_line(q)
    drive(d, square(8, 0.1), t)
    wait(d, wait_length.submit(1.0).result(), bounds=(8, 400))
    drive(d, square(8, 0.1), t)


def test_a_run_time_wait_is_exported_in_its_controllers_cycles():
    qpu = _mixed(links=LINKS)
    lowered = pulse.compile(answer_wait, device=qpu).lower(qpu.qudits[2])
    wait = next(e for e in lowered.export_controller()["events"] if e["kind"] == "wait")
    assert (wait["min_cycles"], wait["max_cycles"]) == (2, 100)
    assert wait["length"]["op"] == "pulse.own_cycles" and wait["length"]["den"] == 4


def test_each_controller_exports_the_reference_clock():
    qpu = _mixed(reference_clock_ghz=0.25)
    lowered = pulse.compile(mixed, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    for index in (0, 1):
        assert lowered.export_controller(controller=index)["reference_clock_ghz"] == 0.25


def test_a_controller_faster_than_the_program_clock_counts_more_cycles():
    qpu = _device(4.0, 1.0, clock_ghz=2.0)
    lowered = pulse.compile(shaped_twice_plain, device=qpu).lower(qpu.qudits[0], qpu.qudits[2])
    program = lowered.export_controller(controller=0)
    assert program["clock_ghz"] == 4.0 and program["program_clock_ghz"] == 2.0
    drive = next(e for e in program["events"] if e["kind"] == "drive")
    assert drive["duration_cycles"] == 16 and drive["waveform"]["params"]["duration"] == 16.0
    drive = next(
        e for e in lowered.export_controller(controller=1)["events"] if e["kind"] == "drive"
    )
    assert drive["duration_cycles"] == 4


@pulse.kernel
def shaped_twice_plain(a, b):
    da, ta = get_drive_line(a)
    db, tb = get_drive_line(b)
    drive(da, square(8, 0.1), ta)
    drive(db, square(8, 0.1), tb)


def test_a_one_controller_device_exports_as_before():
    qpu = _device(0.5)
    lowered = pulse.compile(mixed, schedule="asap", device=qpu).lower(qpu.qudits[0], qpu.qudits[1])
    program = lowered.export_controller()
    assert "controller" not in program and program == lowered.export_controller(controller=0)
    assert "controller" not in lowered.export_emulation()
    with pytest.raises(ValueError, match="has one controller"):
        lowered.export_controller(controller=1)


# ── Budgets between controllers ──────────────────────────────────────────────
#
# Without a shared reference clock, each controller's start is known to one
# of its own cycles: 2 program cycles at 1 GHz and 4 at 0.5 GHz.


@pulse.kernel
def branch_across(a, b):
    ra, rta = get_readout_line(a)
    db, tb = get_drive_line(b)
    m = readout(ra, square(100, 0.05), rta)
    sync(ra, db)
    if m:
        drive(db, square(8, 0.1), tb)


def _forwarding(reference=None, skew=0):
    controllers = [
        pulse.Controller(
            drive_lines=[0, 1],
            readout_lines=[0, 1],
            clock_ghz=1.0,
            feedback_latency_cycles=3,
            peer_latency_cycles={1: 2},
        ),
        pulse.Controller(
            drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=0.5, clock_skew_cycles=skew
        ),
    ]
    system = pulse.ControlSystem(controllers=controllers, reference_clock_ghz=reference)
    return _device(system=system, clock_ghz=2.0)


def _gap(qpu, schedule="rcp"):
    lowered = pulse.compile(branch_across, schedule=schedule, device=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    return drive.start - readout.end


def test_an_outcome_sent_to_another_controller_is_budgeted_with_both_start_uncertainties():
    qpu = _forwarding()
    # Feedback 3 and peer 2 cycles at 1 GHz are 10 program cycles; the start
    # uncertainty adds 2 + 4.
    with pytest.raises(ControllerLimitError) as caught:
        _gap(qpu, schedule="asap")
    [message] = caught.value.violations
    assert message.startswith(
        "controllers[1]: the drive on line 1 at 100 branches on the readout of controllers[0] "
        "ending at 100, whose outcome reaches it 16 cycles later, at 116 (10 of feedback and "
        "peer latency, and 6 of the two controllers' start uncertainty, without a shared "
        "reference_clock_ghz); start it at 116 or later: schedule with rcp"
    )
    # rcp waits for it.
    assert _gap(qpu) == 16
    # A shared reference leaves the latency, and a skew adds its cycles, the
    # drive starting on its line's grid of 4.
    assert _gap(_forwarding(reference=0.5)) == 12
    assert _gap(_forwarding(reference=0.5, skew=1)) == 16
    assert _gap(_forwarding(skew=1)) == 20


def test_an_outcome_used_on_its_own_controller_needs_only_its_feedback_latency():
    qpu = _forwarding()
    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(feedback, schedule="asap", device=qpu).lower(qpu.qudits[0])
    [message] = caught.value.violations
    assert "needs 3 cycles of feedback latency" in message


def _call_across(reference=None, reach_both=True):
    a = pulse.Controller(drive_lines=[0, 1], readout_lines=[0, 1], clock_ghz=1.0)
    b = pulse.Controller(drive_lines=[2, 3], readout_lines=[2, 3], clock_ghz=0.5)
    link = Link(fixed_ns=10, controllers=None if reach_both else [a])
    system = pulse.ControlSystem(
        controllers=[a, b], links={"controller": link}, reference_clock_ghz=reference
    )
    return _device(system=system, clock_ghz=2.0)


def _due(qpu, b=2):
    lowered = pulse.compile(read_then_correct, device=qpu).lower(qpu.qudits[0], qpu.qudits[b])
    [call] = lowered.classical_report()
    return call.due_cycles - call.submit_cycles


def test_an_answer_on_a_link_between_controllers_is_budgeted_with_their_start_uncertainty():
    shared = _due(_call_across(reference=0.5))
    # 1 ns at 1 GHz and 2 ns at 0.5 GHz: 6 program cycles more.
    assert _due(_call_across()) == shared + 6
    # A link that reaches one controller joins none.
    assert _due(_call_across(reach_both=False), b=1) == _due(_call_across(reference=0.5), b=1)


@pulse.kernel
def branch_on_both(a, b):
    ra, rta = get_readout_line(a)
    da, ta = get_drive_line(a)
    db, tb = get_drive_line(b)
    m = readout(ra, square(100, 0.05), rta)
    sync(da, db)
    if m:
        drive(da, square(8, 0.1), ta)
        drive(db, square(8, 0.1), tb)


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
def test_rcp_waits_for_an_outcome_only_as_long_as_it_takes_to_reach_each_controller(schedule):
    qpu = _forwarding()
    lowered = pulse.compile(branch_on_both, schedule=schedule, device=qpu).lower(
        qpu.qudits[0], qpu.qudits[2]
    )
    readout = next(e for e in lowered.schedule if e.kind == "readout")
    mine, other = (e for e in lowered.schedule if e.kind == "drive")
    # On the reader: its feedback latency, 3 cycles at 1 GHz (alap_rcp then
    # ends it with the other); on the other controller: 16, as in
    # test_an_outcome_sent_to_another_controller_is_budgeted_with_both_start_uncertainties.
    assert mine.start - readout.end == (6 if schedule == "rcp" else 16)
    assert other.start - readout.end == 16
    # One controller's program waits its own feedback latency alone.
    lowered = pulse.compile(feedback, schedule=schedule, device=qpu).lower(qpu.qudits[0])
    readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
    assert drive.start - readout.end == 6


@pulse.kernel
def chosen_across(a, b):
    ra, rta = get_readout_line(a)
    db, tb = get_drive_line(b)
    m = readout(ra, square(100, 0.05), rta)
    if m:
        value = 0.3
    else:
        value = 0.1
    drive(db, square(8, value), tb)


def test_rcp_waits_for_an_outcome_a_pulse_computes_with_to_reach_its_controller(monkeypatch):
    monkeypatch.setattr(pulse.compiler, "_can_select_at_runtime", lambda target, kind: True)
    qpu = _forwarding()

    def gap(b):
        lowered = pulse.compile(chosen_across, schedule="rcp", device=qpu).lower(
            qpu.qudits[0], qpu.qudits[b]
        )
        readout, drive = (e for e in lowered.schedule if e.kind in ("readout", "drive"))
        return drive.start - readout.end

    assert gap(1) == 6
    assert gap(2) == 16
    # Scheduled too early, the check names it.
    early = pulse.compile(chosen_across, schedule="asap", device=qpu)
    with pytest.raises(ControllerLimitError, match="computes with the readout of controllers"):
        early.lower(qpu.qudits[0], qpu.qudits[2])
