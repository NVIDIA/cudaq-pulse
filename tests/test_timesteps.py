# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timesteps: a controller that starts and times operations every
``cycles_per_timestep`` of its cycles puts its lines on whole timesteps."""

from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem, _clocks, _waits, controller
from cudaq_pulse.controller import ControllerLimitError, DriveLine, check

from tests._handles import Q

SCHEDULES = ("asap", "alap", "rcp", "alap_rcp")
TIMED = ("drive", "readout", "wait")


def _steps(cycles_per_timestep: int, **kwargs) -> ControlSystem:
    limits = Controller(cycles_per_timestep=cycles_per_timestep, **kwargs)
    return ControlSystem(controllers=[limits])


def _delayed_steps(cycles_per_timestep: int, **kwargs) -> ControlSystem:
    """A controller of timesteps of *cycles_per_timestep* playing qudits 0
    and 1, qudit 0's drive line delayed 3 ns."""
    box = Controller(
        cycles_per_timestep=cycles_per_timestep,
        drive_lines=[DriveLine(0, delay_ns=3.0), 1],
        readout_lines=[0, 1],
        **kwargs,
    )
    return ControlSystem(controllers=[box])


@pulse.kernel
def short(q):
    d = get_drive_line(q)
    drive(d, square(22, 0.1))


@pulse.kernel
def stepped(q, r):
    d = get_drive_line(q)
    e = get_drive_line(r)
    drive(d, square(pulse.round_duration(d, 22), 0.1))
    drive(e, square(pulse.round_duration(e, 22, "down"), 0.1))
    wait(d, pulse.round_duration(d, 3 * pulse.ns))
    sync(d, e)
    drive(e, square(8, 0.1))


def test_a_length_off_the_timestep_is_an_error_naming_the_line():
    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(short, control_system=_steps(4)).lower(Q[3])
    (message,) = caught.value.violations
    assert message.startswith(
        "line d3 moves in timesteps of 4 cycles; 22 cycles isn't a whole number of timesteps"
    )
    assert "pulse.round_duration(line, 22) gives 24, or 20 rounding down" in message


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_operations_start_and_last_whole_timesteps(schedule):
    lowered = pulse.compile(stepped, schedule=schedule, control_system=_steps(4)).lower(Q[0], Q[1])
    timed = [e for e in lowered.schedule if e.kind in TIMED]
    assert timed and all(e.start % 4 == 0 and e.duration_cycles % 4 == 0 for e in timed)
    drives = sorted(e.duration_cycles for e in lowered.schedule if e.kind == "drive")
    waits = [e.duration_cycles for e in lowered.schedule if e.kind == "wait"]
    assert drives == [8, 20, 24] and 8 in waits


def test_a_timestep_is_whole_program_cycles_of_whole_steps():
    assert _clocks.timestep(Fraction(1), 4) == 4
    # At half the program clock, 3 own cycles are 6 program cycles.
    assert _clocks.timestep(Fraction(1, 2), 3) == 6
    # At 3/4 of it, 2 own cycles are 8/3 program cycles: 8 of them are 3 steps.
    assert _clocks.timestep(Fraction(3, 4), 2) == 8
    # At twice it, 4 own cycles are 2 program cycles.
    assert _clocks.timestep(Fraction(2), 4) == 2
    assert _clocks.timestep(Fraction(1, 4), 1) == 4


def test_the_export_counts_whole_steps_of_the_controllers_own_cycles():
    limits = Controller(clock_ghz=1.0, cycles_per_timestep=2)
    machine = ControlSystem(controllers=[limits])
    lowered = pulse.compile(stepped, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    timed = [e for e in lowered.schedule if e.kind in TIMED]
    assert all(e.start % 4 == 0 and e.duration_cycles % 4 == 0 for e in timed)
    program = lowered.export_to_controller()
    assert program["clock_ghz"] == 1.0 and program["program_clock_ghz"] == 2.0
    events = [e for e in program["events"] if e["kind"] in TIMED]
    assert all(e["start_cycles"] % 2 == 0 and e["duration_cycles"] % 2 == 0 for e in events)
    assert program["limits"]["cycles_per_timestep"] == 2


def test_on_the_program_clock_the_export_keeps_its_clock():
    lowered = pulse.compile(stepped, schedule="asap", control_system=_steps(4)).lower(Q[0], Q[1])
    program = lowered.export_to_controller()
    assert program["clock_ghz"] == lowered.quantum_system.clock_ghz
    assert "program_clock_ghz" not in program
    starts = [e["start_cycles"] for e in program["events"] if e["kind"] in TIMED]
    assert starts and all(s % 4 == 0 for s in starts)


@pulse.classical(max_latency_ns=200)
def _pause() -> int:
    return 8


@pulse.kernel
def paced(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    t = _pause.submit().result()
    for _ in range(3):
        drive(d, square(16, 0.1))
        wait(d, t, bounds=(4, 64))
    sync(d, r)
    if readout(r, square(20, 0.05)):
        drive(d, square(8, 0.1))


@pulse.kernel
def rolled(q):
    d = get_drive_line(q)
    for _ in range(100):
        drive(d, gaussian(16, 0.1, 4.0))
        wait(d, 4)


@pytest.mark.parametrize(
    "limits",
    [Controller(cycles_per_timestep=2), Controller(sample_rate_ghz=1.0)],
)
@pytest.mark.parametrize("kernel", [paced, rolled])
def test_format_2_on_the_program_clock_in_timesteps_unrolls_to_format_1(kernel, limits):
    lowered = pulse.compile(kernel, control_system=ControlSystem(controllers=[limits])).lower(Q[0])
    flat = lowered.export_to_controller(version=1)
    structured = lowered.export_to_controller(version=2)
    assert structured["clock_ghz"] == flat["clock_ghz"] == lowered.quantum_system.clock_ghz
    assert controller.unroll(structured)["events"] == flat["events"]
    if kernel is rolled:
        assert [e["kind"] for e in structured["events"]] == ["loop"]
    else:
        # The loop of waits for an answer stays rolled; each of its three
        # waits converts its run-time length in timesteps.
        assert [e["kind"] for e in structured["events"]].count("loop") == 1
        unrolled = controller.unroll(structured)["events"]
        waits = [e for e in unrolled if e["kind"] == "wait" and "length" in e]
        assert len(waits) == 3 and all(w["length"]["timestep"] == 2 for w in waits)


def test_format_2_refuses_a_controller_on_its_own_clock():
    machine = ControlSystem(controllers=[Controller(clock_ghz=1.0)])
    lowered = pulse.compile(paced, control_system=machine).lower(Q[0])
    assert lowered.export_to_controller(version=1)["clock_ghz"] == 1.0
    with pytest.raises(pulse.UnsupportedError, match="export version 1 for this machine"):
        lowered.export_to_controller(version=2)


def test_one_cycle_per_timestep_changes_nothing():
    plain = pulse.compile(stepped).lower(Q[0], Q[1])
    one = pulse.compile(stepped, control_system=_steps(1)).lower(Q[0], Q[1])
    assert one.mlir() == plain.mlir()
    assert _clocks.plan(Controller(), 2.0) is None
    assert _clocks.plan(Controller(cycles_per_timestep=4), 2.0) is not None


def test_a_line_delay_moves_a_line_by_whole_timesteps():
    # 3 ns is 6 cycles of the 2 GHz clock: one timestep of 4, and 1 ns of fine delay.
    machine = _delayed_steps(4, max_fine_delay_ns=1.5)
    lowered = pulse.compile(stepped, schedule="asap", control_system=machine).lower(Q[0], Q[1])
    first = [e.start for e in lowered.schedule if e.kind == "drive" and e.qudit == 0]
    assert first == [4]
    lines = {(x["role"], x["qudit"]): x for x in lowered.export_to_controller()["lines"]}
    assert lines[("drive", 0)]["fine_delay_ns"] == pytest.approx(1.0)
    # Without fine delay, the 1 ns left over is 2 cycles past a timestep.
    exact = _delayed_steps(4)
    with pytest.raises(
        pulse.CompilationError, match="its 2 ns timesteps, and the controller has no"
    ):
        pulse.compile(stepped, schedule="asap", control_system=exact).lower(Q[0], Q[1])


@pytest.mark.parametrize("bad", [0, -2, 2.0, True])
def test_cycles_per_timestep_is_a_positive_int(bad):
    with pytest.raises(ValueError, match="cycles_per_timestep"):
        Controller(cycles_per_timestep=bad)


# ── Sample rates ─────────────────────────────────────────────────────────────

# 2.4 GHz samples on a 1 GHz controller of the 2 GHz program clock: 12
# samples every 5 of its cycles, 10 program cycles.
AWG = Controller(clock_ghz=1.0, sample_rate_ghz=2.4)


def ramp(t):
    return t / 100


@pulse.kernel
def shaped(q):
    d = get_drive_line(q)
    drive(d, custom(pulse.round_duration(d, 18), ramp))


def test_a_fractional_sample_rate_puts_lines_on_whole_samples():
    assert _clocks.plan(AWG, 2.0).controllers[0].grid == 10
    assert _clocks.sample_rate(AWG, 2.0) == (6, 5)
    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(short, control_system=ControlSystem(controllers=[AWG])).lower(Q[3])
    (message,) = caught.value.violations
    assert message.startswith(
        "line d3 moves in timesteps of 10 cycles; 22 cycles isn't a whole number of timesteps"
    )
    assert "pulse.round_duration(line, 22) gives 30, or 20 rounding down" in message


def test_custom_is_sampled_at_a_fractional_rate():
    lowered = pulse.compile(shaped, control_system=ControlSystem(controllers=[AWG])).lower(Q[0])
    (drive,) = [e for e in lowered.export_to_controller()["events"] if e["kind"] == "drive"]
    # 20 program cycles are 10 of the controller's, 24 samples.
    assert drive["duration_cycles"] == 10
    assert len(drive["waveform"]["samples"]) == 24


def test_waveform_memory_counts_samples_at_the_rate():
    lowered = pulse.compile(shaped, control_system=ControlSystem(controllers=[AWG])).lower(Q[0])
    limits = dataclasses.replace(AWG, waveform_memory_samples=20)
    (found,) = check(lowered, limits)
    assert "its waveforms take 24 samples" in found


@pulse.kernel
def waits_for(q, tau: float):
    d = get_drive_line(q)
    drive(d, square(8, 0.1))
    wait(d, tau, bounds=(0, 400))


def test_only_a_timestep_grid_checks_variable_waits_against_it():
    one = pulse.compile(waits_for).lower(Q[0], 64.0)
    assert type(one.waits) is _waits._WaitLengths and one.waits.grids == ()
    grid = pulse.compile(waits_for, control_system=_steps(4)).lower(Q[0], 64.0)
    assert type(grid.waits) is _waits._GridWaitLengths and grid.waits.grids == (4,)
    with pytest.raises(ValueError, match="moves in timesteps of 4 cycles"):
        pulse.compile(waits_for, control_system=_steps(4)).lower(Q[0], 66.0)


def test_a_whole_sample_rate_on_the_clock_changes_nothing():
    assert _clocks.plan(Controller(sample_rate_ghz=4.0), 2.0) is None
    plain = pulse.compile(stepped).lower(Q[0], Q[1])
    two = pulse.compile(
        stepped, control_system=ControlSystem(controllers=[Controller(sample_rate_ghz=4.0)])
    )
    assert two.lower(Q[0], Q[1]).mlir() == plain.mlir()


@pytest.mark.parametrize("bad", [0, -1.0, True, "2.4", float("nan")])
def test_sample_rate_ghz_is_a_positive_number(bad):
    with pytest.raises(ValueError, match="sample_rate_ghz"):
        Controller(sample_rate_ghz=bad)


def test_a_sample_rate_without_a_whole_cycle_ratio_is_an_error():
    with pytest.raises(ValueError, match="sample rate has no whole-cycle ratio"):
        _clocks.plan(Controller(sample_rate_ghz=math.pi), 2.0)
