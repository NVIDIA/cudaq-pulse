# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Timesteps: a controller that starts and times operations every
``cycles_per_timestep`` of its cycles puts its lines on whole timesteps."""

from __future__ import annotations

from fractions import Fraction

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import _clocks
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import ControllerLimitError, ControllerLimits

from tests._handles import Q

SCHEDULES = ("asap", "alap", "rcp", "alap_rcp")
TIMED = ("drive", "readout", "wait")


def _steps(cycles_per_timestep: int, **kwargs) -> MachineModel:
    limits = ControllerLimits(cycles_per_timestep=cycles_per_timestep, **kwargs)
    return MachineModel(controller=limits)


@pulse.kernel
def short(q):
    d, t = get_drive_line(q)
    drive(d, square(22, 0.1), t)


@pulse.kernel
def stepped(q, r):
    d, t = get_drive_line(q)
    e, u = get_drive_line(r)
    drive(d, square(pulse.round_duration(d, 22), 0.1), t)
    drive(e, square(pulse.round_duration(e, 22, "down"), 0.1), u)
    wait(d, pulse.round_duration(d, 3 * pulse.ns))
    sync(d, e)
    drive(e, square(8, 0.1), u)


def test_a_length_off_the_timestep_is_an_error_naming_the_line():
    with pytest.raises(ControllerLimitError) as caught:
        pulse.compile(short, machine=_steps(4)).lower(Q[3])
    (message,) = caught.value.violations
    assert message.startswith(
        "line d3 moves in timesteps of 4 cycles; 22 cycles isn't a whole number of timesteps"
    )
    assert "pulse.round_duration(line, 22) gives 24, or 20 rounding down" in message


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_operations_start_and_last_whole_timesteps(schedule):
    lowered = pulse.compile(stepped, schedule=schedule, machine=_steps(4)).lower(Q[0], Q[1])
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
    limits = ControllerLimits(clock_ghz=1.0, cycles_per_timestep=2)
    machine = MachineModel(controller=limits)
    lowered = pulse.compile(stepped, schedule="asap", machine=machine).lower(Q[0], Q[1])
    timed = [e for e in lowered.schedule if e.kind in TIMED]
    assert all(e.start % 4 == 0 and e.duration_cycles % 4 == 0 for e in timed)
    program = lowered.export_controller()
    assert program["clock_ghz"] == 1.0 and program["program_clock_ghz"] == 2.0
    events = [e for e in program["events"] if e["kind"] in TIMED]
    assert all(e["start_cycles"] % 2 == 0 and e["duration_cycles"] % 2 == 0 for e in events)
    assert program["limits"]["cycles_per_timestep"] == 2


def test_on_the_program_clock_the_export_keeps_its_clock():
    lowered = pulse.compile(stepped, schedule="asap", machine=_steps(4)).lower(Q[0], Q[1])
    program = lowered.export_controller()
    assert program["clock_ghz"] == lowered.target.clock_ghz
    assert "program_clock_ghz" not in program
    starts = [e["start_cycles"] for e in program["events"] if e["kind"] in TIMED]
    assert starts and all(s % 4 == 0 for s in starts)


def test_one_cycle_per_timestep_changes_nothing():
    plain = pulse.compile(stepped).lower(Q[0], Q[1])
    one = pulse.compile(stepped, machine=_steps(1)).lower(Q[0], Q[1])
    assert one.mlir() == plain.mlir()
    assert _clocks.plan(ControllerLimits(), 2.0) is None
    assert _clocks.plan(ControllerLimits(cycles_per_timestep=4), 2.0) is not None


def test_a_line_delay_moves_a_line_by_whole_timesteps():
    # 3 ns is 6 cycles of the 2 GHz clock: one timestep of 4, and 1 ns of fine delay.
    machine = _steps(4, max_fine_delay_ns=1.5)
    machine = MachineModel(controller=machine.controller, line_delays_ns={("drive", 0): 3.0})
    lowered = pulse.compile(stepped, schedule="asap", machine=machine).lower(Q[0], Q[1])
    first = [e.start for e in lowered.schedule if e.kind == "drive" and e.qubit == 0]
    assert first == [4]
    lines = {(x["role"], x["qudit"]): x for x in lowered.export_controller()["lines"]}
    assert lines[("drive", 0)]["fine_delay_ns"] == pytest.approx(1.0)
    # Without fine delay, the 1 ns left over is 2 cycles past a timestep.
    exact = MachineModel(controller=ControllerLimits(cycles_per_timestep=4))
    exact = MachineModel(controller=exact.controller, line_delays_ns={("drive", 0): 3.0})
    with pytest.raises(
        pulse.CompilationError, match="its 2 ns timesteps, and the controller has no"
    ):
        pulse.compile(stepped, schedule="asap", machine=exact).lower(Q[0], Q[1])


@pytest.mark.parametrize("bad", [0, -2, 2.0, True])
def test_cycles_per_timestep_is_a_positive_int(bad):
    with pytest.raises(ValueError, match="cycles_per_timestep"):
        ControllerLimits(cycles_per_timestep=bad)
