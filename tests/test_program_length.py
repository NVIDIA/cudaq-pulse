# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A program lasts at most its controller's longest program, and at most
2**60 cycles without one; a loop's length multiplies without overflow.
The module checks it as it is built, op after op, and a call checks it
again with the lengths its floats give its waits.
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import ControllerLimits

from tests._handles import Q

PAST_THE_BACKSTOP = r"the program lasts more than 2\*\*60 cycles"


def _waits(trips, cycles):
    @pulse.kernel
    def waits(q):
        d = get_drive_line(q)
        for _ in range(trips):
            wait(d, cycles)

    return waits


def test_a_program_of_2_60_cycles_compiles():
    assert _waits(128, 2**53).lower(Q[0]) is not None


@pytest.mark.parametrize("trips", [129, 300, 2000])
def test_a_program_past_2_60_cycles_is_an_error_at_its_loop(trips):
    with pytest.raises(CompilationError, match=PAST_THE_BACKSTOP) as info:
        _waits(trips, 2**53).lower(Q[0])
    assert "for _ in range(trips):" in "\n".join(info.value.__notes__)


def test_a_loop_length_past_2_63_does_not_wrap():
    with pytest.raises(CompilationError, match=PAST_THE_BACKSTOP):
        _waits(2**62, 4).lower(Q[0])

    @pulse.kernel
    def nested(q):
        d = get_drive_line(q)
        for _ in range(2**31):
            for _ in range(2**31):
                wait(d, 4)

    with pytest.raises(CompilationError, match=PAST_THE_BACKSTOP):
        nested.lower(Q[0])


def test_waits_of_run_time_length_are_checked_when_bound():
    @pulse.kernel
    def variable(q, n: float):
        d = get_drive_line(q)
        for _ in range(2**12):
            wait(d, n, bounds=(0, 2**53))

    compiled = pulse.compile(variable)
    assert compiled.lower(Q[0], 4.0) is not None
    with pytest.raises(ValueError, match=PAST_THE_BACKSTOP):
        compiled.lower(Q[0], float(2**52))


def test_a_program_past_the_controllers_longest_program_names_the_limit():
    # 5000 ns on the 2 GHz clock of Q's device: 10,000 cycles.
    machine = MachineModel(controller=ControllerLimits(max_program_duration=5_000))
    assert pulse.compile(_waits(100, 100), machine=machine).lower(Q[0]) is not None
    with pytest.raises(CompilationError, match=r"ControllerLimits\.max_program_duration"):
        pulse.compile(_waits(101, 100), machine=machine).lower(Q[0])


def test_the_longest_program_may_be_a_time():
    assert ControllerLimits(max_program_duration=10 * pulse.us).max_program_duration == 10_000
    assert ControllerLimits(max_program_duration=1.5).max_program_duration == 1.5
    machine = MachineModel(controller=ControllerLimits(max_program_duration=10 * pulse.us))
    with pytest.raises(CompilationError, match=r"ControllerLimits\.max_program_duration"):
        pulse.compile(_waits(1000, 100), machine=machine).lower(Q[0])


@pytest.mark.parametrize("wrong", [0, -1, True, math.nan, math.inf, "8"])
def test_max_program_duration_is_positive_or_none(wrong):
    with pytest.raises(pulse.InvalidValueError, match="max_program_duration"):
        ControllerLimits(max_program_duration=wrong)


def test_a_long_rolled_loop_past_the_controllers_longest_program_fails_fast():
    machine = MachineModel(controller=ControllerLimits(max_program_duration=10**6))
    with pytest.raises(CompilationError, match=r"ControllerLimits\.max_program_duration"):
        pulse.compile(_waits(10**15, 4), machine=machine).lower(Q[0])
