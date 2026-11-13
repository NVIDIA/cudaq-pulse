# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A program lasts at most its controller's longest program, and at most
2**60 cycles without one; a loop's length multiplies without overflow.
The module checks it as it is built, op after op, and a call checks it
again with the lengths its floats give its waits.
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError, Controller, ControlSystem

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
    # 5000 ns on the 2 GHz clock of Q's quantum system: 10,000 cycles.
    machine = ControlSystem(controllers=[Controller(max_program_duration=5_000)])
    assert pulse.compile(_waits(100, 100), control_system=machine).lower(Q[0]) is not None
    with pytest.raises(CompilationError, match=r"Controller\.max_program_duration"):
        pulse.compile(_waits(101, 100), control_system=machine).lower(Q[0])


def test_the_longest_program_may_be_a_time():
    assert Controller(max_program_duration=10 * pulse.us).max_program_duration == 10_000
    assert Controller(max_program_duration=1.5).max_program_duration == 1.5
    machine = ControlSystem(controllers=[Controller(max_program_duration=10 * pulse.us)])
    with pytest.raises(CompilationError, match=r"Controller\.max_program_duration"):
        pulse.compile(_waits(1000, 100), control_system=machine).lower(Q[0])


@pytest.mark.parametrize("wrong", [0, -1, True, math.nan, math.inf, "8"])
def test_max_program_duration_is_positive_or_none(wrong):
    with pytest.raises(pulse.InvalidValueError, match="max_program_duration"):
        Controller(max_program_duration=wrong)


def test_a_long_rolled_loop_past_the_controllers_longest_program_fails_fast():
    machine = ControlSystem(controllers=[Controller(max_program_duration=10**6)])
    with pytest.raises(CompilationError, match=r"Controller\.max_program_duration"):
        pulse.compile(_waits(10**15, 4), control_system=machine).lower(Q[0])


_LONGEST = ControlSystem(controllers=[Controller(max_program_duration=5_000)])
# Loops stay rolled, so a wait in one is one wait run several times.
_ROLLED = {"unroll_threshold": 0}


def _held(d, n, bounds):
    if bounds is None:
        wait(d, n)
    else:
        wait(d, n, bounds=bounds)


def _sweep(trips, bounds, written=None):
    @pulse.kernel
    def sweep(q, x: float, y: float):
        d = get_drive_line(q)
        a, b = (x, y) if written is None else written
        drive(d, square(100, 0.1))
        for _ in range(trips):
            _held(d, a, bounds if written is None else None)
        if trips == 1:
            _held(d, b, bounds if written is None else None)

    return sweep


@pytest.mark.parametrize("trips, bounds", [(3, (4, 10**6)), (3, None), (1, (4, 10**6)), (1, None)])
def test_a_call_fits_exactly_when_its_lengths_written_in_fit(trips, bounds):
    # One wait three times, folded into its bounds; or two once, summed.
    compiled = pulse.compile(_sweep(trips, bounds), control_system=_LONGEST, **_ROLLED)
    lowest = 0 if bounds is None else bounds[0]
    values = [4.0, 3299.4, 3299.5, 3300.49, 3300.5, 4950.0, 9895.5, 9896.5, 20_000.0]
    outcomes = set()
    for x in values:
        for y in values:
            written = (math.floor(x + 0.5), math.floor(y + 0.5))
            try:
                pulse.compile(
                    _sweep(trips, None, written), control_system=_LONGEST, **_ROLLED
                ).lower(Q[0], 0.0, 0.0)
                fits = True
            except CompilationError as error:
                assert "max_program_duration" in str(error)
                fits = False
            outcomes.add(fits)
            if fits:
                assert compiled.lower(Q[0], x, y) is not None, (x, y)
            else:
                with pytest.raises(ValueError, match=r"Controller\.max_program_duration"):
                    compiled.lower(Q[0], x, y)
    assert outcomes == {True, False}
    waits = compiled.lower(Q[0], float(lowest), float(lowest)).generic.waits
    # The most the program may last is its own test only with two waits.
    assert (10_000 in waits.ok.__code__.co_consts) == (trips == 1)


def test_waits_whose_bounds_keep_the_program_short_test_no_length():
    waits = pulse.compile(_sweep(3, None), **_ROLLED).lower(Q[0], 4.0, 4.0).generic.waits
    assert waits.length is not None and waits.length[3] == [3]
    assert 2**60 not in waits.ok.__code__.co_consts
