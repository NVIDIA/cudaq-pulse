# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A program lasts at most its controller's longest program, and at most
2**60 cycles without one; a loop's length multiplies without overflow.
The module checks it as it is built, op after op, and a call checks it
again with the lengths its floats give its waits.
"""

from __future__ import annotations

import dis
import math
import pickle

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError, Controller, ControlSystem
from cudaq_pulse.controller import DriveLine

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


def _delayed(cycles):
    # Q's quantum system runs a 2 GHz clock: a cycle is 0.5 ns.
    return ControlSystem(controllers=[Controller(drive_lines=[DriveLine(0, delay_ns=cycles / 2)])])


@pytest.mark.parametrize("cycles", [2**61, 10**19, 2**64, 10**30])
def test_a_line_delay_longer_than_a_program_is_an_error(cycles):
    with pytest.raises(CompilationError, match=r"\(its delay_ns\), more than a program lasts"):
        pulse.compile(_waits(1, 4), control_system=_delayed(cycles)).lower(Q[0])


def test_a_rolled_loop_its_line_delay_takes_past_2_60_cycles_is_an_error():
    # 64 trips of 2**53 cycles: 2**59 cycles, which a delay of 2**58 keeps short enough.
    loop = pulse.compile(_waits(64, 2**53), control_system=_delayed(2**58), **_ROLLED)
    assert loop.lower(Q[0]).duration / pulse.ns == pytest.approx((2**59 + 2**58) / 2)
    longer = pulse.compile(_waits(64, 2**53), control_system=_delayed(2**59 + 2**20), **_ROLLED)
    with pytest.raises(CompilationError, match=PAST_THE_BACKSTOP + " with its lines delayed"):
        longer.lower(Q[0])


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


# Unrolled, each trip's wait is a wait of its own, as each of a surface-code
# round's thousands of waits is: past about ten thousand of them, a check
# whose expression nested a level deeper per wait no longer compiled.
_MANY = 20_000


def _many_waits(shape):
    @pulse.kernel
    def many(q, tau: float):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        sync(d, r)
        if shape == "in an if":
            if m:
                for _ in range(_MANY):
                    wait(d, tau)
                    drive(d, square(8, 0.1))
        else:
            for _ in range(_MANY):
                wait(d, tau)
                drive(d, square(8, 0.1))

    return many


@pytest.mark.slow
@pytest.mark.parametrize("shape", ["in sequence", "in an if"])
def test_a_call_checks_twenty_thousand_waits_of_one_float(shape):
    # 1 ms on the 2 GHz clock of Q's quantum system: 2,000,000 cycles.
    machine = ControlSystem(controllers=[Controller(max_program_duration=10**6)])
    # Its loop unrolled: 20,000 trips of three calls (wait, drive, square).
    compiled = pulse.compile(_many_waits(shape), control_system=machine, unroll_threshold=3 * _MANY)
    waits = compiled.lower(Q[0], 4.0).generic.waits
    assert len(waits.waits) == len(waits.length[3]) == _MANY
    assert len(waits.length) == (5 if shape == "in an if" else 4)
    # With each wait k cycles long the program lasts its cycles + 20,000 k.
    cycles, most = waits.length[:2]
    k = (most - cycles) // _MANY
    again = pickle.loads(pickle.dumps(waits))
    assert again.ok.__code__.co_code == waits.ok.__code__.co_code
    for checked in (waits, again):
        checked.check([k + 0.49])
        with pytest.raises(ValueError, match=r"Controller\.max_program_duration"):
            checked.check([k + 0.5])
    # The kernel takes two lines, so op after op it may last twice the
    # longest program; its schedule may not, and decides.
    assert waits.length[1:3] == (4_000_000, 2_000_000)
    fits = (2_000_000 - cycles) // _MANY
    assert compiled.lower(Q[0], fits + 0.49) is not None
    with pytest.raises(ValueError, match=r"lasts \d+ cycles, more than controllers\[0\]'s"):
        compiled.lower(Q[0], fits + 1.0)
    with pytest.raises(ValueError, match=r"Controller\.max_program_duration"):
        compiled.lower(Q[0], k + 0.5)
    # One length for the waits of one float, and one test of their bounds.
    assert len({id(w.length) for w in waits.waits}) == len({id(w.length) for w in again.waits}) == 1
    compares = [op for op in dis.get_instructions(waits.ok) if op.opname == "COMPARE_OP"]
    assert len(compares) < 10


# ── The scheduled program ────────────────────────────────────────────────────


@pulse.kernel
def drive_in_if(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    bit = readout(r, square(400, 0.05))
    if bit:
        drive(d, square(40, 0.3))


@pulse.kernel
def repeat_until_success(q):
    d = get_drive_line(q)
    r = get_readout_line(q)
    m = readout(r, square(400, 0.05))
    for _ in range(100):
        if m:
            drive(d, square(360, 0.5))
            sync(d, r)
            m = readout(r, square(400, 0.05))


def _latency_system(limit_ns):
    controller = Controller(feedback_latency_cycles=400, max_program_duration=limit_ns)
    return ControlSystem(controllers=[controller])


@pytest.mark.parametrize("schedule", ["rcp", "alap_rcp"])
@pytest.mark.parametrize(
    ("kernel", "limit_ns", "length"),
    [(drive_in_if, 300, 840), (repeat_until_success, 40_000, 116_400)],
)
def test_a_schedule_that_waits_for_feedback_past_the_longest_program_is_an_error(
    kernel, limit_ns, length, schedule
):
    # Played op after op the programs fit (440 and 76,400 cycles of 600 and
    # 80,000); each waits 400 cycles after every readout it branches on.
    # Before, the schedule was not checked and both compiled.
    machine = _latency_system(limit_ns)
    with pytest.raises(CompilationError) as caught:
        pulse.compile(kernel, schedule=schedule, control_system=machine).lower(Q[0])
    assert str(caught.value) == (
        f"kernel '{kernel.__name__}' lasts {length} cycles, more than controllers[0]'s "
        f"max_program_duration of {limit_ns} ns ({2 * limit_ns} cycles), waiting 400 cycles "
        "of feedback latency after the readouts its operations depend on; let its operations "
        "overlap, shorten them or raise max_program_duration"
    )
    # Long enough, it compiles.
    longer = _latency_system(length / 2)
    lowered = pulse.compile(kernel, schedule=schedule, control_system=longer).lower(Q[0])
    assert max(e.start_cycles + e.duration_cycles for e in lowered.schedule) == length


def test_each_controllers_longest_program_bounds_the_lines_it_plays():
    # Nothing checks several controllers as the module is built.
    controllers = [
        Controller(drive_lines=[0], readout_lines=[0], max_program_duration=100),
        Controller(drive_lines=[1], readout_lines=[1]),
    ]
    machine = ControlSystem(controllers=controllers)
    waits = _waits(3, 100)
    assert pulse.compile(waits, control_system=machine).lower(Q[1]) is not None
    with pytest.raises(CompilationError) as caught:
        pulse.compile(waits, control_system=machine).lower(Q[0])
    assert str(caught.value) == (
        "kernel 'waits' lasts 300 cycles, more than controllers[0]'s max_program_duration "
        "of 100 ns (200 cycles); let its operations overlap, shorten them or raise "
        "max_program_duration"
    )


def _together(lines, cycles):
    @pulse.kernel
    def together(a, b, c):
        for q in (a, b, c)[:lines]:
            drive(get_drive_line(q), square(cycles, 0.1))

    return together


def test_operations_that_play_together_count_once():
    # 500 ns on the 2 GHz clock of Q's quantum system: 1000 cycles.
    machine = ControlSystem(controllers=[Controller(max_program_duration=500)])
    lowered = pulse.compile(_together(3, 400), control_system=machine).lower(Q[0], Q[1], Q[2])
    assert max(e.start_cycles + e.duration_cycles for e in lowered.schedule) == 400
    # Each line alone may last as long as the program.
    assert pulse.compile(_together(3, 1000), control_system=machine).lower(Q[0], Q[1], Q[2])

    @pulse.kernel
    def in_two_parts(a, b, c):
        lines = [get_drive_line(q) for q in (a, b, c)]
        for line in lines:
            drive(line, square(600, 0.1))
        sync(*lines)
        drive(lines[0], square(404, 0.1))

    # Op after op it sums to 2204 cycles, which three lines may; scheduled
    # it lasts 1004, which the controller does not run.
    with pytest.raises(CompilationError) as caught:
        pulse.compile(in_two_parts, control_system=machine).lower(Q[0], Q[1], Q[2])
    assert str(caught.value) == (
        "kernel 'in_two_parts' lasts 1004 cycles, more than controllers[0]'s "
        "max_program_duration of 500 ns (1000 cycles); let its operations overlap, shorten "
        "them or raise max_program_duration"
    )


def test_one_line_is_checked_as_it_is_built():
    machine = ControlSystem(controllers=[Controller(max_program_duration=500)])
    assert pulse.compile(_together(1, 1000), control_system=machine).lower(Q[0], Q[1], Q[2])
    with pytest.raises(CompilationError, match="the program lasts more than 1000 cycles"):
        pulse.compile(_together(1, 1004), control_system=machine).lower(Q[0], Q[1], Q[2])


def test_the_sum_stops_a_kernel_no_schedule_can_save():
    @pulse.kernel
    def runaway(a, b):
        d = get_drive_line(a)
        e = get_drive_line(b)
        for _ in range(10**15):
            drive(d, square(100, 0.1))
            drive(e, square(100, 0.1))

    machine = ControlSystem(controllers=[Controller(max_program_duration=10**6)])
    with pytest.raises(CompilationError, match="the program lasts more than 2000000 cycles"):
        pulse.compile(runaway, control_system=machine).lower(Q[0], Q[1])

    @pulse.kernel
    def in_turn(a, b, trips: int):
        d = get_drive_line(a)
        e = get_drive_line(b)
        for _ in range(trips):
            drive(d, square(100, 0.1))
            sync(d, e)
            drive(e, square(100, 0.1))
            sync(d, e)

    # Two lines that take turns: 10,000 trips sum to the longest program
    # and last it; one more trip is past it as scheduled, and twice as many
    # and one are past what any schedule of two lines lasts.
    compiled = pulse.compile(in_turn, control_system=machine, unroll_threshold=0)
    assert compiled.lower(Q[0], Q[1], 10_000) is not None
    with pytest.raises(CompilationError, match="lasts 2000200 cycles, more than controllers"):
        compiled.lower(Q[0], Q[1], 10_001)
    with pytest.raises(CompilationError, match="the program lasts more than 2000000 cycles"):
        compiled.lower(Q[0], Q[1], 20_001)


def test_lines_taken_in_a_called_kernel_count():
    @pulse.kernel
    def one(q):
        drive(get_drive_line(q), square(900, 0.1))

    @pulse.kernel
    def all_three(a, b, c):
        one(a)
        one(b)
        one(c)

    # 2700 cycles op after op on three lines, 900 as scheduled, under 1000.
    machine = ControlSystem(controllers=[Controller(max_program_duration=500)])
    lowered = pulse.compile(all_three, control_system=machine).lower(Q[0], Q[1], Q[2])
    assert max(e.start_cycles + e.duration_cycles for e in lowered.schedule) == 900


def test_without_a_limit_no_line_is_counted():
    # Only a builder given a limit wraps its buffer to count its lines.
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    builder = PackedIRBuilder()
    assert "get_buffer" not in vars(builder)
    builder.limit_cycles(1000)
    assert "get_buffer" in vars(builder)
