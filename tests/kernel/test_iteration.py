# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``for`` loops over Python iterables unroll exactly like ``range(len())``."""

import itertools
import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse.kernel import _emitter
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q
from tests._ir import python_ir

AMPS = [0.1, 0.2, 0.3]
DURS = (16, 32, 48)


def _text(kernel, qudit_count=1):
    args = [Q[0:qudit_count]] if qudit_count > 1 else [Q[0]]
    compiled = kernel.lower(*args)
    return re.sub(r"module @\w+", "module", compiled.mlir())


def _same(a, b, qudit_count=1):
    ta, tb = _text(a, qudit_count), _text(b, qudit_count)
    assert ta == tb
    args = [Q[0:qudit_count]] if qudit_count > 1 else [Q[0]]
    assert python_ir(a, *args).op_count() == python_ir(b, *args).op_count()


def test_list_and_tuple():
    @pulse.kernel
    def over_list(q):
        d = get_drive_line(q)
        for a in AMPS:
            drive(d, square(16, a))
        for n in DURS:
            wait(d, n)

    @pulse.kernel
    def by_index(q):
        d = get_drive_line(q)
        for i in range(len(AMPS)):
            drive(d, square(16, AMPS[i]))
        for i in range(len(DURS)):
            wait(d, DURS[i])

    _same(over_list, by_index)


def test_enumerate_zip_reversed_and_dict_items():
    phases = {"a": 0.5, "b": -0.25}

    @pulse.kernel
    def builtins(q):
        d = get_drive_line(q)
        for i, a in enumerate(AMPS):
            drive(d, square(DURS[i], a))
        for a, n in zip(AMPS, DURS, strict=True):
            drive(d, square(n, a))
        for a in reversed(AMPS):
            drive(d, square(16, a))
        for _, phase in phases.items():
            shift_phase(d.tone, phase)

    @pulse.kernel
    def by_index(q):
        d = get_drive_line(q)
        for i in range(3):
            drive(d, square(DURS[i], AMPS[i]))
        for i in range(3):
            drive(d, square(DURS[i], AMPS[i]))
        for i in range(2, -1, -1):
            drive(d, square(16, AMPS[i]))
        shift_phase(d.tone, 0.5)
        shift_phase(d.tone, -0.25)

    _same(builtins, by_index)


def test_lines_and_tones_from_lists_stay_threaded():
    @pulse.kernel(unroll_threshold=0)
    def zipped(qudits):
        lines = []
        tones = []
        for q in qudits:
            d = get_drive_line(q)
            t = d.tone
            lines.append(d)
            tones.append(t)
        for _ in range(2):
            for d, t in zip(lines, tones, strict=True):
                drive(d, square(16, 0.1), tone=t)
        for d in lines:
            wait(d, 8)

    @pulse.kernel(unroll_threshold=0)
    def by_index(qudits):
        lines = []
        tones = []
        for i in range(len(qudits)):
            d = get_drive_line(qudits[i])
            lines.append(d)
            tones.append(d.tone)
        for _ in range(2):
            for i in range(len(lines)):
                drive(lines[i], square(16, 0.1), tone=tones[i])
        for i in range(len(lines)):
            wait(lines[i], 8)

    _same(zipped, by_index, qudit_count=3)


def test_iterating_a_qudit_vector():
    qudits = Q[0:3]
    assert list(qudits) == [qudits[i] for i in range(3)]

    @pulse.kernel
    def each(qs):
        for q in qs:
            d = get_drive_line(q)
            drive(d, square(16, 0.1))

    @pulse.kernel
    def by_index(qs):
        for i in range(len(qs)):
            d = get_drive_line(qs[i])
            drive(d, square(16, 0.1))

    _same(each, by_index, qudit_count=3)


def test_break_continue_and_else():
    @pulse.kernel
    def control(q):
        d = get_drive_line(q)
        for n in [8, 16, 0, 32, 64]:
            if n == 0:
                continue
            if n > 40:
                break
            wait(d, n)
        else:
            wait(d, 1000)
        for n in (4,):
            wait(d, n)
        else:
            wait(d, 2)

    @pulse.kernel
    def flat(q):
        d = get_drive_line(q)
        wait(d, 8)
        wait(d, 16)
        wait(d, 32)
        wait(d, 4)
        wait(d, 2)

    _same(control, flat)


def test_nested_inside_a_retained_range_loop():
    @pulse.kernel
    def nested(q):
        d = get_drive_line(q)
        for _ in range(64):
            for a in AMPS:
                drive(d, square(16, a))

    @pulse.kernel
    def by_index(q):
        d = get_drive_line(q)
        for _ in range(64):
            for i in range(len(AMPS)):
                drive(d, square(16, AMPS[i]))

    _same(nested, by_index)
    assert "scf.for" in _text(nested)


def test_range_loops_are_still_retained():
    @pulse.kernel
    def rolled(q):
        d = get_drive_line(q)
        for _ in range(512):
            drive(d, square(16, 0.1))

    ir = python_ir(rolled, Q[0])
    assert [op.kind for op in ir.ops].count("scf.for") == 1
    assert ir.op_count() == 5


def test_mutating_the_iterated_list_follows_python():
    @pulse.kernel
    def grows(q):
        d = get_drive_line(q)
        waits = [4, 8]
        for n in waits:
            if n < 8:
                waits.append(n * 4)
            wait(d, n)

    @pulse.kernel
    def flat(q):
        d = get_drive_line(q)
        wait(d, 4)
        wait(d, 8)
        wait(d, 16)

    _same(grows, flat)


def test_an_endless_iterator_stops_at_the_unroll_limit(monkeypatch):
    monkeypatch.setattr(_emitter, "MAX_UNROLLED_ITERATIONS", 50)

    @pulse.kernel
    def endless(q):
        d = get_drive_line(q)
        for n in itertools.count(1):
            wait(d, n)

    with pytest.raises(CompilationError, match="did not finish after 50 iterations"):
        endless(Q[0])


@pytest.mark.parametrize("bounds", [(2**64,), (-(2**64), 0), (0, 2**65, 2)])
@pytest.mark.parametrize("unroll", [None, 0])
def test_a_range_too_long_for_a_loop_names_the_limit_and_the_line(bounds, unroll):
    @pulse.kernel
    def too_long(q):
        d = get_drive_line(q)
        for _ in range(*bounds):
            wait(d, 4)

    compiled = pulse.compile(too_long, **({} if unroll is None else {"unroll_threshold": unroll}))
    with pytest.raises(CompilationError, match=r"runs more than 2\*\*63 - 1 times") as info:
        compiled.lower(Q[0])
    assert "for _ in range(*bounds)" in "\n".join(info.value.__notes__)


def test_the_longest_range_a_loop_can_run_counts_without_wrapping():
    @pulse.kernel
    def longest(q):
        d = get_drive_line(q)
        for _ in range(2**63 - 1):
            wait(d, 4)

    with pytest.raises(CompilationError, match=r"the program lasts more than 2\*\*60 cycles"):
        longest.lower(Q[0])


def _holding(n):
    return ControlSystem(controllers=[Controller(max_instructions=n)])


def _growing_waits(d, k):
    for n in range(k):
        wait(d, 4 * (n + 1))


def test_an_unrolled_range_past_the_instruction_memory_names_the_loop_and_its_size():
    @pulse.kernel
    def reads_index(q):
        d = get_drive_line(q)
        for n in range(10**15):
            wait(d, 4 + 4 * (n % 2))

    with pytest.raises(CompilationError) as info:
        pulse.compile(reads_index, control_system=_holding(4096)).lower(Q[0])
    assert str(info.value).startswith(
        "for loop over range(0, 1000000000000000) unrolls to about "
        "1,000,000,000,000,000 operations (1,000,000,000,000,000 iterations of about 1), "
        "more than the 4,096 the controller holds (Controller.max_instructions)"
    )
    assert "for n in range(10**15):" in "\n".join(info.value.__notes__)


def test_a_loop_over_an_iterable_past_the_instruction_memory_names_the_loop():
    @pulse.kernel
    def over_list(q):
        d = get_drive_line(q)
        for n in [4, 8] * 10**4:
            wait(d, n)
            wait(d, n)

    with pytest.raises(CompilationError) as info:
        pulse.compile(over_list, control_system=_holding(4096)).lower(Q[0])
    assert str(info.value).startswith("for loop over a Python iterable unrolls to about 40,0")
    assert "(20,000 iterations of about 2), more than the 4,096" in str(info.value)
    assert "for n in [4, 8] * 10**4:" in "\n".join(info.value.__notes__)


def test_the_instruction_limit_error_names_the_unrolled_loop_in_a_helper_and_its_caller():
    @pulse.kernel
    def calls(q):
        d = get_drive_line(q)
        for _ in range(3):
            _growing_waits(d, 10**9)

    with pytest.raises(CompilationError, match=r"range\(0, 1000000000\) unrolls") as info:
        pulse.compile(calls, control_system=_holding(100)).lower(Q[0])
    note = "\n".join(info.value.__notes__)
    assert note.index("_growing_waits(d, 10**9)") < note.index("for n in range(k):")


@pytest.mark.parametrize("carries_state", [False, True])
def test_a_program_of_exactly_max_instructions_compiles_and_one_more_does_not(carries_state):
    @pulse.kernel
    def eleven(q):
        d = get_drive_line(q)
        for n in range(10):
            wait(d, 4 * (n + 1))

    @pulse.kernel
    def eleven_counted(q):
        # The loop is traced rolled first and rolled back: it changes count.
        d = get_drive_line(q)
        count = 0
        for _ in range(10):
            count += 1
            wait(d, 4)

    kernel = eleven_counted if carries_state else eleven
    lowered = pulse.compile(kernel, control_system=_holding(11), unroll_threshold=0).lower(Q[0])
    assert lowered.metrics.op_count == 11
    with pytest.raises(CompilationError, match=r"iterations of about 1\), more than the 10 "):
        pulse.compile(kernel, control_system=_holding(10), unroll_threshold=0).lower(Q[0])


def test_a_rolled_loop_holds_its_body_once_against_the_instruction_memory():
    @pulse.kernel
    def rolled(q):
        d = get_drive_line(q)
        for _ in range(1000):
            wait(d, 4)

    ops = rolled.lower(Q[0]).metrics.op_count
    assert ops < 10
    assert pulse.compile(rolled, control_system=_holding(ops)).lower(Q[0]).metrics.op_count == ops


def test_a_program_past_the_instruction_memory_outside_any_loop_says_so():
    @pulse.kernel
    def straight(q):
        d = get_drive_line(q)
        wait(d, 4)
        wait(d, 8)
        wait(d, 12)

    with pytest.raises(CompilationError, match=r"^the program holds 3 operations, more than the 2"):
        pulse.compile(straight, control_system=_holding(2)).lower(Q[0])


def test_without_an_instruction_memory_unrolling_has_no_size_limit():
    @pulse.kernel
    def long(q):
        d = get_drive_line(q)
        for n in range(20_000):
            wait(d, 4 + 4 * (n % 2))

    assert long.lower(Q[0]).metrics.op_count == 20_001
    machine = ControlSystem(controllers=[Controller()])
    assert pulse.compile(long, control_system=machine).lower(Q[0]).metrics.op_count == 20_001


@pytest.mark.parametrize("wrong", [0, -1, True, 1.5, "8"])
def test_max_instructions_is_a_positive_int_or_none(wrong):
    with pytest.raises(pulse.InvalidValueError, match="max_instructions must be a positive int"):
        Controller(max_instructions=wrong)


def test_exports_list_max_instructions_only_when_it_is_set():
    assert "max_instructions" not in Controller().as_dict()
    assert Controller(max_instructions=64).as_dict()["max_instructions"] == 64


def test_iterating_a_pulse_value_is_an_error():
    @pulse.kernel
    def over_line(q):
        d = get_drive_line(q)
        for x in d:  # noqa: B007
            wait(d, 4)

    with pytest.raises(CompilationError, match="cannot iterate over a pulse drive_line value"):
        over_line(Q[0])


def test_range_errors_name_their_cause():
    @pulse.kernel
    def float_bound(q):
        d = get_drive_line(q)
        for _ in range(2.5):
            wait(d, 4)

    with pytest.raises(CompilationError, match=r"integers known at compile time; got \(float\)"):
        float_bound(Q[0])

    @pulse.kernel
    def symbolic_bound(q, n: float):
        d = get_drive_line(q)
        for _ in range(n):
            wait(d, 4)

    # A float loop bound is specialized on, and range() of a float fails as
    # it does in Python.
    with pytest.raises(CompilationError, match=r"integers known at compile time; got \(float\)"):
        symbolic_bound.lower(Q[0], 3.0)
    assert symbolic_bound.static_parameters == ("q", "n")


def test_while_loops_point_at_for_loops():
    @pulse.kernel
    def spin(q):
        d = get_drive_line(q)
        n = 0
        while n < 3:
            wait(d, 4)
            n += 1

    with pytest.raises(CompilationError, match="use a for loop"):
        spin(Q[0])
