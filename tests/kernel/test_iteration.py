# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""``for`` loops over Python iterables unroll exactly like ``range(len())``."""

import itertools
import re

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel import _emitter
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q
from tests._ir import python_ir

AMPS = [0.1, 0.2, 0.3]
DURS = (16, 32, 48)


def _text(kernel, n_qubits=1):
    args = [Q[0:n_qubits]] if n_qubits > 1 else [Q[0]]
    compiled = kernel.lower(*args)
    return re.sub(r"module @\w+", "module", compiled.mlir())


def _same(a, b, n_qubits=1):
    ta, tb = _text(a, n_qubits), _text(b, n_qubits)
    assert ta == tb
    args = [Q[0:n_qubits]] if n_qubits > 1 else [Q[0]]
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
        for a, n in zip(AMPS, DURS):
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
    def zipped(qubits):
        lines = []
        tones = []
        for q in qubits:
            d = get_drive_line(q)
            t = d.tone
            lines.append(d)
            tones.append(t)
        for _ in range(2):
            for d, t in zip(lines, tones):
                drive(d, square(16, 0.1), tone=t)
        for d in lines:
            wait(d, 8)

    @pulse.kernel(unroll_threshold=0)
    def by_index(qubits):
        lines = []
        tones = []
        for i in range(len(qubits)):
            d = get_drive_line(qubits[i])
            lines.append(d)
            tones.append(d.tone)
        for _ in range(2):
            for i in range(len(lines)):
                drive(lines[i], square(16, 0.1), tone=tones[i])
        for i in range(len(lines)):
            wait(lines[i], 8)

    _same(zipped, by_index, n_qubits=3)


def test_iterating_a_qubit_vector():
    qubits = Q[0:3]
    assert list(qubits) == [qubits[i] for i in range(3)]

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

    _same(each, by_index, n_qubits=3)


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


def test_iterating_a_pulse_value_is_an_error():
    @pulse.kernel
    def over_line(q):
        d = get_drive_line(q)
        for x in d:
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
