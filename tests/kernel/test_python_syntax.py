# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Python syntax inside kernels: what traces, and how the rest is rejected."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError

from tests._handles import Q
from tests._ir import python_ir


def _kinds(kernel, *args) -> list[str]:
    return [op.kind for op in python_ir(kernel, *args).ops]


def _compiles(kernel, *args) -> None:
    """Lowers through the MLIR verifier, which checks linear use of lines."""
    pulse.compile(kernel).lower(*args)


# ── Star-argument and ** calls ───────────────────────────────────────


def test_sync_star_lines_from_the_friction_log():
    @pulse.kernel
    def k(qubits):
        lines = []
        for q in qubits:
            d, t = get_drive_line(q)
            lines.append(d)
        sync(*lines)
        for d in lines:
            wait(d, 8)

    ops = python_ir(k, Q[0:3]).ops
    (sync_op,) = [op for op in ops if op.kind == "pulse.sync"]
    assert len(sync_op.operands) == 3
    # Each wait consumes the synced line, not the one sync consumed.
    waits = [op for op in ops if op.kind == "pulse.wait"]
    assert [w.operands[0] for w in waits] == list(sync_op.results)
    _compiles(k, Q[0:3])


def test_star_arguments_mix_with_positional_and_keyword_arguments():
    def helper(d, t, n, amp=0.1):
        drive(d, square(n, amp), t)

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        args = (d, t)
        opts = {"amp": 0.3}
        helper(*args, 16, **opts)
        helper(d, t, *[8], **{"amp": 0.2})
        drive(d, gaussian(*(40, 0.1), **{"sigma": 10.0}), t)

    ops = python_ir(k, Q[0]).ops
    squares = [
        (op.attrs["duration"], op.attrs["amplitude"]) for op in ops if op.kind == "pulse.square"
    ]
    assert squares == [(16, 0.3), (8, 0.2)]
    assert [op.attrs["sigma"] for op in ops if op.kind == "pulse.gaussian"] == [10.0]
    _compiles(k, Q[0])


def test_star_calls_to_python_functions_run_at_trace_time():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        widths = [8, 24, 16]
        drive(d, square(max(*widths), 0.1), t)

    (sq,) = [op for op in python_ir(k, Q[0]).ops if op.kind == "pulse.square"]
    assert sq.attrs["duration"] == 24


def test_tuple_displays_with_a_star_and_unary_plus():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        widths = (*[8, 16], 24)
        drive(d, square(widths[-1], +0.25), t)

    (sq,) = [op for op in python_ir(k, Q[0]).ops if op.kind == "pulse.square"]
    assert (sq.attrs["duration"], sq.attrs["amplitude"]) == (24, 0.25)


def test_star_unpacking_a_line_is_rejected():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        sync(*d)

    with pytest.raises(CompilationError, match="cannot unpack a pulse drive_line value with"):
        python_ir(k, Q[0])


def test_double_star_needs_a_mapping_and_distinct_keys():
    @pulse.kernel
    def not_a_mapping(q):
        d, t = get_drive_line(q)
        drive(d, square(**[1, 2]), t)

    @pulse.kernel
    def repeated(q):
        d, t = get_drive_line(q)
        drive(d, square(duration=8, **{"duration": 8, "amplitude": 0.1}), t)

    with pytest.raises(CompilationError, match="argument after \\*\\* must be a mapping, not list"):
        python_ir(not_a_mapping, Q[0])
    with pytest.raises(CompilationError, match="multiple values for keyword argument 'duration'"):
        python_ir(repeated, Q[0])


# ── Helper signatures, decorated helpers and methods ─────────────────


def _pulse_star(*lines, width=16):
    sync(*lines)


def _pulse_kwonly(d, t, *, n=20):
    drive(d, square(n, 0.1), t)


def _pulse_kwargs(d, t, **opts):
    drive(d, square(opts["n"], opts.get("amp", 0.1)), t)


def test_helpers_take_star_args_keyword_only_and_double_star_parameters():
    @pulse.kernel
    def k(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        _pulse_kwonly(d0, t0, n=14)
        _pulse_kwonly(d1, t1)
        _pulse_star(d0, d1, width=8)
        _pulse_kwargs(d0, t0, n=12, amp=0.3)
        # The lines the helpers advanced are the caller's current lines.
        wait(d0, 4)
        wait(d1, 4)

    ops = python_ir(k, Q[0], Q[1]).ops
    squares = [
        (op.attrs["duration"], op.attrs["amplitude"]) for op in ops if op.kind == "pulse.square"
    ]
    assert squares == [(14, 0.1), (20, 0.1), (12, 0.3)]
    (sync_op,) = [op for op in ops if op.kind == "pulse.sync"]
    waits = [op for op in ops if op.kind == "pulse.wait"]
    assert waits[1].operands[0] is sync_op.results[1]
    _compiles(k, Q[0], Q[1])


def test_helper_binding_errors_name_the_helper():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        _pulse_kwonly(d, t, 14)

    with pytest.raises(CompilationError, match="_pulse_kwonly: too many positional arguments"):
        python_ir(k, Q[0])


def _passthrough(f):
    import functools

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@_passthrough
def _decorated_x(d, t, n):
    drive(d, square(n, 0.1), t)


@_passthrough
def _decorated_wave(n):
    return square(n, 0.2)


def test_decorated_helpers_from_the_friction_log_are_inlined():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        _decorated_x(d, t, 14)
        drive(d, _decorated_wave(10), t)

    ops = python_ir(k, Q[0]).ops
    assert [op.attrs["duration"] for op in ops if op.kind == "pulse.square"] == [14, 10]
    _compiles(k, Q[0])


class _Pulses:
    amplitude = 0.25

    def make(self, duration):
        return square(duration, self.amplitude)

    def play(self, d, t):
        drive(d, self.make(8), t)

    def scaled(self, x):
        return self.amplitude * x


PULSES = _Pulses()


def test_methods_that_use_pulse_ops_are_inlined():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, PULSES.make(40), t)
        PULSES.play(d, t)
        drive(d, square(4, PULSES.scaled(2.0)), t)

    ops = python_ir(k, Q[0]).ops
    squares = [
        (op.attrs["duration"], op.attrs["amplitude"]) for op in ops if op.kind == "pulse.square"
    ]
    assert squares == [(40, 0.25), (8, 0.25), (4, 0.5)]
    _compiles(k, Q[0])


def _make(duration, amplitude):
    return square(duration, amplitude)


def test_partials_of_pulse_helpers_are_inlined():
    import functools

    wave = functools.partial(_make, amplitude=0.1)

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, wave(8), t)

    (sq,) = [op for op in python_ir(k, Q[0]).ops if op.kind == "pulse.square"]
    assert (sq.attrs["duration"], sq.attrs["amplitude"]) == (8, 0.1)


def test_a_pulse_op_in_code_run_as_python_is_a_compilation_error():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        waves = list(map(_make, [8], [0.1]))
        drive(d, waves[0], t)

    with pytest.raises(CompilationError, match=r"list\(\) calls the pulse op square\(\) but ran"):
        python_ir(k, Q[0])


# ── Lines and tones held in containers ───────────────────────────────


def test_line_tone_pairs_in_a_list_from_the_friction_log():
    @pulse.kernel
    def k(q0, q1):
        lines = []
        for q in (q0, q1):
            lines.append(get_drive_line(q))  # list of (line, tone) tuples
        drive(lines[0][0], gaussian(40, 0.1, 10.0), lines[0][1])
        drive(lines[0][0], gaussian(40, 0.1, 10.0), lines[0][1])
        d, t = lines[0]
        drive(d, gaussian(40, 0.1, 10.0), t)
        d, t = lines[0]
        drive(d, gaussian(40, 0.1, 10.0), t)

    drives = [op for op in python_ir(k, Q[0], Q[1]).ops if op.kind == "pulse.drive"]
    # Each drive consumes the line and tone the one before it produced.
    for before, after in zip(drives, drives[1:]):
        assert after.operands[0] is before.results[0]
        assert after.operands[2] is before.results[1]
    _compiles(k, Q[0], Q[1])


def test_lines_in_dicts_and_nested_lists_stay_current():
    @pulse.kernel
    def k(qubits):
        by_name = {}
        nested = []
        for i, q in enumerate(qubits):
            by_name[f"q{i}"] = get_drive_line(q)
            nested.append([get_drive_line(q)])
        for _ in range(3):
            d, t = by_name["q0"]
            drive(d, square(8, 0.1), t)
            drive(nested[1][0][0], square(8, 0.1), nested[1][0][1])

    # Rolled: d and t enter the loop with the line and tone of the dict entry.
    assert _kinds(k, Q[0:2]).count("scf.for") == 1
    _compiles(k, Q[0:2])


def test_a_rolled_loop_carries_a_line_read_from_a_tuple():
    @pulse.kernel
    def k(q):
        pairs = [get_drive_line(q)]
        for _ in range(20):
            drive(pairs[0][0], square(8, 0.1), pairs[0][1])
        drive(pairs[0][0], square(8, 0.2), pairs[0][1])

    kinds = _kinds(k, Q[0])
    assert kinds.count("scf.for") == 1
    _compiles(k, Q[0])


def test_pop_insert_extend_and_del_on_lists_of_lines_from_the_friction_log():
    @pulse.kernel
    def popped(q0, q1):
        ds, ts = [], []
        for q in (q0, q1):
            d, t = get_drive_line(q)
            ds.append(d)
            ts.append(t)
        d = ds.pop()
        t = ts.pop()
        drive(d, square(10, 0.1), t)
        drive(ds[0], square(10, 0.1), ts[0])

    @pulse.kernel
    def reshaped(q0, q1, q2):
        ds, ts = [], []
        for q in (q0, q1):
            d, t = get_drive_line(q)
            ds.append(d)
            ts.append(t)
        d2, t2 = get_drive_line(q2)
        ds.insert(0, d2)
        ts.insert(0, t2)
        for d, t in zip(ds, ts):
            drive(d, square(10, 0.1), t)
        ds.extend([ds.pop(0)])
        ts.extend([ts.pop(0)])
        del ds[0]
        del ts[0]
        for d, t in zip(ds, ts):
            drive(d, square(10, 0.1), t)

    _compiles(popped, Q[0], Q[1])
    _compiles(reshaped, Q[0], Q[1], Q[2])
    assert _kinds(reshaped, Q[0], Q[1], Q[2]).count("pulse.drive") == 5


def test_del_errors_are_compilation_errors():
    @pulse.kernel
    def k(q):
        xs = [1]
        del xs[3]

    with pytest.raises(CompilationError, match="del failed: list\\[3\\]"):
        python_ir(k, Q[0])


# ── Comprehensions and set displays ──────────────────────────────────

ALL = (0, 1, 2)
AMPS = (0.1, 0.2, 0.3)


def test_comprehensions_from_the_friction_log():
    @pulse.kernel
    def probe(qubits):
        lines = [get_drive_line(qubits[i]) for i in ALL]
        sync(*[line for line, _ in lines])
        amps = [0.1 * i for i in range(3)]
        total = sum([a for a in AMPS])
        widths = {i: 8 * (i + 1) for i in ALL}
        for i, (line, tone) in enumerate(lines):
            drive(line, square(widths[i], amps[i] + total), tone)

    ops = python_ir(probe, Q[0:3]).ops
    (sync_op,) = [op for op in ops if op.kind == "pulse.sync"]
    assert len(sync_op.operands) == 3
    drives = [op for op in ops if op.kind == "pulse.drive"]
    assert [d.operands[0] for d in drives] == list(sync_op.results)
    squares = [op.attrs for op in ops if op.kind == "pulse.square"]
    assert [s["duration"] for s in squares] == [8, 16, 24]
    assert [round(s["amplitude"], 9) for s in squares] == [0.6, 0.7, 0.8]
    _compiles(probe, Q[0:3])


def test_a_comprehension_collects_measurements():
    @pulse.kernel
    def rounds(q):
        r, rt = get_readout_line(q)
        ms = [readout(r, square(100, 0.05), rt) for _ in range(12)]
        assert len(ms) == 12

    kinds = _kinds(rounds, Q[0])
    assert kinds.count("pulse.readout") == 12
    assert "scf.for" not in kinds
    _compiles(rounds, Q[0])


def test_nested_filtered_set_and_dict_comprehensions():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        grid = [[i * j for j in range(3)] for i in range(3)]
        flat = [x for row in grid for x in row if x % 2 == 0]
        unique = {x % 3 for x in flat}
        pairs = {k: v for k, v in zip("ab", (8, 16))}
        literal = {8, 16}
        for n in sorted(unique | literal):
            drive(d, square(n + pairs["a"], 0.1), t)

    ops = python_ir(k, Q[0]).ops
    assert [op.attrs["duration"] for op in ops if op.kind == "pulse.square"] == [8, 9, 10, 16, 24]


def test_a_comprehension_restores_its_variable():
    @pulse.kernel
    def keeps(q):
        d, t = get_drive_line(q)
        x = 8
        ys = [x for x in (16, 24)]
        drive(d, square(x + len(ys), 0.1), t)

    (sq,) = [op for op in python_ir(keeps, Q[0]).ops if op.kind == "pulse.square"]
    assert sq.attrs["duration"] == 10


def test_comprehension_inside_a_rolled_loop_and_over_range():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        for _ in range(20):
            ws = [8 for j in range(2)]
            drive(d, square(ws[0], 0.1), t)

    assert _kinds(k, Q[0]).count("scf.for") == 1
    _compiles(k, Q[0])


def test_generator_expressions_and_lambdas_say_what_to_write_instead():
    @pulse.kernel
    def genexpr(q):
        d, t = get_drive_line(q)
        drive(d, square(sum(a for a in (8, 8)), 0.1), t)

    @pulse.kernel
    def lam(q):
        d, t = get_drive_line(q)
        scale = lambda a: a * 2  # noqa: E731
        drive(d, square(8, scale(0.1)), t)

    @pulse.kernel
    def nested(q):
        def inner():
            return 8

        d, t = get_drive_line(q)

    with pytest.raises(CompilationError, match=r"generator expressions .*sum\(\[x for x in xs\]\)"):
        python_ir(genexpr, Q[0])
    with pytest.raises(CompilationError, match="lambdas are not supported.*module level"):
        python_ir(lam, Q[0])
    with pytest.raises(CompilationError, match="nested functions are not supported.*module level"):
        python_ir(nested, Q[0])


def test_try_is_still_rejected_next_to_a_comprehension():
    @pulse.kernel
    def k(q):
        xs = [i for i in range(2)]
        try:
            xs.append(get_drive_line(q))
        except ValueError:
            pass

    with pytest.raises(CompilationError, match="try and with statements"):
        python_ir(k, Q[0])


# ── Starred assignment ───────────────────────────────────────────────


def test_starred_assignment_from_the_friction_log():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        first, *rest = AMPS
        a, *middle, z = (8, 16, 24, 32)
        *init, last = [8]
        drive(d, square(a + len(middle) + last, first + sum(rest)), t)
        assert middle == [16, 24] and init == [] and z == 32

    (sq,) = [op for op in python_ir(k, Q[0]).ops if op.kind == "pulse.square"]
    assert sq.attrs["duration"] == 18
    assert round(sq.attrs["amplitude"], 9) == 0.6


def test_starred_assignment_of_lines_keeps_them_current():
    @pulse.kernel
    def k(qubits):
        first, *others = [get_drive_line(q) for q in qubits]
        d, t = first
        drive(d, square(8, 0.1), t)
        for line, tone in others:
            drive(line, square(8, 0.1), tone)
            drive(line, square(8, 0.1), tone)
        drive(d, square(8, 0.1), t)

    _compiles(k, Q[0:3])


def test_starred_assignment_errors():
    @pulse.kernel
    def short(q):
        _a, _b, *_rest = (1,)

    @pulse.kernel
    def line(q):
        d, t = get_drive_line(q)
        _a, *_rest = d

    with pytest.raises(CompilationError, match="expected at least 2, got 1"):
        python_ir(short, Q[0])
    with pytest.raises(CompilationError, match="cannot unpack a pulse drive_line value"):
        python_ir(line, Q[0])


def test_slice_assignment():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        xs = [8, 8, 8, 8]
        start = 1
        xs[start:3] = [16]
        xs[:1] = [24, 32]
        for n in xs:
            drive(d, square(n, 0.1), t)

    ops = python_ir(k, Q[0]).ops
    assert [op.attrs["duration"] for op in ops if op.kind == "pulse.square"] == [24, 32, 16, 8]


def test_assigning_into_a_tuple_is_a_compilation_error():
    @pulse.kernel
    def k(q):
        xs = (1, 2)
        xs[0] = 3

    with pytest.raises(CompilationError, match=r"assignment failed: tuple\[0\]"):
        python_ir(k, Q[0])
