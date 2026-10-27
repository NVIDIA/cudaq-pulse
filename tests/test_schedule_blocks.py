# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Kernels called with their own schedule= are scheduled alone, then placed whole.

A called kernel without schedule= is inlined and scheduled with its caller,
as before. compile(schedule=...) sets only the compiled kernel's policy, and
compile(flatten_schedules=True) ignores every called kernel's policy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cudaq_pulse as pulse
import pytest

from tests._handles import Q

_ROOT = Path(__file__).resolve().parents[1]


def _timeline(compiled):
    return [(e.kind, e.line, e.start_cycles, e.duration_cycles) for e in compiled.schedule]


def _starts(compiled):
    return [e.start_cycles for e in compiled.schedule]


def _qubits(n):
    return [Q[i] for i in range(n)]


@pulse.kernel(schedule="asap")
def asap_pair(db, tb, dc, tc):
    drive(db, square(100, 0.1), tb)
    drive(dc, square(40, 0.1), tc)


@pulse.kernel(schedule="alap")
def alap_pair(db, tb, dc, tc):
    drive(db, square(100, 0.1), tb)
    drive(dc, square(40, 0.1), tc)


@pulse.kernel
def plain_pair(db, tb, dc, tc):
    drive(db, square(100, 0.1), tb)
    drive(dc, square(40, 0.1), tc)


def _after_long_drive(pair):
    """Line a runs 200 while `pair` drives lines b and c."""

    @pulse.kernel
    def top(qa, qb, qc):
        da, ta = get_drive_line(qa)
        db, tb = get_drive_line(qb)
        dc, tc = get_drive_line(qc)
        drive(da, square(200, 0.1), ta)
        pair(db, tb, dc, tc)

    return top


# ── Declaring a policy ───────────────────────────────────────────────


def test_bare_and_called_decorator_forms_agree():
    @pulse.kernel()
    def called(q):
        d, t = get_drive_line(q)
        drive(d, square(40, 0.1), t)

    @pulse.kernel
    def bare(q):
        d, t = get_drive_line(q)
        drive(d, square(40, 0.1), t)

    q = Q[0]
    assert _timeline(called.lower(q)) == _timeline(bare.lower(q))


def test_unknown_schedule_is_rejected_where_the_kernel_is_declared():
    with pytest.raises(ValueError, match="Unknown kernel schedule 'fastest'"):
        pulse.kernel(schedule="fastest")


@pytest.mark.parametrize("policy", ["rcp", "alap_rcp"])
def test_resource_constrained_schedules_are_for_whole_programs(policy):
    with pytest.raises(ValueError, match=r"pass it to cudaq_pulse.compile"):
        pulse.kernel(schedule=policy)


# ── Placement ────────────────────────────────────────────────────────


def test_asap_block_keeps_its_shape_in_an_alap_program():
    top = _after_long_drive(asap_pair)
    blocked = top.lower(*_qubits(3))
    flat = pulse.compile(top, flatten_schedules=True).lower(*_qubits(3))
    # The block ends with line a; inside it, both lines start together.
    assert _timeline(blocked) == [
        ("drive", 0, 0, 200),
        ("drive", 1, 100, 100),
        ("drive", 2, 100, 40),
    ]
    # Flat ALAP moves the short pulse to the end.
    assert _timeline(flat)[2] == ("drive", 2, 160, 40)


def test_compile_schedule_sets_only_the_compiled_kernels_policy():
    top = _after_long_drive(alap_pair)
    compiled = pulse.compile(top, schedule="asap").lower(*_qubits(3))
    # The ALAP block right-aligns line c inside it even in an ASAP program.
    assert _timeline(compiled) == [
        ("drive", 0, 0, 200),
        ("drive", 1, 0, 100),
        ("drive", 2, 60, 40),
    ]


def test_compiled_kernel_defaults_to_its_own_schedule():
    @pulse.kernel(schedule="asap")
    def top(qa, qb):
        da, ta = get_drive_line(qa)
        db, tb = get_drive_line(qb)
        drive(da, square(200, 0.1), ta)
        drive(db, square(40, 0.1), tb)

    qubits = _qubits(2)
    assert _starts(top.lower(*qubits)) == [0, 0]
    assert _starts(pulse.compile(top, schedule="alap").lower(*qubits)) == [0, 160]


def test_kernel_without_schedule_is_scheduled_with_its_caller():
    def helper(db, tb, dc, tc):
        drive(db, square(100, 0.1), tb)
        drive(dc, square(40, 0.1), tc)

    qubits = _qubits(3)
    as_kernel = _after_long_drive(plain_pair).lower(*qubits)
    as_helper = _after_long_drive(helper).lower(*qubits)
    assert as_kernel.mlir() == as_helper.mlir()
    assert "schedule_block" not in as_kernel.mlir()


@pytest.mark.parametrize("policy", ["asap", "alap"])
def test_flatten_schedules_ignores_called_kernels_policies(policy):
    qubits = _qubits(3)
    for pair in (asap_pair, alap_pair):
        flat = pulse.compile(
            _after_long_drive(pair), schedule=policy, flatten_schedules=True
        ).lower(*qubits)
        plain = pulse.compile(_after_long_drive(plain_pair), schedule=policy).lower(*qubits)
        assert _timeline(flat) == _timeline(plain)


def test_three_levels_of_blocks_compose():
    @pulse.kernel(schedule="asap")
    def inner(x, tx, y, ty):
        drive(x, square(40, 0.1), tx)
        drive(y, square(20, 0.1), ty)

    @pulse.kernel(schedule="alap")
    def middle(x, tx, y, ty):
        drive(x, square(100, 0.1), tx)
        inner(x, tx, y, ty)

    @pulse.kernel(schedule="asap")
    def outer(x, tx, y, ty):
        drive(y, square(60, 0.1), ty)
        middle(x, tx, y, ty)

    @pulse.kernel
    def top(qx, qy, qz):
        x, tx = get_drive_line(qx)
        y, ty = get_drive_line(qy)
        z, tz = get_drive_line(qz)
        drive(z, square(300, 0.1), tz)
        outer(x, tx, y, ty)

    compiled = top.lower(*_qubits(3))
    # Same placement as the three_levels lit test.
    assert sorted((e.line, e.start_cycles, e.duration_cycles) for e in compiled.schedule) == [
        (0, 160, 100),
        (0, 260, 40),
        (1, 160, 60),
        (1, 260, 20),
        (2, 0, 300),
    ]


def test_block_in_a_rolled_loop_keeps_the_loop():
    @pulse.kernel(schedule="asap")
    def echo(d, t):
        drive(d, square(40, 0.1), t)
        wait(d, 60)

    @pulse.kernel
    def top(q):
        d, t = get_drive_line(q)
        for _ in range(8):
            echo(d, t)

    compiled = pulse.compile(top, schedule="asap").lower(Q[0])
    assert "scf.for" in compiled.mlir()
    assert [(e.kind, e.start_cycles) for e in compiled.schedule][:4] == [
        ("drive", 0),
        ("wait", 40),
        ("drive", 100),
        ("wait", 140),
    ]


def test_measurement_from_a_block_controls_the_caller():
    @pulse.kernel(schedule="alap")
    def measure(d, t, r, rt):
        sync(d, r)
        m = readout(r, square(400, 0.05), rt)
        if m:
            drive(d, square(40, 0.1), t)
        return m

    @pulse.kernel
    def top(qa, qb):
        da, ta = get_drive_line(qa)
        r, rt = get_readout_line(qa)
        db, tb = get_drive_line(qb)
        drive(da, square(100, 0.1), ta)
        m = measure(da, ta, r, rt)
        if m:
            drive(db, square(40, 0.1), tb)

    events = top.lower(*_qubits(2)).schedule
    readout_event = next(e for e in events if e.kind == "readout")
    feedback = events[-1]
    assert readout_event.start_cycles == 100
    assert feedback.line == 2 and feedback.start_cycles == 500


def test_tone_threaded_into_a_block_waits_for_its_pulse():
    @pulse.kernel(schedule="asap")
    def on_target(db, tb):
        drive(db, square(40, 0.1), tb)

    @pulse.kernel
    def top(qa, qb):
        da, ta = get_drive_line(qa)
        db, tb = get_drive_line(qb)
        drive(da, square(300, 0.1), tb)  # cross-resonance at qubit b's tone
        on_target(db, tb)

    assert _starts(top.lower(*_qubits(2))) == [0, 300]


# ── Parametric kernels and resource limits ───────────────────────────


def test_a_float_parameter_does_not_change_a_blocks_schedule():
    @pulse.kernel(schedule="asap")
    def pulse_pair(db, tb, dc, tc, amp: float):
        drive(db, square(100, amp), tb)
        drive(dc, square(40, amp), tc)

    @pulse.kernel
    def top(qa, qb, qc, amp: float):
        da, ta = get_drive_line(qa)
        db, tb = get_drive_line(qb)
        dc, tc = get_drive_line(qc)
        drive(da, square(200, 0.1), ta)
        pulse_pair(db, tb, dc, tc, amp)

    qubits = _qubits(3)
    compiled = pulse.compile(top)
    first = compiled.lower(*qubits, 0.2)
    # The module is scheduled, blocks placed and inlined, with amp symbolic.
    assert "%arg0: f64" in first.generic.mlir()
    assert "schedule_block" not in first.generic.mlir()
    assert _starts(first) == [0, 100, 100]
    assert _timeline(compiled.lower(*qubits, 0.5)) == _timeline(first)
    assert compiled.cache_info().compiled == 1


@pytest.mark.parametrize("policy", ["rcp", "alap_rcp"])
def test_resource_constrained_programs_reject_blocks(policy):
    top = _after_long_drive(asap_pair)
    with pytest.raises(pulse.CompilationError, match="flatten_schedules=True"):
        pulse.compile(top, schedule=policy).lower(*_qubits(3))
    flat = pulse.compile(top, schedule=policy, flatten_schedules=True).lower(*_qubits(3))
    assert len(flat.schedule) == 3


def test_native_scheduler_rejects_resource_limits_with_blocks():
    from cudaq_pulse.compiler import PulseModuleBuilder
    from cudaq_pulse.kernel.bytecode_bridge import _trace_kernel_with_builder
    from cudaq_pulse.kernel.packed_ir_builder import PackedIRBuilder

    builder = PackedIRBuilder()
    _trace_kernel_with_builder(_after_long_drive(asap_pair).__wrapped__, builder, _qubits(3))
    module = PulseModuleBuilder().build_from_packed(
        builder.get_buffer(), 2.0, builder.n_qubits, builder.get_freq_array()
    )
    assert "pulse.schedule_block" in module.print()
    with pytest.raises(RuntimeError, match="does not support kernels with their own schedule"):
        module.schedule("rcp")


# ── Programs without blocks ──────────────────────────────────────────


def _compile_time_workloads():
    spec = importlib.util.spec_from_file_location(
        "compile_time", _ROOT / "benchmarks" / "compiler" / "compile_time.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module.WORKLOADS


@pytest.mark.parametrize("policy", ["asap", "alap", "rcp"])
def test_programs_without_blocks_schedule_as_before(policy):
    for _name, kernel_fn, n_qubits, _freq in _compile_time_workloads():
        args = _qubits(n_qubits)
        default = pulse.compile(kernel_fn, schedule=policy).lower(*args)
        flat = pulse.compile(kernel_fn, schedule=policy, flatten_schedules=True).lower(*args)
        assert "schedule_block" not in default.mlir()
        assert default.mlir() == flat.mlir()


# ── Blocks in rolled loops ───────────────────────────────────────────


@pulse.kernel(schedule="asap")
def asap_layer(d0, t0, d1, t1):
    drive(d0, square(40, 0.1), t0)
    drive(d1, square(16, 0.1), t1)


def test_a_block_in_a_rolled_loop_keeps_its_timing_each_iteration():
    @pulse.kernel
    def repeated(q0, q1, n: int):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        for _ in range(n):
            asap_layer(d0, t0, d1, t1)

    @pulse.kernel
    def once(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        asap_layer(d0, t0, d1, t1)

    qubits = _qubits(2)
    single = _timeline(once.lower(*qubits))
    for n in (2, 8):
        lowered = repeated.lower(*qubits, n)
        assert lowered.mlir().count("scf.for") == 1
        timeline = _timeline(lowered)
        assert len(timeline) == 2 * n
        period = 40
        for k in range(n):
            assert [
                (kind, line, start - k * period, dur)
                for kind, line, start, dur in timeline[2 * k : 2 * k + 2]
            ] == single


def test_a_callee_that_acquires_its_lines_in_a_rolled_loop_stays_rolled():
    @pulse.kernel(schedule="asap")
    def layer(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        drive(d0, square(40, 0.1), t0)
        drive(d1, square(16, 0.1), t1)

    @pulse.kernel
    def repeated(q0, q1, n: int):
        for _ in range(n):
            layer(q0, q1)

    lowered = repeated.lower(*_qubits(2), 4)
    assert lowered.mlir().count("scf.for") == 1
    starts = sorted(e.start_cycles for e in lowered.schedule if e.line == 0)
    assert starts == [0, 40, 80, 120]


def test_kernel_schedules_are_checked_where_kernels_are_declared():
    with pytest.raises(ValueError, match="applies to a whole program"):
        pulse.kernel(schedule="rcp")(lambda q: None)
    assert pulse.kernel(schedule="asap")(lambda q: None).schedule == "asap"


def test_a_block_with_a_float_length_wait_keeps_its_shape_when_called():
    @pulse.kernel(schedule="asap")
    def delayed(da, ta, db, tb, tau: float):
        wait(da, tau, bounds=(0, 400))
        drive(da, square(40, 0.1), ta)
        drive(db, square(10, 0.1), tb)

    @pulse.kernel
    def top(qa, qb, qc, tau: float):
        da, ta = get_drive_line(qa)
        db, tb = get_drive_line(qb)
        dc, tc = get_drive_line(qc)
        drive(dc, square(1000, 0.1), tc)
        delayed(da, ta, db, tb, tau)

    events = top.lower(*_qubits(3), 37.0).schedule
    # Placed whole at the end of the ALAP program, the ASAP block still
    # starts both of its lines together.
    assert [(e.kind, e.line, e.start_cycles) for e in events] == [
        ("drive", 2, 0),
        ("wait", 0, 923),
        ("drive", 0, 960),
        ("drive", 1, 923),
    ]
