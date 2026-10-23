# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Kernels that call other kernels inside loops.

A call inlines the callee into the caller's trace. A ``range()`` loop whose
induction variable is never read stays rolled as one ``scf.for`` with the
callee's ops in its body; reading the variable, or iterating a Python value,
unrolls it. Each case compiles the looped kernel and a hand-unrolled
equivalent, and checks the loop structure, the op counts, the schedule and
the CPU simulation against it.

Only the helpers under "Call mechanics" know how a kernel is traced,
compiled and simulated; the kernels and assertions do not.
"""

from __future__ import annotations

import math
from collections import Counter

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.targets import Qubit, Target
from tests._ir import python_ir

FREQ_HZ = 5.0e9
# With H = amp X / 2 at 1 rad/ns per unit amplitude, 16 vtu (8 ns) at
# amplitude 0.05 rotates by 0.4 rad: small enough that sequences of
# pulses with different phases or frequencies end in distinct states.
AMP = 0.05
SIGNIFICANT = 1e-3
UNFUSED = ("verify", "virtual_z")

# ── Call mechanics ───────────────────────────────────────────────────


def _qubits(n, qvec):
    return [pulse.qvec_ref(n)] if qvec else [pulse.qudit_ref() for _ in range(n)]


def _device(n):
    return pulse.device({"target": _target(n)})


def _trace(kernel, n=1, qvec=False):
    """The kernel's IR, as a list of ops."""
    return python_ir(kernel, *_qubits(n, qvec)).ops


def _compile(kernel, n=1, qvec=False, passes=None):
    """The compiled, scheduled kernel."""
    return pulse.compile(kernel, passes=passes, device=_device(n)).lower(*_qubits(n, qvec))


def _compile_symbolic(kernel):
    """Compile a one-qudit kernel ahead of its other arguments."""
    return pulse.compile(kernel, device=_device(1))


def _evaluate(compiled, *values):
    """The compiled kernel, lowered at concrete values of its other arguments."""
    return compiled.lower(pulse.qudit_ref(), *values)


def _simulate(kernel, n=1, qvec=False, **kwargs):
    """The CPU simulation of the kernel."""
    lowered = _compile(kernel, n, qvec, passes=("verify", "canonicalize", "virtual_z", "fusion"))
    return pulse.simulate(lowered, **kwargs)


# ── Checks ───────────────────────────────────────────────────────────


def _target(n):
    return Target(
        name="loop-calls",
        qubits={
            q: Qubit(
                index=q,
                frequency_hz=FREQ_HZ + 0.1e9 * q,
                anharmonicity_hz=-200.0e6,
                t1_us=0.0,
                t2_star_us=0.0,
                drive_params={"amplitude_scale_rad_per_ns": 1.0},
            )
            for q in range(n)
        },
    )


def _kinds(ops):
    return Counter(op.kind for op in ops)


def _loop_bodies(ops):
    """The op kinds directly inside each scf.for, outermost first."""
    bodies, open_ = [], []
    for op in ops:
        if open_:
            open_[-1].append(op.kind)
        if op.kind == "scf.for":
            body = []
            bodies.append(body)
            open_.append(body)
        elif op.kind == "scf.for_end":
            open_.pop()
    return [[k for k in body if k != "scf.for_end"] for body in bodies]


def _schedule(compiled):
    """``(kind, qubit, start, duration)`` of each event, in time order."""
    events = [(e.kind, e.qubit, e.start_vtu, e.duration_vtu) for e in compiled.schedule]
    return sorted(events, key=lambda e: (e[2], e[1], e[0]))


def _length(compiled):
    return max((e.start_vtu + e.duration_vtu for e in compiled.schedule), default=0)


def _verify(compiled):
    """The scheduled module re-parses and passes the pulse verifier."""
    module = MLIRPipeline().parse_to_module(compiled.mlir())
    module.run_passes(["pulse-verify"])


def check(kernel, reference, *, loops, n=1, qvec=False, sim=True, **sim_kwargs):
    """Compare *kernel* with its hand-unrolled *reference*.

    *loops* is the number of ``scf.for`` ops *kernel* traces to. Returns
    the kernel's ops and its compiled form for case-specific checks.
    """
    ops = _trace(kernel, n, qvec)
    assert _kinds(ops)["scf.for"] == loops
    assert _kinds(_trace(reference, n, qvec))["scf.for"] == 0

    compiled = _compile(kernel, n, qvec)
    expected = _compile(reference, n, qvec)
    _verify(compiled)
    _verify(expected)
    assert _length(compiled) == _length(expected)
    # Fusion merges adjacent drives of the unrolled reference only, so
    # events are compared without it.
    compiled = _compile(kernel, n, qvec, passes=UNFUSED)
    assert _schedule(compiled) == _schedule(_compile(reference, n, qvec, passes=UNFUSED))

    if sim:
        result = _simulate(kernel, n, qvec, **sim_kwargs)
        want = _simulate(reference, n, qvec, **sim_kwargs)
        assert result.times_ns == want.times_ns
        np.testing.assert_allclose(result.density_matrix, want.density_matrix, atol=1e-9)
        got, ref = sorted(result.branches), sorted(want.branches)
        assert [o for o, _ in got] == [o for o, _ in ref]
        assert [p for _, p in got] == pytest.approx([p for _, p in ref])
    return ops, compiled


def _distinct(a, b, n=1, qvec=False):
    """The two kernels end in measurably different states."""
    rho_a = _simulate(a, n, qvec).density_matrix
    rho_b = _simulate(b, n, qvec).density_matrix
    assert np.abs(rho_a - rho_b).max() > SIGNIFICANT


# ── Callees ──────────────────────────────────────────────────────────


@pulse.kernel
def pulse16(d, t):
    drive(d, square(16, AMP), t)


@pulse.kernel
def idle(d, n):
    wait(d, n)


@pulse.kernel
def rotate(d, t, phase):
    t = shift_phase(t, phase)
    drive(d, square(16, AMP), t)


@pulse.kernel
def chirp(d, t):
    t = shift_frequency(t, 20.0e6)
    drive(d, square(16, AMP), t)


@pulse.kernel
def retune(d, t, freq):
    t = set_frequency(t, freq)
    drive(d, square(16, AMP), t)


@pulse.kernel
def on_qubit(q):
    d, t = get_drive_line(q)
    drive(d, square(16, AMP), t)


# ── A callee in a rolled loop ────────────────────────────────────────


def test_callee_in_a_range_loop_stays_rolled():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(4):
            pulse16(d, t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)

    ops, compiled = check(looped, unrolled, loops=1)
    assert _loop_bodies(ops) == [["pulse.square", "pulse.drive", "scf.yield"]]
    assert _kinds(ops)["pulse.drive"] == 1
    assert [e[2] for e in _schedule(compiled)] == [0, 16, 32, 48]
    assert _length(compiled) == 64


def test_plain_helper_and_kernel_callees_trace_alike():
    def helper(d, t):
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def with_kernel(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            pulse16(d, t)

    @pulse.kernel
    def with_helper(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            helper(d, t)

    assert _compile(with_kernel).mlir().replace("with_kernel", "k") == _compile(
        with_helper
    ).mlir().replace("with_helper", "k")


# ── The loop variable as a callee argument ───────────────────────────


def test_loop_variable_as_a_wait_duration_unrolls():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for i in range(4):
            idle(d, 8 * (i + 1))
            pulse16(d, t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for n in (8, 16, 24, 32):
            wait(d, n)
            drive(d, square(16, AMP), t)

    ops, compiled = check(looped, unrolled, loops=0)
    assert [op.attrs["duration"] for op in ops if op.kind == "pulse.wait"] == [8, 16, 24, 32]
    drives = [e[2] for e in _schedule(compiled) if e[0] == "drive"]
    assert drives == [8, 40, 80, 128]
    assert _length(compiled) == 144


def test_loop_variable_as_a_waveform_duration():
    @pulse.kernel
    def stretched(d, t, n):
        drive(d, square(n, AMP), t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for i in range(1, 4):
            stretched(d, t, 8 * i)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        drive(d, square(8, AMP), t)
        drive(d, square(16, AMP), t)
        drive(d, square(24, AMP), t)

    _, compiled = check(looped, unrolled, loops=0)
    assert [(e[2], e[3]) for e in _schedule(compiled)] == [(0, 8), (8, 16), (24, 24)]


def test_loop_variable_in_float_arithmetic():
    @pulse.kernel
    def scaled(d, t, k):
        drive(d, square(16, AMP * (1.0 + 0.5 * k)), t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for i in range(4):
            rotate(d, t, 0.3 * i + 0.1)
            scaled(d, t, float(i))

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for i in range(4):
            t = shift_phase(t, 0.3 * i + 0.1)
            drive(d, square(16, AMP), t)
            drive(d, square(16, AMP * (1.0 + 0.5 * i)), t)

    ops, compiled = check(looped, unrolled, loops=0)
    phases = [op.attrs["phase_rad"] for op in ops if op.kind == "pulse.shift_phase"]
    assert phases == pytest.approx([0.1, 0.4, 0.7, 1.0])
    assert _length(compiled) == 128


# ── Qudits chosen by the loop ────────────────────────────────────────


def test_qudit_indexed_by_the_loop_variable_unrolls():
    @pulse.kernel
    def looped(qs):
        for i in range(len(qs)):
            on_qubit(qs[i])

    @pulse.kernel
    def unrolled(qs):
        d0, t0 = get_drive_line(qs[0])
        drive(d0, square(16, AMP), t0)
        d1, t1 = get_drive_line(qs[1])
        drive(d1, square(16, AMP), t1)
        d2, t2 = get_drive_line(qs[2])
        drive(d2, square(16, AMP), t2)

    ops, compiled = check(looped, unrolled, loops=0, n=3, qvec=True)
    assert _kinds(ops)["pulse.get_drive_line"] == 3
    assert _schedule(compiled) == [("drive", q, 0, 16) for q in range(3)]


def test_iterating_a_list_or_tuple_of_qudits():
    @pulse.kernel
    def over_list(q0, q1, q2):
        for q in [q0, q2]:
            on_qubit(q)
        for q in (q1,):
            on_qubit(q)

    @pulse.kernel
    def unrolled(q0, q1, q2):
        on_qubit(q0)
        on_qubit(q2)
        on_qubit(q1)

    _, compiled = check(over_list, unrolled, loops=0, n=3)
    assert sorted(e[1] for e in _schedule(compiled)) == [0, 1, 2]


def test_lines_held_in_lists_stay_threaded_through_a_rolled_loop():
    @pulse.kernel
    def looped(q0, q1):
        lines, tones = [], []
        for q in (q0, q1):
            d, t = get_drive_line(q)
            lines.append(d)
            tones.append(t)
        for _ in range(3):
            for i in range(len(lines)):
                rotate(lines[i], tones[i], 0.5 * (i + 1))

    @pulse.kernel
    def unrolled(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        for _k in (0, 1, 2):
            t0 = shift_phase(t0, 0.5)
            drive(d0, square(16, AMP), t0)
            t1 = shift_phase(t1, 1.0)
            drive(d1, square(16, AMP), t1)

    ops, _ = check(looped, unrolled, loops=1, n=2)
    for_end = next(op for op in ops if op.kind == "scf.for_end")
    assert {"lines[0]", "lines[1]", "tones[0]", "tones[1]"} <= set(for_end.attrs["iter_args"])


# ── Loops on both sides of the call ──────────────────────────────────


def test_callee_with_its_own_loop_nests_the_loops():
    @pulse.kernel
    def train(d, t):
        for _ in range(3):
            t = shift_phase(t, 0.4)
            drive(d, square(16, AMP), t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(2):
            train(d, t)
            wait(d, 8)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1):
            for _ in (0, 1, 2):
                t = shift_phase(t, 0.4)
                drive(d, square(16, AMP), t)
            wait(d, 8)

    ops, compiled = check(looped, unrolled, loops=2)
    outer, inner = _loop_bodies(ops)
    assert outer.count("scf.for") == 1 and "pulse.wait" in outer
    assert inner == ["pulse.shift_phase", "pulse.square", "pulse.drive", "scf.yield"]
    drives = [e[2] for e in _schedule(compiled) if e[0] == "drive"]
    assert drives == [0, 16, 32, 56, 72, 88]
    assert _length(compiled) == 112


# ── Several calls per iteration ──────────────────────────────────────


def test_same_callee_twice_per_iteration():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            rotate(d, t, 0.6)
            rotate(d, t, -0.2)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            t = shift_phase(t, 0.6)
            drive(d, square(16, AMP), t)
            t = shift_phase(t, -0.2)
            drive(d, square(16, AMP), t)

    ops, compiled = check(looped, unrolled, loops=1)
    assert _kinds(ops)["pulse.drive"] == 2
    assert _length(compiled) == 96


def test_different_callees_in_one_body():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            pulse16(d, t)
            idle(d, 8)
            rotate(d, t, 0.9)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            drive(d, square(16, AMP), t)
            wait(d, 8)
            t = shift_phase(t, 0.9)
            drive(d, square(16, AMP), t)

    ops, compiled = check(looped, unrolled, loops=1)
    assert _loop_bodies(ops)[0].count("pulse.drive") == 2
    assert [e[2] for e in _schedule(compiled)] == [0, 16, 24, 40, 56, 64, 80, 96, 104]
    assert _length(compiled) == 120


# ── Branches around and inside the call ──────────────────────────────

ENABLED = True


def test_callee_under_a_compile_time_if():
    @pulse.kernel
    def flag(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            if ENABLED:
                pulse16(d, t)
            else:
                idle(d, 16)

    @pulse.kernel
    def by_parity(q):
        d, t = get_drive_line(q)
        for i in range(4):
            if i % 2 == 0:
                rotate(d, t, 0.7)
            else:
                idle(d, 8)

    @pulse.kernel
    def flag_unrolled(q):
        d, t = get_drive_line(q)
        drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def parity_unrolled(q):
        d, t = get_drive_line(q)
        t = shift_phase(t, 0.7)
        drive(d, square(16, AMP), t)
        wait(d, 8)
        t = shift_phase(t, 0.7)
        drive(d, square(16, AMP), t)
        wait(d, 8)

    check(flag, flag_unrolled, loops=1)
    _, compiled = check(by_parity, parity_unrolled, loops=0)
    assert _length(compiled) == 48


_PI = math.pi / 8.0  # 16 vtu (8 ns) at this amplitude is a pi rotation


@pulse.kernel
def measure_and_reset(d, t, r, rt):
    sync(d, r)
    bit = readout(r, square(64, 0.05), rt)
    if bit:
        drive(d, square(16, _PI), t)


def test_measurement_feedback_inside_a_called_kernel():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        for _ in range(2):
            drive(d, square(16, _PI / 2), t)
            measure_and_reset(d, t, r, rt)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        drive(d, square(16, _PI / 2), t)
        sync(d, r)
        bit = readout(r, square(64, 0.05), rt)
        if bit:
            drive(d, square(16, _PI), t)
        drive(d, square(16, _PI / 2), t)
        sync(d, r)
        bit = readout(r, square(64, 0.05), rt)
        if bit:
            drive(d, square(16, _PI), t)

    ops, compiled = check(looped, unrolled, loops=1)
    body = _loop_bodies(ops)[0]
    assert body.count("pulse.readout") == 1 and body.count("scf.if") == 1
    readouts = [e[2] for e in _schedule(compiled) if e[0] == "readout"]
    assert readouts == [16, 112]
    result = _simulate(looped)
    # Each round resets to |0>: the ensemble ends in the ground state.
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-7)
    assert [m.iteration for m in result.measurements] == [(0,), (1,)]
    assert sorted(o for o, _ in result.branches) == [(0, 0), (0, 1), (1, 0), (1, 1)]


@pulse.kernel
def measure(r, rt):
    return readout(r, square(64, 0.05), rt)


def test_measurement_returned_by_a_callee_controls_the_loop_body():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        for _ in range(2):
            drive(d, square(16, _PI / 2), t)
            sync(d, r)
            if measure(r, rt):
                pulse16(d, t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        for _ in (0, 1):
            drive(d, square(16, _PI / 2), t)
            sync(d, r)
            if readout(r, square(64, 0.05), rt):
                drive(d, square(16, AMP), t)

    check(looped, unrolled, loops=1)


# ── Tone state carried across iterations ─────────────────────────────


def test_shift_phase_in_a_callee_accumulates_across_iterations():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(4):
            rotate(d, t, 0.8)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2, 3):
            t = shift_phase(t, 0.8)
            drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def without_phase(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2, 3, 4):
            drive(d, square(16, AMP), t)

    ops, _ = check(looped, unrolled, loops=1)
    for_end = next(op for op in ops if op.kind == "scf.for_end")
    assert "t" in for_end.attrs["iter_args"]
    _distinct(looped, without_phase)


def test_shift_frequency_in_a_callee_accumulates_across_iterations():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            chirp(d, t)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            t = shift_frequency(t, 20.0e6)
            drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def one_shift(q):
        d, t = get_drive_line(q)
        t = shift_frequency(t, 20.0e6)
        for _ in (0, 1, 2, 3):
            drive(d, square(16, AMP), t)

    check(looped, unrolled, loops=1)
    _distinct(looped, one_shift)


def test_set_frequency_in_a_callee_holds_after_the_loop():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            retune(d, t, FREQ_HZ + 40.0e6)
            chirp(d, t)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            t = set_frequency(t, FREQ_HZ + 40.0e6)
            drive(d, square(16, AMP), t)
            t = shift_frequency(t, 20.0e6)
            drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)

    check(looped, unrolled, loops=1)


def test_a_callee_that_acquires_its_own_line_each_iteration():
    @pulse.kernel
    def fresh(q):
        d, t = get_drive_line(q)
        t = shift_phase(t, 0.5)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def looped(q):
        for _ in range(2):
            fresh(q)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        t = shift_phase(t, 0.5)
        drive(d, square(16, AMP), t)
        d, t = get_drive_line(q)
        t = shift_phase(t, 0.5)
        drive(d, square(16, AMP), t)

    # Every get_drive_line is a new line with its own timeline and tone, so
    # the loop unrolls: one rolled acquisition would stand for both.
    ops, compiled = check(looped, unrolled, loops=0)
    assert _kinds(ops)["pulse.get_drive_line"] == 2
    assert _schedule(compiled) == [("drive", 0, 0, 16), ("drive", 0, 0, 16)]


def test_a_callee_that_allocates_a_qudit_each_iteration():
    @pulse.kernel
    def scratch():
        on_qubit(qudit_ref())

    @pulse.kernel
    def looped(q):
        for _ in range(2):
            scratch()

    ops = _trace(looped)
    assert _kinds(ops)["scf.for"] == 0
    assert _kinds(ops)["pulse.qudit_alloc"] == 2


# ── sync across lines ────────────────────────────────────────────────


@pulse.kernel
def staggered(d0, t0, d1, t1):
    drive(d0, square(32, AMP), t0)
    sync(d0, d1)
    drive(d1, square(16, AMP), t1)


def test_callee_that_syncs_two_lines():
    @pulse.kernel
    def looped(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        for _ in range(3):
            staggered(d0, t0, d1, t1)

    @pulse.kernel
    def unrolled(q0, q1):
        d0, t0 = get_drive_line(q0)
        d1, t1 = get_drive_line(q1)
        for _ in (0, 1, 2):
            drive(d0, square(32, AMP), t0)
            sync(d0, d1)
            drive(d1, square(16, AMP), t1)

    ops, compiled = check(looped, unrolled, loops=1, n=2)
    assert _loop_bodies(ops)[0].count("pulse.sync") == 1
    # Line 0 runs back to back. ALAP moves each line-1 drive up against the
    # next sync, which waits for line 0's next drive; the last one follows
    # the final sync.
    assert _schedule(compiled) == [
        ("drive", 0, 0, 32),
        ("drive", 0, 32, 32),
        ("drive", 1, 48, 16),
        ("drive", 0, 64, 32),
        ("drive", 1, 80, 16),
        ("drive", 1, 96, 16),
    ]
    assert _length(compiled) == 112


# ── Kernels nested three deep ────────────────────────────────────────


@pulse.kernel
def inner_b(d, t):
    t = shift_phase(t, 0.3)
    drive(d, square(16, AMP), t)


@pulse.kernel
def middle_a(d, t):
    inner_b(d, t)
    wait(d, 8)
    inner_b(d, t)


def test_kernels_nested_three_deep():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            middle_a(d, t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            t = shift_phase(t, 0.3)
            drive(d, square(16, AMP), t)
            wait(d, 8)
            t = shift_phase(t, 0.3)
            drive(d, square(16, AMP), t)

    ops, compiled = check(looped, unrolled, loops=1)
    assert _kinds(ops)["pulse.drive"] == 2
    assert _length(compiled) == 120


# ── Return values ────────────────────────────────────────────────────


def test_callee_returning_a_tone():
    @pulse.kernel
    def advance(d, t):
        t = shift_phase(t, 0.45)
        drive(d, square(16, AMP), t)
        return t

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            t = advance(d, t)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            t = shift_phase(t, 0.45)
            drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)

    check(looped, unrolled, loops=1)


def test_callee_returning_a_python_value():
    @pulse.kernel
    def cycles(k):
        return 8 * k

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            wait(d, cycles(2))
            pulse16(d, t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            wait(d, 16)
            drive(d, square(16, AMP), t)

    _, compiled = check(looped, unrolled, loops=1)
    assert _length(compiled) == 96


# ── Trip counts ──────────────────────────────────────────────────────


def test_zero_iterations():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(0):
            rotate(d, t, 0.5)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        drive(d, square(16, AMP), t)

    ops, compiled = check(looped, unrolled, loops=1)
    loop = next(op for op in ops if op.kind == "scf.for")
    assert loop.attrs["ub"] == 0
    assert _schedule(compiled) == [("drive", 0, 0, 16)]


def test_zero_iterations_with_the_loop_variable_read():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for i in range(0):
            idle(d, i + 8)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        drive(d, square(16, AMP), t)

    check(looped, unrolled, loops=0)


def test_one_iteration():
    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(1):
            rotate(d, t, 0.5)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        t = shift_phase(t, 0.5)
        drive(d, square(16, AMP), t)
        drive(d, square(16, AMP), t)

    _, compiled = check(looped, unrolled, loops=1)
    assert _schedule(compiled) == [("drive", 0, 0, 16), ("drive", 0, 16, 16)]


# ── Globals and closures ─────────────────────────────────────────────

SCALE = 2.0


@pulse.kernel
def scaled_pulse(d, t):
    drive(d, square(16, AMP * SCALE), t)


def test_callee_reading_globals_and_closures():
    phase = 0.35

    @pulse.kernel
    def captured(d, t):
        t = shift_phase(t, phase)
        scaled_pulse(d, t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            captured(d, t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            t = shift_phase(t, 0.35)
            drive(d, square(16, AMP * 2.0), t)

    check(looped, unrolled, loops=1)


def test_callee_consuming_a_global_list_follows_python():
    waits = [8, 16, 24]

    @pulse.kernel
    def next_wait(d):
        wait(d, waits.pop())

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            next_wait(d)
            pulse16(d, t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for n in (24, 16, 8):
            wait(d, n)
            drive(d, square(16, AMP), t)

    ops = _trace(looped)
    # Each iteration reads a different value: the loop must unroll, and
    # trying to roll it must not consume an element.
    assert _kinds(ops)["scf.for"] == 0
    assert [op.attrs["duration"] for op in ops if op.kind == "pulse.wait"] == [24, 16, 8]
    assert waits == []
    waits[:] = [8, 16, 24]
    compiled = _compile(looped)
    assert waits == []
    assert _schedule(compiled) == _schedule(_compile(unrolled))


def test_callee_building_a_temporary_list_stays_rolled():
    @pulse.kernel
    def twice(d, t):
        amps = []
        amps.append(AMP)
        amps.append(-AMP)
        for a in amps:
            drive(d, square(16, a), t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(3):
            twice(d, t)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2):
            drive(d, square(16, AMP), t)
            drive(d, square(16, -AMP), t)

    check(looped, unrolled, loops=1)


def test_callee_mutating_a_global_inside_a_measured_branch_is_rejected():
    log = []

    @pulse.kernel
    def noted(d, t, r, rt):
        sync(d, r)
        if readout(r, square(64, 0.05), rt):
            log.append(1)
            drive(d, square(16, AMP), t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        for _ in range(2):
            noted(d, t, r, rt)

    with pytest.raises(CompilationError, match="measurement-conditioned branch may only change"):
        _trace(looped)


# ── Unsupported calls fail clearly ───────────────────────────────────


def test_unbounded_recursion_is_an_error():
    @pulse.kernel
    def forever(d, t):
        drive(d, square(16, AMP), t)
        forever(d, t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(2):
            forever(d, t)

    with pytest.raises(CompilationError, match="nest more than 64 deep"):
        _trace(looped)


def test_bounded_recursion_unrolls():
    @pulse.kernel
    def repeat(d, t, n):
        if n:
            drive(d, square(16, AMP), t)
            repeat(d, t, n - 1)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(2):
            repeat(d, t, 3)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        for _ in (0, 1, 2, 3, 4, 5):
            drive(d, square(16, AMP), t)

    ops, _ = check(looped, unrolled, loops=1)
    assert _loop_bodies(ops)[0].count("pulse.drive") == 3


def test_variadic_callee_is_an_error():
    with pytest.raises(TypeError, match=r"cannot take \*args"):

        @pulse.kernel
        def variadic(d, t, *amps):
            pass

    def spread(d, t, *amps):
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(2):
            spread(d, t, 0.1, 0.2)

    with pytest.raises(CompilationError, match=r"do not support \*args"):
        _trace(looped)


def test_keyword_and_default_arguments_bind():
    @pulse.kernel
    def shaped(d, t, amp=AMP, n=16):
        drive(d, square(n, amp), t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(2):
            shaped(d, t, n=32)

    @pulse.kernel
    def unrolled(q):
        d, t = get_drive_line(q)
        drive(d, square(32, AMP), t)
        drive(d, square(32, AMP), t)

    check(looped, unrolled, loops=1)


def test_lines_packed_in_a_tuple_are_rejected():
    @pulse.kernel
    def from_pair(pair):
        d, t = pair
        t = shift_phase(t, 0.5)
        drive(d, square(16, AMP), t)

    @pulse.kernel
    def looped(q):
        d, t = get_drive_line(q)
        for _ in range(2):
            from_pair((d, t))

    # The callee's rebinding cannot reach the caller's names, so the second
    # iteration reuses a consumed tone; the verifier rejects it.
    with pytest.raises(RuntimeError, match="carry it through iter_args"):
        _compile(looped)


def test_symbolic_kernel_parameters_reach_the_callee():
    @pulse.kernel
    def looped(q, amp: float, n: int):
        d, t = get_drive_line(q)
        for _ in range(3):
            shaped(d, t, amp, n)

    @pulse.kernel
    def shaped(d, t, amp, n):
        drive(d, square(n, amp), t)

    compiled = _compile_symbolic(looped)
    assert compiled.parameters == ("amp",)
    concrete = _evaluate(compiled, 0.1, 24)
    assert "scf.for" in concrete.mlir()
    assert _schedule(concrete) == [("drive", 0, 24 * k, 24) for k in range(3)]
