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
from cudaq_pulse.targets import Qubit, TargetDescription

from tests._handles import Q
from tests._ir import python_ir

FREQ_HZ = 5.0e9
# With H = amp X / 2 at 1 rad/ns per unit amplitude, 16 cycles (8 ns) at
# amplitude 0.05 rotates by 0.4 rad: small enough that sequences of
# pulses with different phases or frequencies end in distinct states.
AMP = 0.05
SIGNIFICANT = 1e-3
UNFUSED = ("verify", "fold_constants", "remove_unused", "fold_phases")
# The loops here are a few iterations long, which the default would unroll.
ROLLED = 0

# ── Call mechanics ───────────────────────────────────────────────────


def _qubits(n, qvec, dev=None):
    handles = Q if dev is None else dev.qudit_handles
    return [handles[0:n]] if qvec else [handles[i] for i in range(n)]


def _device(n):
    return pulse.device({"target": _target(n)})


def _trace(kernel, n=1, qvec=False, args=()):
    """The kernel's IR, as a list of ops."""
    return python_ir(kernel, *_qubits(n, qvec), *args, unroll_threshold=ROLLED).ops


def _compile(kernel, n=1, qvec=False, passes=None, args=()):
    """The compiled, scheduled kernel; *args* follow the qudits."""
    dev = _device(n)
    compiled = pulse.compile(kernel, passes=passes, device=dev, unroll_threshold=ROLLED)
    return compiled.lower(*_qubits(n, qvec, dev), *args)


def _compile_symbolic(kernel):
    """Compile a one-qudit kernel ahead of its other arguments."""
    return pulse.compile(kernel, device=_device(1), unroll_threshold=ROLLED)


def _evaluate(compiled, *values):
    """The compiled kernel, lowered at concrete values of its other arguments."""
    return compiled.lower(compiled.device.qudit_handles[0], *values)


def _simulate(kernel, n=1, qvec=False, args=(), levels=2, **kwargs):
    """The CPU simulation of the kernel, with two levels per qudit."""
    passes = (*UNFUSED, "merge_waits", "simplify_control_flow", "fuse_pulses")
    lowered = _compile(kernel, n, qvec, passes=passes, args=args)
    return pulse.simulate(lowered, levels=levels, **kwargs)


# ── Checks ───────────────────────────────────────────────────────────


def _target(n):
    return TargetDescription(
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
    events = [(e.kind, e.qubit, e.start_cycles, e.duration_cycles) for e in compiled.schedule]
    return sorted(events, key=lambda e: (e[2], e[1], e[0]))


def _length(compiled):
    return max((e.start_cycles + e.duration_cycles for e in compiled.schedule), default=0)


def _verify(compiled):
    """The scheduled module re-parses and passes the pulse verifier."""
    module = MLIRPipeline().parse_to_module(compiled.mlir())
    module.run_passes(["pulse-verify"])


def check(kernel, reference, *, loops, n=1, qvec=False, sim=True, args=(), **sim_kwargs):
    """Compare *kernel* with its hand-unrolled *reference*.

    *loops* is the number of ``scf.for`` ops *kernel* traces to; *args*
    follow the qudits in every call. Returns the kernel's ops and its
    compiled form for case-specific checks.
    """
    ops = _trace(kernel, n, qvec, args)
    assert _kinds(ops)["scf.for"] == loops
    assert _kinds(_trace(reference, n, qvec, args))["scf.for"] == 0

    compiled = _compile(kernel, n, qvec, args=args)
    expected = _compile(reference, n, qvec, args=args)
    _verify(compiled)
    _verify(expected)
    assert _length(compiled) == _length(expected)
    # fuse_pulses merges adjacent drives of the unrolled reference only, so
    # events are compared without it.
    compiled = _compile(kernel, n, qvec, passes=UNFUSED, args=args)
    unfused = _compile(reference, n, qvec, passes=UNFUSED, args=args)
    assert _schedule(compiled) == _schedule(unfused)

    if sim:
        result = _simulate(kernel, n, qvec, args, **sim_kwargs)
        want = _simulate(reference, n, qvec, args, **sim_kwargs)
        assert result.times_ns == want.times_ns
        np.testing.assert_allclose(result.density_matrix, want.density_matrix, atol=1e-9)
        got = sorted(result.branches, key=lambda b: b.levels)
        ref = sorted(want.branches, key=lambda b: b.levels)
        assert [b.levels for b in got] == [b.levels for b in ref]
        assert [b.probability for b in got] == pytest.approx([b.probability for b in ref])
    return ops, compiled


def _distinct(a, b, n=1, qvec=False):
    """The two kernels end in measurably different states."""
    rho_a = _simulate(a, n, qvec).density_matrix
    rho_b = _simulate(b, n, qvec).density_matrix
    assert np.abs(rho_a - rho_b).max() > SIGNIFICANT


# ── Callees ──────────────────────────────────────────────────────────


@pulse.kernel
def pulse16(d):
    drive(d, square(16, AMP))


@pulse.kernel
def idle(d, n):
    wait(d, n)


@pulse.kernel
def rotate(d, t, phase):
    t = shift_phase(t, phase)
    drive(d, square(16, AMP), tone=t)


@pulse.kernel
def chirp(d):
    shift_frequency(d.tone, 20.0e6)
    drive(d, square(16, AMP))


@pulse.kernel
def retune(d, freq):
    set_frequency(d.tone, freq)
    drive(d, square(16, AMP))


@pulse.kernel
def on_qubit(q):
    d = get_drive_line(q)
    drive(d, square(16, AMP))


# ── A callee in a rolled loop ────────────────────────────────────────


def test_callee_in_a_range_loop_stays_rolled():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(4):
            pulse16(d)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        drive(d, square(16, AMP))
        drive(d, square(16, AMP))
        drive(d, square(16, AMP))
        drive(d, square(16, AMP))

    ops, compiled = check(looped, unrolled, loops=1)
    assert _loop_bodies(ops) == [["pulse.square", "pulse.drive", "scf.yield"]]
    assert _kinds(ops)["pulse.drive"] == 1
    assert [e[2] for e in _schedule(compiled)] == [0, 16, 32, 48]
    assert _length(compiled) == 64


def test_plain_helper_and_kernel_callees_trace_alike():
    def helper(d, t):
        drive(d, square(16, AMP), tone=t)

    @pulse.kernel
    def with_kernel(q):
        d = get_drive_line(q)
        for _ in range(3):
            pulse16(d)

    @pulse.kernel
    def with_helper(q):
        d = get_drive_line(q)
        for _ in range(3):
            helper(d, d.tone)

    assert _compile(with_kernel).mlir().replace("with_kernel", "k") == _compile(
        with_helper
    ).mlir().replace("with_helper", "k")


# ── The loop variable as a callee argument ───────────────────────────


def test_loop_variable_as_a_wait_duration_unrolls():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for i in range(4):
            idle(d, 8 * (i + 1))
            pulse16(d)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for n in (8, 16, 24, 32):
            wait(d, n)
            drive(d, square(16, AMP))

    ops, compiled = check(looped, unrolled, loops=0)
    assert [op.attrs["duration"] for op in ops if op.kind == "pulse.wait"] == [8, 16, 24, 32]
    drives = [e[2] for e in _schedule(compiled) if e[0] == "drive"]
    assert drives == [8, 40, 80, 128]
    assert _length(compiled) == 144


def test_loop_variable_as_a_waveform_duration():
    @pulse.kernel
    def stretched(d, n):
        drive(d, square(n, AMP))

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for i in range(1, 4):
            stretched(d, 8 * i)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        drive(d, square(8, AMP))
        drive(d, square(16, AMP))
        drive(d, square(24, AMP))

    _, compiled = check(looped, unrolled, loops=0)
    assert [(e[2], e[3]) for e in _schedule(compiled)] == [(0, 8), (8, 16), (24, 24)]


def test_loop_variable_in_float_arithmetic():
    @pulse.kernel
    def scaled(d, t, k):
        drive(d, square(16, AMP * (1.0 + 0.5 * k)), tone=t)

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for i in range(4):
            rotate(d, d.tone, 0.3 * i + 0.1)
            scaled(d, d.tone, float(i))

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for i in range(4):
            shift_phase(d.tone, 0.3 * i + 0.1)
            drive(d, square(16, AMP))
            drive(d, square(16, AMP * (1.0 + 0.5 * i)))

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
        d0 = get_drive_line(qs[0])
        drive(d0, square(16, AMP))
        d1 = get_drive_line(qs[1])
        drive(d1, square(16, AMP))
        d2 = get_drive_line(qs[2])
        drive(d2, square(16, AMP))

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
            d = get_drive_line(q)
            lines.append(d)
            tones.append(d.tone)
        for _ in range(3):
            for i in range(len(lines)):
                rotate(lines[i], tones[i], 0.5 * (i + 1))

    @pulse.kernel
    def unrolled(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        for _k in (0, 1, 2):
            shift_phase(d0.tone, 0.5)
            drive(d0, square(16, AMP))
            shift_phase(d1.tone, 1.0)
            drive(d1, square(16, AMP))

    ops, _ = check(looped, unrolled, loops=1, n=2)
    for_end = next(op for op in ops if op.kind == "scf.for_end")
    # lines[1] and tones[1] hold d and d.tone, which carry them.
    assert for_end.attrs["iter_args"] == ["d", "d.tone", "tones[0]", "lines[0]"]


# ── Tones kept in a container across a rolled loop ───────────────────
#
# A line or tone kept in a list is not rebound when its tone is retuned:
# names read from the list later hold an old value of the channel. The loop
# carries the channel, not those names.

DETUNE_HZ = 20.0e6


def _retuned(q):
    return FREQ_HZ + 0.1e9 * q + DETUNE_HZ


def _rolled(compiled):
    """The compiled module keeps its loops and its retunes."""
    text = compiled.mlir()
    assert "scf.for" in text and "pulse.set_frequency" in text
    return text


def test_tones_retuned_in_a_list_stay_rolled():
    @pulse.kernel
    def looped(qs):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (_retuned(0), _retuned(1))):
            set_frequency(d.tone, f)
        for _ in range(3):
            for d in lines:
                drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        for _k in (0, 1, 2):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))

    @pulse.kernel
    def native(qs):
        lines = [get_drive_line(q) for q in qs]
        for _ in range(3):
            for d in lines:
                drive(d, square(16, AMP))

    _, compiled = check(looped, unrolled, loops=1, n=2, qvec=True)
    _rolled(_compile(looped, 2, True))
    assert _length(compiled) == 3 * 16
    _distinct(looped, native, n=2, qvec=True)


def test_several_tones_retuned_in_a_list():
    @pulse.kernel
    def looped(qs):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (_retuned(0), _retuned(1), _retuned(2))):
            set_frequency(d.tone, f)
        for _ in range(2):
            for d in lines:
                drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        d2 = get_drive_line(qs[2])
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        set_frequency(d2.tone, _retuned(2))
        for _k in (0, 1):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))
            drive(d2, square(16, AMP))

    check(looped, unrolled, loops=1, n=3, qvec=True)
    _rolled(_compile(looped, 3, True))


def test_tones_retuned_in_a_tuple():
    @pulse.kernel
    def looped(q0, q1):
        lines = (get_drive_line(q0), get_drive_line(q1))
        for d, f in zip(lines, (_retuned(0), _retuned(1))):
            set_frequency(d.tone, f)
        for _ in range(3):
            for d in lines:
                drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        for _k in (0, 1, 2):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))

    check(looped, unrolled, loops=1, n=2)
    _rolled(_compile(looped, 2))


def test_tones_retuned_in_a_dict():
    @pulse.kernel
    def looped(q0, q1):
        lines = {"a": get_drive_line(q0), "b": get_drive_line(q1)}
        set_frequency(lines["a"].tone, _retuned(0))
        set_frequency(lines["b"].tone, _retuned(1))
        for _ in range(3):
            for d in lines.values():
                drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        for _k in (0, 1, 2):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))

    check(looped, unrolled, loops=1, n=2)
    _rolled(_compile(looped, 2))


def test_tones_in_a_list_retuned_to_runtime_floats():
    @pulse.kernel
    def looped(qs, f0: float, f1: float):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (f0, f1)):
            set_frequency(d.tone, f)
        for _ in range(3):
            for d in lines:
                drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(qs, f0: float, f1: float):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        set_frequency(d0.tone, f0)
        set_frequency(d1.tone, f1)
        for _k in (0, 1, 2):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))

    args = (_retuned(0), _retuned(1))
    check(looped, unrolled, loops=1, n=2, qvec=True, args=args)
    _rolled(_compile(looped, 2, True, args=args))
    generic = pulse.compile(looped, device=_device(2))
    assert generic.parameters == ("f0", "f1")


def test_tones_retuned_in_a_list_through_nested_loops():
    @pulse.kernel
    def looped(qs):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (_retuned(0), _retuned(1))):
            set_frequency(d.tone, f)
        for _ in range(2):
            for _ in range(3):
                for d in lines:
                    drive(d, square(16, AMP))
            for d in lines:
                wait(d, 8)

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        for _j in (0, 1):
            for _k in (0, 1, 2):
                drive(d0, square(16, AMP))
                drive(d1, square(16, AMP))
            wait(d0, 8)
            wait(d1, 8)

    ops, _ = check(looped, unrolled, loops=2, n=2, qvec=True)
    outer, inner = _loop_bodies(ops)
    assert "scf.for" in outer and inner.count("pulse.drive") == 2
    assert _rolled(_compile(looped, 2, True)).count("scf.for") == 2


def test_tones_retuned_in_a_list_reach_a_self_scheduled_callee():
    @pulse.kernel(schedule="asap")
    def burst(lines):
        for _ in range(3):
            for d in lines:
                drive(d, square(16, AMP))

    @pulse.kernel
    def looped(qs):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (_retuned(0), _retuned(1))):
            set_frequency(d.tone, f)
        burst(lines)
        for _ in range(2):
            burst(lines)

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        for _k in (0, 1, 2, 3, 4, 5, 6, 7, 8):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))

    check(looped, unrolled, loops=3, n=2, qvec=True)
    _rolled(_compile(looped, 2, True))


def test_tones_retuned_in_a_list_iterated_in_another_order():
    # The names left by the retuning loop enter on qudit 1 and leave on
    # qudit 0; the body rebinds them before reading them.
    @pulse.kernel
    def looped(qs):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (_retuned(0), _retuned(1))):
            set_frequency(d.tone, f)
        for _ in range(3):
            for d in lines[::-1]:
                drive(d, square(16, AMP))
        drive(d, square(16, 2 * AMP))

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        for _k in (0, 1, 2):
            drive(d1, square(16, AMP))
            drive(d0, square(16, AMP))
        drive(d0, square(16, 2 * AMP))

    check(looped, unrolled, loops=1, n=2, qvec=True)
    _rolled(_compile(looped, 2, True))


def test_a_name_read_before_it_moves_to_another_channel_unrolls():
    # The first iteration drives qudit 1 through the names the retuning loop
    # left, the others qudit 0: the iterations differ, so the loop unrolls.
    @pulse.kernel
    def looped(qs):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (_retuned(0), _retuned(1))):
            set_frequency(d.tone, f)
        for _ in range(3):
            drive(d, square(16, 2 * AMP))
            for d in lines[::-1]:
                drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        drive(d1, square(16, 2 * AMP))
        for k in (0, 1, 2):
            if k:
                drive(d0, square(16, 2 * AMP))
            drive(d1, square(16, AMP))
            drive(d0, square(16, AMP))

    check(looped, unrolled, loops=0, n=2, qvec=True)


def test_a_name_read_before_it_is_rebound_on_its_channel_stays_rolled():
    @pulse.kernel
    def looped(qs):
        lines = [get_drive_line(q) for q in qs]
        for d, f in zip(lines, (_retuned(0), _retuned(1))):
            set_frequency(d.tone, f)
        for _ in range(3):
            drive(d, square(16, 2 * AMP))
            for d in lines:
                drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        set_frequency(d0.tone, _retuned(0))
        set_frequency(d1.tone, _retuned(1))
        for _k in (0, 1, 2):
            drive(d1, square(16, 2 * AMP))
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))

    check(looped, unrolled, loops=1, n=2, qvec=True)
    _rolled(_compile(looped, 2, True))


# ── Qudits named in a rolled loop body ───────────────────────────────
#
# A qudit is allocated before any loop, so a name a rolled body binds to
# one (``for q in qs``) is not carried: every iteration leaves the same
# qudit on it.


def _gathered(qs):
    """Drive each qudit and return its line, in a list built as it goes."""
    drives = []
    for q in qs:
        d = get_drive_line(q)
        drive(d, square(16, AMP))
        drives.append(d)
    return drives


def test_lines_gathered_in_a_list_in_a_rolled_body():
    @pulse.kernel
    def looped(qs):
        for _ in range(3):
            drives = []
            for q in qs:
                d = get_drive_line(q)
                drive(d, square(16, AMP))
                drives.append(d)
            sync(*drives)

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        d2 = get_drive_line(qs[2])
        for _k in (0, 1, 2):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))
            drive(d2, square(16, AMP))
            sync(d0, d1, d2)

    ops, _ = check(looped, unrolled, loops=1, n=3, qvec=True)
    for_end = next(op for op in ops if op.kind == "scf.for_end")
    assert "q" not in for_end.attrs["iter_args"]
    (body,) = _loop_bodies(ops)
    assert body.count("pulse.drive") == 3 and "pulse.get_drive_line" not in body


def test_lines_gathered_in_a_list_through_nested_loops():
    @pulse.kernel
    def looped(qs):
        for _ in range(2):
            for _ in range(3):
                drives = []
                for q in qs:
                    d = get_drive_line(q)
                    drive(d, square(16, AMP))
                    drives.append(d)
                sync(*drives)
            for q in qs[::-1]:
                d = get_drive_line(q)
                wait(d, 8)

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        for _j in (0, 1):
            for _k in (0, 1, 2):
                drive(d0, square(16, AMP))
                drive(d1, square(16, AMP))
                sync(d0, d1)
            wait(d1, 8)
            wait(d0, 8)

    ops, _ = check(looped, unrolled, loops=2, n=2, qvec=True)
    outer, inner = _loop_bodies(ops)
    assert "scf.for" in outer and inner.count("pulse.drive") == 2


def test_lines_gathered_by_a_helper_in_a_rolled_body():
    @pulse.kernel
    def looped(qs):
        for _ in range(3):
            sync(*_gathered(qs))

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        for _k in (0, 1, 2):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))
            sync(d0, d1)

    check(looped, unrolled, loops=1, n=2, qvec=True)


def test_a_line_list_and_qudit_from_a_rolled_body_are_used_after_it():
    # After the loop the list holds the lines the last iteration left, and
    # q the last qudit of the list.
    @pulse.kernel
    def looped(qs):
        for _ in range(3):
            drives = []
            for q in qs:
                d = get_drive_line(q)
                drive(d, square(16, AMP))
                drives.append(d)
            sync(*drives)
        drive(drives[0], square(16, 2 * AMP), tone=get_drive_line(qs[0]).tone)
        sync(*drives)
        d = get_drive_line(q)
        drive(d, square(32, AMP))

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        for _k in (0, 1, 2):
            drive(d0, square(16, AMP))
            drive(d1, square(16, AMP))
            sync(d0, d1)
        drive(d0, square(16, 2 * AMP))
        sync(d0, d1)
        drive(d1, square(32, AMP))

    _, compiled = check(looped, unrolled, loops=1, n=2, qvec=True)
    assert _length(compiled) == 3 * 16 + 16 + 32


def test_a_qudit_name_rebound_before_it_is_read_stays_rolled():
    @pulse.kernel
    def looped(qs):
        q = qs[0]
        for _ in range(3):
            q = qs[1]
            d = get_drive_line(q)
            drive(d, square(16, AMP))
        d = get_drive_line(q)
        drive(d, square(16, 2 * AMP))

    @pulse.kernel
    def unrolled(qs):
        d1 = get_drive_line(qs[1])
        for _k in (0, 1, 2):
            drive(d1, square(16, AMP))
        drive(d1, square(16, 2 * AMP))

    check(looped, unrolled, loops=1, n=2, qvec=True)


def test_a_qudit_name_read_before_it_is_rebound_unrolls():
    # The first iteration drives qudit 0, the others qudit 1.
    @pulse.kernel
    def looped(qs):
        q = qs[0]
        for _ in range(3):
            d = get_drive_line(q)
            drive(d, square(16, AMP))
            q = qs[1]

    @pulse.kernel
    def unrolled(qs):
        d0 = get_drive_line(qs[0])
        d1 = get_drive_line(qs[1])
        drive(d0, square(16, AMP))
        drive(d1, square(16, AMP))
        drive(d1, square(16, AMP))

    check(looped, unrolled, loops=0, n=2, qvec=True)


def test_a_qudit_name_keeps_its_value_after_a_loop_that_does_not_run():
    @pulse.kernel
    def looped(qs):
        q = qs[0]
        for _ in range(0):
            for q in qs:
                d = get_drive_line(q)
                drive(d, square(16, AMP))
        d = get_drive_line(q)
        drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(qs):
        d = get_drive_line(qs[0])
        drive(d, square(16, AMP))

    # The lines the body acquires are acquired before the loop, so the
    # simulation holds qudit 1 as well.
    _, compiled = check(looped, unrolled, loops=1, n=2, qvec=True, sim=False)
    assert _schedule(compiled) == [("drive", 0, 0, 16)]


def test_a_measured_branch_choosing_a_qudit_is_rejected():
    @pulse.kernel
    def looped(qs):
        r = get_readout_line(qs[0])
        for _ in range(3):
            m = readout(r, square(16, AMP))
            q = qs[0]
            if m:
                q = qs[1]
            d = get_drive_line(q)
            drive(d, square(16, AMP))

    with pytest.raises(CompilationError, match="'q' would depend on a measurement result"):
        _compile(looped, 2, True)


# ── Loops on both sides of the call ──────────────────────────────────


def test_callee_with_its_own_loop_nests_the_loops():
    @pulse.kernel
    def train(d):
        for _ in range(3):
            shift_phase(d.tone, 0.4)
            drive(d, square(16, AMP))

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(2):
            train(d)
            wait(d, 8)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1):
            for _ in (0, 1, 2):
                shift_phase(d.tone, 0.4)
                drive(d, square(16, AMP))
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
        d = get_drive_line(q)
        for _ in range(3):
            rotate(d, d.tone, 0.6)
            rotate(d, d.tone, -0.2)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            shift_phase(d.tone, 0.6)
            drive(d, square(16, AMP))
            shift_phase(d.tone, -0.2)
            drive(d, square(16, AMP))

    ops, compiled = check(looped, unrolled, loops=1)
    assert _kinds(ops)["pulse.drive"] == 2
    assert _length(compiled) == 96


def test_different_callees_in_one_body():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(3):
            pulse16(d)
            idle(d, 8)
            rotate(d, d.tone, 0.9)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            drive(d, square(16, AMP))
            wait(d, 8)
            shift_phase(d.tone, 0.9)
            drive(d, square(16, AMP))

    ops, compiled = check(looped, unrolled, loops=1)
    assert _loop_bodies(ops)[0].count("pulse.drive") == 2
    assert [e[2] for e in _schedule(compiled)] == [0, 16, 24, 40, 56, 64, 80, 96, 104]
    assert _length(compiled) == 120


# ── Branches around and inside the call ──────────────────────────────

ENABLED = True


def test_callee_under_a_compile_time_if():
    @pulse.kernel
    def flag(q):
        d = get_drive_line(q)
        for _ in range(3):
            if ENABLED:
                pulse16(d)
            else:
                idle(d, 16)

    @pulse.kernel
    def by_parity(q):
        d = get_drive_line(q)
        for i in range(4):
            if i % 2 == 0:
                rotate(d, d.tone, 0.7)
            else:
                idle(d, 8)

    @pulse.kernel
    def flag_unrolled(q):
        d = get_drive_line(q)
        drive(d, square(16, AMP))
        drive(d, square(16, AMP))
        drive(d, square(16, AMP))

    @pulse.kernel
    def parity_unrolled(q):
        d = get_drive_line(q)
        shift_phase(d.tone, 0.7)
        drive(d, square(16, AMP))
        wait(d, 8)
        shift_phase(d.tone, 0.7)
        drive(d, square(16, AMP))
        wait(d, 8)

    check(flag, flag_unrolled, loops=1)
    _, compiled = check(by_parity, parity_unrolled, loops=0)
    assert _length(compiled) == 48


_PI = math.pi / 8.0  # 16 cycles (8 ns) at this amplitude is a pi rotation


@pulse.kernel
def measure_and_reset(d, r):
    sync(d, r)
    bit = readout(r, square(64, 0.05))
    if bit:
        drive(d, square(16, _PI))


def test_measurement_feedback_inside_a_called_kernel():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(2):
            drive(d, square(16, _PI / 2))
            measure_and_reset(d, r)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        drive(d, square(16, _PI / 2))
        sync(d, r)
        bit = readout(r, square(64, 0.05))
        if bit:
            drive(d, square(16, _PI))
        drive(d, square(16, _PI / 2))
        sync(d, r)
        bit = readout(r, square(64, 0.05))
        if bit:
            drive(d, square(16, _PI))

    ops, compiled = check(looped, unrolled, loops=1)
    body = _loop_bodies(ops)[0]
    assert body.count("pulse.readout") == 1 and body.count("scf.if") == 1
    readouts = [e[2] for e in _schedule(compiled) if e[0] == "readout"]
    assert readouts == [16, 112]
    result = _simulate(looped)
    # Each round resets to |0>: the ensemble ends in the ground state.
    np.testing.assert_allclose(result.density_matrix, np.diag([1.0, 0.0]), atol=1e-7)
    assert [m.iteration for m in result.measurements] == [(0,), (1,)]
    assert sorted(b.levels for b in result.branches) == [(0, 0), (0, 1), (1, 0), (1, 1)]


@pulse.kernel
def measure(r, rt):
    return readout(r, square(64, 0.05), tone=rt)


def test_measurement_returned_by_a_callee_controls_the_loop_body():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(2):
            drive(d, square(16, _PI / 2))
            sync(d, r)
            if measure(r, r.tone):
                pulse16(d)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in (0, 1):
            drive(d, square(16, _PI / 2))
            sync(d, r)
            if readout(r, square(64, 0.05)):
                drive(d, square(16, AMP))

    check(looped, unrolled, loops=1)


# ── Tone state carried across iterations ─────────────────────────────


def test_shift_phase_in_a_callee_accumulates_across_iterations():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(4):
            rotate(d, d.tone, 0.8)
        drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2, 3):
            shift_phase(d.tone, 0.8)
            drive(d, square(16, AMP))
        drive(d, square(16, AMP))

    @pulse.kernel
    def without_phase(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2, 3, 4):
            drive(d, square(16, AMP))

    ops, _ = check(looped, unrolled, loops=1)
    for_end = next(op for op in ops if op.kind == "scf.for_end")
    assert "d.tone" in for_end.attrs["iter_args"]
    _distinct(looped, without_phase)


def test_shift_frequency_in_a_callee_accumulates_across_iterations():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(3):
            chirp(d)
        drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            shift_frequency(d.tone, 20.0e6)
            drive(d, square(16, AMP))
        drive(d, square(16, AMP))

    @pulse.kernel
    def one_shift(q):
        d = get_drive_line(q)
        shift_frequency(d.tone, 20.0e6)
        for _ in (0, 1, 2, 3):
            drive(d, square(16, AMP))

    check(looped, unrolled, loops=1)
    _distinct(looped, one_shift)


def test_set_frequency_in_a_callee_holds_after_the_loop():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(3):
            retune(d, FREQ_HZ + 40.0e6)
            chirp(d)
        drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            set_frequency(d.tone, FREQ_HZ + 40.0e6)
            drive(d, square(16, AMP))
            shift_frequency(d.tone, 20.0e6)
            drive(d, square(16, AMP))
        drive(d, square(16, AMP))

    check(looped, unrolled, loops=1)


def test_a_callee_that_acquires_its_own_line_each_iteration():
    @pulse.kernel
    def fresh(q):
        d = get_drive_line(q)
        shift_phase(d.tone, 0.5)
        drive(d, square(16, AMP))

    @pulse.kernel
    def looped(q):
        for _ in range(2):
            fresh(q)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        shift_phase(d.tone, 0.5)
        drive(d, square(16, AMP))
        d = get_drive_line(q)
        shift_phase(d.tone, 0.5)
        drive(d, square(16, AMP))

    # get_drive_line returns the qudit's one drive channel: it is acquired
    # once, before the loop, and the pulses play one after the other.
    ops, compiled = check(looped, unrolled, loops=1)
    assert _kinds(ops)["pulse.get_drive_line"] == 1
    assert _schedule(compiled) == [("drive", 0, 0, 16), ("drive", 0, 16, 16)]


# ── sync across lines ────────────────────────────────────────────────


@pulse.kernel
def staggered(d0, d1):
    drive(d0, square(32, AMP))
    sync(d0, d1)
    drive(d1, square(16, AMP))


def test_callee_that_syncs_two_lines():
    @pulse.kernel
    def looped(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        for _ in range(3):
            staggered(d0, d1)

    @pulse.kernel
    def unrolled(q0, q1):
        d0 = get_drive_line(q0)
        d1 = get_drive_line(q1)
        for _ in (0, 1, 2):
            drive(d0, square(32, AMP))
            sync(d0, d1)
            drive(d1, square(16, AMP))

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
def inner_b(d):
    shift_phase(d.tone, 0.3)
    drive(d, square(16, AMP))


@pulse.kernel
def middle_a(d):
    inner_b(d)
    wait(d, 8)
    inner_b(d)


def test_kernels_nested_three_deep():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(3):
            middle_a(d)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            shift_phase(d.tone, 0.3)
            drive(d, square(16, AMP))
            wait(d, 8)
            shift_phase(d.tone, 0.3)
            drive(d, square(16, AMP))

    ops, compiled = check(looped, unrolled, loops=1)
    assert _kinds(ops)["pulse.drive"] == 2
    assert _length(compiled) == 120


# ── Return values ────────────────────────────────────────────────────


def test_callee_returning_a_tone():
    @pulse.kernel
    def advance(d, t):
        t = shift_phase(t, 0.45)
        drive(d, square(16, AMP), tone=t)
        return t

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        t = d.tone
        for _ in range(3):
            t = advance(d, t)
        drive(d, square(16, AMP), tone=t)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            shift_phase(d.tone, 0.45)
            drive(d, square(16, AMP))
        drive(d, square(16, AMP))

    check(looped, unrolled, loops=1)


def test_callee_returning_a_python_value():
    @pulse.kernel
    def cycles(k):
        return 8 * k

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(3):
            wait(d, cycles(2))
            pulse16(d)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            wait(d, 16)
            drive(d, square(16, AMP))

    _, compiled = check(looped, unrolled, loops=1)
    assert _length(compiled) == 96


# ── Trip counts ──────────────────────────────────────────────────────


def test_zero_iterations():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(0):
            rotate(d, d.tone, 0.5)
        drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        drive(d, square(16, AMP))

    ops, compiled = check(looped, unrolled, loops=1)
    loop = next(op for op in ops if op.kind == "scf.for")
    assert loop.attrs["ub"] == 0
    assert _schedule(compiled) == [("drive", 0, 0, 16)]


def test_zero_iterations_with_the_loop_variable_read():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for i in range(0):
            idle(d, i + 8)
        drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        drive(d, square(16, AMP))

    check(looped, unrolled, loops=0)


def test_one_iteration():
    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(1):
            rotate(d, d.tone, 0.5)
        drive(d, square(16, AMP))

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        shift_phase(d.tone, 0.5)
        drive(d, square(16, AMP))
        drive(d, square(16, AMP))

    _, compiled = check(looped, unrolled, loops=1)
    assert _schedule(compiled) == [("drive", 0, 0, 16), ("drive", 0, 16, 16)]


# ── Globals and closures ─────────────────────────────────────────────

SCALE = 2.0


@pulse.kernel
def scaled_pulse(d):
    drive(d, square(16, AMP * SCALE))


def test_callee_reading_globals_and_closures():
    phase = 0.35

    @pulse.kernel
    def captured(d):
        shift_phase(d.tone, phase)
        scaled_pulse(d)

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(3):
            captured(d)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            shift_phase(d.tone, 0.35)
            drive(d, square(16, AMP * 2.0))

    check(looped, unrolled, loops=1)


def test_callee_consuming_a_global_list_follows_python():
    waits = [8, 16, 24]

    @pulse.kernel
    def next_wait(d):
        wait(d, waits.pop())

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(3):
            next_wait(d)
            pulse16(d)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for n in (24, 16, 8):
            wait(d, n)
            drive(d, square(16, AMP))

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
    def twice(d):
        amps = []
        amps.append(AMP)
        amps.append(-AMP)
        for a in amps:
            drive(d, square(16, a))

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(3):
            twice(d)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2):
            drive(d, square(16, AMP))
            drive(d, square(16, -AMP))

    check(looped, unrolled, loops=1)


def test_callee_mutating_a_global_inside_a_measured_branch_is_rejected():
    log = []

    @pulse.kernel
    def noted(d, r):
        sync(d, r)
        if readout(r, square(64, 0.05)):
            log.append(1)
            drive(d, square(16, AMP))

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        for _ in range(2):
            noted(d, r)

    with pytest.raises(CompilationError, match="measurement-conditioned branch may only change"):
        _trace(looped)


# ── Unsupported calls fail clearly ───────────────────────────────────


def test_unbounded_recursion_is_an_error():
    @pulse.kernel
    def forever(d, t):
        drive(d, square(16, AMP), tone=t)
        forever(d, t)

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(2):
            forever(d, d.tone)

    with pytest.raises(CompilationError, match="nest more than 64 deep"):
        _trace(looped)


def test_bounded_recursion_unrolls():
    @pulse.kernel
    def repeat(d, t, n):
        if n:
            drive(d, square(16, AMP), tone=t)
            repeat(d, t, n - 1)

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(2):
            repeat(d, d.tone, 3)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2, 3, 4, 5):
            drive(d, square(16, AMP))

    ops, _ = check(looped, unrolled, loops=1)
    assert _loop_bodies(ops)[0].count("pulse.drive") == 3


def test_variadic_kernel_is_an_error_and_variadic_helper_is_inlined():
    with pytest.raises(TypeError, match=r"cannot take \*args"):

        @pulse.kernel
        def variadic(d, t, *amps):
            pass

    def spread(d, t, *amps):
        for amp in amps:
            drive(d, square(16, amp), tone=t)

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(2):
            spread(d, d.tone, AMP, AMP)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        for _ in (0, 1, 2, 3):
            drive(d, square(16, AMP))

    check(looped, unrolled, loops=1)


def test_keyword_and_default_arguments_bind():
    @pulse.kernel
    def shaped(d, t, amp=AMP, n=16):
        drive(d, square(n, amp), tone=t)

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(2):
            shaped(d, d.tone, n=32)

    @pulse.kernel
    def unrolled(q):
        d = get_drive_line(q)
        drive(d, square(32, AMP))
        drive(d, square(32, AMP))

    check(looped, unrolled, loops=1)


def test_lines_packed_in_a_tuple_reach_the_callee():
    @pulse.kernel
    def from_pair(pair):
        d, t = pair
        t = shift_phase(t, 0.5)
        drive(d, square(16, AMP), tone=t)

    @pulse.kernel
    def looped(q):
        d = get_drive_line(q)
        for _ in range(2):
            from_pair((d, d.tone))

    # The callee advances the qudit's drive channel, which the loop carries,
    # so the second iteration plays after the first.
    assert [e[2] for e in _schedule(_compile(looped))] == [0, 16]


def test_symbolic_kernel_parameters_reach_the_callee():
    @pulse.kernel
    def looped(q, amp: float, n: int):
        d = get_drive_line(q)
        for _ in range(3):
            shaped(d, d.tone, amp, n)

    @pulse.kernel
    def shaped(d, t, amp, n):
        drive(d, square(n, amp), tone=t)

    compiled = _compile_symbolic(looped)
    assert compiled.parameters == ("amp",)
    concrete = _evaluate(compiled, 0.1, 24)
    assert "scf.for" in concrete.mlir()
    assert _schedule(concrete) == [("drive", 0, 24 * k, 24) for k in range(3)]
