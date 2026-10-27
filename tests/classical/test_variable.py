# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Waits whose length comes from a classical answer: barriers and epochs."""

from __future__ import annotations

import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import Bits, Float, Integer
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._handles import Q
from tests.property_settings import examples

POLICIES = ("asap", "alap", "rcp", "alap_rcp")


@pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=50)
def next_tau(x):
    return 2.0 * x


@pulse.classical(args=(Bits(1),), results=(Integer(),), max_latency_ns=50)
def tau_index(bits):
    return 10 * bits[0]


def _mlir(kernel, *args, **options):
    return pulse.compile(kernel, **options).lower(*args).mlir()


@pulse.kernel
def single(q, tau0: float):
    d, t = get_drive_line(q)
    tau = next_tau.submit(tau0).result()
    drive(d, square(40, 0.1), t)
    wait(d, tau, bounds=(16, 4000))
    drive(d, square(40, 0.1), t)


def test_a_wait_from_an_answer_is_a_variable_wait():
    text = _mlir(single, Q[0], 10.0)
    assert re.search(
        r"pulse.wait .*classical.variable, .*pulse.max_vtu = 4000 : i64, "
        r"pulse.min_vtu = 16 : i64, pulse.variable",
        text,
    )
    # Rounded to the nearest cycle: (x + 0.5) cast to an integer.
    assert "arith.addf" in text and "arith.fptosi" in text


def test_integer_answers_and_no_bounds():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        n = tau_index.submit([readout(r, square(100, 0.05), rt)]).result()
        wait(d, n)
        drive(d, square(40, 0.1), t)

    text = _mlir(k, Q[0])
    assert re.search(r"classical.variable, .*pulse.min_vtu = 0 : i64, pulse.variable", text)
    assert "pulse.max_vtu" not in text


def test_bounds_errors():
    def bad_bounds(q):
        d, t = get_drive_line(q)
        wait(d, next_tau.submit(1.0).result(), bounds=(10, 5))

    def fixed_outside_bounds(q):
        d, t = get_drive_line(q)
        wait(d, 30, bounds=(0, 20))

    def not_a_pair(q):
        d, t = get_drive_line(q)
        wait(d, next_tau.submit(1.0).result(), bounds=5)

    for body, message in (
        (bad_bounds, "lowest <= highest"),
        (fixed_outside_bounds, "outside its bounds"),
        (not_a_pair, r"takes \(lowest, highest\)"),
    ):
        with pytest.raises(pulse.CompilationError, match=message):
            pulse.compile(pulse.kernel(body)).lower(Q[0])

    # A concrete length within its bounds is a plain wait.
    @pulse.kernel
    def fixed_with_bounds(q):
        d, t = get_drive_line(q)
        wait(d, 10, bounds=(0, 20))

    assert "pulse.variable" not in pulse.compile(fixed_with_bounds).lower(Q[0]).mlir()


def test_a_variable_wait_cannot_be_in_a_block_or_a_branch_arm():
    @pulse.kernel(schedule="asap")
    def inner(d, x):
        wait(d, x)

    @pulse.kernel
    def in_block(q):
        d, t = get_drive_line(q)
        inner(d, next_tau.submit(1.0).result())

    @pulse.kernel
    def in_branch(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        x = next_tau.submit(1.0).result()
        if readout(r, square(100, 0.05), rt):
            wait(d, x)

    with pytest.raises(RuntimeError, match="cannot be inside a pulse.schedule_block"):
        pulse.compile(in_block).lower(Q[0])
    with pytest.raises(RuntimeError, match="cannot be inside one arm of an scf.if"):
        pulse.compile(in_branch).lower(Q[0])
    # Flattened, the kernel is no block, and the wait is allowed.
    pulse.compile(in_block, flatten_schedules=True).lower(Q[0])


@pytest.mark.parametrize("policy", POLICIES)
def test_epochs_and_their_shift_ranges(policy):
    @pulse.kernel
    def k(q, p):
        d, t = get_drive_line(q)
        e, u = get_drive_line(p)
        drive(e, square(400, 0.01), u)
        a = next_tau.submit(1.0).result()
        wait(d, a, bounds=(8, 100))
        drive(d, square(40, 0.1), t)
        b = next_tau.submit(a).result()
        wait(d, b)
        drive(e, square(40, 0.1), u)

    low = pulse.compile(k, schedule=policy).lower(Q[0], Q[1])
    epochs = low.epochs()
    assert [e.opened_by for e in epochs] == ["start", "wait", "wait"]
    assert epochs[1].max_shift_vtu == 92 and epochs[2].max_shift_vtu is None
    # The first barrier waits for the 400-cycle drive on the other line.
    assert epochs[1].start_vtu == 400 + 8
    calls = low.classical_report()
    assert [c.epoch for c in calls] == [0, 1]


def test_a_program_without_variable_waits_has_one_epoch():
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, square(40, next_tau.submit(1.0).result()), t)

    assert pulse.compile(k).lower(Q[0]).epochs() == [pulse.compiler.Epoch(0, "start", -1, 0, 0, 0)]


# ── Epoch invariants ─────────────────────────────────────────────────────────

_STEP = st.tuples(st.sampled_from(("drive", "readout", "variable_wait", "call")), st.integers(0, 2))


def _program(steps):
    @pulse.kernel
    def k(q0, q1, q2):
        qs = (q0, q1, q2)
        x = 1.0
        for kind, q in steps:
            d, t = get_drive_line(qs[q])
            r, rt = get_readout_line(qs[q])
            if kind == "drive":
                drive(d, square(30, 0.1), t)
            elif kind == "readout":
                readout(r, square(50, 0.05), rt)
            elif kind == "call":
                x = next_tau.submit(x).result()
            elif kind == "variable_wait":
                wait(d, next_tau.submit(x).result(), bounds=(4, None))

    return k


@settings(max_examples=examples(40), deadline=None)
@given(steps=st.lists(_STEP, min_size=1, max_size=20), policy=st.sampled_from(POLICIES))
def test_epoch_invariants(steps, policy):
    low = pulse.compile(_program(steps), schedule=policy).lower(Q[0], Q[1], Q[2])
    events = low.schedule
    barriers = [
        i
        for i, e in enumerate(events)
        if e.kind == "wait" and e.link is None and e.duration_vtu == 4
    ]
    variable = sum(1 for kind, _ in steps if kind == "variable_wait")
    assert len(low.epochs()) == variable + 1
    for i in barriers:
        start = events[i].start_vtu
        end = start + events[i].duration_vtu
        for j, e in enumerate(events):
            if j < i:
                assert e.start_vtu + e.duration_vtu <= start
            elif j > i:
                assert e.start_vtu >= end


def test_a_wait_from_a_float_parameter_beside_calls_is_no_barrier():
    @pulse.kernel
    def k(q, p, tau: float):
        d, t = get_drive_line(q)
        e, u = get_drive_line(p)
        drive(e, square(400, 0.01), u)
        wait(d, tau, bounds=(0, 100))
        drive(d, square(40, 0.1), t)
        a = next_tau.submit(1.0).result()
        wait(d, a, bounds=(8, 100))
        drive(d, square(40, 0.1), t)

    compiled = pulse.compile(k, schedule="asap")
    low = compiled.lower(Q[0], Q[1], 10.4)
    waits = [line for line in low.mlir().splitlines() if "pulse.wait" in line]
    assert len(waits) == 2
    # The parameter's wait: bound and rounded, no barrier; the answer's: a barrier.
    assert "classical.variable" not in waits[0] and "duration_vtu = 10 : i64" in waits[0]
    assert "classical.variable" in waits[1] and "pulse.min_vtu = 8" in waits[1]
    drives = [e for e in low.schedule if e.kind == "drive" and e.line == 0]
    assert drives[0].start == 10
    assert [e.opened_by for e in low.epochs()] == ["start", "wait"]
    # One compilation for every length; a length outside its bounds is a call error.
    compiled.lower(Q[0], Q[1], 50.0)
    assert compiled.cache_info().compiled == 1
    with pytest.raises(ValueError, match="bounds"):
        compiled.lower(Q[0], Q[1], 120.0)


@pulse.kernel
def offset_by_tau(q, p, tau: float):
    d, t = get_drive_line(q)
    e, u = get_drive_line(p)
    drive(e, square(400, 0.01), u)
    drive(d, square(40, 0.1), t)
    a = next_tau.submit(10.0).result()
    wait(d, a + tau, bounds=(8, 1000))
    drive(d, square(40, 0.1), t)


def test_a_wait_from_an_answer_and_a_float_compiles_once():
    compiled = pulse.compile(offset_by_tau)
    # Each call checks the float part, the length with its answer at 0,
    # against the bounds; the answer adds to it at run time.
    for k in range(1000):
        low = compiled.lower(Q[0], Q[1], 8.0 + 0.5 * k)
    assert compiled.cache_info().compiled == 1
    # The answer keeps the wait a barrier that opens an epoch, at its lowest
    # length until it is known.
    assert [e.opened_by for e in low.epochs()] == ["start", "wait"]
    (wait_event,) = [e for e in low.schedule if e.kind == "wait" and e.link is None]
    assert wait_event.start == 400 and wait_event.duration_vtu == 8
    assert "%arg0: f64" in low.generic.mlir()


@pytest.mark.parametrize(
    ("tau", "message"),
    [
        (1500.0, r"next_tau\[0\] \+ tau \(its answers at 0\) = 1500.0 .* outside its bounds"),
        (2.0, r"= 2.0 rounds to 2 clock cycles, outside its bounds \(8, 1000\)"),
        (float("nan"), "must be a finite number"),
    ],
)
def test_the_float_part_of_a_wait_from_an_answer_is_checked_at_call_time(tau, message):
    compiled = pulse.compile(offset_by_tau)
    with pytest.raises(ValueError, match=message):
        compiled.lower(Q[0], Q[1], tau)
    compiled.lower(Q[0], Q[1], 30.0)
    assert compiled.cache_info().compiled == 1


@pytest.mark.parametrize("tau", [30.4, 980.0])
def test_a_wait_from_an_answer_and_a_float_runs_their_sum_clamped(tau):
    from cudaq_pulse.targets import Qubit, TargetDescription

    qubits = {i: Qubit(i, 5e9 + 1e8 * i, -2e8, 2.0, 0.0) for i in range(2)}
    dev = pulse.device({"target": TargetDescription(name="decay", qubits=qubits)})
    q, p = dev.qudit_handles
    length = min(max(round(2.0 * 10.0 + tau), 8), 1000)

    @pulse.kernel
    def fixed(q, p):
        d, t = get_drive_line(q)
        e, u = get_drive_line(p)
        drive(e, square(400, 0.01), u)
        drive(d, square(40, 0.1), t)
        sync(d, e)
        wait(d, length)
        drive(d, square(40, 0.1), t)

    mixed = pulse.simulate(pulse.compile(offset_by_tau, device=dev).lower(q, p, tau))
    reference = pulse.simulate(pulse.compile(fixed, device=dev).lower(q, p))
    np.testing.assert_allclose(mixed.density_matrix, reference.density_matrix, atol=1e-9)


@pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=50, latency_mode="blocking")
def blocking_tau(x):
    return x


@pytest.mark.parametrize("barrier", ["variable", "blocking", "variable_in_loop"])
def test_pulses_either_side_of_a_barrier_are_not_fused(barrier):
    @pulse.kernel
    def k(q, p):
        d, t = get_drive_line(q)
        e, u = get_drive_line(p)
        drive(e, square(400, 0.05), u)
        if barrier == "variable":
            wait(d, next_tau.submit(10.0).result(), bounds=(8, 100))
        elif barrier == "blocking":
            drive(d, square(40, blocking_tau.submit(0.1).result()), t)
        else:
            for _ in range(2):
                wait(d, next_tau.submit(10.0).result(), bounds=(8, 100))
        drive(e, square(30, 0.05), u)

    low = pulse.compile(k, unroll_threshold=0).lower(Q[0], Q[1])
    spectator = [e for e in low.schedule if e.kind == "drive" and e.qubit == 1]
    assert [e.end - e.start for e in spectator] == [400, 30]
    assert spectator[1].start >= low.epochs()[-1].start_vtu
