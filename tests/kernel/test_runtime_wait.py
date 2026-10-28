# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A wait whose length a float parameter sets compiles once.

Each program is compiled twice from one function: with its lengths ``float``
(one module, rescheduled per call) and with them static (specialized on
every value). The two must schedule every value alike.
"""

from __future__ import annotations

import types

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q

pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")

POLICIES = ("alap", "asap", "rcp", "alap_rcp")


def _static_twin(fn: types.FunctionType) -> pulse.Kernel:
    """*fn* compiled for every value of its float parameters, as a float
    parameter the program needs while compiling is."""
    twin = pulse.kernel(fn)
    twin._promote(set(twin._split[1]))
    return twin


def _events(lowered) -> list[tuple]:
    return [(e.kind, e.line, e.start_cycles, e.duration_cycles) for e in lowered.schedule]


def _assert_same_schedules(fn, qudits, sweeps, policies=POLICIES):
    runtime = pulse.kernel(fn)
    static = _static_twin(fn)
    for policy in policies:
        compiled = pulse.compile(runtime, schedule=policy)
        reference = pulse.compile(static, schedule=policy)
        for values in sweeps:
            got = _events(compiled.lower(*qudits, *values))
            want = _events(reference.lower(*qudits, *values))
            assert got == want, (policy, values)
        assert compiled.cache_info().compiled == 1, policy
    assert runtime.runtime_parameters == static.static_parameters[len(qudits) :]


TAUS = [(0.0,), (1.0,), (7.0,), (40.0,), (333.0,), (2000.0,)]


def t1(q, tau: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, square(40, 0.5), t)
    wait(d, tau)
    sync(d, r)
    readout(r, square(200, 0.05), rt)


def ramsey(q, tau: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, gaussian(40, 0.25, 10.0), t)
    wait(d, tau)
    drive(d, gaussian(40, 0.25, 10.0), t)
    sync(d, r)
    readout(r, square(200, 0.05), rt)


def echo(q, tau: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    half = tau / 2
    drive(d, gaussian(40, 0.25, 10.0), t)
    wait(d, half)
    drive(d, gaussian(40, 0.5, 10.0), t)
    wait(d, half)
    drive(d, gaussian(40, 0.25, 10.0), t)
    sync(d, r)
    readout(r, square(200, 0.05), rt)


def parallel(a, b, tau_a: float, tau_b: float):
    da, ta = get_drive_line(a)
    db, tb = get_drive_line(b)
    ra, rta = get_readout_line(a)
    drive(da, square(40, 0.5), ta)
    drive(db, square(24, 0.5), tb)
    wait(da, tau_a)
    wait(db, tau_b)
    drive(db, square(16, 0.5), tb)
    sync(da, db, ra)
    readout(ra, square(100, 0.05), rta)


def dd(q, tau: float):
    d, t = get_drive_line(q)
    drive(d, gaussian(40, 0.25, 10.0), t)
    for _ in range(4):
        wait(d, tau)
        drive(d, gaussian(40, 0.5, 10.0), t)
        wait(d, tau)
    drive(d, gaussian(40, 0.25, 10.0), t)


def cpmg(q, tau: float):
    d, t = get_drive_line(q)
    half = tau / 2
    drive(d, gaussian(40, 0.25, 10.0), t)
    wait(d, half)
    for _ in range(6):
        drive(d, gaussian(40, 0.5, 10.0), t)
        wait(d, tau)
    drive(d, gaussian(40, 0.5, 10.0), t)
    wait(d, half)
    drive(d, gaussian(40, 0.25, 10.0), t)


def stepped(q, tau: float):
    d, t = get_drive_line(q)
    for i in range(3):
        drive(d, square(8, 0.5), t)
        wait(d, tau + 10 * i)


def _idle(line, length):
    wait(line, length)


@pulse.kernel(schedule="asap")
def _block(line, tone, length: float):
    drive(line, square(16, 0.5), tone)
    wait(line, length)
    drive(line, square(16, 0.5), tone)


def helpers(a, b, tau: float):
    da, ta = get_drive_line(a)
    db, tb = get_drive_line(b)
    drive(da, square(40, 0.5), ta)
    _idle(da, tau)
    _block(db, tb, 2 * tau)
    sync(da, db)
    drive(da, square(40, 0.5), ta)


def branch(q, tau: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    m = readout(r, square(100, 0.05), rt)
    sync(d, r)
    if m:
        wait(d, tau)
        drive(d, square(40, 0.5), t)
    drive(d, square(40, 0.5), t)


def mixed(q, amp: float, phase: float, tau: float, total: float):
    d, t = get_drive_line(q)
    shift_phase(t, phase)
    drive(d, square(40, amp), t)
    wait(d, 2 * tau + 40)
    drive(d, square(40, -amp), t)
    wait(d, total - tau)
    drive(d, square(40, amp / 2), t)


@pytest.mark.parametrize("fn", [t1, ramsey, dd, stepped], ids=lambda f: f.__name__)
def test_one_wait_length_schedules_as_its_specializations(fn):
    _assert_same_schedules(fn, [Q[0]], TAUS)


def test_an_echo_halves_its_length_at_run_time():
    _assert_same_schedules(echo, [Q[0]], [(0.0,), (2.0,), (80.0,), (4000.0,)])


def test_a_cpmg_train_uses_one_half_length_around_a_rolled_loop():
    _assert_same_schedules(cpmg, [Q[0]], [(0.0,), (2.0,), (80.0,), (4000.0,)])
    assert "scf.for" in pulse.compile(pulse.kernel(cpmg)).lower(Q[0], 80.0).generic.mlir()


def test_two_lines_waiting_different_lengths_sync_on_the_longer():
    sweeps = [(0.0, 500.0), (500.0, 0.0), (250.0, 250.0), (1000.0, 3.0), (3.0, 1000.0)]
    _assert_same_schedules(parallel, [Q[0], Q[1]], sweeps)


def test_two_lines_waiting_one_length_wait_in_parallel():
    def same(a, b, tau: float):
        parallel(a, b, tau, tau)

    sweeps = [(0.0,), (16.0,), (900.0,)]
    _assert_same_schedules(same, [Q[0], Q[1]], sweeps)
    lowered = pulse.compile(pulse.kernel(same)).lower(Q[0], Q[1], 900.0)
    # Concurrent, not one after the other as behind a barrier.
    waits = [e for e in lowered.schedule if e.kind == "wait"]
    assert [(w.start_cycles, w.duration_cycles) for w in waits] == [(40, 900), (24, 900)]


def test_helpers_and_called_kernels_take_the_length():
    _assert_same_schedules(helpers, [Q[0], Q[1]], TAUS, policies=("alap", "asap"))


def test_a_wait_in_a_measurement_branch_takes_the_length():
    _assert_same_schedules(branch, [Q[0]], TAUS, policies=("alap", "asap"))


def test_lengths_mix_with_other_float_parameters():
    sweeps = [(0.1, 0.0, 0.0, 0.0), (0.3, 1.5, 20.0, 100.0), (0.2, -0.5, 300.0, 900.0)]
    _assert_same_schedules(mixed, [Q[0]], sweeps)
    lowered = pulse.compile(pulse.kernel(mixed)).lower(Q[0], 0.3, 1.5, 20.0, 100.0)
    assert lowered.arguments == (0.3, 1.5, 20.0, 100.0)


def test_a_sweep_compiles_once_and_never_evicts():
    kernel = pulse.kernel(t1, cache_size=4)
    for i in range(200):
        kernel.lower(Q[0], 13.0 * i)
    info = kernel.cache_info()
    assert (info.compiled, info.evicted) == (1, 0)
    assert kernel.runtime_parameters == ("tau",)


@pytest.mark.parametrize(
    ("tau", "cycles"),
    [(0.0, 0), (0.49, 0), (-0.5, 0), (2.5, 3), (99.5, 100), (100.4999, 100), (1e6 + 0.5, 1000001)],
)
def test_a_length_rounds_to_the_nearest_cycle_ties_up(tau, cycles):
    lowered = pulse.compile(pulse.kernel(t1)).lower(Q[0], tau)
    wait_event = next(e for e in lowered.schedule if e.kind == "wait")
    readout_event = lowered.schedule[-1]
    assert (wait_event.start_cycles, wait_event.duration_cycles) == (40, cycles)
    assert readout_event.start_cycles == 40 + cycles


def test_the_module_rounds_and_the_call_binds():
    lowered = pulse.compile(pulse.kernel(t1)).lower(Q[0], 7.4)
    generic = lowered.generic.mlir()
    assert "%arg0: f64" in generic
    assert "arith.fptosi" in generic
    assert "pulse.variable_wait" in generic and "pulse.min_cycles = 0" in generic
    bound = lowered.mlir()
    assert "%arg0" not in bound
    assert "duration_cycles = 7 : i64, pulse.min_cycles = 0" in bound


def test_without_values_a_wait_stands_at_its_lowest_length():
    @pulse.kernel
    def k(q, tau: float):
        d, t = get_drive_line(q)
        wait(d, tau, bounds=(100, None))
        drive(d, square(40, 0.5), t)

    compiled = pulse.compile(k, device=Q[0].device)
    generic = compiled.lower(Q[0], 250.0).generic
    assert _events(generic) == [("wait", 0, 0, 100), ("drive", 0, 100, 40)]
    assert _events(compiled.lower(Q[0], 250.0)) == [("wait", 0, 0, 250), ("drive", 0, 250, 40)]


def test_a_bound_call_lowers_to_llvm_and_the_generic_module_does_not():
    @pulse.kernel
    def k(q, tau: float):
        d, t = get_drive_line(q)
        drive(d, square(40, 0.5), t)
        wait(d, tau)
        drive(d, square(40, 0.5), t)

    lowered = pulse.compile(k).lower(Q[0], 120.0)
    assert "llvm.func" in lowered.mlir("llvm")
    with pytest.raises(ValueError, match="float parameters"):
        lowered.generic.mlir("llvm")


# ── Bounds ───────────────────────────────────────────────────────────────


def _bounded(bounds):
    @pulse.kernel
    def k(q, tau: float):
        d, t = get_drive_line(q)
        drive(d, square(40, 0.5), t)
        wait(d, tau, bounds=bounds)

    return pulse.compile(k)


def test_lengths_within_the_bounds_run():
    compiled = _bounded((100, 4000))
    for tau, cycles in ((100.0, 100), (99.5, 100), (4000.4, 4000), (1234.0, 1234)):
        assert compiled.lower(Q[0], tau).schedule[-1].duration_cycles == cycles


@pytest.mark.parametrize(
    ("tau", "message"),
    [
        (
            99.49,
            r"k\(\): the wait length tau = 99\.49 rounds to 99 clock cycles, outside its "
            r"bounds \(100, 4000\)",
        ),
        (4000.5, r"rounds to 4001 clock cycles, outside its bounds \(100, 4000\)"),
    ],
)
def test_a_length_outside_the_bounds_fails_the_call(tau, message):
    compiled = _bounded((100, 4000))
    with pytest.raises(ValueError, match=message):
        compiled.lower(Q[0], tau)
    with pytest.raises(ValueError, match=message):
        compiled(Q[0], tau)


def test_bounds_may_be_positional_and_open_above():
    @pulse.kernel
    def k(q, tau: float):
        d, t = get_drive_line(q)
        wait(d, tau, (10, None))

    compiled = pulse.compile(k)
    assert compiled.lower(Q[0], 1e9).schedule[0].duration_cycles == 10**9
    with pytest.raises(ValueError, match=r"outside its bounds \(10, None\)"):
        compiled.lower(Q[0], 9.0)


@pytest.mark.parametrize(
    ("bounds", "message"),
    [
        ((5,), r"bounds= takes \(lowest, highest\) in clock cycles"),
        ("x", r"bounds= takes \(lowest, highest\) in clock cycles"),
        ((-1, 3), r"bounds= must satisfy 0 <= lowest <= highest"),
        ((5, 2), r"bounds= must satisfy 0 <= lowest <= highest"),
        ((0.5, 2), r"bounds is a float \(0.5\)\. Durations are an int number of cycles"),
    ],
)
def test_malformed_bounds_fail_while_compiling(bounds, message):
    with pytest.raises(CompilationError, match=message):
        _bounded(bounds).lower(Q[0], 10.0)


def test_bounds_must_be_known_while_compiling():
    @pulse.kernel
    def k(q, tau: float, top: float):
        d, t = get_drive_line(q)
        wait(d, tau, bounds=(0, top))

    with pytest.raises(CompilationError, match="bounds of a wait must be known"):
        k.lower(Q[0], 1.0, 2.0)


def test_bounds_check_a_length_known_while_compiling():
    @pulse.kernel
    def k(q, tau):
        d, t = get_drive_line(q)
        wait(d, tau, bounds=(0, 40))

    assert k.lower(Q[0], 30).schedule[0].duration_cycles == 30
    with pytest.raises(CompilationError, match=r"wait duration 50 is outside its bounds \(0, 40\)"):
        k.lower(Q[0], 50)


# ── Errors at call time ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("tau", "message"),
    [
        (
            -0.51,
            r"t1\(\): the wait length tau = -0\.51 rounds to -1 clock cycles; a wait cannot "
            r"be negative",
        ),
        (-40.0, r"rounds to -40 clock cycles; a wait cannot be negative"),
        (float("nan"), r"the wait length tau is nan; it must be a finite number of clock cycles"),
        (float("inf"), r"the wait length tau is inf"),
        (2.0**60, r"outside its bounds \(0, None\)"),
    ],
)
def test_a_negative_or_infinite_length_fails_the_call(tau, message):
    compiled = pulse.compile(pulse.kernel(t1))
    with pytest.raises(ValueError, match=message):
        compiled.lower(Q[0], tau)


def test_an_error_names_the_expression():
    @pulse.kernel
    def k(q, tau: float, n: float):
        d, t = get_drive_line(q)
        wait(d, 2 * tau - n / 4)

    with pytest.raises(ValueError, match=r"the wait length 2 \* tau - n / 4 = -1\.0 rounds"):
        k.lower(Q[0], 1.5, 16.0)
    with pytest.raises(ValueError, match=r"the wait length 2 \* tau - n / 4 is nan|is -inf"):
        k.lower(Q[0], float("-inf"), 16.0)


def test_an_error_names_a_negated_expression():
    @pulse.kernel
    def k(q, tau: float, t0: float):
        d, t = get_drive_line(q)
        wait(d, -(tau - t0))

    assert k.lower(Q[0], 10.0, 50.0).schedule[0].duration_cycles == 40
    with pytest.raises(ValueError, match=r"the wait length -\(tau - t0\) = -40\.0 rounds"):
        k.lower(Q[0], 50.0, 10.0)


def test_a_division_by_a_zero_argument_fails_the_call():
    @pulse.kernel
    def k(q, total: float, n: float):
        d, t = get_drive_line(q)
        wait(d, total / n)

    assert k.lower(Q[0], 100.0, 4.0).schedule[0].duration_cycles == 25
    with pytest.raises(ValueError, match="the wait length total / n is nan"):
        k.lower(Q[0], 100.0, 0.0)


def test_a_length_that_needs_more_than_arithmetic_stays_static():
    @pulse.kernel
    def k(q, tau: float):
        d, t = get_drive_line(q)
        wait(d, tau % 7)

    with pytest.warns(RuntimeWarning, match="float argument 'tau'"):
        assert k.lower(Q[0], 30.0).schedule[0].duration_cycles == 2
    assert k.runtime_parameters == ()
