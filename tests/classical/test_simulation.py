# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Simulating programs that call classical functions."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import Bits, Float, Integer
from cudaq_pulse.targets import Qubit, TargetDescription

PI_AMP = math.pi / 20  # square(40, PI_AMP) is a pi pulse at 2 GHz


def _device(t1_us=0.0, n=1, readout_params=None):
    qubits = {
        i: Qubit(
            index=i,
            frequency_hz=5e9 + 1e8 * i,
            anharmonicity_hz=-2e8,
            t1_us=t1_us,
            t2_star_us=0.0,
            readout_params=readout_params or {},
        )
        for i in range(n)
    }
    return pulse.device({"target": TargetDescription(name="sim", qubits=qubits)})


@pulse.classical(args=(Bits(1),), results=(Bits(1),), max_latency_ns=0)
def decoder(bits):
    return [bits[0]]


@pulse.kernel
def active_reset(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    result = readout(r, square(1000, 0.05), rt)
    sync(d, r)
    if result:
        drive(d, square(40, PI_AMP), t)


@pulse.kernel
def decoded_reset(q):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    result = readout(r, square(1000, 0.05), rt)
    sync(d, r)
    if decoder.submit([result]).result()[0]:
        drive(d, square(40, PI_AMP), t)


PLUS = [2**-0.5, 2**-0.5]


def _by_outcome(result):
    """Branch probabilities summed per reported outcome (paths with one
    report but different true levels are separate branches)."""
    total: dict = {}
    for outcomes, w in result.branches:
        total[outcomes] = total.get(outcomes, 0.0) + w
    return total


@pytest.mark.parametrize("shots", [None, 500])
def test_a_decoder_reproduces_active_reset_exactly(shots):
    dev = _device()
    q = dev.qudit_handles[0]
    a = pulse.simulate(
        pulse.compile(active_reset, device=dev).lower(q), initial_state=PLUS, shots=shots, seed=3
    )
    b = pulse.simulate(
        pulse.compile(decoded_reset, device=dev).lower(q), initial_state=PLUS, shots=shots, seed=3
    )
    np.testing.assert_allclose(b.density_matrix, a.density_matrix, atol=1e-12)
    assert b.branches == a.branches
    if shots:
        assert (b.records == a.records).all()
    assert [[c.answer for c in calls] for calls in b.calls] == [[[0]], [[1]]]


def test_an_answer_amplitude_is_bound_per_path():
    amplitude = pulse.classical(
        lambda bits: PI_AMP * bits[0],
        name="amplitude",
        args=(Bits(1),),
        results=(Float(),),
        max_latency_ns=0,
    )

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        m = readout(r, square(1000, 0.05), rt)
        a = amplitude.submit([m]).result()
        sync(d, r)
        drive(d, square(40, a), t)

    dev = _device()
    q = dev.qudit_handles[0]
    result = pulse.simulate(pulse.compile(k, device=dev).lower(q), initial_state=PLUS)
    reset = pulse.simulate(pulse.compile(active_reset, device=dev).lower(q), initial_state=PLUS)
    np.testing.assert_allclose(result.density_matrix, reset.density_matrix, atol=1e-9)


@pulse.classical(args=(Float(),), results=(Float(),), max_latency_ns=0)
def constant(x):
    return x


@pulse.kernel
def variable_t1(q, tau: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    w = constant.submit(tau).result()
    drive(d, square(40, PI_AMP), t)
    wait(d, w)
    sync(d, r)
    readout(r, square(200, 0.05), rt)


@pulse.kernel
def fixed_t1(q, tau: int):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, square(40, PI_AMP), t)
    sync(d, r)
    wait(d, tau)
    sync(d, r)
    readout(r, square(200, 0.05), rt)


@pytest.mark.parametrize("tau", [0, 400, 4000, 20000])
def test_a_variable_wait_matches_the_fixed_wait_and_t1_decay(tau):
    t1_us = 5.0
    dev = _device(t1_us=t1_us)
    q = dev.qudit_handles[0]
    variable = pulse.simulate(pulse.compile(variable_t1, device=dev).lower(q, float(tau)))
    fixed = pulse.simulate(pulse.compile(fixed_t1, device=dev).lower(q, tau))
    p_variable = dict(variable.branches)[(1,)]
    p_fixed = dict(fixed.branches)[(1,)]
    assert p_variable == pytest.approx(p_fixed, abs=1e-9)
    # The pi pulse (40 cycles) plus the wait, then decay until the readout
    # ends: P(1) at the projection is exp(-(wait + readout) / T1).
    clock = dev.clock_ghz
    elapsed_ns = (tau + 200) / clock
    assert p_variable == pytest.approx(math.exp(-elapsed_ns / (t1_us * 1e3)), rel=2e-3)


def test_variable_waits_differ_per_path():
    adaptive = pulse.classical(
        lambda bits: 4000.0 * bits[0],
        name="adaptive",
        args=(Bits(1),),
        results=(Float(),),
        max_latency_ns=0,
    )

    @pulse.kernel
    def k(q, p):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        e, u = get_drive_line(p)
        f, ft = get_readout_line(p)
        m = readout(r, square(200, 0.05), rt)
        w = adaptive.submit([m]).result()
        drive(e, square(40, PI_AMP), u)
        wait(e, w)
        sync(e, f)
        readout(f, square(200, 0.05), ft)

    dev = _device(t1_us=5.0, n=2)
    qs = dev.qudit_handles
    result = pulse.simulate(
        pulse.compile(k, device=dev).lower(qs[0], qs[1]), initial_state=np.kron([1, 0], PLUS)
    )
    branches = dict(result.branches)
    clock = dev.clock_ghz
    t1 = 5e3
    for first, waited in ((0, 0), (1, 4000)):
        p1 = branches[(first, 1)] / (branches[(first, 0)] + branches[(first, 1)])
        assert p1 == pytest.approx(math.exp(-(waited + 200) / clock / t1), rel=3e-3)
    assert result.times_ns[1] > 0


def test_a_d2_repetition_round_flags_double_flips_with_probability_p_squared():
    p = 0.2
    both = pulse.classical(
        lambda bits: [bits[0] & bits[1]],
        name="both_flipped",
        args=(Bits(2),),
        results=(Bits(1),),
        max_latency_ns=0,
    )

    @pulse.kernel
    def k(a, b, flag):
        ra, rta = get_readout_line(a)
        rb, rtb = get_readout_line(b)
        f, ft = get_drive_line(flag)
        ma = readout(ra, square(200, 0.05), rta)
        mb = readout(rb, square(200, 0.05), rtb)
        sync(f, ra, rb)
        if both.submit([ma, mb]).result()[0]:
            drive(f, square(40, PI_AMP), ft)

    dev = _device(n=3)
    noisy = np.diag([1 - p, p])
    rho = np.kron(np.diag([1.0, 0.0]), np.kron(noisy, noisy))  # qudit 0 least significant
    result = pulse.simulate(
        pulse.compile(k, device=dev).lower(*dev.qudit_handles[:3]), initial_state=rho
    )
    dims = result.dims
    flag_excited = sum(
        result.density_matrix[i, i].real for i in range(math.prod(dims)) if (i // 4) % 2 == 1
    )
    assert flag_excited == pytest.approx(p**2, abs=1e-6)


def test_timeouts_take_the_fallback_at_the_modelled_rate():
    slow = pulse.classical(
        lambda bits: [1],
        name="slow",
        args=(Bits(1),),
        results=(Bits(1),),
        min_latency_ns=0,
        nominal_latency_ns=500,
        max_latency_ns=1000,
    )

    @pulse.kernel
    def k(q, p):
        r, rt = get_readout_line(q)
        e, u = get_drive_line(p)
        m = readout(r, square(100, 0.05), rt)
        ok, fix = slow.submit([m]).result(timeout=500 * pulse.ns)
        if ok:
            drive(e, square(40, PI_AMP), u)

    dev = _device(n=2)
    qs = dev.qudit_handles[:2]
    low = pulse.compile(k, device=dev).lower(*qs)
    # Triangular latency on [0, 1000] with mode 500: P(L <= 500) = 1/2.
    exact = pulse.simulate(low, latency_branching=True)
    assert np.trace(exact.density_matrix).real == pytest.approx(1.0)
    p1 = sum(exact.density_matrix[i, i].real for i in (2, 3))
    assert p1 == pytest.approx(0.5, abs=1e-9)
    shots = pulse.simulate(low, shots=4000, seed=11)
    ok_weight = sum(w for (_, w), calls in zip(shots.branches, shots.calls) if calls[0].in_time)
    assert ok_weight == pytest.approx(0.5, abs=1e-9)
    # The sampled shots split between in time and late within 4 sigma.
    excited = shots.density_matrix[2, 2].real + shots.density_matrix[3, 3].real
    assert excited == pytest.approx(0.5, abs=1e-9)
    # Without shots or branching the nominal latency decides.
    nominal = pulse.simulate(low)
    assert all(c[0].in_time for c in nominal.calls)


def test_readout_errors_are_reported_and_seen_by_calls():
    seen = []

    def record(bits):
        seen.append(bits[0])
        return [bits[0]]

    dev = _device(readout_params={"error_1to0": 0.1, "error_0to1": 0.05})
    q = dev.qudit_handles[0]
    low = pulse.compile(decoded_reset, device=dev).lower(q)
    result = pulse.simulate(
        low, initial_state=[0, 1], readout="model", handlers={"decoder": record}
    )
    branches = _by_outcome(result)
    assert branches[(0,)] == pytest.approx(0.1)
    assert branches[(1,)] == pytest.approx(0.9)
    assert sorted(seen) == [0, 1]
    # Reported 0 means no reset pulse: the qubit stays in |1>.
    assert result.density_matrix[1, 1].real == pytest.approx(0.1)


def test_decay_during_readout_adds_to_the_one_to_zero_error():
    dev = _device(t1_us=10.0, readout_params={"error_1to0": 0.0, "ro_dur": 500.0})
    q = dev.qudit_handles[0]

    @pulse.kernel
    def measure(q):
        r, rt = get_readout_line(q)
        readout(r, square(2, 0.05), rt)

    low = pulse.compile(measure, device=dev).lower(q)
    plain = _by_outcome(pulse.simulate(low, initial_state=[0, 1], readout="model"))
    decayed = _by_outcome(pulse.simulate(low, initial_state=[0, 1], readout="model_with_decay"))
    extra = 1 - math.exp(-500 / (2 * 10e3))
    # The extra error applies to what is in |1> when the readout ends.
    assert decayed[(0,)] - plain.get((0,), 0.0) == pytest.approx(plain[(1,)] * extra, rel=1e-6)


def test_handlers_override_and_are_required_for_external_functions():
    from cudaq_pulse import classical

    remote = classical.external("remote", args=(Bits(1),), results=(Integer(),), max_latency_ns=0)

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        n = remote.submit([readout(r, square(100, 0.05), rt)]).result()
        if n == 3:
            drive(d, square(40, PI_AMP), t)

    dev = _device()
    low = pulse.compile(k, device=dev).lower(dev.qudit_handles[0])
    with pytest.raises(
        KeyError, match="classical function 'remote' is external and has no handler"
    ):
        pulse.simulate(low)
    result = pulse.simulate(low, handlers={"remote": lambda bits: 3})
    assert result.density_matrix[1, 1].real == pytest.approx(1.0, abs=1e-6)


def test_calls_in_a_rolled_loop_are_bound_per_iteration():
    counter = pulse.classical(
        lambda x: x + 1.0, name="counter", args=(Float(),), results=(Float(),), max_latency_ns=0
    )

    @pulse.kernel(unroll_threshold=0)
    def k(q):
        d, t = get_drive_line(q)
        x = 0.0
        for _ in range(4):
            x = counter.submit(x).result()
        drive(d, square(40, PI_AMP * x / 4), t)

    dev = _device()
    result = pulse.simulate(pulse.compile(k, device=dev).lower(dev.qudit_handles[0]))
    assert [c.answer for c in result.calls[0]] == [1.0, 2.0, 3.0, 4.0]
    assert result.density_matrix[1, 1].real == pytest.approx(1.0, abs=1e-6)


@pulse.kernel
def answered_wait(q, tau: float):
    d, t = get_drive_line(q)
    w = constant.submit(tau).result()
    drive(d, square(40, PI_AMP), t)
    wait(d, w)


def test_whole_program_lowering_rejects_unanswered_calls():
    dev = _device()
    lowered = pulse.compile(answered_wait, device=dev).lower(dev.qudit_handles[0], 400.0)
    with pytest.raises(RuntimeError, match="evolves in segments between their answers"):
        lowered.mlir("llvm")


def test_a_segment_with_bound_answers_lowers_to_llvm_without_classical_ops():
    from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline

    dev = _device()
    lowered = pulse.compile(decoded_reset, device=dev).lower(dev.qudit_handles[0])
    module = lowered._concrete().simulation_module(*lowered.options.schedule_args())
    segment = module.placeholders()
    segment.set_attributes(
        "{pulse.path = [array<i64: 0, 1>], qop.t_start = 0.0 : f64, "
        "qop.t_end = 600.0 : f64, qop.num_steps = 10 : i64}"
    )
    qop = MLIRPipeline().run_pulse_to_qop(segment.print())
    assert "classical" not in qop
    assert MLIRPipeline().jit_pulse(segment.print(), []) is not None


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["decoded_reset", "variable_wait"])
def test_gpu_simulation_of_calls_matches_the_cpu(case):
    dev = _device(t1_us=5.0)
    q = dev.qudit_handles[0]
    if case == "decoded_reset":
        lowered = pulse.compile(decoded_reset, device=dev).lower(q)
        kwargs = {"initial_state": PLUS}
    else:
        lowered = pulse.compile(variable_t1, device=dev).lower(q, 4000.0)
        kwargs = {}
    cpu = pulse.simulate(lowered, backend="cpu", **kwargs)
    gpu = pulse.simulate(lowered, backend="gpu", **kwargs)
    assert _by_outcome(gpu) == pytest.approx(_by_outcome(cpu), abs=1e-6)
    assert np.allclose(gpu.density_matrix, cpu.density_matrix, atol=1e-6)


def test_calls_on_a_high_index_qudit_use_its_mode():
    dev = _device(n=3, readout_params={"error_1to0": 0.1})
    q = dev.qudit_handles[2]
    a = pulse.simulate(pulse.compile(active_reset, device=dev).lower(q), initial_state=PLUS)
    b = pulse.simulate(pulse.compile(decoded_reset, device=dev).lower(q), initial_state=PLUS)
    assert b.qudits == (2,)
    assert b.measurements[0].qudit == 2
    assert _by_outcome(b) == pytest.approx(_by_outcome(a), abs=1e-9)
    assert np.allclose(b.density_matrix, a.density_matrix, atol=1e-9)
    modelled = pulse.simulate(
        pulse.compile(decoded_reset, device=dev).lower(q), initial_state=PLUS, readout="model"
    )
    assert _by_outcome(modelled)[(0,)] == pytest.approx(0.5 + 0.5 * 0.1, abs=1e-6)


@pytest.mark.parametrize("schedule", ["asap", "alap"])
@pytest.mark.parametrize("where", ["future_in", "future_out"])
def test_an_answer_crossing_a_block_is_bound_in_simulation(schedule, where):
    @pulse.kernel(schedule=schedule)
    def correct(future, d, t):
        if future.result()[0]:
            drive(d, square(40, PI_AMP), t)

    @pulse.kernel(schedule=schedule)
    def measure(d, r, rt):
        result = readout(r, square(1000, 0.05), rt)
        sync(d, r)
        return decoder.submit([result])

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        if where == "future_in":
            result = readout(r, square(1000, 0.05), rt)
            sync(d, r)
            correct(decoder.submit([result]), d, t)
        else:
            if measure(d, r, rt).result()[0]:
                drive(d, square(40, PI_AMP), t)

    dev = _device()
    q = dev.qudit_handles[0]
    result = pulse.simulate(pulse.compile(k, device=dev).lower(q), initial_state=PLUS)
    reset = pulse.simulate(pulse.compile(active_reset, device=dev).lower(q), initial_state=PLUS)
    np.testing.assert_allclose(result.density_matrix, reset.density_matrix, atol=1e-9)
    assert [[c.answer for c in calls] for calls in result.calls] == [[[0]], [[1]]]


def test_simulation_keeps_a_blocks_own_schedule():
    @pulse.kernel(schedule="asap")
    def early(d, t, e, u):
        drive(d, square(40, PI_AMP), t)
        drive(e, square(100, 0.0), u)

    @pulse.kernel
    def k(q, p):
        d, t = get_drive_line(q)
        e, u = get_drive_line(p)
        drive(e, square(4000, constant.submit(0.0).result()), u)
        early(d, t, e, u)

    dev = _device(t1_us=1.0, n=2)
    low = pulse.compile(k, device=dev).lower(*dev.qudit_handles)
    pi = next(e for e in low.schedule if e.kind == "drive" and e.qubit == 0)
    # The ASAP block ends with the program: its pi pulse ends 60 cycles
    # before the end, where a flattened ALAP schedule would end it at the end.
    end = max(e.end for e in low.schedule)
    assert end - pi.end == 60
    excited = 1.0 - pulse.simulate(low).density_matrix[0, 0].real

    @pulse.kernel
    def pi_alone(q):
        d, t = get_drive_line(q)
        drive(d, square(40, PI_AMP), t)

    alone = pulse.simulate(pulse.compile(pi_alone, device=dev).lower(dev.qudit_handles[0]))
    at_end = 1.0 - alone.density_matrix[0, 0].real
    assert excited == pytest.approx(at_end * math.exp(-60 / 2.0 / 1000.0), abs=1e-4)
