# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Calls that take IQ points synthesized from the dispersive readout model."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import IQ, Bits
from cudaq_pulse.runtime._calls import IQModel
from cudaq_pulse.targets import Qubit, TargetDescription

PI_AMP = math.pi / 20
PLUS = [2**-0.5, 2**-0.5]


def _device(t1_us=0.0, **readout):
    params = {"chi_hz": 1e6, "kappa_eff_hz": 2e6, "ro_photons": 4.0, "ro_efficiency": 0.5}
    params.update(readout)
    qubit = Qubit(
        index=0,
        frequency_hz=5e9,
        anharmonicity_hz=-2e8,
        t1_us=t1_us,
        t2_star_us=0.0,
        readout_params=params,
    )
    return pulse.device({"target": TargetDescription(name="sim", qubits={0: qubit})})


def _centers(dev, amplitude=0.05, duration_ns=None):
    model = IQModel(dev.target, (0,), decay=False)
    tau = duration_ns if duration_ns is not None else 1000 / dev.clock_ghz
    return [model.center(0, k, math.nan, amplitude, tau) for k in (0, 1)]


def _threshold(dev):
    c0, c1 = _centers(dev)
    axis = (c1 - c0) / abs(c1 - c0)
    middle = (c0 + c1) / 2

    @pulse.classical(args=(IQ(1),), results=(Bits(1),), max_latency_ns=0)
    def discriminate(points):
        (i, q) = points[0]
        return [int(((complex(i, q) - middle) * axis.conjugate()).real > 0)]

    return discriminate


def _iq_reset(discriminate):
    @pulse.kernel
    def iq_reset(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        result = readout(r, square(1000, 0.05), rt)
        sync(d, r)
        if discriminate.submit([result]).result()[0]:
            drive(d, square(40, PI_AMP), t)

    return iq_reset


def test_discrimination_errors_match_the_gaussian_overlap():
    dev = _device(ro_photons=0.3)
    q = dev.qudit_handles[0]
    discriminate = _threshold(dev)
    shots = 4000
    result = pulse.simulate(
        pulse.compile(_iq_reset(discriminate), device=dev).lower(q),
        initial_state=PLUS,
        shots=shots,
        seed=11,
    )
    c0, c1 = _centers(dev)
    expected = 0.5 * math.erfc(abs(c1 - c0) / 2)  # sigma^2 = 1/2 per quadrature
    assert 0.1 < expected < 0.25
    tolerance = 4 * math.sqrt(expected * (1 - expected) / shots)
    wrong = sum(
        w
        for calls, (outcomes, w) in zip(result.calls, result.branches)
        if calls[0].answer[0] != outcomes[0]
    )
    assert wrong == pytest.approx(expected, abs=tolerance)
    # A wrong answer leaves |1> in place or flips |0> to |1>.
    p1 = float(np.real(result.density_matrix[1, 1]))
    assert p1 == pytest.approx(expected, abs=tolerance)
    assert result.records is not None and len(result.records) == shots


def test_iq_points_scale_with_the_readout_amplitude():
    dev = _device(ro_amp=0.05)
    q = dev.qudit_handles[0]
    seen: dict[float, list] = {0.05: [], 0.1: []}

    for amplitude in seen:

        @pulse.classical(args=(IQ(1),), results=(Bits(1),), max_latency_ns=0, name="record")
        def record(points, amplitude=amplitude):
            seen[amplitude].append(complex(*points[0]))
            return [0]

        @pulse.kernel
        def probe(q, a: float):
            r, rt = get_readout_line(q)
            result = readout(r, square(1000, a), rt)
            record.submit([result]).result()

        pulse.simulate(pulse.compile(probe, device=dev).lower(q, amplitude), shots=600, seed=5)

    mean_low = np.mean(seen[0.05])
    mean_high = np.mean(seen[0.1])
    assert abs(mean_high / mean_low) == pytest.approx(2.0, rel=0.1)
    # Without ro_amp the reference is the amplitude played: 4 photons at any.
    free = IQModel(_device().target, (0,), decay=False)
    tau = 1000 / dev.clock_ghz
    assert abs(free.center(0, 0, math.nan, 0.1, tau)) == pytest.approx(
        abs(free.center(0, 0, math.nan, 0.05, tau))
    )


def test_the_centers_follow_the_readout_frequency():
    dev = _device(ro_freq=7e9)
    model = IQModel(dev.target, (0,), decay=False)
    tau = 500.0
    # On resonance with the resonator pulled by |1> (f_r + chi), |1> rings
    # up most; on resonance with |0> (f_r - chi), |0> does.
    on_one = [abs(model.center(0, k, 7e9 + 1e6, 0.05, tau)) for k in (0, 1)]
    on_zero = [abs(model.center(0, k, 7e9 - 1e6, 0.05, tau)) for k in (0, 1)]
    assert on_one[1] > on_one[0]
    assert on_zero[0] > on_zero[1]
    assert on_one[1] == pytest.approx(on_zero[0])


def test_relaxation_during_the_window_pulls_one_toward_zero():
    t1_us = 1.0
    dev = _device(t1_us=t1_us, ro_integration=500.0)
    tau, t1 = 500.0, t1_us * 1e3
    model = IQModel(dev.target, (0,), decay=True)
    c0 = model.center(0, 0, math.nan, 0.05, tau)
    c1 = model.center(0, 1, math.nan, 0.05, tau)
    rng = np.random.default_rng(2)
    points = [complex(*model.sample(rng, 0, 1, math.nan, 0.05, tau)) for _ in range(20000)]
    # E[point] = c0 + (c1 - c0) E[min(t, tau)] / tau, t ~ Exp(T1).
    fraction = (t1 / tau) * (1 - math.exp(-tau / t1))
    expected = c0 + (c1 - c0) * fraction
    assert abs(np.mean(points) - expected) < 0.03 * abs(c1 - c0)


def test_iq_calls_need_shots_and_are_seeded():
    dev = _device()
    q = dev.qudit_handles[0]
    lowered = pulse.compile(_iq_reset(_threshold(dev)), device=dev).lower(q)
    with pytest.raises(ValueError, match="simulate with shots"):
        pulse.simulate(lowered, initial_state=PLUS)
    a = pulse.simulate(lowered, initial_state=PLUS, shots=50, seed=4)
    b = pulse.simulate(lowered, initial_state=PLUS, shots=50, seed=4)
    assert np.array_equal(a.density_matrix, b.density_matrix)


def test_depletion_time_follows_kappa():
    dev = _device(kappa_eff_hz=1e6)
    model = IQModel(dev.target, (0,), decay=False)
    kappa = 2 * math.pi * 1e6 * 1e-9
    assert model.depletion_ns(0) == pytest.approx(math.log(100) / kappa)


def test_emulation_draws_iq_points_at_the_reported_levels():
    from cudaq_pulse.classical import emulation

    dev = _device()
    lowered = pulse.compile(_iq_reset(_threshold(dev)), device=dev).lower(dev.qudit_handles[0])
    result = emulation.emulate(lowered, shots=200, outcomes=lambda q, _, rng: 1, seed=3)
    answers = [traces[0].answer for traces in result.calls]
    assert all(a == [1] for a in answers)  # 5 sigma apart: no errors expected
    points = [traces[0].inputs[0][0] for traces in result.calls]
    _, c1 = _centers(dev)
    assert abs(complex(*np.mean(points, axis=0)) - c1) < 0.2


def test_the_krinner_twin_separates_its_readout_states():
    from cudaq_pulse.classical import emulation
    from cudaq_pulse.targets import transmon_krinner_17q

    dev = pulse.device({"target": transmon_krinner_17q()})
    q = dev.qudit_handles[0]
    target = dev.target
    model = IQModel(target, (0,), decay=False)
    params = target.qubits[0].readout_params
    tau = float(params["ro_integration"])
    c0, c1 = (model.center(0, k, params["ro_freq"], 1.0, tau) for k in (0, 1))
    assert abs(c1 - c0) > 3  # a high-fidelity single-shot readout

    record = pulse.classical(
        lambda points: [0], name="krinner_iq", args=(IQ(1),), results=(Bits(1),), max_latency_ns=0
    )

    @pulse.kernel
    def probe(q):
        r, rt = get_readout_line(q)
        result = readout(r, square(400, 0.1), rt)
        record.submit([result]).result()

    lowered = pulse.compile(probe, device=dev).lower(q)
    result = emulation.emulate(lowered, shots=20, outcomes=lambda q, _, rng: 0, seed=1)
    points = [traces[0].inputs[0][0] for traces in result.calls]
    assert all(math.isfinite(i) and math.isfinite(qv) for i, qv in points)
