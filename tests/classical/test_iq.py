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
from cudaq_pulse.runtime._calls import IQModel
from cudaq_pulse.targets import DispersiveReadout, Qubit, ReadoutPulse, TargetDescription

PI_AMP = math.pi / 20
PLUS = [2**-0.5, 2**-0.5]


def _device(t1_us=0.0, **readout):
    fields = {"dispersive_shift_hz": 1e6, "linewidth_hz": 2e6, "efficiency": 0.5}
    tuned = {k: readout.pop(k) for k in ReadoutPulse._fields if k in readout}
    fields.update(readout)
    qubit = Qubit(
        index=0,
        frequency_hz=5e9,
        anharmonicity_hz=-2e8,
        t1_us=t1_us,
        t2_star_us=0.0,
        readout=DispersiveReadout(**fields),
        calibrations={pulse.gates.READOUT: ReadoutPulse(**tuned)},
    )
    return pulse.device(TargetDescription(name="sim", qubits={0: qubit}))


def _centers(dev, amplitude=0.05, duration_ns=None):
    model = IQModel(dev.target, (0,), decay=False)
    tau = duration_ns if duration_ns is not None else 1000 / dev.clock_ghz
    return [model.center(0, k, math.nan, amplitude, tau) for k in (0, 1)]


def _threshold(dev):
    c0, c1 = _centers(dev)
    axis = (c1 - c0) / abs(c1 - c0)
    middle = (c0 + c1) / 2

    @pulse.classical(max_latency_ns=0)
    def discriminate(points: list[complex]) -> pulse.Bits:
        return [int(((points[0] - middle) * axis.conjugate()).real > 0)]

    return discriminate


def _iq_reset(discriminate):
    @pulse.kernel
    def iq_reset(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        result = readout(r, square(1000, 0.05))
        sync(d, r)
        if discriminate.submit([result]).result()[0]:
            drive(d, square(40, PI_AMP))

    return iq_reset


def test_discrimination_errors_match_the_gaussian_overlap():
    dev = _device(photon_number=0.3)
    q = dev.qudit_handles[0]
    discriminate = _threshold(dev)
    shots = 4000
    result = pulse.simulate(
        pulse.compile(_iq_reset(discriminate), device=dev).lower(q),
        initial_state=PLUS,
        shots=shots,
        seed=11,
        levels=2,
    )
    c0, c1 = _centers(dev)
    expected = 0.5 * math.erfc(abs(c1 - c0) / 2)  # sigma^2 = 1/2 per quadrature
    assert 0.1 < expected < 0.25
    tolerance = 4 * math.sqrt(expected * (1 - expected) / shots)
    wrong = sum(b.probability for b in result.branches if b.calls[0].answer[0] != b.levels[0])
    assert wrong == pytest.approx(expected, abs=tolerance)
    # A wrong answer leaves |1> in place or flips |0> to |1>.
    p1 = float(np.real(result.density_matrix[1, 1]))
    assert p1 == pytest.approx(expected, abs=tolerance)
    assert result.records is not None and len(result.records) == shots


def test_iq_points_scale_with_the_readout_amplitude():
    dev = _device(amplitude=0.05)
    q = dev.qudit_handles[0]
    seen: dict[float, list] = {0.05: [], 0.1: []}

    for amplitude in seen:

        @pulse.classical(max_latency_ns=0, name="record")
        def record(point: complex) -> bool:
            seen[amplitude].append(point)  # noqa: B023
            return False

        @pulse.kernel
        def probe(q, a: float):
            r = get_readout_line(q)
            result = readout(r, square(1000, a))
            record.submit(result).result()

        pulse.simulate(
            pulse.compile(probe, device=dev).lower(q, amplitude), shots=600, seed=5, levels=2
        )

    mean_low = np.mean(seen[0.05])
    mean_high = np.mean(seen[0.1])
    assert abs(mean_high / mean_low) == pytest.approx(2.0, rel=0.1)
    # Without an amplitude the reference is the amplitude played: 4 photons at any.
    free = IQModel(_device().target, (0,), decay=False)
    tau = 1000 / dev.clock_ghz
    assert abs(free.center(0, 0, math.nan, 0.1, tau)) == pytest.approx(
        abs(free.center(0, 0, math.nan, 0.05, tau))
    )


def test_iq_points_follow_a_readout_amplitude_computed_from_an_answer():
    from cudaq_pulse.classical import emulation

    dev = _device(amplitude=0.05)
    q = dev.qudit_handles[0]
    seen: list[complex] = []

    @pulse.classical(max_latency_ns=0)
    def setting(x: float) -> float:
        return 0.2

    @pulse.classical(max_latency_ns=0, name="record")
    def record(point: complex) -> bool:
        seen.append(point)
        return False

    @pulse.kernel
    def probe(q):
        a = setting.submit(0.0).result()
        r = get_readout_line(q)
        record.submit(readout(r, square(1000, a * 0.5))).result()

    lowered = pulse.compile(probe, device=dev).lower(q)
    emulation.emulate(lowered, shots=400, outcomes=lambda q, _, rng: 0, seed=2)
    pulse.simulate(lowered, initial_state=[1, 0], shots=400, seed=2, levels=2)
    # Played at 0.1, twice the readout's amplitude: the points are twice as far.
    (c0, _), (r0, _) = _centers(dev, amplitude=0.1), _centers(dev, amplitude=0.05)
    assert c0 == pytest.approx(2 * r0)
    assert abs(np.mean(seen[:400]) - c0) < 0.15
    assert abs(np.mean(seen[400:]) - c0) < 0.15


def test_a_readout_the_model_cannot_evaluate_is_an_error():
    dev = _device(amplitude=0.05)
    q = dev.qudit_handles[0]

    @pulse.classical(max_latency_ns=0, name="record")
    def record(point: complex) -> bool:
        return False

    @pulse.kernel
    def probe(q):
        r = get_readout_line(q)
        record.submit(readout(r, custom_samples([0.05] * 200))).result()

    lowered = pulse.compile(probe, device=dev).lower(q)
    message = r"qudit 0 depend on its pulse's peak amplitude, .* READOUT calibration's amplitude"
    with pytest.raises(pulse.UnsupportedError, match=message):
        pulse.simulate(lowered, shots=10, seed=2, levels=2)
    # Without a readout amplitude the pulse's does not matter.
    plain = _device()
    pulse.simulate(
        pulse.compile(probe, device=plain).lower(plain.qudit_handles[0]), shots=10, seed=2, levels=2
    )
    # Nor, without a readout frequency, does the frequency; with one, an unknown
    # frequency is an error.
    assert math.isfinite(
        abs(IQModel(_device().target, (0,), False).center(0, 0, math.nan, 0.1, 500))
    )
    model = IQModel(_device(frequency_hz=7e9).target, (0,), decay=False)
    message = r"its tone's frequency, .*DispersiveReadout's frequency_hz"
    with pytest.raises(pulse.UnsupportedError, match=message):
        model.center(0, 0, math.nan, 0.1, 500)


def test_the_centers_follow_the_readout_frequency():
    dev = _device(frequency_hz=7e9)
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
    dev = _device(t1_us=t1_us, integration_ns=500.0)
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
        pulse.simulate(lowered, initial_state=PLUS, levels=2)
    a = pulse.simulate(lowered, initial_state=PLUS, shots=50, seed=4, levels=2)
    b = pulse.simulate(lowered, initial_state=PLUS, shots=50, seed=4, levels=2)
    assert np.array_equal(a.density_matrix, b.density_matrix)


def test_depletion_time_follows_kappa():
    dev = _device(linewidth_hz=1e6)
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
    assert abs(np.mean(points) - c1) < 0.2


def test_the_17_qubit_twin_separates_its_readout_states():
    from cudaq_pulse.classical import emulation
    from cudaq_pulse.targets import transmon_17q

    dev = pulse.device(transmon_17q())
    q = dev.qudit_handles[0]
    target = dev.target
    model = IQModel(target, (0,), decay=False)
    given = target.qubits[0].readout
    tau = float(dev.qudits[0].get_calibration(pulse.gates.READOUT).integration_ns)
    c0, c1 = (model.center(0, k, given.frequency_hz, 1.0, tau) for k in (0, 1))
    assert abs(c1 - c0) > 3  # a high-fidelity single-shot readout

    @pulse.classical(name="transmon_17q_iq", max_latency_ns=0)
    def record(points: list[complex]) -> pulse.Bits:
        return [0]

    @pulse.kernel
    def probe(q):
        r = get_readout_line(q)
        result = readout(r, square(400, 0.1))
        record.submit([result]).result()

    lowered = pulse.compile(probe, device=dev).lower(q)
    result = emulation.emulate(lowered, shots=200, outcomes=lambda q, _, rng: 0, seed=1)
    points = [traces[0].inputs[0][0] for traces in result.calls]
    assert all(math.isfinite(p.real) and math.isfinite(p.imag) for p in points)
    # The readout line's own tone is at the resonator, so the points gather
    # at its |0> point.
    assert abs(np.mean(points) - c0) < 0.25
