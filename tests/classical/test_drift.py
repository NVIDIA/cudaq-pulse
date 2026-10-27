# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The drifting plant: its processes, its truth trace, and its surrogates
against full simulation of the same device."""

from __future__ import annotations

import json
import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.runtime.drift import DriftingPlant, OrnsteinUhlenbeck, TelegraphT1
from cudaq_pulse.targets import Qubit, TargetDescription

PI_AMP = math.pi / 20


def _target(t1_us=20.0):
    qubit = Qubit(index=0, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=t1_us, t2_star_us=0.0)
    return TargetDescription(name="sim", qubits={0: qubit})


def test_telegraph_t1_switches_at_its_rate_and_spends_half_its_time_high():
    plant = DriftingPlant(_target(), t1=TelegraphT1(5.0, 30.0, rate_hz=10.0), seed=1)
    dt = 1e-3
    for _ in range(20000):
        plant.advance(dt)
    t1 = np.array(plant.truth_trace()["t1_us"])
    assert set(np.unique(t1)) == {5.0, 30.0}
    assert np.mean(t1 == 30.0) == pytest.approx(0.5, abs=0.05)
    switches = np.count_nonzero(np.diff(t1))
    # 10 Hz for 20 s, less the pairs that cancel within one step.
    expected = 200 * math.exp(-2 * 10 * dt)
    assert switches == pytest.approx(expected, rel=0.2)


def test_ornstein_uhlenbeck_drift_has_its_variance_and_correlation():
    sigma, tau, dt = 50e3, 0.5, 0.05
    plant = DriftingPlant(_target(), frequency=OrnsteinUhlenbeck(sigma, tau), seed=2)
    for _ in range(40000):
        plant.advance(dt)
    x = np.array(plant.truth_trace()["frequency_offset_hz"])
    assert np.std(x) == pytest.approx(sigma, rel=0.1)
    lag = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert lag == pytest.approx(math.exp(-dt / tau), abs=0.02)


def test_the_truth_trace_is_json_and_seeded():
    def run(seed):
        plant = DriftingPlant(
            _target(),
            t1=TelegraphT1(5.0, 30.0, 1.0),
            frequency=OrnsteinUhlenbeck(1e4, 1.0),
            amplitude=OrnsteinUhlenbeck(0.01, 1.0),
            seed=seed,
        )
        for _ in range(50):
            plant.advance(0.1)
        return plant.truth_trace()

    trace = run(5)
    assert len(trace["time_s"]) == 51
    assert trace["time_s"][-1] == pytest.approx(5.0)
    assert json.loads(json.dumps(trace)) == trace
    assert run(5) == trace and run(6) != trace


@pulse.kernel
def t1_shot(q, wait_ns: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, square(40, PI_AMP), t)
    wait(d, wait_ns)
    sync(d, r)
    readout(r, square(40, 0.05), rt)


@pulse.kernel
def rabi(q, scale: float):
    d, t = get_drive_line(q)
    drive(d, square(40, PI_AMP * scale), t)


@pulse.kernel
def ramsey(q, phase: float):
    d, t = get_drive_line(q)
    drive(d, square(40, PI_AMP / 2), t)
    wait(d, 400)
    shift_phase(t, phase)
    drive(d, square(40, PI_AMP / 2), t)


def _plant():
    plant = DriftingPlant(
        _target(),
        t1=TelegraphT1(4.0, 25.0, 1.0),
        frequency=OrnsteinUhlenbeck(1e6, 1.0),
        amplitude=OrnsteinUhlenbeck(0.05, 1.0),
        seed=11,
    )
    plant.advance(0.3)
    return plant


def test_the_surrogates_match_simulating_the_plant():
    plant = _plant()
    state = plant.state
    target = plant.target()
    assert target.qubits[0].t1_us == state.t1_us
    assert target.qubits[0].frequency_hz == pytest.approx(5e9 + state.frequency_offset_hz)
    dev = pulse.device({"target": _target()})
    q = dev.qudit_handles[0]
    compiled_t1 = pulse.compile(t1_shot, device=dev)
    for wait_cycles in (0.0, 4000.0, 16000.0):
        result = pulse.simulate(compiled_t1.lower(q, wait_cycles), plant=target, dt_ns=1.0)
        p1 = dict(result.branches).get((1,), 0.0)
        # The surrogate counts the wait; the pi pulse and readout add 40 ns.
        elapsed = wait_cycles / dev.clock_ghz + 40
        assert p1 == pytest.approx(plant.p_survive(elapsed), abs=0.03)
    # On resonance the amplitude scale stretches the rotation exactly.
    resonant = DriftingPlant(_target(t1_us=0.0), amplitude=OrnsteinUhlenbeck(0.05, 1.0), seed=3)
    compiled_rabi = pulse.compile(rabi, device=dev)
    for scale in (0.5, 1.0):
        result = pulse.simulate(compiled_rabi.lower(q, scale), plant=resonant.target())
        assert result.density_matrix[1, 1].real == pytest.approx(
            resonant.p_rabi(math.pi * scale), abs=1e-6
        )
    # Ramsey: the simulated sequence gives (1 + cos(2 pi d T - phase)) / 2,
    # which is the estimators' convention at phase pi - phase.
    offset_only = DriftingPlant(_target(t1_us=0.0), frequency=OrnsteinUhlenbeck(1e6, 1.0), seed=4)
    compiled_ramsey = pulse.compile(ramsey, device=dev)
    separation_ns = (400 + 40) / dev.clock_ghz
    for phase in (0.0, 1.0, 2.5):
        result = pulse.simulate(compiled_ramsey.lower(q, phase), plant=offset_only.target())
        expected = offset_only.p_ramsey(separation_ns, math.pi - phase)
        assert result.density_matrix[1, 1].real == pytest.approx(expected, abs=0.03)


def test_the_plant_must_define_the_programs_qudits():
    dev = pulse.device({"target": _target()})
    lowered = pulse.compile(rabi, device=dev).lower(dev.qudit_handles[0], 1.0)
    empty = TargetDescription(name="none", qubits={})
    with pytest.raises(ValueError, match="does not define qudits"):
        pulse.simulate(lowered, plant=empty)
    with pytest.raises(TypeError, match="TargetDescription"):
        pulse.simulate(lowered, plant=dev)
