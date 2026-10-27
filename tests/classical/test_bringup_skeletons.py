# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Skeletons of fast bring-up experiments, with toy estimators, each on a
generic transmon and on the Krinner 17-qubit twin.

Each skeleton is expressed, scheduled, simulated and, where it calls
estimators while the QPU runs, exported for emulation. They check the
structure (what is called with what, what the timeline does) and that the
estimators recover what the simulated device is.
"""

from __future__ import annotations

import dataclasses
import json
import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.classical import Bits, Float, Floats, Integer, Link, random_index
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.classical.loop import EstimateLog, ParameterStore
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.runtime._calls import IQModel
from cudaq_pulse.runtime.drift import DriftingPlant, OrnsteinUhlenbeck, TelegraphT1
from cudaq_pulse.targets import transmon_generic, transmon_krinner_17q

PI_AMP = math.pi / 20  # square(40, PI_AMP) is a pi pulse (unit drive scale)
CLOCK = 2.0  # GHz on both targets
MACHINE = MachineModel(links={"controller": Link(fixed_ns=4, feedback_ns=8)})


def _generic():
    # The generic target has no readout parameters; give it a dispersive
    # readout so the readout skeletons have something to optimize.
    target = transmon_generic(1)
    qubit = dataclasses.replace(
        target.qubits[0],
        readout_params={
            "ro_freq": 7.0e9,
            "chi_hz": -1.5e6,
            "kappa_eff_hz": 4e6,
            "ro_integration": 400.0,
            "error_0to1": 0.01,
            "error_1to0": 0.03,
        },
    )
    return dataclasses.replace(target, qubits={0: qubit})


TARGETS = {"generic": _generic, "krinner": transmon_krinner_17q}


@pytest.fixture(params=sorted(TARGETS))
def target(request):
    return TARGETS[request.param]()


def _device(target):
    return pulse.device({"target": target})


# ── K11: adaptive T1 with a variable wait ───────────────────────────────────

T1_BOUNDS = (200, 20000)  # 0.1 to 10 us


@pulse.classical(
    args=(Bits(1), Float(), Float(), Float()),
    results=(Float(), Float(), Float()),
    max_latency_ns=200,
    link="controller",
)
def t1_estimator(survived, alpha, beta, wait_vtu):
    alpha, beta = cal.t1_update(alpha, beta, bool(survived[0]), wait_vtu / CLOCK)
    wait_vtu = min(max(cal.t1_wait(alpha, beta) * CLOCK, T1_BOUNDS[0]), T1_BOUNDS[1])
    return alpha, beta, wait_vtu


@pulse.kernel
def adaptive_t1(q, rounds: int):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    alpha, beta, tau = 1.0, 5000.0, 10000.0
    drive(d, square(40, PI_AMP), t)
    wait(d, 10000)
    for _ in range(rounds):
        sync(d, r)
        m = readout(r, square(400, 0.05), rt)
        alpha, beta, tau = t1_estimator.submit([m], alpha, beta, tau).result()
        sync(d, r)
        if m:
            drive(d, square(40, PI_AMP), t)
        drive(d, square(40, PI_AMP), t)
        wait(d, tau, bounds=T1_BOUNDS)
    sync(d, r)
    readout(r, square(400, 0.05), rt)


def test_k11_adaptive_t1_tracks_its_posterior_per_path(target):
    dev = _device(target)
    lowered = pulse.compile(adaptive_t1, device=dev, machine=MACHINE).lower(dev.qudit_handles[0], 2)
    assert [e.opened_by for e in lowered.epochs()] == ["start", "wait", "wait"]
    result = pulse.simulate(lowered, dt_ns=4.0)
    assert sum(p for _, p in result.branches) == pytest.approx(1.0)
    for calls, (outcomes, _) in zip(result.calls, result.branches):
        alpha, beta, tau = 1.0, 5000.0, 10000.0
        for call, outcome in zip(calls, outcomes):
            alpha, beta, tau = t1_estimator(np.array([outcome]), alpha, beta, tau)
            assert call.answer == pytest.approx((alpha, beta, tau))
    # The first shot survives the 5 us wait (plus the 200 ns readout) at
    # exp(-t / T1).
    t1_ns = target.qubits[0].t1_us * 1e3
    survived = sum(p for outcomes, p in result.branches if outcomes[0] == 1)
    assert survived == pytest.approx(math.exp(-(5000 + 200) / t1_ns), abs=0.02)
    artifact = json.loads(json.dumps(lowered.export_emulation()))
    assert len(artifact["epochs"]) == 3
    assert artifact["manifest"]["functions"][0]["link_name"] == "controller"


# ── K12: frequency binary search ────────────────────────────────────────────

SIGMA0 = 2e6
W0, PHASE0 = cal.ramsey_setting(0.0, SIGMA0)
RAMSEY_BOUNDS = (20, 40000)


@pulse.classical(
    args=(Bits(1), Float(), Float(), Float(), Float()),
    results=(Float(), Float(), Float(), Float()),
    max_latency_ns=300,
    link="controller",
)
def frequency_estimator(bit, mean, sigma, wait_vtu, phase):
    # The Ramsey wait counts from the first pulse's centre to the second's.
    mean, sigma = cal.frequency_update(mean, sigma, int(bit[0]), wait_vtu / CLOCK + 20.0, phase)
    wait_ns, phase = cal.ramsey_setting(mean, sigma)
    wait_vtu = min(max((wait_ns - 20.0) * CLOCK, RAMSEY_BOUNDS[0]), RAMSEY_BOUNDS[1])
    return mean, sigma, wait_vtu, phase


@pulse.kernel
def binary_search(q, rounds: int):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    mean, sigma = 0.0, SIGMA0
    tau = (W0 - 20.0) * CLOCK
    phase = PHASE0
    drive(d, square(40, PI_AMP / 2), t)
    wait(d, int(tau))
    shift_phase(t, math.pi - phase)
    drive(d, square(40, PI_AMP / 2), t)
    for _ in range(rounds):
        sync(d, r)
        m = readout(r, square(400, 0.05), rt)
        mean, sigma, tau, phase = frequency_estimator.submit([m], mean, sigma, tau, phase).result()
        sync(d, r)
        if m:
            drive(d, square(40, PI_AMP), t)
        drive(d, square(40, PI_AMP / 2), t)
        wait(d, tau, bounds=RAMSEY_BOUNDS)
        shift_phase(t, math.pi - phase)
        drive(d, square(40, PI_AMP / 2), t)
    sync(d, r)
    readout(r, square(400, 0.05), rt)


def test_k12_frequency_binary_search_follows_a_detuned_plant(target):
    dev = _device(target)
    offset = 300e3
    plant = DriftingPlant(target, frequency=OrnsteinUhlenbeck(1.0, 1.0), seed=0)
    detuned = dataclasses.replace(
        target,
        qubits={
            **target.qubits,
            0: dataclasses.replace(
                target.qubits[0], frequency_hz=target.qubits[0].frequency_hz + offset
            ),
        },
    )
    lowered = pulse.compile(binary_search, device=dev, machine=MACHINE).lower(
        dev.qudit_handles[0], 1
    )
    result = pulse.simulate(lowered, plant=detuned, dt_ns=1.0)
    # The first shot's outcome follows the surrogate's fringe (decay lowers
    # the contrast a little).
    plant.state = dataclasses.replace(plant.state, frequency_offset_hz=offset)
    p1 = sum(p for outcomes, p in result.branches if outcomes[0] == 1)
    assert p1 == pytest.approx(plant.p_ramsey(W0, PHASE0), abs=0.04)
    for calls, (outcomes, _) in zip(result.calls, result.branches):
        (call,) = calls
        mean, sigma, _, _ = call.answer
        expected = cal.frequency_update(0.0, SIGMA0, outcomes[0], W0, PHASE0)
        assert (mean, sigma) == pytest.approx(expected)
    # On average the estimate moves toward the true detuning.
    moved = sum(p * calls[0].answer[0] for calls, (_, p) in zip(result.calls, result.branches))
    assert moved > 0


# ── K13: three-point estimators ─────────────────────────────────────────────


@pulse.kernel
def t1_shot(q, wait_vtu: int):
    d, t = get_drive_line(q)
    drive(d, square(40, PI_AMP), t)
    wait(d, wait_vtu)


@pulse.kernel
def ramsey_shot(q, wait_vtu: int, phase: float):
    d, t = get_drive_line(q)
    drive(d, square(40, PI_AMP / 2), t)
    wait(d, wait_vtu)
    shift_phase(t, phase)
    drive(d, square(40, PI_AMP / 2), t)


def test_k13_three_point_decay_and_phase_recover_the_device(target):
    dev = _device(target)
    q = dev.qudit_handles[0]
    compiled = pulse.compile(t1_shot, device=dev)
    spacing_vtu = 8000
    p = [
        pulse.simulate(compiled.lower(q, 2000 + k * spacing_vtu), dt_ns=4.0)
        .density_matrix[1, 1]
        .real
        for k in range(3)
    ]
    t1_us = cal.three_point_decay(*p, spacing=spacing_vtu / CLOCK) / 1e3
    assert t1_us == pytest.approx(target.qubits[0].t1_us, rel=0.02)
    # A detuned plant puts a phase 2 pi d T on the Ramsey fringe.
    offset = 250e3
    detuned = dataclasses.replace(
        target,
        qubits={
            **target.qubits,
            0: dataclasses.replace(
                target.qubits[0], frequency_hz=target.qubits[0].frequency_hz + offset
            ),
        },
    )
    ramsey = pulse.compile(ramsey_shot, device=dev)
    wait_vtu, theta, delta = 600, math.pi / 2, 0.5
    # The simulated fringe is a - b cos(theta + e) at kernel phase pi - theta.
    p = [
        pulse.simulate(ramsey.lower(q, wait_vtu, math.pi - (theta + k * delta)), plant=detuned)
        .density_matrix[1, 1]
        .real
        for k in (-1, 0, 1)
    ]
    error = cal.three_point_phase(*p, theta=theta, delta=delta)
    separation_ns = (wait_vtu + 40) / CLOCK
    assert error == pytest.approx(2 * math.pi * offset * separation_ns * 1e-9, abs=0.02)


# ── K14: golden-section spectroscopy and Nelder-Mead readout ────────────────


@pulse.kernel
def spectroscopy(q, detuning_hz: float):
    d, t = get_drive_line(q)
    shift_frequency(t, detuning_hz)
    drive(d, square(800, math.pi / 400), t)  # a 400 ns pi pulse


def test_k14_golden_section_finds_the_qubit_frequency(target):
    dev = _device(target)
    q = dev.qudit_handles[0]
    offset = 1.3e6
    detuned = dataclasses.replace(
        target,
        qubits={
            **target.qubits,
            0: dataclasses.replace(
                target.qubits[0], frequency_hz=target.qubits[0].frequency_hz + offset
            ),
        },
    )
    compiled = pulse.compile(spectroscopy, device=dev)
    state = cal.GoldenSection.start(-4e6, 4e6, tolerance=2e4)
    x, done, steps = state.next_point(), False, 0
    while not done:
        p1 = (
            pulse.simulate(compiled.lower(q, x), plant=detuned, dt_ns=1.0).density_matrix[1, 1].real
        )
        state, x, done = cal.golden_section_step(state, 1 - p1)
        steps += 1
    assert x == pytest.approx(offset, abs=4e4)
    assert steps <= 25
    assert compiled.cache_info().compiled == 1


def test_k14_nelder_mead_optimizes_readout_frequency_and_amplitude(target):
    # The cost of a readout setting: minus the separation of the |0> and |1>
    # points, plus a penalty on photons (amplitude squared).
    model = IQModel(target, (0,), decay=False)
    ro_freq = target.qubits[0].readout_params["ro_freq"]
    tau = target.qubits[0].readout_params["ro_integration"]

    def cost(point):
        offset_mhz, amplitude = point
        f = ro_freq + offset_mhz * 1e6
        c0, c1 = (model.center(0, k, f, amplitude, tau) for k in (0, 1))
        return -abs(c1 - c0) + 2.0 * amplitude * amplitude

    state = cal.NelderMead.start([1.5, 0.4], step=0.5, tolerance=1e-4)
    point, done, steps = state.next_point(), False, 0
    while not done and steps < 400:
        state, point, done = cal.nelder_mead_step(state, cost(point))
        steps += 1
    assert done
    grid = [(f, a) for f in np.linspace(-4, 4, 161) for a in np.linspace(0.05, 3, 119)]
    best = min(grid, key=cost)
    assert cost(point) <= cost(best) + 1e-3


GOLDEN = cal.GoldenSection.start(-4e6, 4e6, tolerance=3e6)


@pulse.classical(
    args=(Bits(1), Floats(7)),
    results=(Floats(7), Float(), Float()),
    max_latency_ns=400,
    link="controller",
)
def golden_step(bit, state):
    # A toy cost from one shot: 0 if the qubit was excited, else 1.
    state = cal.GoldenSection.from_floats(state)
    state, x, done = cal.golden_section_step(state, 1.0 - bit[0])
    return state.as_floats(), x, float(done)


@pulse.kernel
def golden_in_shot(q, iterations: int):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    state = GOLDEN.as_floats()
    x = GOLDEN.next_point()
    done = 0.0
    for _ in range(iterations):
        if done < 0.5:
            shift_frequency(t, x)
            drive(d, square(800, math.pi / 400), t)
            shift_frequency(t, -x)
        sync(d, r)
        m = readout(r, square(400, 0.05), rt)
        state, x, done = golden_step.submit([m], state).result()
        sync(d, r)
        if m:
            drive(d, square(40, PI_AMP), t)


def test_k14_golden_section_in_a_shot_uses_the_converged_pattern(target):
    dev = _device(target)
    lowered = pulse.compile(golden_in_shot, device=dev, machine=MACHINE).lower(
        dev.qudit_handles[0], 4
    )
    assert len(lowered.classical_report()) == 4
    result = pulse.simulate(lowered, dt_ns=1.0)
    for calls, (outcomes, _) in zip(result.calls, result.branches):
        state, done = GOLDEN, False
        for call, outcome in zip(calls, outcomes):
            state, x, done = cal.golden_section_step(state, 1.0 - outcome)
            assert call.answer[1] == pytest.approx(x)
            assert call.answer[2] == float(done)
        # The tolerance is reached in at most three costs.
        assert calls[-1].answer[2] == 1.0


# ── K15: closed-loop cadence between shots ──────────────────────────────────

STORE = ParameterStore(wait_vtu=4000.0)
READ_WAIT = STORE.reader("wait_vtu", link="controller", bounds=(0.0, 40000.0))


@pulse.kernel
def cadence_shot(q, amplitude: float):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    drive(d, square(40, amplitude), t)
    wait(d, READ_WAIT.submit().result(), bounds=(0, 40000))
    sync(d, r)
    readout(r, square(400, 0.05), rt)


def test_k15_the_closed_loop_runs_ten_thousand_iterations_on_one_compile(target):
    dev = _device(target)
    q = dev.qudit_handles[0]
    compiled = pulse.compile(cadence_shot, device=dev, machine=MACHINE)
    log = EstimateLog()
    for i in range(10_000):
        STORE.set("wait_vtu", 2000.0 + (i % 100) * 20.0, iteration=i)
        lowered = compiled.lower(q, PI_AMP * (1 + 1e-4 * (i % 10)))
        if i % 2500 == 0:
            result = pulse.simulate(lowered, dt_ns=4.0)
            log.record(i, "p_survive", dict(result.branches).get((1,), 0.0))
    assert compiled.cache_info().compiled == 1
    assert len(log) == 4


# ── K16: tracking a drifting plant against its truth ────────────────────────


def test_k16_tracking_a_drifting_plant_against_its_truth(target):
    plant = DriftingPlant(
        target,
        t1=TelegraphT1(target.qubits[0].t1_us / 4, target.qubits[0].t1_us, rate_hz=0.5),
        frequency=OrnsteinUhlenbeck(100e3, correlation_s=5.0),
        seed=21,
    )
    log = EstimateLog()
    alpha, beta = 1.0, 10_000.0
    mean, sigma = 0.0, 1e6
    for i in range(6000):
        wait_ns = cal.t1_wait(alpha, beta)
        alpha, beta = cal.t1_update(
            alpha, beta, bool(plant.sample(plant.p_survive(wait_ns))), wait_ns
        )
        t, phase = cal.ramsey_setting(mean, sigma, max_wait_ns=2000.0)
        mean, sigma = cal.frequency_update(
            mean, sigma, plant.sample(plant.p_ramsey(t, phase)), t, phase
        )
        if (i + 1) % 150 == 0:
            log.record(i, "t1_us", beta / alpha / 1e3, truth=plant.state.t1_us)
            log.record(
                i, "offset_hz", mean, uncertainty=sigma, truth=plant.state.frequency_offset_hz
            )
            alpha, beta = 2.0, 2.0 * beta / alpha
            sigma = max(sigma, 200e3)  # forget, to follow the drift
        plant.advance(1e-3)
    assert len(log.series("t1_us")) == 40
    split = (plant.t1.low_us + plant.t1.high_us) / 2
    right = sum((e.value > split) == (e.truth > split) for e in log.series("t1_us"))
    assert right >= 30
    assert log.tracking_error("offset_hz") < 50e3
    assert len(plant.truth_trace()["time_s"]) == 6001


# ── K17: hidden and exposed latency ─────────────────────────────────────────


def test_k17_latency_within_depletion_is_hidden(target):
    depleting = dataclasses.replace(
        target,
        qubits={
            **target.qubits,
            0: dataclasses.replace(
                target.qubits[0],
                readout_params={**target.qubits[0].readout_params, "depletion_ns": 250.0},
            ),
        },
    )
    dev = _device(depleting)
    for latency_ns, exposed_ns in ((200.0, 0.0), (400.0, 150.0)):
        decide = pulse.classical(
            lambda bits: [bits[0]],
            name="decide",
            args=(Bits(1),),
            results=(Bits(1),),
            max_latency_ns=latency_ns - 0.5,
        )

        @pulse.kernel
        def feedback(q):
            d, t = get_drive_line(q)
            r, rt = get_readout_line(q)
            m = readout(r, square(400, 0.05), rt)
            if decide.submit([m]).result()[0]:
                drive(d, square(40, PI_AMP), t)

        (call,) = pulse.compile(feedback, device=dev).lower(dev.qudit_handles[0]).classical_report()
        assert call.exposed_vtu / CLOCK == pytest.approx(exposed_ns, abs=1.0)


# ── K18: readout models ─────────────────────────────────────────────────────


def test_k18_readout_models_on_the_target(target):
    params = dict(target.qubits[0].readout_params)
    params.setdefault("error_0to1", 0.02)
    params.setdefault("error_1to0", 0.05)
    params["ro_dur"] = 400.0
    modelled = dataclasses.replace(
        target,
        qubits={**target.qubits, 0: dataclasses.replace(target.qubits[0], readout_params=params)},
    )
    dev = _device(modelled)

    @pulse.kernel
    def excite_and_measure(q):
        d, t = get_drive_line(q)
        r, rt = get_readout_line(q)
        drive(d, square(40, PI_AMP), t)
        sync(d, r)
        readout(r, square(800, 0.05), rt)

    lowered = pulse.compile(excite_and_measure, device=dev).lower(dev.qudit_handles[0])
    ideal = dict(pulse.simulate(lowered, dt_ns=1.0).branches)
    model = pulse.simulate(lowered, dt_ns=1.0, readout="model")
    decay = pulse.simulate(lowered, dt_ns=1.0, readout="model_with_decay")
    reported_zero = lambda r: sum(p for o, p in r.branches if o == (0,))  # noqa: E731
    # Asymmetric errors: |1> is reported 0 at error_1to0 (plus the little
    # it decayed before the readout); decay during the window adds more.
    p1 = ideal.get((1,), 0.0)
    assert reported_zero(model) == pytest.approx(
        (1 - p1) * (1 - params["error_0to1"]) + p1 * params["error_1to0"], abs=1e-6
    )
    assert reported_zero(decay) > reported_zero(model)
    # The dispersive IQ model separates the states on this target.
    iq = IQModel(modelled, (0,), decay=False)
    c0, c1 = (iq.center(0, k, params.get("ro_freq", math.nan), 1.0, 400.0) for k in (0, 1))
    assert abs(c1 - c0) > 1.0
    assert iq.depletion_ns(0) > 0


# ── K19: N-way switch and random source: a randomized benchmarking skeleton ─

# The Pauli group up to phase, as two bits (x, z): I = 0, X = 1, Z = 2,
# Y = 3; a product is the exclusive or.
PAULI = random_index(4, name="pauli", link="controller")


@pulse.classical(
    args=(Integer(), Integer(), Integer()),
    results=(Integer(),),
    max_latency_ns=50,
    link="controller",
)
def recovery(a, b, c):
    return a ^ b ^ c


def _apply(d, t, index):
    if index < 2:
        if index == 1:
            drive(d, square(40, PI_AMP), t)
    else:
        if index == 2:
            shift_phase(t, math.pi)
        else:
            shift_phase(t, math.pi / 2)
            drive(d, square(40, PI_AMP), t)
            shift_phase(t, -math.pi / 2)


@pulse.kernel
def pauli_rb(q):
    d, t = get_drive_line(q)
    a = PAULI.submit().result()
    _apply(d, t, a)
    b = PAULI.submit().result()
    _apply(d, t, b)
    c = PAULI.submit().result()
    _apply(d, t, c)
    _apply(d, t, recovery.submit(a, b, c).result())


def test_k19_pauli_benchmarking_returns_to_zero_unless_miscalibrated(target):
    dev = _device(target)
    lowered = pulse.compile(pauli_rb, device=dev, machine=MACHINE).lower(dev.qudit_handles[0])
    ideal = pulse.simulate(lowered, initial_state=[1, 0], dt_ns=1.0)
    assert len(ideal.branches) == 64  # every sequence of three Paulis
    # Each sequence and its recovery multiply to the identity.
    assert ideal.density_matrix[0, 0].real == pytest.approx(1.0, abs=0.01)
    plant = DriftingPlant(target, amplitude=OrnsteinUhlenbeck(1e-9, 1.0), seed=1)
    plant.state = dataclasses.replace(plant.state, amplitude_scale=1.1)
    drifted = pulse.simulate(lowered, initial_state=[1, 0], dt_ns=1.0, plant=plant.target())
    assert drifted.density_matrix[0, 0].real < ideal.density_matrix[0, 0].real - 0.01
    shots = pulse.simulate(lowered, initial_state=[1, 0], dt_ns=1.0, shots=200, seed=3)
    assert shots.records is not None and len(shots.records) == 200
