# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive calibration against a drifting quantum system.

Builds on ``19_classical_calls.py``. On qubit 0 of the 17-qubit transmon
twin (``transmon_surface17``), this example:

1. Runs a frequency binary search inside one shot: after each Ramsey shot an
   estimator on the controller link updates a Gaussian posterior over the
   detuning and picks the next wait (a variable wait) and phase.
2. Tracks a drifting quantum system between shots: T1 switches between two values
   and the frequency wanders. A parameter store holds what the programs run
   with, an estimate log records each estimate with the truth, and the
   kernel is compiled once for the whole loop. The loop draws its shots from
   the true values' outcome probabilities for speed and checks one
   iteration against full simulation.

The estimators and outcome probabilities come from
``cudaq_pulse.classical.calibration``; the drift from
``cudaq_pulse.quantum_systems.drift``.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import math

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse import ControlSystem
from cudaq_pulse.classical import Link
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.classical.loop import EstimateLog, ParameterStore
from cudaq_pulse.quantum_systems import transmon_surface17
from cudaq_pulse.quantum_systems.drift import OrnsteinUhlenbeck, ParameterDrift, TelegraphT1

# square(pi_duration, pi_amp) is a pi pulse; at half the amplitude, a pi/2.
pulse_cal = {
    "pi_duration": 20 * pulse.ns,
    "pi_amp": math.pi / 20,
    "readout_duration": 200 * pulse.ns,
}
pi_duration_ns = pulse_cal["pi_duration"] / pulse.ns
control_system = ControlSystem(links={"controller": Link(fixed_ns=4, feedback_ns=8)})

# ── 1. Frequency binary search in one shot ──────────────────────────────────

initial_sigma_hz = 2e6
initial_wait_ns, initial_phase = cal.ramsey_setting(0.0, initial_sigma_hz)
wait_bounds = (10 * pulse.ns, 20 * pulse.us)
wait_bounds_ns = tuple(bound / pulse.ns for bound in wait_bounds)


@pulse.classical(max_latency_ns=300, link="controller")
def frequency_estimator(
    bit: pulse.Bits,
    mean_hz: float,
    sigma_hz: float,
    gap_ns: float,
    phase: float,
) -> tuple[float, float, float, float]:
    """Update the detuning posterior from one Ramsey shot; pick the next."""
    # The Ramsey wait counts from the first pi/2 pulse's centre to the
    # second's; the gap between the pulses is a pulse shorter.
    mean_hz, sigma_hz = cal.frequency_update(
        mean_hz, sigma_hz, bit[0], gap_ns + pi_duration_ns, phase
    )
    wait_ns, phase = cal.ramsey_setting(mean_hz, sigma_hz)
    gap_ns = min(max(wait_ns - pi_duration_ns, wait_bounds_ns[0]), wait_bounds_ns[1])
    return mean_hz, sigma_hz, gap_ns, phase


@pulse.kernel
def binary_search(q, rounds: int):
    d = get_drive_line(q)
    r = get_readout_line(q)
    pi_pulse = square(pulse_cal["pi_duration"], pulse_cal["pi_amp"])
    half_pi_pulse = square(pulse_cal["pi_duration"], pulse_cal["pi_amp"] / 2)
    readout_pulse = square(pulse_cal["readout_duration"], 0.05)
    mean, sigma = 0.0, initial_sigma_hz
    gap_ns = initial_wait_ns - pi_duration_ns
    phase = initial_phase
    # A Ramsey shot: pi/2, wait, pi/2 with its tone shifted by pi - phase.
    drive(d, half_pi_pulse)
    wait(d, pulse.round_duration(d, gap_ns * pulse.ns, "down"))
    shift_phase(d.tone, math.pi - phase)
    drive(d, half_pi_pulse)
    for _ in range(rounds):
        sync(d, r)
        m = readout(r, readout_pulse)
        mean, sigma, gap_ns, phase = frequency_estimator.submit(
            [m], mean, sigma, gap_ns, phase
        ).result()
        sync(d, r)
        if m:  # active reset
            drive(d, pi_pulse)
        drive(d, half_pi_pulse)
        wait(d, gap_ns * pulse.ns, bounds=wait_bounds)
        shift_phase(d.tone, math.pi - phase)
        drive(d, half_pi_pulse)
    sync(d, r)
    readout(r, readout_pulse)


# ── 2. Tracking a drifting quantum system between shots ─────────────────────────────


@pulse.kernel
def t1_shot(q, wait_ns: int):
    d = get_drive_line(q)
    drive(d, square(pulse_cal["pi_duration"], pulse_cal["pi_amp"]))
    wait(d, wait_ns * pulse.ns)


if __name__ == "__main__":
    qs = transmon_surface17()
    q = qs.qudits[0]
    qudit = qs.physical_qudits[0]

    print("=== 1. Frequency binary search in one shot (2 rounds) ===")
    offset_hz = 400e3
    detuned = qs.replace(
        qudits={
            **qs.physical_qudits,
            0: qudit.replace(frequency_hz=qudit.frequency_hz + offset_hz),
        }
    )
    scheduled = pulse.compile(
        binary_search, quantum_system=qs, control_system=control_system
    ).lower(q, 2)
    result = pulse.simulate(scheduled, quantum_system=detuned, dt_ns=0.25)
    print(f"  true detuning {offset_hz / 1e3:.0f} kHz; prior 0 +- {initial_sigma_hz / 1e6:.0f} MHz")
    for branch in sorted(result.branches, key=lambda branch: -branch.probability):
        mean, sigma, _, _ = branch.answers("frequency_estimator")[-1]
        print(
            f"  outcomes {branch.levels}  p = {branch.probability:.3f}"
            f"  estimate {mean / 1e3:+7.0f} +- {sigma / 1e3:.0f} kHz"
        )
    expected = sum(
        branch.probability * branch.answers("frequency_estimator")[-1][0]
        for branch in result.branches
    )
    print(f"  mean estimate over branches: {expected / 1e3:+.0f} kHz")

    print("\n=== 2. Tracking a drifting quantum system between shots ===")
    # T1 switches between its calibrated value and a quarter of it, and the
    # frequency wanders: drift far faster than a real chip's.
    drift = ParameterDrift(
        frequency=OrnsteinUhlenbeck(100e3, correlation_time=5 * pulse.s),
        t1=TelegraphT1(low_fraction=0.25, dwell_time=2 * pulse.s),
        amplitude=None,
    )
    qpu = qs.replace(parameter_drift={0: drift}, parameter_drift_seed=21)
    shots = np.random.default_rng(21)
    store = ParameterStore(alpha=1.0, beta=10_000.0, mean_hz=0.0, sigma_hz=1e6)
    log = EstimateLog()
    compiled = pulse.compile(t1_shot, quantum_system=qpu)
    iterations, block = 6000, 150
    for i in range(iterations):
        alpha, beta, mean, sigma = store.values("alpha", "beta", "mean_hz", "sigma_hz")
        true = qpu.true_qudits[0]
        # T1: excite, wait, measure.
        wait_ns = cal.t1_wait(alpha, beta)
        survived = shots.random() < cal.survival_probability(wait_ns, true.t1_us)
        alpha, beta = cal.t1_update(alpha, beta, bool(survived), wait_ns)
        # Frequency: one Ramsey shot.
        t, phase = cal.ramsey_setting(mean, sigma, max_wait_ns=2000.0)
        offset_hz = true.frequency_hz - qudit.frequency_hz
        outcome = int(shots.random() < cal.ramsey_probability(t, phase, offset_hz))
        mean, sigma = cal.frequency_update(mean, sigma, outcome, t, phase)
        if (i + 1) % block == 0:
            log.record(
                i,
                "t1_us",
                beta / alpha / 1e3,
                truth=true.t1_us,
                time_s=qpu.parameter_drift_time / pulse.s,
            )
            log.record(i, "offset_hz", mean, uncertainty=sigma, truth=offset_hz)
            # Forget, to follow the drift.
            alpha, beta = 2.0, 2.0 * beta / alpha
            sigma = max(sigma, 200e3)
        store.update(
            {"alpha": alpha, "beta": beta, "mean_hz": mean, "sigma_hz": sigma}, iteration=i
        )
        qpu.advance(1 * pulse.ms)
    for e in log.series("t1_us")[:6]:
        print(f"  t = {e.time_s:5.2f} s  T1 estimate {e.value:5.1f} us  true {e.truth:5.1f} us")
    split = qudit.t1_us * (1 + 0.25) / 2
    blocks = [(e.value, e.truth) for e in log.series("t1_us") if e.truth is not None]
    right = sum((value > split) == (truth > split) for value, truth in blocks)
    print(f"  T1 state right in {right} of {len(blocks)} blocks of {block} shots")
    print(f"  frequency tracking error (rms): {log.tracking_error('offset_hz') / 1e3:.1f} kHz")
    print(f"  store version {store.version} after {iterations} iterations")

    # One iteration against full simulation: the program is compiled from the
    # calibrated values and plays on the true ones as they are now. They
    # differ by the qubit's thermal population, which the probability leaves out.
    wait_ns = 8000
    t1_now = qpu.true_qudits[0].t1_us
    p_full = pulse.simulate(compiled.lower(qpu.qudits[0], wait_ns), dt_ns=0.25).level_population(1)
    print(
        f"  check at {wait_ns / 1e3:.0f} us: simulated P(1) = {p_full:.3f}, "
        f"from the true T1 {cal.survival_probability(wait_ns + 20, t1_now):.3f}"
    )
