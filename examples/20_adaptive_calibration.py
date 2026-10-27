# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Adaptive calibration against a drifting device.

Builds on ``19_classical_calls.py``. On qubit 0 of the Krinner 17-qubit
transmon twin, this example:

1. Runs a frequency binary search inside one shot: after each Ramsey shot an
   estimator on the controller link updates a Gaussian posterior over the
   detuning and picks the next wait (a variable wait) and phase.
2. Tracks a drifting device between shots: T1 switches between two values
   and the frequency wanders. A parameter store holds what the programs run
   with, an estimate log records each estimate with the truth, and the
   kernel is compiled once for the whole loop. The loop uses the plant's
   surrogates for speed and checks one iteration against full simulation.

The estimators come from ``cudaq_pulse.classical.calibration``; the plant
from ``cudaq_pulse.runtime.drift``.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import dataclasses
import math

import cudaq_pulse as pulse
from cudaq_pulse.classical import Bits, Float, Link
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.classical.loop import EstimateLog, ParameterStore
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.runtime.drift import DriftingPlant, OrnsteinUhlenbeck, TelegraphT1
from cudaq_pulse.targets import transmon_krinner_17q

PI_AMP = math.pi / 20  # square(40, PI_AMP) is a pi pulse
CLOCK_GHZ = 2.0
MACHINE = MachineModel(links={"controller": Link(fixed_ns=4, feedback_ns=8)})

# ── 1. Frequency binary search in one shot ──────────────────────────────────

SIGMA0_HZ = 2e6
WAIT0_NS, PHASE0 = cal.ramsey_setting(0.0, SIGMA0_HZ)
BOUNDS_CYCLES = (20, 40000)


@pulse.classical(
    args=(Bits(1), Float(), Float(), Float(), Float()),
    results=(Float(), Float(), Float(), Float()),
    max_latency_ns=300,
    link="controller",
)
def frequency_estimator(bit, mean_hz, sigma_hz, wait_cycles, phase):
    """Update the detuning posterior from one Ramsey shot; pick the next."""
    # The Ramsey wait counts from the first pi/2 pulse's centre to the second's.
    mean_hz, sigma_hz = cal.frequency_update(
        mean_hz, sigma_hz, int(bit[0]), wait_cycles / CLOCK_GHZ + 20.0, phase
    )
    wait_ns, phase = cal.ramsey_setting(mean_hz, sigma_hz)
    wait_cycles = min(max((wait_ns - 20.0) * CLOCK_GHZ, BOUNDS_CYCLES[0]), BOUNDS_CYCLES[1])
    return mean_hz, sigma_hz, wait_cycles, phase


@pulse.kernel
def binary_search(q, rounds: int):
    d, t = get_drive_line(q)
    r, rt = get_readout_line(q)
    mean, sigma = 0.0, SIGMA0_HZ
    tau = (WAIT0_NS - 20.0) * CLOCK_GHZ
    phase = PHASE0
    # A Ramsey shot: pi/2, wait, pi/2 with its tone shifted by pi - phase.
    drive(d, square(40, PI_AMP / 2), t)
    wait(d, int(tau))
    shift_phase(t, math.pi - phase)
    drive(d, square(40, PI_AMP / 2), t)
    for _ in range(rounds):
        sync(d, r)
        m = readout(r, square(400, 0.05), rt)
        mean, sigma, tau, phase = frequency_estimator.submit([m], mean, sigma, tau, phase).result()
        sync(d, r)
        if m:  # active reset
            drive(d, square(40, PI_AMP), t)
        drive(d, square(40, PI_AMP / 2), t)
        wait(d, tau, bounds=BOUNDS_CYCLES)
        shift_phase(t, math.pi - phase)
        drive(d, square(40, PI_AMP / 2), t)
    sync(d, r)
    readout(r, square(400, 0.05), rt)


# ── 2. Tracking a drifting device between shots ─────────────────────────────


@pulse.kernel
def t1_shot(q, wait_ns: int):
    d, t = get_drive_line(q)
    drive(d, square(40, PI_AMP), t)
    wait(d, wait_ns * 2)


if __name__ == "__main__":
    target = transmon_krinner_17q()
    dev = pulse.device({"target": target})
    q = dev.qudit_handles[0]
    qubit = target.qubits[0]

    print("=== 1. Frequency binary search in one shot (2 rounds) ===")
    offset_hz = 400e3
    detuned = dataclasses.replace(
        target,
        qubits={
            **target.qubits,
            0: dataclasses.replace(qubit, frequency_hz=qubit.frequency_hz + offset_hz),
        },
    )
    lowered = pulse.compile(binary_search, device=dev, machine=MACHINE).lower(q, 2)
    result = pulse.simulate(lowered, plant=detuned, dt_ns=1.0)
    print(f"  true detuning {offset_hz / 1e3:.0f} kHz; prior 0 +- {SIGMA0_HZ / 1e6:.0f} MHz")
    for calls, (outcomes, p) in sorted(zip(result.calls, result.branches), key=lambda c: -c[1][1]):
        mean, sigma, _, _ = calls[-1].answer
        print(
            f"  outcomes {outcomes}  p = {p:.3f}  estimate {mean / 1e3:+7.0f} +- {sigma / 1e3:.0f} kHz"
        )
    expected = sum(p * calls[-1].answer[0] for calls, (_, p) in zip(result.calls, result.branches))
    print(f"  mean estimate over paths: {expected / 1e3:+.0f} kHz")

    print("\n=== 2. Tracking a drifting device between shots ===")
    telegraph = TelegraphT1(qubit.t1_us / 4, qubit.t1_us, rate_hz=0.5)
    plant = DriftingPlant(
        target,
        t1=telegraph,
        frequency=OrnsteinUhlenbeck(100e3, correlation_s=5.0),
        seed=21,
    )
    store = ParameterStore(alpha=1.0, beta=10_000.0, mean_hz=0.0, sigma_hz=1e6)
    log = EstimateLog()
    compiled = pulse.compile(t1_shot, device=dev)
    iterations, block = 6000, 150
    for i in range(iterations):
        alpha, beta, mean, sigma = store.values("alpha", "beta", "mean_hz", "sigma_hz")
        # T1: excite, wait, measure.
        wait_ns = cal.t1_wait(alpha, beta)
        alpha, beta = cal.t1_update(
            alpha, beta, bool(plant.sample(plant.p_survive(wait_ns))), wait_ns
        )
        # Frequency: one Ramsey shot.
        t, phase = cal.ramsey_setting(mean, sigma, max_wait_ns=2000.0)
        mean, sigma = cal.frequency_update(
            mean, sigma, plant.sample(plant.p_ramsey(t, phase)), t, phase
        )
        if (i + 1) % block == 0:
            log.record(
                i, "t1_us", beta / alpha / 1e3, truth=plant.state.t1_us, time_s=plant.state.time_s
            )
            log.record(
                i, "offset_hz", mean, uncertainty=sigma, truth=plant.state.frequency_offset_hz
            )
            # Forget, to follow the drift.
            alpha, beta = 2.0, 2.0 * beta / alpha
            sigma = max(sigma, 200e3)
        store.update(
            {"alpha": alpha, "beta": beta, "mean_hz": mean, "sigma_hz": sigma}, iteration=i
        )
        plant.advance(1e-3)
    for e in log.series("t1_us")[:6]:
        print(f"  t = {e.time_s:5.2f} s  T1 estimate {e.value:5.1f} us  true {e.truth:5.1f} us")
    split = (telegraph.low_us + telegraph.high_us) / 2
    blocks = [(e.value, e.truth) for e in log.series("t1_us") if e.truth is not None]
    right = sum((value > split) == (truth > split) for value, truth in blocks)
    print(f"  T1 state right in {right} of {len(blocks)} blocks of {block} shots")
    print(f"  frequency tracking error (rms): {log.tracking_error('offset_hz') / 1e3:.1f} kHz")
    print(f"  store version {store.version} after {iterations} iterations")

    # One iteration against full simulation of the plant as it is now; they
    # differ by the qubit's thermal population, which the surrogate leaves out.
    wait_ns = 8000
    p_full = (
        pulse.simulate(compiled.lower(q, wait_ns), plant=plant.target(), dt_ns=4.0)
        .density_matrix[1, 1]
        .real
    )
    print(
        f"  check at {wait_ns / 1e3:.0f} us: simulated P(1) = {p_full:.3f}, "
        f"surrogate {plant.p_survive(wait_ns + 20):.3f}"
    )
