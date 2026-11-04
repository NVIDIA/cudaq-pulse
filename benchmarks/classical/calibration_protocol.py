# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The calibration protocol of examples/21 end to end, over many seeds.

The protocol is that of

    M. A. Marciniak, R. T. Birke, J. B. Severin, F. Berritta, D. Kjær,
    F. Nilsson, S. N. Themadath, S. Kallatt, J. L. Webb, K. Bentsen, T. Madsen,
    Z. Sun, S. Krøjer, C. W. Warren, J. Hastrup and M. Kjaergaard,
    "Millisecond-Scale Calibration and Benchmarking of Superconducting Qubits",
    arXiv:2602.11912 (2026).

with adaptive T1 estimation after

    F. Berritta, J. Benestad, J. A. Krzywda, O. Krause, M. A. Marciniak,
    S. Krøjer, C. W. Warren, E. Hogedal, A. Nylander, I. Ahmad, A. Osman,
    J. Biznárová, M. Rommel, A. F. Roudsari, J. Bylander, G. Tancredi,
    J. Danon, J. Hastrup, F. Kuemmeth and M. Kjaergaard, "Real-time adaptive
    tracking of fluctuating relaxation rates in superconducting qubits",
    arXiv:2506.09576 (2025),

and the frequency binary search of

    F. Berritta, J. Benestad, L. Pahl, M. Mathews, J. A. Krzywda, R. Assouly,
    Y. Sung, D. K. Kim, B. M. Niedzielski, K. Serniak, M. E. Schwartz,
    J. L. Yoder, A. Chatterjee, J. A. Grover, J. Danon, W. D. Oliver and
    F. Kuemmeth, "Efficient Qubit Calibration by Binary-Search Hamiltonian
    Tracking", arXiv:2501.05386 (2025).

Every number here comes from pulse.simulate running the example's kernels,
with the estimators as classical functions on the controller link, one
sampled path per shot of the protocol:

- adaptive T1 on a device like that of Berritta et al. (2025):
  fixed-frequency, T1 160 us, assignment errors 0.11 and 0.14, prior
  (3, 450 us), 50 shots; the paper reports 159 us, 68% interval
  [131, 202] us;
- the frequency binary search, 15 shots without reset,
  from a 1 MHz prior over detunings drawn from it;
- the calibration chain of Marciniak et al. (2026) on the
  17-qubit transmon twin's qubit 0 at T1 = 18.3 us; the paper reports T1 =
  18.3 +- 4.7 us from the decay estimator;
- for each step on its own: compile time, simulation time per protocol
  shot and per step, the step's time to decision on the device's timeline,
  and the peak memory of simulating it.

    python benchmarks/classical/calibration_protocol.py [--att 400]
        [--search 200] [--chain 6] [--seed 1]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import runpy
import time
import tracemalloc
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.runtime import _propagate
from cudaq_pulse.targets import transmon_17q

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "21_calibration_protocol.py"
P = runpy.run_path(str(_EXAMPLE), run_name="calibration_protocol")
CLOCK = P["clock_ghz"]
PI_AMP = P["nominal_pi_amplitude"]


def _bootstrap(values, stat, rng, n=2000):
    values = np.asarray(values)
    draws = [stat(values[rng.integers(0, len(values), len(values))]) for _ in range(n)]
    return float(stat(values)), *map(float, np.quantile(draws, [0.025, 0.975]))


def _att_target():
    base = transmon_17q()
    qubit = base.qubits[0]
    att = dataclasses.replace(
        qubit,
        frequency_hz=4.13e9,
        anharmonicity_hz=-210e6,
        t1_us=160.0,
        t2_star_us=100.0,
        thermal_population=0.0,
        readout=qubit.readout.replace(error_1to0=0.11, error_0to1=0.14),
    )
    return dataclasses.replace(base, qubits={**base.qubits, 0: att})


tracked_t1 = P["tracked_t1"]
ramsey = P["ramsey"]
amplitude = P["amplitude"]
decay = P["decay"]
assignment = P["assignment"]
TRAIN = P["pulses_per_train"]


@pulse.kernel
def att_paper(q):
    # Prior (3, 450 us) as a rate in clock cycles; a 800 us passive reset.
    tracked_t1(q, PI_AMP, 3.0, 450_000.0 * CLOCK, 0.11, 0.14, 1_600_000)


@pulse.kernel
def step_assignment(q):
    assignment(q, PI_AMP, P["settle_time"])


@pulse.kernel
def step_ramsey(q):
    ramsey(q, PI_AMP, 0.0)


@pulse.kernel
def step_pi(q):
    amplitude(q, PI_AMP, TRAIN, PI_AMP)


@pulse.kernel
def step_half_pi(q):
    amplitude(q, PI_AMP * 0.5, 2 * TRAIN, PI_AMP)


@pulse.kernel
def step_decay(q):
    decay(q, PI_AMP, 20_000.0 * CLOCK)


@pulse.kernel
def step_tracked(q):
    tracked_t1(q, PI_AMP, 3.0, 3.0 * 20_000.0 * CLOCK, 0.03, 0.02, P["settle_time"])


def _shares(result, shots):
    """Each path's last calls and its number of shots."""
    for branch in result.branches:
        yield branch.calls, round(branch.probability * shots)


def adaptive_t1(n, seed):
    target = _att_target()
    dev = pulse.device({"target": target})
    lowered = pulse.compile(att_paper, device=dev, machine=P["machine"]).lower(dev.qudit_handles[0])
    start = time.perf_counter()
    result = pulse.simulate(
        lowered, shots=n, seed=seed, branches="sampled", readout="model", levels=2
    )
    seconds = time.perf_counter() - start
    estimates, posteriors = [], []
    for calls, count in _shares(result, n):
        shape, rate, _ = calls[-1].answer
        estimates += [rate / shape / CLOCK / 1e3] * count
        posteriors += [(shape, rate)] * count
    rng = np.random.default_rng(seed)
    order = np.argsort(estimates)
    shape, rate = posteriors[int(order[len(order) // 2])]
    # The rate is per clock cycle: T1 in us is 1 / (rate x clock x 1e3).
    draws = 1e-3 / (CLOCK * rng.gamma(shape, 1.0 / rate, 200_000))
    lo, hi = np.quantile(draws, [0.16, 0.84])
    return {
        "trajectories": n,
        "median_T1_us": _bootstrap(estimates, np.median, rng),
        "median_run_68pct_interval_us": [float(lo), float(hi)],
        "spread_16_84_us": list(map(float, np.quantile(estimates, [0.16, 0.84]))),
        "paper": "159 us, 68% interval [131, 202] us (true T1 unknown, about 160 us)",
        "reference_implementation": "156 us, [125, 209] us at a true 160 us",
        "seconds_per_trajectory": seconds / n,
    }


def binary_search(n, seed):
    target = transmon_17q()
    dev = pulse.device({"target": target})
    lowered = pulse.compile(P["binary_search"], device=dev, machine=P["machine"]).lower(
        dev.qudit_handles[0], 15
    )
    rng = np.random.default_rng(seed)
    errors, sigmas = [], []
    start = time.perf_counter()
    for k in range(n):
        offset = float(rng.normal(0.0, 1e6))
        result = pulse.simulate(
            lowered,
            plant=P["search_plant"](target, offset),
            shots=1,
            seed=seed * 100_003 + k,
            branches="sampled",
            readout="model",
            levels=2,
        )
        mean, sigma, _, _ = result.branches[0].calls[-1].answer
        errors.append(mean - offset)
        sigmas.append(sigma)
    seconds = time.perf_counter() - start
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
    return {
        "trajectories": n,
        "median_sigma_kHz": [v / 1e3 for v in _bootstrap(sigmas, np.median, rng)],
        "rms_error_kHz": [v / 1e3 for v in _bootstrap(errors, rms, rng)],
        "reference_implementation": "366 kHz, rms error 647 kHz",
        "paper": "67 kHz in one run; the equations narrow by about 0.37 per 15 shots",
        "seconds_per_trajectory": seconds / n,
    }


def chain(n, seed):
    target = transmon_17q()
    dev = pulse.device({"target": target})
    lowered = pulse.compile(P["calibration_chain"], device=dev, machine=P["machine"]).lower(
        dev.qudit_handles[0], 20_000.0 * CLOCK
    )
    plant = P["chain_plant"](target)
    rows = []
    start = time.perf_counter()
    for k in range(n):
        result = pulse.simulate(
            lowered,
            plant=plant,
            shots=1,
            seed=seed * 7919 + k,
            branches="sampled",
            readout="model",
            levels=2,
        )
        rows.append(P["chain_results"](result.branches[0]))
    seconds = time.perf_counter() - start
    truth = {
        "detuning (kHz)": 60.0,
        "pi amplitude": PI_AMP / 1.015,
        "pi/2 amplitude": PI_AMP / 2 / 1.015,
        "T1, decay estimator (us)": 18.3,
        "T1, adaptive (us)": 18.3,
    }
    out = {"trajectories": n, "seconds_per_trajectory": seconds / n}
    for name, value in truth.items():
        values = np.array([row[name] for row in rows])
        se = values.std(ddof=1) / math.sqrt(n) if n > 1 else math.nan
        out[name] = {
            "truth": value,
            "mean": float(values.mean()),
            "sd": float(values.std(ddof=1)) if n > 1 else math.nan,
            "ci95": [float(values.mean() - 1.96 * se), float(values.mean() + 1.96 * se)],
        }
    errors = np.array([row["readout errors"] for row in rows])
    out["readout errors"] = {
        "plant": [0.03, 0.02],
        "mean": errors.mean(axis=0).tolist(),
        "note": "measured with a passive reset, as the adaptive tracker uses",
    }
    out["paper_decay_T1"] = "18.3 +- 4.7 us"
    return out


def steps(seed):
    target = transmon_17q()
    dev = pulse.device({"target": target})
    q = dev.qudit_handles[0]
    plant = P["chain_plant"](target)
    shots_per_step = {
        "readout errors": (step_assignment, 2 * P["shots_per_setting"] + 1),
        "Ramsey frequency": (step_ramsey, 3 * P["shots_per_setting"]),
        "pi amplitude": (step_pi, 3 * P["shots_per_setting"]),
        "pi/2 amplitude": (step_half_pi, 3 * P["shots_per_setting"]),
        "decay T1": (step_decay, 3 * P["shots_per_setting"]),
        "adaptive T1": (step_tracked, P["tracker_shots"]),
    }
    out = {}
    for name, (kernel, shots) in shots_per_step.items():
        start = time.perf_counter()
        lowered = pulse.compile(kernel, device=dev, machine=P["machine"]).lower(q)
        compile_s = time.perf_counter() - start
        start = time.perf_counter()
        result = pulse.simulate(
            lowered, plant=plant, shots=1, seed=seed, branches="sampled", readout="model", levels=2
        )
        simulate_s = time.perf_counter() - start
        # Tracing allocations slows NumPy several times over: measure apart,
        # from cold (no propagators kept).
        _propagate.clear_cache()
        tracemalloc.start()
        pulse.simulate(
            lowered, plant=plant, shots=1, seed=seed, branches="sampled", readout="model", levels=2
        )
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        m = lowered.metrics
        out[name] = {
            "shots": shots,
            "ops": m.op_count,
            "trace_ms": m.trace_ms,
            "compile_ms": compile_s * 1e3,
            "simulate_s": simulate_s,
            "simulate_ms_per_shot": simulate_s / shots * 1e3,
            "time_to_decision_ms": result.times_ns[1] / 1e6,
            "simulate_peak_MB": peak / 2**20,
        }
    out["paper_time_to_decision_ms"] = {
        "Ramsey frequency": 7.5,
        "pi amplitude": 1.1,
        "decay T1": 9.8,
        "adaptive T1 (T1 160 us)": 11.0,
        "note": "set by reset and sequence length; this protocol resets actively in about 1 us",
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--att", type=int, default=400)
    parser.add_argument("--search", type=int, default=200)
    parser.add_argument("--chain", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    report = {
        "adaptive_t1": adaptive_t1(args.att, args.seed),
        "binary_search": binary_search(args.search, args.seed),
        "steps": steps(args.seed),
        "chain": chain(args.chain, args.seed),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
