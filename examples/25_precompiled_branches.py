# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compiling every branch of an adaptive experiment ahead of time.

Builds on ``24_pipelined_calibration.py``. A check measures whether each of
four qudits of the 17-qudit transmon twin (``transmon_17q``) still has a
good pi pulse. Which qudits fail decides the next calibration step: an
amplitude sweep of amplified pi trains on exactly those qudits, with the
others only checked. Four qudits give 15 possible next programs, plus "all
in spec", which needs none.

Worker processes, started with the ``spawn`` method, compile all 15
programs up front, in parallel, and bind and export every point of each
sweep. The kernel and the qudit register travel to them as task arguments,
and the results come back pickled into a table keyed by the check's
outcome. At run time the host does not compile anything: it looks up the
ready programs. The example measures that lookup against compiling on
demand.

The decision is made on the host, between shots. Feedback within a shot
belongs in the kernel (see ``08_readout_and_branching.py``).

What is simulated: the simulator plays the QPU. Each round redraws the
drive strengths of a "true" device by a few percent. The check and the
sweep are simulated against it up to the point where the readouts start,
and the shot outcomes are binomial draws from the populations there; the
readouts themselves are not simulated. The four qudits share no couplings,
so each one's sweep reads out on its own. The compile, the pickling and the
lookup times are real.
"""

import concurrent.futures
import dataclasses
import itertools
import math
import multiprocessing
import statistics
import time

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.targets import MicrowaveDrive, transmon_17q

target = transmon_17q()
qudit_indices = (0, 1, 2, 3)
repetitions = 9
pulse_duration = 40 * pulse.ns
pulse_sigma = 10 * pulse.ns
readout_duration = 300 * pulse.ns
scales = np.linspace(0.95, 1.05, 9)
spec = 0.93  # a qudit whose amplified train reaches P(1) below this is recalibrated
shots = 400
rounds = 4
workers = 4
sample_rate_ghz = 2.0

# The amplitude of a pi pulse: a rotation of pi from the Gaussian's area (ns)
# and the drive's rad/ns per unit amplitude.
area_ns = (pulse_sigma / pulse.ns) * math.sqrt(2 * math.pi) * math.erf(2 / math.sqrt(2))
pi_amplitudes = tuple(math.pi / (area_ns * target.drive_amplitude_scale(q)) for q in qudit_indices)


@pulse.kernel
def pi_trains(qudits, trained, scale: float):
    """An amplified pi train, at *scale* times the pi amplitude, on each qudit
    in *trained*; one pi pulse on every other qudit; then read all of them out."""
    for i in range(len(qudits)):
        d = get_drive_line(qudits[i])
        r = get_readout_line(qudits[i])
        if i in trained:
            for _ in range(repetitions):
                drive(d, gaussian(pulse_duration, pi_amplitudes[i] * scale, pulse_sigma))
        else:
            drive(d, gaussian(pulse_duration, pi_amplitudes[i], pulse_sigma))
        sync(d, r)
        readout(r, square(readout_duration, 0.05))


def every_outcome():
    """Each nonempty set of qudits that can fail the check."""
    return [
        trained
        for size in range(1, len(qudit_indices) + 1)
        for trained in itertools.combinations(range(len(qudit_indices)), size)
    ]


def compile_branch(kernel: pulse.Kernel[...], qudits: pulse.Register, trained: tuple[int, ...]):
    """The sweep for one outcome: each point bound and exported for the controller."""
    program = pulse.compile(kernel, device=qudits[0].device).lower(qudits, trained, 1.0).generic
    sweep = []
    for scale in scales:
        bound = program.bind([scale])
        sweep.append((bound, bound.export_controller(sample_rate_ghz=sample_rate_ghz)))
    return trained, sweep


# ── The simulated QPU ────────────────────────────────────────────────────────


def drifted(factors):
    """The target with each qudit's drive strength multiplied by its factor."""
    qubits = dict(target.qubits)
    for q, factor in zip(qudit_indices, factors, strict=True):
        strength = target.drive_amplitude_scale(q) * factor
        given = qubits[q].drive
        assert isinstance(given, MicrowaveDrive)
        stronger = given.replace(amplitude_scale_rad_per_ns=strength)
        qubits[q] = qubits[q].replace(drive=stronger)
    return dataclasses.replace(target, qubits=qubits)


def populations(program, plant):
    """P(1) of each qudit where the readouts start."""
    drives_end = repetitions * pulse_duration
    result = pulse.simulate(program, levels=2, dt_ns=1.0, plant=plant, t_end=drives_end)
    return [result.level_population(1, qudit=q) for q in qudit_indices]


def best_scale(p1):
    """The scale at the top of a parabola through the sweep's best three points."""
    k = int(np.clip(np.argmax(p1), 1, len(p1) - 2))
    a, b, _ = np.polyfit(scales[k - 1 : k + 2], p1[k - 1 : k + 2], 2)
    return -b / (2 * a)


def median_s(operation, repeat):
    samples = []
    for _ in range(repeat):
        started = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


if __name__ == "__main__":
    device = pulse.device(target)
    qudits = device.qudits[list(qudit_indices)]
    outcomes = every_outcome()
    compile_branch(pi_trains, qudits, outcomes[-1])  # compile once here before timing anything

    print(f"=== Compiling all {len(outcomes)} next programs ahead of time ===")
    started = time.perf_counter()
    serial = dict(compile_branch(pi_trains, qudits, trained) for trained in outcomes)
    serial_s = time.perf_counter() - started
    print(f"  in this process: {serial_s * 1e3:.1f} ms")

    spawn = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(workers, mp_context=spawn) as pool:
        list(pool.map(math.sqrt, range(workers)))
        startup_s = time.perf_counter() - started
        # Warm each worker's compiler, as this process's was warmed above.
        warm = [outcomes[-1]] * workers
        list(pool.map(compile_branch, [pi_trains] * workers, [qudits] * workers, warm))
        started = time.perf_counter()
        futures = [pool.submit(compile_branch, pi_trains, qudits, t) for t in outcomes]
        table = dict(future.result() for future in futures)
        parallel_s = time.perf_counter() - started
    print(
        f"  in {workers} worker processes: {parallel_s * 1e3:.1f} ms, pickled back "
        f"(starting the workers took {startup_s:.2f} s, once)"
    )
    points = sum(len(sweep) for sweep in table.values())
    print(f"  table: {len(table)} outcomes, {points} bound and exported programs")
    same = all(
        a.export_controller() == b.export_controller()
        for t in outcomes
        for (a, _), (b, _) in zip(table[t], serial[t], strict=True)
    )
    print(f"  identical to the programs compiled here: {same}")

    print("\n=== At run time: the ready program against compiling on demand ===")
    some = outcomes[len(outcomes) // 2]
    lookup_s = median_s(lambda: table.get(some), 10_000)
    on_demand_s = median_s(lambda: compile_branch(pi_trains, qudits, some), 20)
    compile_s = median_s(
        lambda: pulse.compile(pi_trains, device=device).lower(qudits, some, 1.0), 20
    )
    print(f"  looking up the ready sweep: {lookup_s * 1e6:.2f} us")
    print(
        f"  compiling, binding and exporting it on demand: {on_demand_s * 1e3:.2f} ms "
        f"(compiling {compile_s * 1e3:.2f} ms of it), {on_demand_s / lookup_s:,.0f}x longer"
    )

    print(f"\n=== {rounds} rounds against a drifting device ===")
    rng = np.random.default_rng(7)
    check = table[tuple(range(len(qudit_indices)))][len(scales) // 2][0]  # every train, scale 1
    for round_ in range(1, rounds + 1):
        factors = 1 + 0.02 * rng.standard_normal(len(qudit_indices))
        plant = drifted(factors)
        measured = [rng.binomial(shots, p) / shots for p in populations(check, plant)]
        outcome = tuple(i for i, p in enumerate(measured) if p < spec)
        sweep = table.get(outcome)
        shown = " ".join(f"{p:.3f}" for p in measured)
        print(f"  round {round_}: check P(1) [{shown}]")
        if sweep is None:
            print("    all in spec: nothing to recalibrate")
            continue
        print(
            f"    recalibrate qudits {list(outcome)}: {len(sweep)} ready programs, "
            f"{len(sweep[0][1]['events'])} controller events each"
        )
        p1 = rng.binomial(shots, [populations(bound, plant) for bound, _ in sweep]) / shots
        for i in outcome:
            print(
                f"    qudit {i}: amplitude x{best_scale(p1[:, i]):.4f} "
                f"(the drift needs x{1 / factors[i]:.4f})"
            )
