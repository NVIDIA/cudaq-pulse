# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-loop calibration with a compile pipeline in worker processes.

Builds on ``20_adaptive_calibration.py``. An optimizer tunes the amplitude
and the detuning of a pi pulse on one qudit of the 17-qudit transmon twin
(``transmon_17q``), measured through a train of 9 pi pulses, which adds up
both errors. Each batch runs a few of its suggestions; their results update
it and it suggests the next batch. Two loops are compared:

- serial: suggest, prepare, run, observe. The QPU waits while the host
  suggests the next settings and prepares their programs.
- pipelined: while batch N runs, the host suggests batch N+1 from what it
  knows so far (results up to batch N-1) and worker processes prepare it,
  so the QPU starts batch N+1 as soon as batch N ends.

Compiled programs pickle, which is what makes the pipeline possible. The
kernel is compiled once, here. Each worker receives the compiled program
(MLIR bytecode, not the kernel's source) and only binds new values and
exports the result for the controller. It never compiles. The workers are
started with the ``spawn`` method, so this works wherever that is the
default too.

What is simulated: the QPU is a stand-in process. It takes a prepared batch,
spends a fixed time on it (an assumed 50 us per shot), and returns counts
drawn from ``pulse.simulate`` of each program against a "true" qudit whose
frequency is 1.5 MHz off the device description's. A real controller would
load the export; the stand-in only checks that it has events. The compile,
the pickling, the hand-offs between processes, the workers' binding and
exporting and the optimizer's run time are real. The best detuning is not
just that offset: the pulse itself shifts the resonance a little (the
simulation keeps three levels), so the amplitude and the detuning are found
together. The optimizer is Gaussian-process Thompson sampling on a grid of
settings, written in NumPy.
"""

import concurrent.futures
import dataclasses
import math
import multiprocessing
import pickle
import time

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.quantum_systems import DeviceDescription, transmon_17q

target = transmon_17q()
pulse_duration = 40 * pulse.ns
pulse_sigma = 10 * pulse.ns
# A pi pulse rotates by pi: the Gaussian's area (ns) times the drive's rad/ns
# per unit amplitude.
area_ns = (pulse_sigma / pulse.ns) * math.sqrt(2 * math.pi) * math.erf(2 / math.sqrt(2))
pi_amplitude = math.pi / (area_ns * target.drive_amplitude_scale(0))

# An odd number of pi pulses ends in |1>; amplitude and frequency errors add up along it.
repetitions = 9
amplitudes = pi_amplitude * np.linspace(0.85, 1.15, 25)
detunings_hz = np.linspace(-2e6, 6e6, 25)
settings = np.array([(a, d) for a in amplitudes for d in detunings_hz])
settings_per_batch = 8
shots_count = 100
shot_time = 50 * pulse.us  # the stand-in QPU's repetition period
batch_time = settings_per_batch * shots_count * shot_time
batches = 20
workers = 4
sample_rate_ghz = 2.0
hidden_offset_hz = 1.5e6  # the true qudit's frequency minus the quantum system's


@pulse.kernel
def pi_train(qudit, repetitions: int, amplitude: float, detuning_hz: float):
    """*repetitions* Gaussian pi pulses on a tone shifted by *detuning_hz*."""
    d = get_drive_line(qudit)
    shift_frequency(d.tone, detuning_hz)
    for _ in range(repetitions):
        drive(d, gaussian(pulse_duration, amplitude, pulse_sigma))


# ── The optimizer ────────────────────────────────────────────────────────────


class ThompsonSampler:
    """Batch Thompson sampling with a Gaussian process over a grid.

    Each suggestion is the best grid point of one draw from the posterior
    over the success probability: a batch spreads out where the posterior
    is unsure and gathers where it is confident.
    """

    def __init__(self, grid, length_scale=0.12, noise=0.05, seed=1):
        lo, hi = grid.min(axis=0), grid.max(axis=0)
        self.unit = (grid - lo) / (hi - lo)
        self.length_scale = length_scale
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.seen: list[int] = []
        self.values: list[float] = []

    def covariance(self, a, b):
        squared = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return 0.1 * np.exp(-0.5 * squared / self.length_scale**2)

    def posterior(self):
        prior = self.covariance(self.unit, self.unit)
        if not self.seen:
            return np.full(len(self.unit), 0.5), prior
        seen, values = self.unit[self.seen], np.array(self.values)
        offset = values.mean()
        noisy = self.covariance(seen, seen) + self.noise**2 * np.eye(len(seen))
        cross = self.covariance(self.unit, seen)
        weights = np.linalg.solve(noisy, cross.T)
        return offset + weights.T @ (values - offset), prior - cross @ weights

    def suggest(self, count):
        mean, cov = self.posterior()
        root = np.linalg.cholesky(cov + 1e-6 * np.eye(len(mean)))
        draws = mean[:, None] + root @ self.rng.standard_normal((len(mean), count))
        return draws.argmax(axis=0)

    def observe(self, indices, values):
        self.seen.extend(int(i) for i in indices)
        self.values.extend(float(v) for v in values)

    def best(self):
        return int(self.posterior()[0].argmax())


# ── Worker processes: bind and export ────────────────────────────────────────

worker_program: dict[str, pulse.Lowered] = {}


def receive_program(program):
    worker_program["generic"] = program


def prepare(setting):
    """One setting bound into the compiled program, and its controller export."""
    amplitude, detuning_hz = setting
    bound = worker_program["generic"].bind([amplitude, detuning_hz])
    return bound, bound.export_to_controller(sample_rate_ghz=sample_rate_ghz)


# ── The stand-in QPU ─────────────────────────────────────────────────────────

qpu_state: dict[str, DeviceDescription] = {}


def power_on(true_system):
    qpu_state["true_system"] = true_system


def excited_population(program, true_system):
    """P(1) at the end of *program*, simulated against *true_system* with three levels."""
    result = pulse.simulate(program, levels=3, dt_ns=0.25, quantum_system=true_system)
    return min(max(result.level_population(1), 0.0), 1.0)


def run_batch(prepared, seed):
    """Spend shots on each prepared program; return counts and when it ran."""
    start = time.monotonic()
    rng = np.random.default_rng(seed)
    counts = []
    for program, controller_program in prepared:
        assert controller_program["events"], "a controller would load these events"
        counts.append(
            int(rng.binomial(shots_count, excited_population(program, qpu_state["true_system"])))
        )
    ends = start + len(prepared) * shots_count * (shot_time / pulse.s)
    overran = time.monotonic() - ends
    time.sleep(max(0.0, -overran))
    return counts, (start, time.monotonic()), max(0.0, overran)


# ── The loop ─────────────────────────────────────────────────────────────────


def calibrate(pool, qpu, pipelined):
    optimizer = ThompsonSampler(settings)
    suggest_s, prepare_s = [], []

    def next_batch():
        started = time.monotonic()
        indices = optimizer.suggest(settings_per_batch)
        suggested = time.monotonic()
        prepared = list(pool.map(prepare, settings[indices]))
        suggest_s.append(suggested - started)
        prepare_s.append(time.monotonic() - suggested)
        return indices, prepared

    began = time.monotonic()
    spans, overruns = [], []
    indices, prepared = next_batch()
    running = qpu.submit(run_batch, prepared, 0)
    for n in range(1, batches + 1):
        last = n == batches
        # Pipelined, while batch n-1 runs: from the results of batches up to n-2.
        upcoming = next_batch() if pipelined and not last else None
        counts, span, overran = running.result()
        spans.append(span)
        overruns.append(overran)
        optimizer.observe(indices, np.array(counts) / shots_count)
        if last:
            break
        if upcoming is None:
            # Serial, the QPU waits: from the results of every batch so far.
            upcoming = next_batch()
        indices, prepared = upcoming
        running = qpu.submit(run_batch, prepared, n)
    wall_s = time.monotonic() - began
    busy_s = sum(end - start for start, end in spans)
    idle_s = (spans[-1][1] - spans[0][0]) - busy_s
    return {
        "wall_s": wall_s,
        "busy_s": busy_s,
        "idle_s": idle_s,
        "suggest_ms": 1e3 * float(np.median(suggest_s)),
        "prepare_ms": 1e3 * float(np.median(prepare_s)),
        "overrun_ms": 1e3 * max(overruns),
        "best": optimizer.best(),
    }


def true_p1(program, true_system, index):
    """P(1) of grid setting *index*, simulated exactly against *true_system*."""
    return excited_population(program.bind(list(settings[index])), true_system)


if __name__ == "__main__":
    quantum_system = pulse.quantum_system(target)
    nominal = target.qudits[0]
    true_qudit = nominal.replace(frequency_hz=nominal.frequency_hz + hidden_offset_hz)
    true_system = dataclasses.replace(target, qudits={**target.qudits, 0: true_qudit})

    print("=== The program, compiled once ===")
    started = time.perf_counter()
    compiled = pulse.compile(pi_train, quantum_system=quantum_system)
    program = compiled.lower(quantum_system.qudits[0], repetitions, pi_amplitude, 0.0).generic
    print(
        f"  compiled in {1e3 * (time.perf_counter() - started):.2f} ms "
        f"(the first compile in a process); float parameters {program.parameters}"
    )
    print(f"  sent to each worker as {len(pickle.dumps(program)):,} pickled bytes")
    started = time.perf_counter()
    for setting in settings[:50]:
        program.bind(list(setting)).export_to_controller(sample_rate_ghz=sample_rate_ghz)
    print(
        f"  binding and exporting one setting: {1e3 * (time.perf_counter() - started) / 50:.3f} ms"
    )
    print(
        f"  a batch is {settings_per_batch} settings x {shots_count} shots x {shot_time} "
        f"= {batch_time / pulse.ms:.0f} ms on the QPU"
    )

    spawn = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    pool = concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=spawn, initializer=receive_program, initargs=(program,)
    )
    qpu = concurrent.futures.ProcessPoolExecutor(
        1, mp_context=spawn, initializer=power_on, initargs=(true_system,)
    )
    results = {}
    with pool, qpu:
        # Start every process, and warm it, before the clock does.
        list(pool.map(prepare, [settings[0]] * workers))
        qpu.submit(run_batch, [], 0).result()
        print(
            f"  starting {workers} workers and the stand-in QPU took "
            f"{time.perf_counter() - started:.2f} s, once"
        )
        for pipelined in (False, True):
            label = "pipelined" if pipelined else "serial"
            results[label] = calibrate(pool, qpu, pipelined)

    for label, r in results.items():
        print(f"\n=== {label.capitalize()}: {batches} batches ===")
        print(
            f"  host per batch: optimizer {r['suggest_ms']:.1f} ms, "
            f"{workers} workers bind and export {r['prepare_ms']:.1f} ms"
        )
        print(
            f"  QPU busy {r['busy_s']:.3f} s, idle waiting on the host "
            f"{r['idle_s']:.3f} s ({100 * r['idle_s'] / (r['busy_s'] + r['idle_s']):.0f}%, "
            f"{1e3 * r['idle_s'] / (batches - 1):.1f} ms between batches)"
        )
        print(f"  wall time {r['wall_s']:.3f} s")
        if r["overrun_ms"] > 0:
            print(f"  (the stand-in QPU overran its shot time by up to {r['overrun_ms']:.1f} ms)")
        amplitude, detuning_hz = settings[r["best"]]
        print(
            f"  best setting: amplitude {amplitude / pi_amplitude:.3f} x nominal, detuning "
            f"{detuning_hz / 1e6:+.2f} MHz, true P(1) {true_p1(program, true_system, r['best']):.4f}"
        )

    print("\n=== Comparison ===")
    exact = [true_p1(program, true_system, i) for i in range(len(settings))]
    amplitude, detuning_hz = settings[int(np.argmax(exact))]
    print(
        f"  best on the whole grid (simulated exactly): P(1) {max(exact):.4f} at "
        f"amplitude {amplitude / pi_amplitude:.3f} x nominal, detuning {detuning_hz / 1e6:+.2f} MHz"
    )
    serial, piped = results["serial"], results["pipelined"]
    print(f"  QPU idle: serial {serial['idle_s']:.3f} s, pipelined {piped['idle_s']:.3f} s")
    print(
        f"  wall time: serial {serial['wall_s']:.3f} s, pipelined {piped['wall_s']:.3f} s "
        f"({100 * (1 - piped['wall_s'] / serial['wall_s']):.0f}% less)"
    )
