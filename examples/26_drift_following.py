# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Following a drifting qudit by swapping in programs between batches.

Builds on ``20_adaptive_calibration.py`` and ``24_pipelined_calibration.py``.
A qudit's frequency drifts while a loop runs one program batch after batch.
Each batch also takes a few Ramsey shots and hands them to a background
worker process, started with the ``spawn`` method. The worker keeps a
posterior over the frequency, writes the estimate into the calibration at
a fixed resolution and builds the device again. When the device's
fingerprint has changed, it recompiles the program and sends it back
pickled. The loop swaps the fresh program in before its next batch.

The device fingerprint is a hash of everything a compiled program depends
on, so the worker only recompiles when the calibration actually changed.
It keeps its programs by fingerprint, so returning to an earlier
calibration costs no compile either.

The program is a Ramsey fringe at its top: two pi/2 pulses a fixed wait
apart, which end in |1> when the drive is on resonance. A frequency error
turns it away from |1>, so 1 - P(1) measures how well the calibration
follows the drift.

What is simulated: the simulator plays the QPU. Each batch's error is
``pulse.simulate`` of the program in use against the drifting system's true
parameters, and the Ramsey shots are drawn from the true system's analytic
surrogate (ideal pulses, no decay). The drift is far faster than a real
device's, so the run fits in seconds; the batches are real 20 ms waits, and
the worker, the pickling and the compiles are real.

Programs are swapped between batches, by the host. Swapping one in between
shots, inside a running controller program, needs runtime support that
does not exist yet.
"""

import dataclasses
import math
import multiprocessing
import queue
import time

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse.classical import calibration as cal
from cudaq_pulse.quantum_systems import transmon_17q
from cudaq_pulse.runtime.drift import DriftingPlant, OrnsteinUhlenbeck

target = transmon_17q()
qudit_index = 0
pulse_duration = 40 * pulse.ns
pulse_sigma = 10 * pulse.ns
area_ns = (pulse_sigma / pulse.ns) * math.sqrt(2 * math.pi) * math.erf(2 / math.sqrt(2))
half_pi_amplitude = math.pi / 2 / (area_ns * target.drive_amplitude_scale(qudit_index))
fringe_wait = 1 * pulse.us

drift = OrnsteinUhlenbeck(200e3, correlation_s=10.0)
batch_time = 20 * pulse.ms  # 400 shots of 50 us on the QPU
batches = 200
settled = 10  # batches the first estimate takes
ramsey_shots = 16
prior_sigma_hz = 300e3
process_noise_hz = 20e3  # how far the drift can move in one batch
calibration_step_hz = 25e3  # the resolution the calibration is written at


@pulse.kernel
def fringe_top(qudit):
    """pi/2, a wait, pi/2: |1> when the drive is on resonance."""
    d = get_drive_line(qudit)
    drive(d, gaussian(pulse_duration, half_pi_amplitude, pulse_sigma))
    wait(d, fringe_wait)
    drive(d, gaussian(pulse_duration, half_pi_amplitude, pulse_sigma))


def calibrated_system(offset_hz):
    """The device whose qudit frequency is the nominal one plus *offset_hz*."""
    nominal = target.qudits[qudit_index]
    qudit = nominal.replace(frequency_hz=nominal.frequency_hz + offset_hz)
    return pulse.quantum_system(
        dataclasses.replace(target, qudits={**target.qudits, qudit_index: qudit})
    )


# ── The background worker ────────────────────────────────────────────────────


def calibration_worker(records_in, updates_out):
    """Estimate the frequency from Ramsey records; recompile when the device changes."""
    mean_hz, sigma_hz = 0.0, prior_sigma_hz
    start = calibrated_system(0.0)
    programs = {start.fingerprint: fringe_top.lower(start.qudits[qudit_index])}
    current = start.fingerprint
    counts = {"estimates": 0, "unchanged": 0, "compiled": 0, "reused": 0}
    updates_out.put("ready")
    while (message := records_in.get()) is not None:
        batch, records = message
        sigma_hz = math.hypot(sigma_hz, process_noise_hz)
        for wait_ns, phase, outcome in records:
            mean_hz, sigma_hz = cal.frequency_update(mean_hz, sigma_hz, outcome, wait_ns, phase)
        counts["estimates"] += 1
        written_hz = calibration_step_hz * round(mean_hz / calibration_step_hz)
        quantum_system = calibrated_system(written_hz)
        fresh = None
        if quantum_system.fingerprint == current:
            counts["unchanged"] += 1
        else:
            if quantum_system.fingerprint in programs:
                counts["reused"] += 1
            else:
                programs[quantum_system.fingerprint] = fringe_top.lower(
                    quantum_system.qudits[qudit_index]
                )
                counts["compiled"] += 1
            current = quantum_system.fingerprint
            fresh = (written_hz, programs[current])
        updates_out.put((batch, mean_hz, sigma_hz, fresh))
    updates_out.put(counts)


# ── The loop ─────────────────────────────────────────────────────────────────


def error_against(program, drifting):
    result = pulse.simulate(program, levels=3, dt_ns=0.25, quantum_system=drifting.now())
    return 1 - result.level_population(1)


def follow(swapping, seed=5):
    drifting = DriftingPlant(target, qudit_index, frequency=drift, seed=seed)
    in_use_hz = 0.0
    program = fringe_top.lower(calibrated_system(0.0).qudits[qudit_index])
    mean_hz, sigma_hz = 0.0, prior_sigma_hz
    spawn = multiprocessing.get_context("spawn")
    records_in, updates_out = spawn.Queue(), spawn.Queue()
    worker = spawn.Process(target=calibration_worker, args=(records_in, updates_out))
    worker.start()
    # Its start and first compile would otherwise show up as unswapped batches.
    assert updates_out.get() == "ready"
    trace, lags, swaps = [], [], 0
    for batch in range(batches):
        started = time.monotonic()
        # Between batches: take whatever the worker has finished.
        while True:
            try:
                sent, mean_hz, sigma_hz, fresh = updates_out.get_nowait()
            except queue.Empty:
                break
            if fresh is not None and swapping:
                in_use_hz, program = fresh
                swaps += 1
                lags.append(batch - sent)
        trace.append(
            (drifting.state.frequency_offset_hz, in_use_hz, error_against(program, drifting))
        )
        records = []
        for _ in range(ramsey_shots):
            wait_ns, phase = cal.ramsey_setting(mean_hz, sigma_hz, max_wait_ns=2000.0)
            records.append((wait_ns, phase, drifting.sample(drifting.p_ramsey(wait_ns, phase))))
        records_in.put((batch, records))
        time.sleep(max(0.0, started + batch_time / pulse.s - time.monotonic()))
        drifting.advance(batch_time / pulse.s)
    records_in.put(None)
    while not isinstance(counts := updates_out.get(), dict):
        pass
    worker.join()
    return np.array(trace), counts, swaps, lags


if __name__ == "__main__":
    print("=== The device fingerprint follows the calibration ===")
    for offset_hz in (0.0, 0.0, 25e3):
        print(
            f"  qudit frequency +{offset_hz / 1e3:4.0f} kHz: fingerprint "
            f"{calibrated_system(offset_hz).fingerprint}"
        )

    runs = {}
    for swapping in (False, True):
        runs[swapping] = follow(swapping)
    off, _, _, _ = runs[False]
    on, counts, swaps, lags = runs[True]

    print(f"\n=== {batches} batches of {batch_time}, drift sigma {drift.sigma / 1e3:.0f} kHz ===")
    print("  time    true offset   calibration in use   error, swaps off   error, swaps on")
    for batch in range(0, batches, 25):
        elapsed = batch * batch_time / pulse.s
        print(
            f"  {elapsed:4.2f} s  {on[batch, 0] / 1e3:+8.1f} kHz   "
            f"{on[batch, 1] / 1e3:+8.1f} kHz          {off[batch, 2]:.4f}             "
            f"{on[batch, 2]:.4f}"
        )
    print(f"  mean error: swaps off {off[:, 2].mean():.4f}, swaps on {on[:, 2].mean():.4f}")
    settled_off, settled_on = off[settled:, 2], on[settled:, 2]
    print(
        f"  after the first {settled} batches: mean swaps off {settled_off.mean():.4f}, "
        f"on {settled_on.mean():.4f}; worst off {settled_off.max():.4f}, "
        f"on {settled_on.max():.4f}"
    )
    floor = error_against(
        fringe_top.lower(calibrated_system(0.0).qudits[qudit_index]),
        DriftingPlant(target, qudit_index),
    )
    print(f"  (on resonance the error is {floor:.4f}: decay and thermal population)")
    tracking = np.sqrt(np.mean((on[settled:, 0] - on[settled:, 1]) ** 2))
    print(f"  calibration tracking error (rms, settled): {tracking / 1e3:.1f} kHz")

    print("\n=== What the worker did ===")
    print(
        f"  {counts['estimates']} estimates: the device built from {counts['unchanged']} "
        f"of them had the fingerprint already in use, so nothing was recompiled"
    )
    print(
        f"  changed {counts['compiled'] + counts['reused']} times: compiled "
        f"{counts['compiled']}, reused {counts['reused']} kept by fingerprint"
    )
    print(
        f"  {swaps} programs swapped in, a median {np.median(lags):.0f} batch(es) after "
        f"the Ramsey shots that called for them"
    )
