#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scorer of the challenge "Find the pi pulse".

    python challenges/pi_pulse/score.py my_pulse.py

my_pulse.py defines a kernel named pi_pulse that takes one qudit.

The submission never runs in the scorer's process. A second Python process (_child.py) loads the
file, compiles pi_pulse for qubit Z4 of pulse.quantum_system("transmon_surface17") and writes what
it would send to the drive line: for each pulse its start, its samples, its frequency and its
phase. The scorer reads only those numbers, checks them against the limits below, plays them
itself from level 0 and reports the populations of levels 1 and 2 (three levels, T1 and T2*;
nothing is sampled, so every run prints the same numbers). A submission can therefore change
nothing about the rules, the qubit or the simulation.

Limits (the clock is the system's; the three limits are the challenge's, the system ships none):
only drive pulses, waits and frame changes; at most MAX_DURATION_NS in all; every sample at most
MAX_AMPLITUDE in magnitude; every pulse within MAX_DETUNING_HZ of the qubit's 0-1 frequency.
"""

import json
import math
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse

QUDIT = "Z4"
MAX_DURATION_NS = 20.0  # half of the shipped X pulse
MAX_AMPLITUDE = 0.5  # the shipped X pulse peaks at 0.13
MAX_DETUNING_HZ = 50.0e6
BRONZE_P1 = 0.98
SILVER_P1, SILVER_P2 = 0.999, 2.0e-4
CHILD_TIMEOUT_S = 120
MAX_PULSES = 64
CHILD = Path(__file__).with_name("_child.py")


def one_line(text):
    """*text* from the submission's process as one short line of printable characters."""
    return " ".join("".join(c if c.isprintable() else " " for c in str(text)).split())[:300]


class InvalidSubmissionError(Exception):
    """The submission cannot be scored; the message says why."""


def read_submission(path):
    """The pulses of the file's pi_pulse as plain numbers, from the second process."""
    try:
        done = subprocess.run(
            [sys.executable, str(CHILD), path, QUDIT],
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise InvalidSubmissionError(f"it did not compile within {CHILD_TIMEOUT_S} s") from None
    lines = [line for line in done.stdout.splitlines() if line.startswith("PULSES ")]
    if done.returncode != 0 or not lines:
        last = one_line((done.stderr.strip().splitlines() or ["no message"])[-1])
        raise InvalidSubmissionError(f"it does not load or compile: {last}")
    try:
        data = json.loads(lines[-1][len("PULSES ") :])
        others = [one_line(kind)[:40] for kind in data["others"]]
        total = int(data["total_cycles"])
        pulses = []
        for item in data["pulses"]:
            real, imag = [float(x) for x in item["re"]], [float(x) for x in item["im"]]
            if len(real) != len(imag) or not real:
                raise ValueError("samples")
            pulses.append(
                {
                    "start": int(item["start_cycles"]),
                    "samples": [complex(a, b) for a, b in zip(real, imag, strict=True)],
                    "frequency_hz": float(item["frequency_hz"]),
                    "phase_rad": float(item["phase_rad"]),
                }
            )
    except (KeyError, TypeError, ValueError, OverflowError):
        raise InvalidSubmissionError("its pulses could not be read") from None
    numbers = [x for p in pulses for s in p["samples"] for x in (s.real, s.imag)]
    numbers += [x for p in pulses for x in (p["frequency_hz"], p["phase_rad"])]
    if not all(math.isfinite(x) for x in numbers):
        raise InvalidSubmissionError(
            "a pulse has a sample, frequency or phase that is not a finite number"
        )
    return sorted(pulses, key=lambda p: p["start"]), others, total


def check(pulses, others, total, system, qudit):
    """(problems, duration in ns, peak amplitude) of the pulses against the limits."""
    problems = [f"it has a {kind}; only drive pulses and waits are allowed" for kind in others]
    if len(pulses) > MAX_PULSES:
        problems.append(f"it has {len(pulses)} pulses; the limit is {MAX_PULSES}")
    end, peak, busy = max(total, 0), 0.0, 0  # waits after the last pulse count
    for p in pulses:
        if p["start"] < busy:
            problems.append("two pulses overlap on the drive line")
        busy = p["start"] + len(p["samples"])
        end = max(end, busy)
        peak = max(peak, max(abs(s) for s in p["samples"]))
        detuning_mhz = (p["frequency_hz"] - system.qudit_freq[qudit.index]) / 1e6
        if abs(detuning_mhz) > MAX_DETUNING_HZ / 1e6:
            limit_mhz = MAX_DETUNING_HZ / 1e6
            problems.append(
                f"a pulse is {detuning_mhz:+g} MHz from the qubit; the limit is {limit_mhz:g} MHz"
            )
    duration_ns = end / system.clock_ghz
    if duration_ns > MAX_DURATION_NS:
        problems.append(f"it lasts {duration_ns:g} ns; the limit is {MAX_DURATION_NS:g} ns")
    if peak > MAX_AMPLITUDE:
        problems.append(f"a sample reaches {peak:.4f}; the limit is {MAX_AMPLITUDE:g}")
    return problems, duration_ns, peak


def replay(pulses, system, qudit):
    """Level populations after the scorer plays the pulses itself from level 0."""
    to_ns = 1.0 / system.clock_ghz
    native = system.qudit_freq[qudit.index]
    steps, clock, frequency, frame, last = [], 0, native, 0.0, 0
    for p in pulses:
        # A pulse's phase is its frame's at its start. The scorer's own tone, off the qubit by
        # frequency - native since the last pulse began, has gathered part of it already.
        frame += 2.0 * math.pi * (frequency - native) * (p["start"] - last) * to_ns * 1e-9
        steps.append(
            (
                (p["start"] - clock) * to_ns,
                p["frequency_hz"] - frequency,
                p["phase_rad"] - frame,
                [s.real for s in p["samples"]],
                [s.imag for s in p["samples"]],
                len(p["samples"]),
            )
        )
        clock, last = p["start"] + len(p["samples"]), p["start"]
        frequency, frame = p["frequency_hz"], p["phase_rad"]

    @pulse.kernel
    def played(target):
        line = get_drive_line(target)
        for gap_ns, df, dphi, real, imag, count in steps:
            if gap_ns > 0:
                wait(line, gap_ns * pulse.ns)
            if df != 0.0:
                shift_frequency(line.tone, df)
            if dphi != 0.0:
                shift_phase(line.tone, dphi)
            quadrature = wf_mul(custom_samples(imag), square(count, 1j))
            drive(line, wf_add(custom_samples(real), quadrature))

    program = pulse.compile(played, quantum_system=system, schedule="asap").lower(qudit)
    return pulse.simulate(program, initial_state=[1.0, 0.0, 0.0]).level_populations(qudit)


def main(path):
    system = pulse.quantum_system("transmon_surface17")
    qudit = system.qudit(QUDIT)
    p1 = p2 = 0.0
    try:
        pulses, others, total = read_submission(path)
        problems, duration_ns, peak = check(pulses, others, total, system, qudit)
        if pulses and not problems:
            _, p1, p2 = replay(pulses, system, qudit)
    except InvalidSubmissionError as error:
        print(f"INVALID {error}\nBRONZE fail\nSILVER fail\nSCORE none")
        return 1
    for problem in problems:
        print(f"INVALID {problem}")
    bronze = not problems and p1 >= BRONZE_P1
    silver = not problems and p1 >= SILVER_P1 and p2 <= SILVER_P2
    print(f"BRONZE {'pass' if bronze else 'fail'}  (P1 >= {BRONZE_P1:g})")
    print(f"SILVER {'pass' if silver else 'fail'}  (P1 >= {SILVER_P1:g} and P2 <= {SILVER_P2:g})")
    valid = "no" if problems else "yes"
    numbers = "P1=none P2=none" if problems else f"P1={p1:.6f} P2={p2:.6f}"
    print(f"SCORE {numbers} duration_ns={duration_ns:g} peak_amplitude={peak:.4f} valid={valid}")
    return 1 if problems else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python challenges/pi_pulse/score.py my_pulse.py")
    sys.exit(main(sys.argv[1]))
