#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scorer of the challenge "Benchmark it yourself".

    python challenges/benchmark_it_yourself/score.py my_rb.py [seed]

my_rb.py defines error_per_gate(lab, seed): its estimate, by randomized benchmarking, of the error
of one calibrated X90 pulse of lab.qubit. lab.run(kernel, lab.qubit, ...) plays a kernel once and
returns a result with level_population(i) and level_populations(): exact numbers, nothing is
sampled.

The qubit is hidden. It has the frequency, the anharmonicity and the calibrated pulses of a qubit
like Z4 of transmon_surface17, and a T1 and a T2* that each run of the scorer draws anew. The
submission never runs in the scorer's process: a second Python process (_child.py) loads it and
holds a public copy of the qubit that does not decay. There, lab.run compiles the kernel and sends the scorer
which calibrated pulses it plays and when. The scorer plays them on the hidden qubit and sends the
populations back, and it is the scorer that counts sequences and pulses.

Rules (all the scorer's): a sequence plays only the qubit's calibrated X90 and Y90, back to back
from the start (a frame change before a pulse is allowed, a wait is not), then at most one
calibrated readout at once; a call of error_per_gate plays at least MIN_SEQUENCES sequences and
MIN_PULSES pulses, and at most MAX_PULSES pulses.

The reference is the exact average gate error of the X90 on the hidden qubit: one minus its
fidelity with a quarter turn about X, averaged over the six axis states, in the same three-level
model. It is printed after the estimate, never before.
"""

import json
import math
import secrets
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
from _qubit import benchmark_system

RUNS = 5
T1_RANGE_US = (8.0, 80.0)
T2_RANGE_US = (8.0, 80.0)
BRONZE_FACTOR = 1.5
SILVER_TOLERANCE, SILVER_SEQUENCES = 0.10, 8
MIN_SEQUENCES, MIN_PULSES, MAX_PULSES = 2, 20, 6000  # per call of error_per_gate
PULSE_CYCLES = 80  # a calibrated X90 or Y90: 40 ns at the 2 GHz clock
CHILD = Path(__file__).with_name("_child.py")


class RefusedError(Exception):
    """A sequence the scorer does not play; the message goes back to the submission."""


@pulse.kernel
def played(qudit, steps, measured):
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    x90 = drive_line.get_calibration(pulse.gates.X90)
    y90 = drive_line.get_calibration(pulse.gates.Y90)
    for gap_ns, turn, name in steps:
        if gap_ns > 0:
            wait(drive_line, gap_ns * pulse.ns)
        if turn != 0.0:
            shift_phase(drive_line.tone, turn)
        if name == "Y90":
            drive(drive_line, y90)
        else:
            drive(drive_line, x90)
    if measured:
        sync(drive_line, readout_line)
        readout(readout_line, readout_line.get_calibration(pulse.gates.READOUT))


def replay(request, hidden, count):
    """The populations after one sequence: the scorer plays it itself on the hidden qubit."""
    try:
        pulses = sorted(
            (int(start), str(name), float(phase)) for start, name, phase in request["pulses"]
        )
        readouts = [int(start) for start in request["readouts"]]
        others = [str(kind) for kind in request["others"]]
    except (KeyError, TypeError, ValueError):
        raise RefusedError("the sequence could not be read") from None
    if others:
        raise RefusedError(
            f"a sequence has a {others[0]}; only the calibrated X90 and Y90 and one calibrated readout are played"
        )
    if len(readouts) > 1:
        raise RefusedError("a sequence has at most one readout")
    count["sequences"] += 1
    count["pulses"] += len(pulses)
    if count["pulses"] > MAX_PULSES:
        raise RefusedError(f"more than {MAX_PULSES} pulses in one call of error_per_gate")
    steps, clock, phase = [], 0, 0.0
    for start, name, turn in pulses:
        if name not in ("X90", "Y90") or not math.isfinite(turn):
            raise RefusedError("the sequence could not be read")
        if start != clock:
            raise RefusedError("a sequence plays its pulses back to back, from the start; no waits")
        steps.append((0.0, turn - phase, name))
        clock, phase = start + PULSE_CYCLES, turn
    if readouts and readouts[0] != clock:
        raise RefusedError("the readout follows the last pulse at once; no wait before it")
    program = pulse.compile(played, quantum_system=hidden, schedule="asap").lower(
        hidden.qudits[0], tuple(steps), bool(readouts)
    )
    result = pulse.simulate(program, dt_ns=0.25)
    return [float(result.level_population(level)) for level in range(3)]


def reference(hidden):
    """(error, leakage) of the calibrated X90 on the hidden qubit, from the six axis states."""
    program = pulse.compile(played, quantum_system=hidden, schedule="asap").lower(
        hidden.qudits[0], ((0.0, 0.0, "X90"),), False
    )
    ideal = np.array([[1, -1j], [-1j, 1]]) / math.sqrt(2)
    s = math.sqrt(0.5)
    fidelity = leakage = 0.0
    for state in ((1, 0), (0, 1), (s, s), (s, -s), (s, 1j * s), (s, -1j * s)):
        rho = pulse.simulate(program, initial_state=[*state, 0], dt_ns=0.25).density_matrix
        target = ideal @ np.array(state)
        fidelity += float(np.real(target.conj() @ rho[:2, :2] @ target)) / 6
        leakage += float(np.real(rho[2, 2])) / 6
    return 1 - fidelity, leakage


def one_line(text):
    """*text* from the submission's process as one short line of printable characters."""
    return " ".join("".join(c if c.isprintable() else " " for c in str(text)).split())[:300]


def ask(child):
    line = child.stdout.readline()
    if not line:
        raise EOFError("the submission's process ended")
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("the submission's process sent something that is not a message")
    return message


def one_run(child, seed, hidden, count):
    """The submission's estimate for one hidden qubit; its sequences are played and counted here."""
    child.stdin.write(f"SEED {seed}\n")
    child.stdin.flush()
    while True:
        message = ask(child)
        if "estimate" in message:
            return float(message["estimate"])
        if "error" in message:
            raise ValueError(one_line(message["error"]))
        try:
            answer = {"populations": replay(message["run"], hidden, count)}
        except RefusedError as refused:
            answer = {"error": str(refused)}
        except Exception as error:  # the numbers do not make a playable sequence
            answer = {"error": f"the sequence could not be played: {type(error).__name__}: {error}"}
        child.stdin.write(json.dumps(answer) + "\n")
        child.stdin.flush()


def main(path, seed=None):
    given = seed is not None
    seed = seed if given else secrets.randbelow(2**31)
    rng = np.random.default_rng(seed)
    child = subprocess.Popen(
        [sys.executable, str(CHILD), path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    runs = []
    try:
        first = ask(child)
        if "error" in first:
            raise ValueError(f"it does not load: {one_line(first['error'])}")
        for run in range(1, RUNS + 1):
            t1_us = float(rng.uniform(*T1_RANGE_US))
            t2_us = float(rng.uniform(T2_RANGE_US[0], min(T2_RANGE_US[1], 2 * t1_us)))
            hidden = benchmark_system(t1_us, t2_us)
            count = {"sequences": 0, "pulses": 0}
            estimate = one_run(child, run, hidden, count)
            enough = count["sequences"] >= MIN_SEQUENCES and count["pulses"] >= MIN_PULSES
            if not enough or not 0 < estimate < math.inf:
                raise ValueError(
                    f"run {run} gave {estimate!r} from {count['sequences']} sequences and {count['pulses']} pulses; an estimate is a positive number from at least {MIN_SEQUENCES} sequences and {MIN_PULSES} pulses"
                )
            error, leakage = reference(hidden)
            runs.append((estimate / error, count["sequences"]))
            print(
                f"RUN {run} estimate={estimate:.4e} reference={error:.4e} (leakage {leakage:.1e}) ratio={estimate / error:.4f} sequences={count['sequences']} pulses={count['pulses']}"
            )
    except Exception as problem:  # the file does not load, breaks a rule or the exchange, or fails
        print(f"INVALID {one_line(problem)}\nBRONZE fail\nSILVER fail\nSCORE none")
        return 1
    finally:
        child.kill()
        child.wait(timeout=30)
    worst, most = max(abs(ratio - 1) for ratio, _ in runs), max(sequences for _, sequences in runs)
    bronze = all(1 / BRONZE_FACTOR <= ratio <= BRONZE_FACTOR for ratio, _ in runs)
    silver = worst <= SILVER_TOLERANCE and most <= SILVER_SEQUENCES
    repeat = " (repeat run)" if given else ""
    print(
        f"BRONZE {'pass' if bronze else 'fail'}{repeat}  (every run within a factor {BRONZE_FACTOR:g} of the reference)"
    )
    print(
        f"SILVER {'pass' if silver else 'fail'}{repeat}  (every run within {SILVER_TOLERANCE:.0%} of it, from at most {SILVER_SEQUENCES} sequences; worst {worst:.2%}, most {most})"
    )
    if given:
        print("SCORE none (a run with a given seed repeats known qubits: it is not a score)")
    else:
        print(f"SCORE worst_deviation={worst:.2%} most_sequences={most}")
    print(f"SEED {seed} (give it as the second argument to repeat this run)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python challenges/benchmark_it_yourself/score.py my_rb.py [seed]")
    sys.exit(main(sys.argv[1], *(int(value) for value in sys.argv[2:3])))
