#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scorer of the challenge "Mystery qubit".

    python challenges/mystery_qubit/score.py solution.py [seed]

The solution defines estimate_t1(lab), which returns T1 in microseconds. lab.qubit is the qubit;
lab.run(kernel, lab.qubit, ..., shots_count=n) plays a kernel on it n times and returns the
records, one row per shot, as pulse.simulate(..., shots_count=n).records does.

The solution never runs in the scorer's process. A second Python process (_child.py) loads it.
There, lab.run compiles the kernel for the public description of the qubit (no T1) and sends the scorer what the
kernel would put on the drive and readout lines: starts, samples, frequencies and phases. The
scorer plays those numbers on the hidden qubit and sends the shots back. The hidden T1 values, the
shot tally and the verdicts exist only in the scorer's process.

Each run draws new hidden qubits unless a seed is given; the seed is printed at the end, so a run
can be repeated.
"""

import json
import math
import secrets
import subprocess
import sys
from pathlib import Path

import cudaq_pulse as pulse
import numpy as np
from _qubit import mystery_system

T1_RANGE_US = (20.0, 200.0)
QUBITS = 20
TOLERANCE = 0.05
SHOTS_PER_QUBIT = 100_000  # the most a solution may take on one qubit
RUNS_PER_QUBIT = 200  # the most calls of lab.run on one qubit
SILVER_SHOTS = 10_000  # per qubit, on average
MAX_PULSES = 64  # drive pulses in one kernel
CHUNK = 252  # the most samples one custom_samples() waveform holds
MAX_CYCLES = 4_000_000  # 2 ms at the 2 GHz clock: the longest kernel
CHILD = Path(__file__).with_name("_child.py")


class RefusedError(Exception):
    """A run the scorer does not play; the message goes back to the solution."""


def as_pulse(item):
    real, imag = [float(x) for x in item["re"]], [float(x) for x in item["im"]]
    values = [*real, *imag, float(item["frequency_hz"]), float(item["phase_rad"])]
    if len(real) != len(imag) or not real or not all(math.isfinite(x) for x in values):
        raise RefusedError("a pulse could not be read")
    start = int(item["start_cycles"])
    if not 0 <= start <= MAX_CYCLES or start + len(real) > MAX_CYCLES:
        raise RefusedError(f"a kernel lasts at most {MAX_CYCLES} clock cycles")
    return start, real, imag, float(item["frequency_hz"]), float(item["phase_rad"])


def replay(request, hidden, seed):
    """The records of one run: the scorer plays the numbers itself on the hidden qubit."""
    try:
        shots_count = request["shots_count"]
        drives = sorted(as_pulse(item) for item in request["drives"])
        readouts = request["readouts"]
        others = [str(kind) for kind in request["others"]]
    except (KeyError, TypeError, ValueError):
        raise RefusedError("the kernel could not be read") from None
    if others:
        raise RefusedError(
            f"the kernel has a {others[0]}; drive pulses, waits and one readout are played"
        )
    if len(readouts) != 1:
        raise RefusedError("a shot is one readout; this kernel does not make exactly one")
    try:
        item = readouts[0]
        shape, r_start, r_count = (
            str(item["shape"]),
            int(item["start_cycles"]),
            int(item["duration_cycles"]),
        )
        r_amplitude = complex(float(item["re"]), float(item["im"]))
        r_hz, r_rad = float(item["frequency_hz"]), float(item["phase_rad"])
    except (KeyError, TypeError, ValueError):
        raise RefusedError("the readout could not be read") from None
    if shape != "square":
        raise RefusedError(
            f"the readout pulse is a {shape}; the scorer plays a square readout pulse, as the calibrated one is"
        )
    if not all(math.isfinite(x) for x in (r_amplitude.real, r_amplitude.imag, r_hz, r_rad)):
        raise RefusedError("the readout could not be read")
    if not (0 <= r_start <= MAX_CYCLES and 0 < r_count <= MAX_CYCLES):
        raise RefusedError(f"a kernel lasts at most {MAX_CYCLES} clock cycles")
    if len(drives) > MAX_PULSES:
        raise RefusedError(f"a kernel has at most {MAX_PULSES} drive pulses")
    qudit = hidden.qudits[0]
    to_ns = 1.0 / hidden.clock_ghz
    native = hidden.qudit_freq[qudit.index]
    steps, clock, frequency, frame, last = [], 0, native, 0.0, 0
    for start, real, imag, hz, rad in drives:
        if start < clock:
            raise RefusedError("two drive pulses overlap")
        # A pulse's phase is its frame's at its start. The scorer's own tone, off the qubit by
        # frequency - native since the last pulse began, has gathered part of it already.
        frame += 2.0 * math.pi * (frequency - native) * (start - last) * to_ns * 1e-9
        # A waveform of samples holds at most CHUNK of them: a longer pulse is played in
        # pieces, back to back.
        for at in range(0, len(real), CHUNK):
            piece_re, piece_im = real[at : at + CHUNK], imag[at : at + CHUNK]
            gap = (start - clock) * to_ns if at == 0 else 0.0
            step = (gap, hz - frequency, rad - frame, piece_re, piece_im)
            steps.append((*step, len(piece_re), any(piece_im)))
            frequency, frame = hz, rad
        clock, last = start + len(real), start
    r_wait = r_start * to_ns
    r_df = r_hz - native  # the readout tone starts at the qubit's frequency here

    @pulse.kernel
    def played(target):
        drive_line = get_drive_line(target)
        readout_line = get_readout_line(target)
        for gap, df, dphi, real, imag, count, has_imag in steps:
            if gap > 0:
                wait(drive_line, gap * pulse.ns)
            if df != 0.0:
                shift_frequency(drive_line.tone, df)
            if dphi != 0.0:
                shift_phase(drive_line.tone, dphi)
            if has_imag:
                drive(
                    drive_line,
                    wf_add(custom_samples(real), wf_mul(custom_samples(imag), square(count, 1j))),
                )
            else:
                drive(drive_line, custom_samples(real))
        if r_wait > 0:
            wait(readout_line, r_wait * pulse.ns)
        if r_df != 0.0:
            shift_frequency(readout_line.tone, r_df)
        if r_rad != 0.0:
            shift_phase(readout_line.tone, r_rad)
        return readout(readout_line, square(r_count, r_amplitude))

    program = pulse.compile(played, quantum_system=hidden, schedule="asap").lower(qudit)
    result = pulse.simulate(program, shots_count=shots_count, seed=seed)
    records = np.asarray(result.records).reshape(-1)
    if records.size != shots_count:  # for example a sample too large to simulate
        raise RefusedError("the kernel could not be played for every shot; nothing was counted")
    return records


def one_line(text):
    """*text* from the submission's process as one short line of printable characters."""
    return " ".join("".join(c if c.isprintable() else " " for c in str(text)).split())[:300]


def ask(child):
    line = child.stdout.readline()
    if not line:
        raise EOFError("the solution's process ended")
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("the solution's process sent something that is not a message")
    return message


def one_qubit(child, t1_us, noise, used):
    """The solution's estimate for one hidden qubit; its runs are played and counted here."""
    hidden = mystery_system(t1_us)
    child.stdin.write("QUBIT\n")
    child.stdin.flush()
    while True:
        message = ask(child)
        if "estimate" in message:
            if used["shots"] == 0:
                raise RuntimeError("no shots taken; only lab.run plays this qubit's true T1")
            return float(message["estimate"])
        if "error" in message:
            raise RuntimeError(one_line(message["error"]))
        try:
            request = message["run"]
            shots_count = request.get("shots_count")
            used["runs"] += 1
            too_many = used["runs"] > RUNS_PER_QUBIT
            if (
                too_many
                or type(shots_count) is not int
                or not 0 < shots_count <= SHOTS_PER_QUBIT - used["shots"]
            ):
                raise RefusedError(
                    f"a qubit gives at most {SHOTS_PER_QUBIT} shots in {RUNS_PER_QUBIT} runs"
                )
            records = replay(request, hidden, int(noise.integers(2**31)))
            used["shots"] += shots_count
            answer = {"records": [int(x) for x in records]}
        except RefusedError as refused:
            answer = {"error": str(refused)}
        except Exception as error:  # the numbers do not make a playable kernel
            answer = {"error": f"the kernel could not be played: {type(error).__name__}: {error}"}
        child.stdin.write(json.dumps(answer) + "\n")
        child.stdin.flush()


def main(path, seed=None):
    given = seed is not None
    seed = seed if given else secrets.randbelow(2**31)
    child = subprocess.Popen(
        [sys.executable, str(CHILD), path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        return rounds(child, seed, given)
    except Exception as error:  # the solution's process broke the exchange: nothing is scored
        print(f"INVALID {one_line(error)}\nBRONZE fail\nSILVER fail\nSCORE none")
        return 1
    finally:
        child.kill()
        child.wait(timeout=30)


def rounds(child, seed, given):
    """Every hidden qubit in turn; prints the result. Any error here ends the run as invalid."""
    first = ask(child)
    if "error" in first:
        raise ValueError(f"the solution does not load: {one_line(first['error'])}")
    rng = np.random.default_rng(seed)
    low, high = T1_RANGE_US
    # One qubit from each slice of the range (on a log scale), in shuffled order.
    places = rng.permutation((np.arange(QUBITS) + rng.random(QUBITS)) / QUBITS)
    within = shots = 0
    for index, place in enumerate(places):
        t1_us = low * (high / low) ** place
        used = {"shots": 0, "runs": 0}
        try:
            estimate = one_qubit(child, t1_us, np.random.default_rng([seed, index]), used)
        except RuntimeError as error:  # the solution's own error on this qubit: a miss
            estimate = math.nan
            print(f"qubit {index + 1:2d}: {error}")
        off = abs(estimate / t1_us - 1)
        ok = bool(off <= TOLERANCE)
        within += ok
        shots += used["shots"]
        line = f"qubit {index + 1:2d}: T1 {t1_us:6.1f} us, estimate {estimate:6.1f} us"
        line += f", off by {off:5.1%}"
        print(f"{line}, {used['shots']:6d} shots, {'ok' if ok else 'MISS'}")
    bronze = within == QUBITS
    silver = bronze and shots <= SILVER_SHOTS * QUBITS
    limit = f"at most {SILVER_SHOTS} shots per qubit on average (used {shots / QUBITS:.0f})"
    repeat = " (repeat run)" if given else ""
    print(
        f"BRONZE {'pass' if bronze else 'fail'}{repeat}: {within} of {QUBITS} qubits within {TOLERANCE:.0%}"
    )
    print(f"SILVER {'pass' if silver else 'fail'}{repeat}: bronze, with {limit}")
    if given:
        print("SCORE none (a run with a given seed repeats known qubits: it is not a score)")
    else:
        print(f"SCORE {shots} shots in total" + ("" if bronze else " (bronze failed)"))
    print(f"SEED {seed} (give it as the second argument to repeat this run)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit("usage: python challenges/mystery_qubit/score.py solution.py [seed]")
    sys.exit(main(sys.argv[1], *(int(value) for value in sys.argv[2:3])))
