# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark it yourself: a solution that reaches bronze.

This is examples/14_randomized_benchmarking.py with three changes:

1. the kernel plays the qubit's calibrated X90 and Y90, not a pulse of its own;
2. every depth averages three random sequences;
3. a straight-line fit of the decay, as the coherence tutorial fits T1, gives
   the error per Clifford, and a Clifford is 37/12 pulses on average.
"""

import math
import random

import cudaq_pulse as pulse
import numpy as np

_NATIVE = {
    "X90": np.array([[1, -1j], [-1j, 1]]) / math.sqrt(2),
    "Y90": np.array([[1, -1], [1, 1]]) / math.sqrt(2),
}


def _unitary(gates):
    unitary = np.eye(2, dtype=complex)
    for gate in gates:
        unitary = _NATIVE[gate] @ unitary
    return unitary


def _up_to_phase(unitary):
    """A hashable form of *unitary* with its global phase removed."""
    flat = unitary.ravel()
    first = flat[np.flatnonzero(np.abs(flat) > 1e-9)[0]]
    return tuple(np.round(flat * abs(first) / first, 6))


def _clifford_decompositions():
    """The shortest X90/Y90 word for each of the 24 single-qubit Cliffords."""
    words: dict[tuple[complex, ...], tuple[str, ...]] = {_up_to_phase(np.eye(2)): ()}
    frontier: list[tuple[str, ...]] = [()]
    while frontier:
        longer = []
        for word in frontier:
            for gate in _NATIVE:
                key = _up_to_phase(_unitary(word + (gate,)))
                if key not in words:
                    words[key] = word + (gate,)
                    longer.append(word + (gate,))
        frontier = longer
    return words


clifford_decompositions = _clifford_decompositions()
assert len(clifford_decompositions) == 24
pulses_per_clifford = sum(len(word) for word in clifford_decompositions.values()) / 24


def rb_sequence(depth, seed):
    """*depth* random Cliffords and their inverse, as native gate names."""
    rng = random.Random(seed)
    gates: tuple[str, ...] = ()
    for _ in range(depth):
        gates += rng.choice(list(clifford_decompositions.values()))
    return gates + clifford_decompositions[_up_to_phase(_unitary(gates).conj().T)]


@pulse.kernel
def clifford_sequence(qudit, gates):
    """Apply native gates (``"X90"`` or ``"Y90"``) with the qubit's calibrated
    pulses, then read out.

    *gates* is a tuple of names, so the loop unrolls while tracing.
    """
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)
    x90 = drive_line.get_calibration(pulse.gates.X90)
    y90 = drive_line.get_calibration(pulse.gates.Y90)
    measurement = readout_line.get_calibration(pulse.gates.READOUT)

    for gate in gates:
        if gate == "Y90":
            drive(drive_line, y90)
        else:
            drive(drive_line, x90)

    sync(drive_line, readout_line)
    return readout(readout_line, measurement)


def error_per_gate(lab, seed):
    """The error of one calibrated X90 pulse of ``lab.qubit``, from 12 sequences."""
    depths = [1, 5, 10, 20]
    survival = []
    for depth in depths:
        runs = []
        for repeat in range(3):
            gates = rb_sequence(depth, seed=1000 * seed + 10 * depth + repeat)
            runs.append(lab.run(clifford_sequence, lab.qubit, gates).level_population(0))
        survival.append(np.mean(runs))

    # survival = A * p**depth + 1/2: subtract the 1/2 and the decay is a
    # straight line on a log scale, whose slope is log(p).
    slope, intercept = np.polyfit(depths, np.log(np.array(survival) - 0.5), 1)
    p = math.exp(slope)
    error_per_clifford = (1 - p) / 2
    return error_per_clifford / pulses_per_clifford
