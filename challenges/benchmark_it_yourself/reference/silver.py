# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark it yourself: a solution that reaches silver.

The bronze solution averages the sequences of a depth and fits the mean
against the depth. But a Clifford is anything from 0 to 5 pulses, so two
sequences of the same depth differ by tens of pulses, and it is the pulses
that cost fidelity: most of the scatter of the bronze fit is that difference.

The one idea here: fit every sequence against its own number of pulses.
Eight sequences then give the error per pulse with a scatter below 1 %, and
there is no average number of pulses per Clifford to divide by.

The estimate lands a few percent below the scorer's reference, and that is
not noise. The reference counts the pulse's leakage to level 2 in full, which
randomized benchmarking does not, and the qubit decays during the readout at
the end.

Everything down to the kernel is the bronze solution's, unchanged.
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
    """The error of one calibrated X90 pulse of ``lab.qubit``, from 8 sequences."""
    pulses = []
    survival = []
    for depth in [1, 50]:
        for repeat in range(4):
            gates = rb_sequence(depth, seed=1000 * seed + 10 * depth + repeat)
            pulses.append(len(gates))
            survival.append(lab.run(clifford_sequence, lab.qubit, gates).level_population(0))

    # survival = A * q**pulses + 1/2, with q the decay per pulse: a straight
    # line on a log scale against each sequence's own number of pulses.
    slope, intercept = np.polyfit(pulses, np.log(np.array(survival) - 0.5), 1)
    q = math.exp(slope)
    return (1 - q) / 2
