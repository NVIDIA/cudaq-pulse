# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-qubit randomized benchmarking (RB) at the pulse level.

Draws random Clifford gates from a seed, appends the Clifford that inverts
the sequence, and runs the result as native X90 pulses, with Y90 as an X90
between two frame-phase shifts. Every sequence should return the qubit to
|0>; the CPU simulation shows the survival probability falling slowly with
depth, through T1 and T2 during the pulses.
"""

import math
import random

import cudaq_pulse as pulse
import numpy as np

# A 20 ns DRAG X90 for the 17-qubit transmon device.
cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "half_pi_amp": 0.13,
    "readout_duration": 500 * pulse.ns,
}

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
    """Apply native gates (``"X90"`` or ``"Y90"``), then read out.

    *gates* is a tuple of names, so the loop unrolls while tracing.
    """
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)

    x90 = drag(cal["x_duration"], cal["half_pi_amp"], cal["x_sigma"], cal["drag_beta"])

    for gate in gates:
        if gate == "Y90":
            shift_phase(drive_line.tone, math.pi / 2)
            drive(drive_line, x90)
            shift_phase(drive_line.tone, -math.pi / 2)
        else:
            drive(drive_line, x90)

    sync(drive_line, readout_line)
    readout(readout_line, square(cal["readout_duration"], 0.05))


if __name__ == "__main__":
    dev = pulse.device("transmon_17q")
    qudit = dev.qudits[0]
    rb = pulse.compile(clifford_sequence, device=dev)

    print("=== RB sequence (depth=2, seed=42) ===")
    gates = rb_sequence(2, 42)
    print(f"  native gates: {' '.join(gates)}")
    print(rb.lower(qudit, gates).mlir())

    print("\n=== Survival probability against depth (CPU simulation) ===")
    for depth in [1, 5, 10, 20]:
        gates = rb_sequence(depth, seed=depth)
        compiled_kernel = rb.lower(qudit, gates)
        survival = pulse.simulate(compiled_kernel, dt_ns=0.25).level_population(0)
        print(
            f"  depth={depth:3d} ({len(gates):3d} pulses): P(0) = {survival:.3f}, "
            f"compiled in {compiled_kernel.metrics.total_ms:.3f} ms"
        )
