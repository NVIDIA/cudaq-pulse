# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Single-qubit randomized benchmarking (RB) at the pulse level.

Generates a sequence of random Clifford gates decomposed into native
pi/2 and pi pulses with phase shifts, followed by the inverse Clifford.
Demonstrates how pulse-level programming captures complex gate sequences
as flat, high-performance IR.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import math

import cudaq_pulse as pulse

CLIFFORD_DECOMPOSITIONS = [
    [],
    [("X90",)],
    [("X90",), ("X90",)],
    [("X90",), ("X90",), ("X90",)],
    [("Y90",)],
    [("Y90",), ("Y90",)],
    [("Y90",), ("Y90",), ("Y90",)],
    [("X90",), ("Y90",)],
    [("X90",), ("Y90",), ("Y90",), ("Y90",)],
    [("X90",), ("X90",), ("X90",), ("Y90",)],
    [("Y90",), ("X90",)],
    [("Y90",), ("X90",), ("X90",), ("X90",)],
]


@pulse.kernel
def clifford_sequence(qubit, sequence_length, seed):
    """Apply a sequence of single-qubit Cliffords + inversion + readout.

    The specific Cliffords are determined by `seed` at compile time.
    """
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, 0.25, 10.0, 0.5)

    for _step in range(sequence_length):
        drive(drive_line, half_pi, drive_tone)
        shift_phase(drive_tone, math.pi / 2)
        drive(drive_line, half_pi, drive_tone)
        shift_phase(drive_tone, -math.pi / 2)

    shift_phase(drive_tone, math.pi)
    for _step in range(sequence_length):
        shift_phase(drive_tone, math.pi / 2)
        drive(drive_line, half_pi, drive_tone)
        shift_phase(drive_tone, -math.pi / 2)
        drive(drive_line, half_pi, drive_tone)
    shift_phase(drive_tone, -math.pi)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


@pulse.kernel
def rb_depth_sweep(qubit):
    """Compile RB sequences at multiple depths in a single kernel.

    The outer loop over depths is unrolled at compile time; the inner
    Clifford loops use for-loop capture.
    """
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, 0.25, 10.0, 0.5)

    for depth in range(5):
        for _step in range(depth):
            drive(drive_line, half_pi, drive_tone)
            shift_phase(drive_tone, math.pi / 2)

        sync(drive_line, readout_line)
        readout(readout_line, square(1000, 0.05), readout_tone)
        sync(drive_line, readout_line)


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    print("=== RB Clifford sequence (depth=10) ===")
    compiled_kernel = pulse.compile(clifford_sequence, device=dev).lower(
        dev.qudit_handles[0], 10, 42
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Depth sweep ===")
    for depth in [1, 5, 10, 20]:
        compiled_kernel = pulse.compile(clifford_sequence, device=dev).lower(
            dev.qudit_handles[0], depth, 0
        )
        print(f"  depth={depth:3d}: compiled in {compiled_kernel.metrics.total_ms:.3f} ms")

    print("\n=== RB depth sweep (depths 0-4 in one kernel) ===")
    compiled_kernel = pulse.compile(rb_depth_sweep, device=dev).lower(dev.qudit_handles[0])
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
