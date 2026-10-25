# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Multi-qubit programs with sync barriers and registers of qudit handles.

Shows qudits passed one by one and as a register (a slice of the device's
handles), the sync primitive for cross-line coordination, and patterns for
building multi-qubit pulse programs.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse

# ── 1. Explicit multi-qudit allocation ───────────────────────────────────────


@pulse.kernel
def two_qubit_simultaneous(qubit_0, qubit_1):
    """Drive two qubits simultaneously, then sync them.

    Each qudit is allocated individually inside the kernel.
    """

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    drive(drive_line_0, gaussian(40, 0.25, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, 0.30, 10.0), tone_1)

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(40, 0.25, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, 0.30, 10.0), tone_1)


# ── 2. External allocation — qudits passed as arguments ──────────────────────


@pulse.kernel
def parametric_pi_half(qubit_0, qubit_1, amplitude_0, amplitude_1):
    """Calibration-friendly: pass external qudit refs and amplitudes in."""
    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    drive(drive_line_0, drag(40, amplitude_0, 10.0, 0.5), tone_0)
    drive(drive_line_1, drag(40, amplitude_1, 10.0, 0.5), tone_1)


# ── 3. A register of qudits ──────────────────────────────────────────────────


@pulse.kernel
def qvec_simultaneous_drive(qubit_vector, amplitude):
    """Drive two qubits of a register in parallel.

    dev.qudit_handles[0:4] is a register of four qudits; individual qudits
    are accessed by indexing.
    """
    drive_line_0, tone_0 = get_drive_line(qubit_vector[0])
    drive_line_1, tone_1 = get_drive_line(qubit_vector[1])

    drive(drive_line_0, gaussian(40, amplitude, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, amplitude, 10.0), tone_1)
    sync(drive_line_0, drive_line_1)


# ── 4. Sync patterns ─────────────────────────────────────────────────────────


@pulse.kernel
def sync_after_asymmetric_ops(qubit_0, qubit_1):
    """Sync after different-duration ops to re-align timelines.

    Without sync, the two lines would diverge in time.
    """

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    drive(drive_line_0, square(200, 0.1), tone_0)
    drive(drive_line_1, gaussian(40, 0.25, 10.0), tone_1)

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(40, 0.25, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, 0.25, 10.0), tone_1)


@pulse.kernel
def staggered_readout(qubit_0, qubit_1):
    """Stagger operations across qubits with wait + sync.

    Drive qubit_0, wait, then sync with qubit_1 before a joint operation.
    """

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    drive(drive_line_0, gaussian(40, 0.5, 10.0), tone_0)
    wait(drive_line_0, 500)

    drive(drive_line_1, gaussian(40, 0.5, 10.0), tone_1)

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(40, 0.25, 10.0), tone_0)
    drive(drive_line_1, gaussian(40, 0.25, 10.0), tone_1)


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    print("=== Two-qubit simultaneous drive ===")
    compiled_kernel = pulse.compile(two_qubit_simultaneous, device=dev).lower(
        dev.qudit_handles[0], dev.qudit_handles[1]
    )
    print(compiled_kernel.mlir())

    print("\n=== Parametric pi/2 ===")
    compiled_kernel = pulse.compile(parametric_pi_half, device=dev).lower(
        dev.qudit_handles[0], dev.qudit_handles[1], 0.25, 0.28
    )
    print(compiled_kernel.mlir())

    print("\n=== Register simultaneous drive ===")
    compiled_kernel = pulse.compile(qvec_simultaneous_drive, device=dev).lower(
        dev.qudit_handles[0:4], 0.3
    )
    print(compiled_kernel.mlir())

    print("\n=== Sync after asymmetric ops ===")
    compiled_kernel = pulse.compile(sync_after_asymmetric_ops, device=dev).lower(
        dev.qudit_handles[0], dev.qudit_handles[1]
    )
    print(compiled_kernel.mlir())

    print("\n=== Staggered readout ===")
    compiled_kernel = pulse.compile(staggered_readout, device=dev).lower(
        dev.qudit_handles[0], dev.qudit_handles[1]
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
