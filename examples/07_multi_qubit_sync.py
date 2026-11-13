# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-qubit programs with sync barriers and registers of qudit handles.

Shows qudits passed one by one and as a register (a slice of the device's
handles), the sync primitive for cross-line coordination, and patterns for
building multi-qubit pulse programs.
"""

import cudaq_pulse as pulse

cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "long_duration": 100 * pulse.ns,
    "stagger_wait": 250 * pulse.ns,
}

# ── 1. Qudits passed one by one ──────────────────────────────────────────────


@pulse.kernel
def two_qubit_simultaneous(qubit_0, qubit_1):
    """Drive two qubits simultaneously, then sync them."""

    drive_line_0 = get_drive_line(qubit_0)
    drive_line_1 = get_drive_line(qubit_1)

    drive(drive_line_0, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))
    drive(drive_line_1, gaussian(cal["x_duration"], 0.30, cal["x_sigma"]))

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))
    drive(drive_line_1, gaussian(cal["x_duration"], 0.30, cal["x_sigma"]))


# ── 2. Qudits and amplitudes as parameters ───────────────────────────────────


@pulse.kernel
def parametric_pi_half(qubit_0, qubit_1, amplitude_0, amplitude_1):
    """Calibration-friendly: pass the qudits and their amplitudes in."""
    drive_line_0 = get_drive_line(qubit_0)
    drive_line_1 = get_drive_line(qubit_1)

    x_duration, x_sigma, beta = cal["x_duration"], cal["x_sigma"], cal["drag_beta"]
    drive(drive_line_0, drag(x_duration, amplitude_0, x_sigma, beta))
    drive(drive_line_1, drag(x_duration, amplitude_1, x_sigma, beta))


# ── 3. A register of qudits ──────────────────────────────────────────────────


@pulse.kernel
def qvec_simultaneous_drive(qubit_vector, amplitude):
    """Drive two qubits of a register in parallel.

    dev.qudits[0:4] is a register of four qudits; individual qudits
    are accessed by indexing.
    """
    drive_line_0 = get_drive_line(qubit_vector[0])
    drive_line_1 = get_drive_line(qubit_vector[1])

    drive(drive_line_0, gaussian(cal["x_duration"], amplitude, cal["x_sigma"]))
    drive(drive_line_1, gaussian(cal["x_duration"], amplitude, cal["x_sigma"]))
    sync(drive_line_0, drive_line_1)


# ── 4. Sync patterns ─────────────────────────────────────────────────────────


@pulse.kernel
def sync_after_asymmetric_ops(qubit_0, qubit_1):
    """Sync after different-duration ops to re-align timelines.

    Without sync, the two lines would diverge in time.
    """

    drive_line_0 = get_drive_line(qubit_0)
    drive_line_1 = get_drive_line(qubit_1)

    drive(drive_line_0, square(cal["long_duration"], 0.1))
    drive(drive_line_1, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))
    drive(drive_line_1, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))


@pulse.kernel
def staggered_readout(qubit_0, qubit_1):
    """Stagger operations across qubits with wait + sync.

    Drive qubit_0, wait, then sync with qubit_1 before a joint operation.
    """

    drive_line_0 = get_drive_line(qubit_0)
    drive_line_1 = get_drive_line(qubit_1)

    drive(drive_line_0, gaussian(cal["x_duration"], 0.5, cal["x_sigma"]))
    wait(drive_line_0, cal["stagger_wait"])

    drive(drive_line_1, gaussian(cal["x_duration"], 0.5, cal["x_sigma"]))

    sync(drive_line_0, drive_line_1)

    drive(drive_line_0, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))
    drive(drive_line_1, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))


if __name__ == "__main__":
    dev = pulse.device("transmon_17q")
    print("=== Two-qubit simultaneous drive ===")
    compiled_kernel = pulse.compile(two_qubit_simultaneous, device=dev).lower(
        dev.qudits[0], dev.qudits[1]
    )
    print(compiled_kernel.mlir())

    print("\n=== Parametric pi/2 ===")
    compiled_kernel = pulse.compile(parametric_pi_half, device=dev).lower(
        dev.qudits[0], dev.qudits[1], 0.25, 0.28
    )
    print(compiled_kernel.mlir())

    print("\n=== Register simultaneous drive ===")
    compiled_kernel = pulse.compile(qvec_simultaneous_drive, device=dev).lower(dev.qudits[0:4], 0.3)
    print(compiled_kernel.mlir())

    print("\n=== Sync after asymmetric ops ===")
    compiled_kernel = pulse.compile(sync_after_asymmetric_ops, device=dev).lower(
        dev.qudits[0], dev.qudits[1]
    )
    print(compiled_kernel.mlir())

    print("\n=== Staggered readout ===")
    compiled_kernel = pulse.compile(staggered_readout, device=dev).lower(
        dev.qudits[0], dev.qudits[1]
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
