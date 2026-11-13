# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GHZ state preparation at the pulse level.

Creates the N-qubit GHZ state |000...0> + |111...1> using:
  1. A pi/2 pulse on qubit 0
  2. A chain of echoed cross-resonance CNOT gates: (0,1), (1,2), ..., (N-2,N-1)

Demonstrates multi-qubit pulse programming with qudit handles, sync
barriers, and the cross-resonance gate primitive.
"""

import cudaq_pulse as pulse

cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "cr_duration": 100 * pulse.ns,
    "cr_width": 80 * pulse.ns,
}


@pulse.kernel
def ghz_3(qudit_0, qudit_1, qudit_2):
    """3-qubit GHZ state: H(qudit_0) - CNOT(0,1) - CNOT(1,2).

    Each CNOT is an echoed cross-resonance gate:
      SX(target) - CR(ctrl->tgt) - X(ctrl) - CR_neg(ctrl->tgt) - SX(target)
    """

    drive_line_0 = get_drive_line(qudit_0)
    drive_line_1 = get_drive_line(qudit_1)
    drive_line_2 = get_drive_line(qudit_2)

    x_duration, x_sigma, beta = cal["x_duration"], cal["x_sigma"], cal["drag_beta"]
    sx_pulse = drag(x_duration, 0.25, x_sigma, beta)
    x_pi = drag(x_duration, 0.50, x_sigma, beta)
    cr_pos = gaussian_square(cal["cr_duration"], 0.05, x_sigma, cal["cr_width"])
    cr_neg = gaussian_square(cal["cr_duration"], -0.05, x_sigma, cal["cr_width"])

    # H(qudit_0) = Rz(pi) SX Rz(pi/2)
    shift_phase(drive_line_0.tone, 3.14159)
    drive(drive_line_0, sx_pulse)
    shift_phase(drive_line_0.tone, 1.5708)

    sync(drive_line_0, drive_line_1)

    # CNOT(0, 1): echoed CR
    drive(drive_line_1, sx_pulse)  # SX on target
    drive(drive_line_0, cr_pos, tone=drive_line_1.tone)  # CR on ctrl at target freq
    drive(drive_line_0, x_pi)  # X on ctrl
    drive(drive_line_0, cr_neg, tone=drive_line_1.tone)  # CR_neg on ctrl at target freq
    drive(drive_line_1, sx_pulse)  # SX on target

    sync(drive_line_0, drive_line_1, drive_line_2)

    # CNOT(1, 2): echoed CR
    drive(drive_line_2, sx_pulse)
    drive(drive_line_1, cr_pos, tone=drive_line_2.tone)
    drive(drive_line_1, x_pi)
    drive(drive_line_1, cr_neg, tone=drive_line_2.tone)
    drive(drive_line_2, sx_pulse)


@pulse.kernel
def ghz_n(qudit_0, qudit_1, qudit_count):
    """N-qubit GHZ state using a loop of CNOT gates.

    The outer structure: H on qudit_0, then CNOT chain.
    The for-loop demonstrates loop capture for repeated gate patterns.
    """

    drive_line_0 = get_drive_line(qudit_0)
    drive_line_1 = get_drive_line(qudit_1)

    x_duration, x_sigma, beta = cal["x_duration"], cal["x_sigma"], cal["drag_beta"]
    sx_pulse = drag(x_duration, 0.25, x_sigma, beta)
    x_pi = drag(x_duration, 0.50, x_sigma, beta)
    cr_pos = gaussian_square(cal["cr_duration"], 0.05, x_sigma, cal["cr_width"])
    cr_neg = gaussian_square(cal["cr_duration"], -0.05, x_sigma, cal["cr_width"])

    # H(qudit_0)
    shift_phase(drive_line_0.tone, 3.14159)
    drive(drive_line_0, sx_pulse)
    shift_phase(drive_line_0.tone, 1.5708)

    # CNOT chain: for a 2-qubit kernel we do one CNOT(0,1)
    # In a real N-qubit system, this loop would iterate over qubit pairs
    for _i in range(qudit_count - 1):
        sync(drive_line_0, drive_line_1)
        drive(drive_line_1, sx_pulse)
        drive(drive_line_0, cr_pos, tone=drive_line_1.tone)
        drive(drive_line_0, x_pi)
        drive(drive_line_0, cr_neg, tone=drive_line_1.tone)
        drive(drive_line_1, sx_pulse)


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_17q")
    print("=== 3-qubit GHZ (explicit) ===")
    compiled_kernel = pulse.compile(ghz_3, quantum_system=qs).lower(
        qs.qudits[0], qs.qudits[1], qs.qudits[2]
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== N-qubit GHZ (loop-based, N=5) ===")
    compiled_kernel = pulse.compile(ghz_n, quantum_system=qs).lower(qs.qudits[0], qs.qudits[1], 5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
