# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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
def ghz_3(qubit_0, qubit_1, qubit_2):
    """3-qubit GHZ state: H(qubit_0) - CNOT(0,1) - CNOT(1,2).

    Each CNOT is an echoed cross-resonance gate:
      SX(target) - CR(ctrl->tgt) - X(ctrl) - CR_neg(ctrl->tgt) - SX(target)
    """

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)
    drive_line_2, tone_2 = get_drive_line(qubit_2)

    x_duration, x_sigma, beta = cal["x_duration"], cal["x_sigma"], cal["drag_beta"]
    sx_pulse = drag(x_duration, 0.25, x_sigma, beta)
    x_pi = drag(x_duration, 0.50, x_sigma, beta)
    cr_pos = gaussian_square(cal["cr_duration"], 0.05, x_sigma, cal["cr_width"])
    cr_neg = gaussian_square(cal["cr_duration"], -0.05, x_sigma, cal["cr_width"])

    # H(qubit_0) = Rz(pi) SX Rz(pi/2)
    shift_phase(tone_0, 3.14159)
    drive(drive_line_0, sx_pulse, tone_0)
    shift_phase(tone_0, 1.5708)

    sync(drive_line_0, drive_line_1)

    # CNOT(0, 1): echoed CR
    drive(drive_line_1, sx_pulse, tone_1)  # SX on target
    drive(drive_line_0, cr_pos, tone_1)  # CR on ctrl at target freq
    drive(drive_line_0, x_pi, tone_0)  # X on ctrl
    drive(drive_line_0, cr_neg, tone_1)  # CR_neg on ctrl at target freq
    drive(drive_line_1, sx_pulse, tone_1)  # SX on target

    sync(drive_line_0, drive_line_1, drive_line_2)

    # CNOT(1, 2): echoed CR
    drive(drive_line_2, sx_pulse, tone_2)
    drive(drive_line_1, cr_pos, tone_2)
    drive(drive_line_1, x_pi, tone_1)
    drive(drive_line_1, cr_neg, tone_2)
    drive(drive_line_2, sx_pulse, tone_2)


@pulse.kernel
def ghz_n(qubit_0, qubit_1, qubit_count):
    """N-qubit GHZ state using a loop of CNOT gates.

    The outer structure: H on qubit_0, then CNOT chain.
    The for-loop demonstrates loop capture for repeated gate patterns.
    """

    drive_line_0, tone_0 = get_drive_line(qubit_0)
    drive_line_1, tone_1 = get_drive_line(qubit_1)

    x_duration, x_sigma, beta = cal["x_duration"], cal["x_sigma"], cal["drag_beta"]
    sx_pulse = drag(x_duration, 0.25, x_sigma, beta)
    x_pi = drag(x_duration, 0.50, x_sigma, beta)
    cr_pos = gaussian_square(cal["cr_duration"], 0.05, x_sigma, cal["cr_width"])
    cr_neg = gaussian_square(cal["cr_duration"], -0.05, x_sigma, cal["cr_width"])

    # H(qubit_0)
    shift_phase(tone_0, 3.14159)
    drive(drive_line_0, sx_pulse, tone_0)
    shift_phase(tone_0, 1.5708)

    # CNOT chain: for a 2-qubit kernel we do one CNOT(0,1)
    # In a real N-qubit system, this loop would iterate over qubit pairs
    for _i in range(qubit_count - 1):
        sync(drive_line_0, drive_line_1)
        drive(drive_line_1, sx_pulse, tone_1)
        drive(drive_line_0, cr_pos, tone_1)
        drive(drive_line_0, x_pi, tone_0)
        drive(drive_line_0, cr_neg, tone_1)
        drive(drive_line_1, sx_pulse, tone_1)


if __name__ == "__main__":
    dev = pulse.device("transmon_krinner_17q")
    print("=== 3-qubit GHZ (explicit) ===")
    compiled_kernel = pulse.compile(ghz_3, device=dev).lower(
        dev.qudit_handles[0], dev.qudit_handles[1], dev.qudit_handles[2]
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== N-qubit GHZ (loop-based, N=5) ===")
    compiled_kernel = pulse.compile(ghz_n, device=dev).lower(
        dev.qudit_handles[0], dev.qudit_handles[1], 5
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
