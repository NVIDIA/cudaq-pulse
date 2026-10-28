# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Two-qubit CNOT via echoed cross-resonance.

Demonstrates cross-resonance driving: the control qubit's drive line
is modulated at the target qubit's frequency (via the target's tone).
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
def echoed_cr(qubit_ctrl, qubit_tgt):
    """Echoed cross-resonance CNOT from *qubit_ctrl* to *qubit_tgt*."""
    drive_line_ctrl, tone_ctrl = get_drive_line(qubit_ctrl)
    drive_line_tgt, tone_tgt = get_drive_line(qubit_tgt)

    sync(drive_line_ctrl, drive_line_tgt)

    x_duration, x_sigma, beta = cal["x_duration"], cal["x_sigma"], cal["drag_beta"]
    sx_pulse = drag(x_duration, 0.025, x_sigma, beta)
    x_ctrl = drag(x_duration, 0.047, x_sigma, beta)
    cr = gaussian_square(cal["cr_duration"], 0.10, x_sigma, cal["cr_width"])
    cr_neg = gaussian_square(cal["cr_duration"], -0.10, x_sigma, cal["cr_width"])

    drive(drive_line_tgt, sx_pulse, tone_tgt)
    drive(drive_line_ctrl, cr, tone_tgt)
    drive(drive_line_ctrl, x_ctrl, tone_ctrl)
    drive(drive_line_ctrl, cr_neg, tone_tgt)
    drive(drive_line_tgt, sx_pulse, tone_tgt)


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    compiled_kernel = pulse.compile(echoed_cr, device=dev).lower(
        dev.qudit_handles[0], dev.qudit_handles[1]
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
