# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
def echoed_cr(qudit_ctrl, qudit_tgt):
    """Echoed cross-resonance CNOT from *qudit_ctrl* to *qudit_tgt*."""
    drive_line_ctrl = get_drive_line(qudit_ctrl)
    drive_line_tgt = get_drive_line(qudit_tgt)

    sync(drive_line_ctrl, drive_line_tgt)

    x_duration, x_sigma, beta = cal["x_duration"], cal["x_sigma"], cal["drag_beta"]
    sx_pulse = drag(x_duration, 0.025, x_sigma, beta)
    x_ctrl = drag(x_duration, 0.047, x_sigma, beta)
    cr = gaussian_square(cal["cr_duration"], 0.10, x_sigma, cal["cr_width"])
    cr_neg = gaussian_square(cal["cr_duration"], -0.10, x_sigma, cal["cr_width"])

    drive(drive_line_tgt, sx_pulse)
    drive(drive_line_ctrl, cr, tone=drive_line_tgt.tone)
    drive(drive_line_ctrl, x_ctrl)
    drive(drive_line_ctrl, cr_neg, tone=drive_line_tgt.tone)
    drive(drive_line_tgt, sx_pulse)


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_surface17")
    scheduled = pulse.compile(echoed_cr, quantum_system=qs).lower(qs.qudits[0], qs.qudits[1])
    print(scheduled.mlir())
    print(f"  Compile metrics: {scheduled.metrics}")
