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


@pulse.kernel
def echoed_cr(qubit_ctrl, qubit_tgt):
    """Echoed cross-resonance CNOT from *qubit_ctrl* to *qubit_tgt*."""
    drive_line_ctrl, tone_ctrl = get_drive_line(qubit_ctrl)
    drive_line_tgt, tone_tgt = get_drive_line(qubit_tgt)

    sync(drive_line_ctrl, drive_line_tgt)

    sx_pulse = drag(40, 0.025, 10.0, 0.5)
    x_ctrl = drag(40, 0.047, 10.0, 0.5)
    cr = gaussian_square(200, 0.10, 10.0, 160)
    cr_neg = gaussian_square(200, -0.10, 10.0, 160)

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
