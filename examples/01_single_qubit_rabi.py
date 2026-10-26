# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Single-qubit Rabi oscillation.

Drives a qubit with a square pulse of varying amplitude and simulates each
amplitude on the CPU to trace out the Rabi oscillation. The simplest
end-to-end example: qudit handles from a device, a float parameter that stays
symbolic across a sweep, and the same kernel compiled for a pinned device.
"""

import cudaq_pulse as pulse


@pulse.kernel
def rabi(qubit, duration: int, amplitude: float):
    """Drive *qubit* with a square pulse."""
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, square(duration, amplitude), tone)


@pulse.kernel
def rabi_with_readout(qubit, duration, amplitude):
    """Rabi experiment with measurement — the full circuit."""
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    drive(drive_line, square(duration, amplitude), drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    qubit = dev.qudit_handles[0]

    print("=== Compiled for the qubit's device ===")
    compiled_kernel = rabi.lower(qubit, 100, 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Amplitude sweep (Rabi oscillation, CPU simulation) ===")
    # Pinning the device compiles for it ahead of the call.
    sweep = pulse.compile(rabi, device=dev)
    for amplitude in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        compiled_kernel = sweep.lower(qubit, 100, amplitude)
        p1 = pulse.simulate(compiled_kernel).density_matrix[1, 1].real
        print(f"  amplitude={amplitude:.2f}: P(1) = {p1:.3f}")
    print(f"  compiled {sweep.cache_info().compiled} time(s) for the whole sweep")

    print("\n=== With readout ===")
    compiled_kernel = rabi_with_readout.lower(qubit, 100, 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
