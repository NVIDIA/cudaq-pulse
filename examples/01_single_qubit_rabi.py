# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Single-qubit Rabi oscillation.

Drives a qubit with a square pulse of varying amplitude to trace out
Rabi oscillations. The simplest end-to-end example: qudit handles from a
device, a float parameter that stays symbolic across a sweep, and a kernel
that allocates its qudit itself and is pinned to a device.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse


@pulse.kernel
def rabi_external(qubit, duration: int, amplitude: float):
    """Rabi with qudit allocated externally and passed in."""
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, square(duration, amplitude), tone)


@pulse.kernel
def rabi_internal(duration: int, amplitude: float):
    """Rabi with qudit allocated inside the kernel."""
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, square(duration, amplitude), tone)


@pulse.kernel
def rabi_with_readout(duration, amplitude):
    """Rabi experiment with measurement — the full circuit."""
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    drive(drive_line, square(duration, amplitude), drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    print("=== External allocation ===")
    qubit = dev.qudit_handles[0]
    compiled_kernel = rabi_external.lower(qubit, 100, 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Internal allocation ===")
    compiled_kernel = pulse.compile(rabi_internal, device=dev).lower(100, 0.5)
    print(compiled_kernel.mlir())

    print("\n=== Amplitude sweep (Rabi oscillation) ===")
    sweep = pulse.compile(rabi_external)
    for amplitude in [0.1, 0.2, 0.3, 0.4, 0.5]:
        compiled_kernel = sweep.lower(qubit, 100, amplitude)
        print(
            f"  amplitude={amplitude:.1f}: specialized in {compiled_kernel.metrics.total_ms:.3f} ms"
        )
    print(f"  compiled {sweep.cache_info().compiled} time(s) for the whole sweep")

    print("\n=== With readout ===")
    compiled_kernel = pulse.compile(rabi_with_readout, device=dev).lower(100, 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
