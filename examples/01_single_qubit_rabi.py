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
symbolic across a sweep, durations as times with units, and the same kernel
compiled for a pinned device and for a device with another clock.
"""

import cudaq_pulse as pulse

cal = {"rabi_duration": 50 * pulse.ns, "readout_duration": 500 * pulse.ns}


@pulse.kernel
def rabi(qubit, duration, amplitude: float):
    """Drive *qubit* with a square pulse."""
    drive_line = get_drive_line(qubit)
    drive(drive_line, square(duration, amplitude))


@pulse.kernel
def rabi_with_readout(qubit, duration, amplitude):
    """Rabi experiment with measurement — the full circuit."""
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)

    drive(drive_line, square(duration, amplitude))

    sync(drive_line, readout_line)
    readout(readout_line, square(cal["readout_duration"], 0.05))


if __name__ == "__main__":
    dev = pulse.device("transmon_krinner_17q")
    qubit = dev.qudit_handles[0]

    print("=== Compiled for the qubit's device ===")
    compiled_kernel = rabi.lower(qubit, cal["rabi_duration"], 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Amplitude sweep (Rabi oscillation, CPU simulation) ===")
    # Pinning the device compiles for it ahead of the call.
    sweep = pulse.compile(rabi, device=dev)
    for amplitude in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        compiled_kernel = sweep.lower(qubit, cal["rabi_duration"], amplitude)
        p1 = pulse.simulate(compiled_kernel).level_population(1)
        print(f"  amplitude={amplitude:.2f}: P(1) = {p1:.3f}")
    print(f"  compiled {sweep.cache_info().compiled} time(s) for the whole sweep")

    print("\n=== The same kernel on a device with a 1 GHz clock ===")
    slow = pulse.device({"target": "transmon_krinner_17q", "clock_ghz": 1.0})
    for device in (dev, slow):
        lowered = rabi.lower(device.qudit_handles[0], cal["rabi_duration"], 0.5)
        cycles = lowered.schedule[0].duration_cycles
        print(f"  {device.clock_ghz:g} GHz: {cal['rabi_duration']} is {cycles} cycles")

    print("\n=== With readout ===")
    compiled_kernel = rabi_with_readout.lower(qubit, cal["rabi_duration"], 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
