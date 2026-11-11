# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-qubit Rabi oscillation.

Drives a qubit with a square pulse of varying amplitude and simulates each
amplitude on the CPU to trace out the Rabi oscillation. The simplest
end-to-end example: qudit handles from a quantum system, a float parameter that stays
symbolic across a sweep, durations as times with units, and the same kernel
compiled for a pinned quantum system and for a quantum system with another clock.
"""

import cudaq_pulse as pulse

cal = {"rabi_duration": 50 * pulse.ns, "readout_duration": 500 * pulse.ns}


@pulse.kernel
def rabi(qudit, duration, amplitude: float):
    """Drive *qudit* with a square pulse."""
    drive_line = get_drive_line(qudit)
    drive(drive_line, square(duration, amplitude))


@pulse.kernel
def rabi_with_readout(qudit, duration, amplitude):
    """Rabi experiment with measurement — the full circuit."""
    drive_line = get_drive_line(qudit)
    readout_line = get_readout_line(qudit)

    drive(drive_line, square(duration, amplitude))

    sync(drive_line, readout_line)
    readout(readout_line, square(cal["readout_duration"], 0.05))


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_surface17")
    qudit = qs.qudits[0]

    print("=== Compiled for the qubit's quantum system ===")
    compiled_kernel = rabi.lower(qudit, cal["rabi_duration"], 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Amplitude sweep (Rabi oscillation, CPU simulation) ===")
    # Pinning the quantum system compiles for it ahead of the call.
    sweep = pulse.compile(rabi, quantum_system=qs)
    for amplitude in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]:
        compiled_kernel = sweep.lower(qudit, cal["rabi_duration"], amplitude)
        p1 = pulse.simulate(compiled_kernel).level_population(1)
        print(f"  amplitude={amplitude:.2f}: P(1) = {p1:.3f}")
    print(f"  compiled {sweep.cache_info().compiled} time(s) for the whole sweep")

    print("\n=== The same kernel on a quantum system with a 1 GHz clock ===")
    slow = pulse.quantum_system("transmon_surface17", clock_ghz=1.0)
    for quantum_system in (qs, slow):
        compiled = rabi.lower(quantum_system.qudits[0], cal["rabi_duration"], 0.5)
        cycles = compiled.schedule[0].duration_cycles
        print(f"  {quantum_system.clock_ghz:g} GHz: {cal['rabi_duration']} is {cycles} cycles")

    print("\n=== With readout ===")
    compiled_kernel = rabi_with_readout.lower(qudit, cal["rabi_duration"], 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
