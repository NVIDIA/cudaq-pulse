# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Phase and frequency manipulation on tone channels.

Demonstrates shift_phase, set_phase, shift_frequency, set_frequency —
the four tone-modification primitives that control the rotating frame
of a drive or readout channel.
"""

import math

import cudaq_pulse as pulse

cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "step_duration": 10 * pulse.ns,
    "park_wait": 50 * pulse.ns,
}


@pulse.kernel
def phase_rotation_demo(qudit):
    """Drive X, then shift tone by pi/2 to drive Y, then by pi to drive -X.

    Illustrates how phase shifts rotate the drive axis on the Bloch sphere.
    """
    drive_line = get_drive_line(qudit)

    pi2_pulse = gaussian(cal["x_duration"], 0.25, cal["x_sigma"])

    drive(drive_line, pi2_pulse)

    shift_phase(drive_line.tone, math.pi / 2)
    drive(drive_line, pi2_pulse)

    shift_phase(drive_line.tone, math.pi / 2)
    drive(drive_line, pi2_pulse)

    shift_phase(drive_line.tone, math.pi / 2)
    drive(drive_line, pi2_pulse)


@pulse.kernel
def set_phase_vs_shift_phase(qudit):
    """Contrast between set_phase (absolute) and shift_phase (relative).

    set_phase resets the tone to a fixed angle regardless of history.
    shift_phase accumulates relative to the current phase.
    """
    drive_line = get_drive_line(qudit)
    envelope = gaussian(cal["x_duration"], 0.3, cal["x_sigma"])

    shift_phase(drive_line.tone, math.pi / 4)
    drive(drive_line, envelope)

    shift_phase(drive_line.tone, math.pi / 4)
    drive(drive_line, envelope)

    set_phase(drive_line.tone, 0.0)
    drive(drive_line, envelope)


@pulse.kernel
def chirped_drive(qudit, step_count, total_shift_hz):
    """Frequency-chirped pulse: step the drive frequency across the pulse.

    Useful for spectroscopy sweeps compiled as a single kernel.
    """
    drive_line = get_drive_line(qudit)
    step_size = total_shift_hz / step_count

    for _i in range(step_count):
        drive(drive_line, square(cal["step_duration"], 0.1))
        shift_frequency(drive_line.tone, step_size)


@pulse.kernel
def sideband_modulation(qudit):
    """Use set_frequency to park the drive tone at a sideband.

    Common in flux-tunable transmon architectures where the drive
    line needs to address different transition frequencies.
    """
    drive_line = get_drive_line(qudit)

    set_frequency(drive_line.tone, 5.0e9)
    drive(drive_line, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))
    wait(drive_line, cal["park_wait"])

    set_frequency(drive_line.tone, 5.15e9)
    drive(drive_line, gaussian(cal["x_duration"], 0.25, cal["x_sigma"]))


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_surface17")
    print("=== Phase Rotation (4 x pi/2 = full circle) ===")
    compiled_kernel = pulse.compile(phase_rotation_demo, quantum_system=qs).lower(qs.qudits[0])
    print(compiled_kernel.mlir())

    print("\n=== set_phase vs shift_phase ===")
    compiled_kernel = pulse.compile(set_phase_vs_shift_phase, quantum_system=qs).lower(qs.qudits[0])
    print(compiled_kernel.mlir())

    print("\n=== Chirped Drive (10 steps, 50 MHz sweep) ===")
    compiled_kernel = pulse.compile(chirped_drive, quantum_system=qs).lower(qs.qudits[0], 10, 50e6)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Sideband Modulation ===")
    compiled_kernel = pulse.compile(sideband_modulation, quantum_system=qs).lower(qs.qudits[0])
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
