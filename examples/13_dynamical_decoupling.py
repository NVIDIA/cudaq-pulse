# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamical decoupling pulse sequences: XY4, Uhrig, and CPMG.

Each sequence uses a different refocusing pattern to suppress different
noise spectra. All three use for-loops that capture as rolled scf.for ops
in the IR — demonstrating the kernel's loop-capture capability.
"""

import math

import cudaq_pulse as pulse


@pulse.kernel
def xy4(qudit, sigma, amplitude_pi, tau, cycle_count):
    """XY4 dynamical decoupling: [X - tau - Y - tau - X - tau - Y - tau]^n.

    Suppresses both dephasing and amplitude noise by alternating X and Y
    refocusing axes.
    """
    drive_line = get_drive_line(qudit)

    pi_x = gaussian(4 * sigma, amplitude_pi, sigma)

    for _cycle in range(cycle_count):
        # X refocus
        drive(drive_line, pi_x)
        wait(drive_line, tau)

        # Y refocus = phase shift pi/2, then pi_X, then undo phase
        shift_phase(drive_line.tone, math.pi / 2)
        drive(drive_line, pi_x)
        shift_phase(drive_line.tone, -math.pi / 2)
        wait(drive_line, tau)

        # X refocus
        drive(drive_line, pi_x)
        wait(drive_line, tau)

        # Y refocus
        shift_phase(drive_line.tone, math.pi / 2)
        drive(drive_line, pi_x)
        shift_phase(drive_line.tone, -math.pi / 2)
        wait(drive_line, tau)


@pulse.kernel
def uhrig_dd(qudit, sigma, amplitude_pi, total_cycles, pulse_count):
    """Uhrig dynamical decoupling (UDD).

    Places n refocusing pulses at non-uniform intervals:
        t_j = T * sin^2(pi * j / (2n + 2))

    Optimal for pure dephasing from a soft-cutoff noise spectrum.
    The intervals are computed at compile time from pulse_count, in whole
    clock cycles of a sequence *total_cycles* long.
    """
    drive_line = get_drive_line(qudit)

    pi_pulse = gaussian(4 * sigma, amplitude_pi, sigma)
    dur_pi = pulse.round_duration(drive_line, 4 * sigma)

    prev_time = 0
    for j in range(pulse_count):
        frac = math.sin(math.pi * (j + 1) / (2 * pulse_count + 2)) ** 2
        tj = int(total_cycles * frac)
        gap = tj - prev_time - dur_pi
        if gap > 0:
            wait(drive_line, gap)
        drive(drive_line, pi_pulse)
        prev_time = tj

    remaining = total_cycles - prev_time
    if remaining > 0:
        wait(drive_line, remaining)


@pulse.kernel
def cpmg(qudit, sigma, amplitude_half_pi, amplitude_pi, tau, refocus_count):
    """Carr-Purcell-Meiboom-Gill: pi/2_X - [tau - pi_Y - tau]^n - pi/2_X.

    The standard multi-pulse echo for dephasing suppression.
    """
    drive_line = get_drive_line(qudit)

    half_pi_pulse = gaussian(4 * sigma, amplitude_half_pi, sigma)
    pi_pulse = gaussian(4 * sigma, amplitude_pi, sigma)

    drive(drive_line, half_pi_pulse)

    for _i in range(refocus_count):
        wait(drive_line, tau)
        shift_phase(drive_line.tone, math.pi / 2)
        drive(drive_line, pi_pulse)
        shift_phase(drive_line.tone, -math.pi / 2)
        wait(drive_line, tau)

    drive(drive_line, half_pi_pulse)


@pulse.kernel
def knill_dd(qudit, sigma, amplitude_pi, tau):
    """Knill dynamical decoupling (KDD): a composite-pulse sequence.

    Uses 5 pi-pulses with carefully chosen phases to suppress both
    systematic over/under-rotation and dephasing.
    Phases: [pi/6, 0, pi/2, 0, pi/6]
    """
    drive_line = get_drive_line(qudit)

    pi_pulse = gaussian(4 * sigma, amplitude_pi, sigma)

    phases = [math.pi / 6, 0.0, math.pi / 2, 0.0, math.pi / 6]

    for idx in range(5):
        wait(drive_line, tau)
        shift_phase(drive_line.tone, phases[idx])
        drive(drive_line, pi_pulse)
        shift_phase(drive_line.tone, -phases[idx])


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_surface17")
    sigma = 5 * pulse.ns  # every pulse lasts 4 sigma
    amplitude_pi = 0.50
    amplitude_half_pi = 0.25
    tau = 100 * pulse.ns

    print("=== XY4 (4 cycles) ===")
    compiled_kernel = pulse.compile(xy4, quantum_system=qs).lower(
        qs.qudits[0], sigma, amplitude_pi, tau, 4
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Uhrig DD (8 pulses, T=5000 clock cycles) ===")
    compiled_kernel = pulse.compile(uhrig_dd, quantum_system=qs).lower(
        qs.qudits[0], sigma, amplitude_pi, 5000, 8
    )
    print(compiled_kernel.mlir())

    print("\n=== CPMG (8 refocusing pulses) ===")
    compiled_kernel = pulse.compile(cpmg, quantum_system=qs).lower(
        qs.qudits[0], sigma, amplitude_half_pi, amplitude_pi, tau, 8
    )
    print(compiled_kernel.mlir())

    print("\n=== KDD (Knill DD) ===")
    compiled_kernel = pulse.compile(knill_dd, quantum_system=qs).lower(
        qs.qudits[0], sigma, amplitude_pi, tau
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
