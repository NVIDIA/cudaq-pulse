# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical spin-echo example.

Implements the Hahn echo sequence: pi/2 - tau - pi - tau - pi/2,
which refocuses dephasing due to static frequency detuning and
low-frequency noise.
"""

import cudaq_pulse as pulse
import numpy as np


@pulse.kernel
def hahn_echo(qubit, sigma, amp_half_pi, amp_pi, tau):
    """Hahn echo: pi/2_X - tau - pi_X - tau - pi/2_X."""
    drive_line = get_drive_line(qubit)

    dur_half_pi = 4 * sigma
    dur_pi = 4 * sigma

    half_pi_envelope = gaussian(dur_half_pi, amp_half_pi, sigma)
    pi_envelope = gaussian(dur_pi, amp_pi, sigma)

    # pi/2 pulse about X
    drive(drive_line, half_pi_envelope)

    # Free evolution
    wait(drive_line, tau)

    # pi refocusing pulse about X
    drive(drive_line, pi_envelope)

    # Free evolution
    wait(drive_line, tau)

    # Final pi/2 pulse about X
    drive(drive_line, half_pi_envelope)


@pulse.kernel
def cpmg_echo(qubit, sigma, amp_half_pi, amp_pi, tau, refocus_count):
    """CPMG dynamical decoupling: pi/2_X - [tau - pi_Y - tau]^n - pi/2_X."""
    drive_line = get_drive_line(qubit)

    dur_half_pi = 4 * sigma
    dur_pi = 4 * sigma

    half_pi_envelope = gaussian(dur_half_pi, amp_half_pi, sigma)
    pi_envelope = gaussian(dur_pi, amp_pi, sigma)

    # Initial pi/2 about X
    drive(drive_line, half_pi_envelope)

    for _i in range(refocus_count):
        wait(drive_line, tau)
        # pi about Y = phase shift of pi/2 then pi_X
        shift_phase(drive_line.tone, np.pi / 2)
        drive(drive_line, pi_envelope)
        shift_phase(drive_line.tone, -np.pi / 2)
        wait(drive_line, tau)

    # Final pi/2 about X
    drive(drive_line, half_pi_envelope)


def cycles(time):
    """*time* in cycles of the device's program clock."""
    return time / pulse.ns * dev.clock_ghz


def main():
    sigma = 5 * pulse.ns  # Gaussian sigma; pulses last 4 sigma
    amp_half_pi = 0.25
    amp_pi = 0.50
    tau = 250 * pulse.ns

    print("=== Hahn Echo ===")
    compiled_kernel = pulse.compile(hahn_echo, device=dev).lower(
        dev.qudit_handles[0], sigma, amp_half_pi, amp_pi, tau
    )
    total_hahn = 2 * (4 * sigma) + 4 * sigma + 2 * tau
    print(f"  Expected duration: {cycles(total_hahn):.0f} clock cycles")
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== CPMG-4 Echo ===")
    refocus_count = 4
    compiled_kernel = pulse.compile(cpmg_echo, device=dev).lower(
        dev.qudit_handles[0], sigma, amp_half_pi, amp_pi, tau, refocus_count
    )
    total_cpmg = 2 * (4 * sigma) + refocus_count * (4 * sigma + 2 * tau)
    print(f"  Expected duration: {cycles(total_cpmg):.0f} clock cycles")
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")


if __name__ == "__main__":
    dev = pulse.device("transmon_17q")
    main()
