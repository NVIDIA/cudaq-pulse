# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""T1/T2 measurement with Lindblad dissipator.

Prepares a qubit, applies a wait period, and measures, compiling each
experiment with ``pulse.compile()``. ``pulse.simulate()`` adds Lindblad
dissipators from a target's T1/T2 calibration when it lowers a kernel.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import cudaq_pulse as pulse


@pulse.kernel
def t1_experiment(wait_time):
    """T1 measurement: X pulse -> wait -> readout.

    The wait duration is parameterized to sweep T1 decay.
    """
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    x_pulse = drag(40, 0.5, 10.0, 0.5)
    drive(drive_line, x_pulse, drive_tone)

    wait(drive_line, wait_time)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


@pulse.kernel
def t2_ramsey(tau):
    """T2* (Ramsey) measurement: pi/2 -> tau -> pi/2 -> readout."""
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, 0.25, 10.0, 0.5)

    drive(drive_line, half_pi, drive_tone)
    wait(drive_line, tau)
    drive(drive_line, half_pi, drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


@pulse.kernel
def t2_echo(tau):
    """T2 (Hahn echo) measurement: pi/2 -> tau/2 -> pi -> tau/2 -> pi/2 -> readout."""
    qubit = pulse.qudit_ref()
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, 0.25, 10.0, 0.5)
    pi_pulse = drag(40, 0.50, 10.0, 0.5)
    half_tau = tau / 2

    drive(drive_line, half_pi, drive_tone)
    wait(drive_line, half_tau)
    drive(drive_line, pi_pulse, drive_tone)
    wait(drive_line, half_tau)
    drive(drive_line, half_pi, drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    print("=== T1 experiment (sweep wait times) ===")
    for tau in [100, 500, 1000, 5000, 10000]:
        compiled_kernel = pulse.compile(t1_experiment, device=dev).lower(tau)
        print(f"  tau={tau:6d} VTU: compiled in {compiled_kernel.metrics.total_ms:.3f} ms")

    print("\n=== T2* Ramsey (tau=500) ===")
    compiled_kernel = pulse.compile(t2_ramsey, device=dev).lower(500)
    print(compiled_kernel.mlir())

    print("\n=== T2 Echo (tau=1000) ===")
    compiled_kernel = pulse.compile(t2_echo, device=dev).lower(1000)
    print(compiled_kernel.mlir())

    # pulse.simulate(...) adds the device's T1/T2 Lindblad
    # dissipators when it lowers a kernel; see examples/transmon/krinner_evolve_gpu.py.
