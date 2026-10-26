# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""T1/T2 measurement with Lindblad dissipators.

Prepares a qubit, waits, and measures. ``pulse.simulate()`` evolves each
compiled experiment with the Lindblad dissipators of the target's T1 and T2
and the qubit's thermal population, on the CPU. The T1 sweep is simulated
and compared with the exponential decay T1 predicts; the Ramsey and echo
experiments are compiled and printed.
"""

import math

import cudaq_pulse as pulse

# DRAG pulses of 40 clock cycles (20 ns) with sigma = 10 cycles, calibrated
# for the 17-qubit transmon device's drive scale.
PI_AMPLITUDE = 0.26
HALF_PI_AMPLITUDE = 0.13


@pulse.kernel
def t1_experiment(qubit, wait_time):
    """T1 measurement: X pulse -> wait -> readout.

    The wait duration is parameterized to sweep T1 decay.
    """
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    x_pulse = drag(40, PI_AMPLITUDE, 10.0, 0.5)
    drive(drive_line, x_pulse, drive_tone)

    wait(drive_line, wait_time)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


@pulse.kernel
def t2_ramsey(qubit, tau):
    """T2* (Ramsey) measurement: pi/2 -> tau -> pi/2 -> readout."""
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, HALF_PI_AMPLITUDE, 10.0, 0.5)

    drive(drive_line, half_pi, drive_tone)
    wait(drive_line, tau)
    drive(drive_line, half_pi, drive_tone)

    sync(drive_line, readout_line)
    readout(readout_line, square(1000, 0.05), readout_tone)


@pulse.kernel
def t2_echo(qubit, tau):
    """T2 (Hahn echo) measurement: pi/2 -> tau/2 -> pi -> tau/2 -> pi/2 -> readout."""
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    half_pi = drag(40, HALF_PI_AMPLITUDE, 10.0, 0.5)
    pi_pulse = drag(40, PI_AMPLITUDE, 10.0, 0.5)
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
    qubit = dev.qudit_handles[0]
    spec = dev.target.qubits[0]
    t1_ns = spec.t1_us * 1e3
    print(f"=== T1 experiment on {spec.label} (T1 = {spec.t1_us:.1f} us), CPU simulation ===")
    t1 = pulse.compile(t1_experiment, device=dev)
    first = None
    for wait_cycles in [0, 4000, 10000, 20000]:
        # A 0.5 ns step (one clock cycle) resolves these pulses and keeps
        # the long waits fast.
        result = pulse.simulate(t1.lower(qubit, wait_cycles), dt_ns=0.5)
        p1 = result.density_matrix[1, 1].real
        readout_ns = result.measurements[0].start_ns
        if first is None:
            first = (readout_ns, p1)
        p0 = spec.thermal_population
        expected = p0 + (first[1] - p0) * math.exp(-(readout_ns - first[0]) / t1_ns)
        print(
            f"  wait={wait_cycles:6d} clock cycles: P(1) = {p1:.3f}"
            f"  (T1 decay predicts {expected:.3f})"
        )

    print("\n=== T2* Ramsey (tau=500) ===")
    compiled_kernel = pulse.compile(t2_ramsey, device=dev).lower(qubit, 500)
    print(compiled_kernel.mlir())

    print("\n=== T2 Echo (tau=1000) ===")
    compiled_kernel = pulse.compile(t2_echo, device=dev).lower(qubit, 1000)
    print(compiled_kernel.mlir())
