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

# 20 ns DRAG pulses with sigma = 5 ns, calibrated for the 17-qubit transmon
# device's drive scale.
cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "pi_amp": 0.26,
    "half_pi_amp": 0.13,
    "readout_duration": 500 * pulse.ns,
}


@pulse.kernel
def t1_experiment(qubit, wait_time):
    """T1 measurement: X pulse -> wait -> readout.

    The wait duration is parameterized to sweep T1 decay.
    """
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)

    x_pulse = drag(cal["x_duration"], cal["pi_amp"], cal["x_sigma"], cal["drag_beta"])
    drive(drive_line, x_pulse)

    wait(drive_line, wait_time)

    sync(drive_line, readout_line)
    readout(readout_line, square(cal["readout_duration"], 0.05))


@pulse.kernel
def t2_ramsey(qubit, tau):
    """T2* (Ramsey) measurement: pi/2 -> tau -> pi/2 -> readout."""
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)

    half_pi = drag(cal["x_duration"], cal["half_pi_amp"], cal["x_sigma"], cal["drag_beta"])

    drive(drive_line, half_pi)
    wait(drive_line, tau)
    drive(drive_line, half_pi)

    sync(drive_line, readout_line)
    readout(readout_line, square(cal["readout_duration"], 0.05))


@pulse.kernel
def t2_echo(qubit, tau):
    """T2 (Hahn echo) measurement: pi/2 -> tau/2 -> pi -> tau/2 -> pi/2 -> readout."""
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)

    half_pi = drag(cal["x_duration"], cal["half_pi_amp"], cal["x_sigma"], cal["drag_beta"])
    pi_pulse = drag(cal["x_duration"], cal["pi_amp"], cal["x_sigma"], cal["drag_beta"])
    half_tau = tau / 2

    drive(drive_line, half_pi)
    wait(drive_line, half_tau)
    drive(drive_line, pi_pulse)
    wait(drive_line, half_tau)
    drive(drive_line, half_pi)

    sync(drive_line, readout_line)
    readout(readout_line, square(cal["readout_duration"], 0.05))


if __name__ == "__main__":
    dev = pulse.device("transmon_17q")
    qubit = dev.qudit_handles[0]
    spec = dev.target.qubits[0]
    t1_ns = spec.t1_us * 1e3
    print(f"=== T1 experiment on {spec.label} (T1 = {spec.t1_us:.1f} us), CPU simulation ===")
    t1 = pulse.compile(t1_experiment, device=dev)
    first = None
    for wait_time in [0 * pulse.ns, 2 * pulse.us, 5 * pulse.us, 10 * pulse.us]:
        # A 0.5 ns step (one clock cycle) resolves these pulses and keeps
        # the long waits fast.
        result = pulse.simulate(t1.lower(qubit, wait_time), dt_ns=0.5)
        p1 = result.level_population(1)
        readout_ns = result.measurements[0].start_ns
        if first is None:
            first = (readout_ns, p1)
        p0 = spec.thermal_population
        expected = p0 + (first[1] - p0) * math.exp(-(readout_ns - first[0]) / t1_ns)
        print(
            f"  wait={wait_time / pulse.ns * dev.clock_ghz:6.0f} clock cycles: P(1) = {p1:.3f}"
            f"  (T1 decay predicts {expected:.3f})"
        )

    print("\n=== T2* Ramsey (tau=500) ===")
    ramsey_tau = 250 * pulse.ns
    compiled_kernel = pulse.compile(t2_ramsey, device=dev).lower(qubit, ramsey_tau)
    print(compiled_kernel.mlir())

    print("\n=== T2 Echo (tau=1000) ===")
    echo_tau = 500 * pulse.ns
    compiled_kernel = pulse.compile(t2_echo, device=dev).lower(qubit, echo_tau)
    print(compiled_kernel.mlir())
