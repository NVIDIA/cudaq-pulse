# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Readout, measurement, and measurement-conditioned feedback.

Shows how to use readout channels and acquire IQ data, and how an ``if`` on a
measurement result is kept as a conditional region in the kernel IR.
"""

import math

import cudaq_pulse as pulse

cal = {
    "x_duration": 20 * pulse.ns,
    "x_sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "readout_duration": 500 * pulse.ns,
    "shot_gap": 250 * pulse.ns,
}


@pulse.kernel
def measure_single_qubit(qubit):
    """Prepare |1> and measure.

    Demonstrates the readout channel: get_readout_line returns the readout
    line, and readout() acquires IQ data at the line's own tone.
    """
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)

    x_pulse = drag(cal["x_duration"], 0.5, cal["x_sigma"], cal["drag_beta"])
    drive(drive_line, x_pulse)

    sync(drive_line, readout_line)

    readout_envelope = square(cal["readout_duration"], 0.05)
    readout(readout_line, readout_envelope)


@pulse.kernel
def active_reset(qubit):
    """Active reset: measure, and if excited, apply an X pulse.

    The ``if`` becomes an scf.if region: compiling keeps it as a branch on
    pulse.is_set and schedules it for the longer arm.
    """
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)

    readout_envelope = square(cal["readout_duration"], 0.05)
    result = readout(readout_line, readout_envelope)

    sync(drive_line, readout_line)

    if result:
        # A pi pulse: with H = amplitude * X / 2, this amplitude (rad/ns)
        # rotates by pi in 20 ns.
        x_pulse = square(cal["x_duration"], math.pi / 20)
        drive(drive_line, x_pulse)


@pulse.kernel
def repeated_measurement(qubit, n_shots):
    """Repeated preparation and measurement in a single kernel.

    A for loop wrapping prepare-measure cycles.
    """
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)

    x_half = drag(cal["x_duration"], 0.25, cal["x_sigma"], cal["drag_beta"])

    for _shot in range(n_shots):
        drive(drive_line, x_half)
        sync(drive_line, readout_line)

        readout(readout_line, square(cal["readout_duration"], 0.05))

        sync(drive_line, readout_line)
        wait(drive_line, cal["shot_gap"])


@pulse.kernel
def ramsey_with_readout(qubit, tau):
    """Ramsey experiment: pi/2 - tau - pi/2 - measure.

    Demonstrates the full single-qubit characterization loop with readout.
    """
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)

    half_pi = drag(cal["x_duration"], 0.25, cal["x_sigma"], cal["drag_beta"])

    drive(drive_line, half_pi)

    wait(drive_line, tau)

    drive(drive_line, half_pi)

    sync(drive_line, readout_line)
    readout(readout_line, square(cal["readout_duration"], 0.05))


if __name__ == "__main__":
    dev = pulse.device("transmon_krinner_17q")
    print("=== Measure single qubit ===")
    compiled_kernel = pulse.compile(measure_single_qubit, device=dev).lower(dev.qudit_handles[0])
    print(compiled_kernel.mlir())

    print("\n=== Active reset (measurement-conditioned branch) ===")
    compiled_kernel = pulse.compile(active_reset, device=dev).lower(dev.qudit_handles[0])
    print(compiled_kernel.mlir())

    print("\n=== Simulating it (CPU reference) ===")
    from cudaq_pulse.targets import Qubit, TargetDescription

    target = TargetDescription(
        name="reset-demo",
        qubits={
            0: Qubit(index=0, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=0.0, t2_star_us=0.0)
        },
    )
    # Start in (|0> + |1>) / sqrt(2): each outcome is equally likely, and
    # either way the qubit ends in |0>.
    reset_demo = pulse.device({"target": target})
    lowered = pulse.compile(active_reset, device=reset_demo).lower(reset_demo.qudit_handles[0])
    result = pulse.simulate(lowered, initial_state=[2**-0.5, 2**-0.5, 0.0], shots=1000, seed=7)
    for branch in result.branches:
        print(f"  measured {branch.levels}: p = {branch.probability:.3f}")
    print(f"  final |0> population: {result.level_population(0):.4f}")

    print("\n=== Repeated measurement (5 shots) ===")
    compiled_kernel = pulse.compile(repeated_measurement, device=dev).lower(dev.qudit_handles[0], 5)
    print(compiled_kernel.mlir())

    print("\n=== Ramsey with readout (tau=200) ===")
    tau = 100 * pulse.ns
    compiled_kernel = pulse.compile(ramsey_with_readout, device=dev).lower(
        dev.qudit_handles[0], tau
    )
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
