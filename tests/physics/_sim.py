# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Devices and runs in the reference conventions: amplitudes are Rabi rates."""

import cudaq_pulse as pulse
from cudaq_pulse.targets import Coupling, DeviceDescription, MicrowaveDrive, Qubit

PASSES = (
    "verify",
    "fold_constants",
    "remove_unused",
    "merge_waits",
    "simplify_control_flow",
    "fold_phases",
    "fuse_pulses",
)
CLOCK_GHZ = 2.0


def cycles(ns: float) -> int:
    """A duration in clock cycles."""
    value = ns * CLOCK_GHZ
    assert abs(value - round(value)) < 1e-9, ns
    return int(round(value))


def device(freqs_hz, alphas_hz, couplings=(), t1_us=0.0, t2_us=0.0, **qubit):
    target = DeviceDescription(
        name="reference",
        qubits={
            i: Qubit(
                i,
                f,
                a,
                t1_us,
                t2_us,
                drive=MicrowaveDrive(amplitude_scale_rad_per_ns=1.0),
                **qubit,
            )
            for i, (f, a) in enumerate(zip(freqs_hz, alphas_hz, strict=False))
        },
        couplings=[Coupling((i, j), g) for i, j, g in couplings],
    )
    return pulse.device(target)


def run(dev, kernel, *args, n=None, backend="cpu", levels=2, **simulate):
    """Simulate ``kernel`` on the first *n* qudits of *dev*, then *args*,
    with two levels per qudit unless *levels* says otherwise: the checks
    compare with two-level formulas."""
    n = len(dev.qudits) if n is None else n
    lowered = pulse.compile(kernel, passes=PASSES).lower(*[dev.qudits[i] for i in range(n)], *args)
    return pulse.simulate(lowered, backend=backend, levels=levels, **simulate)


def basis(levels, *digits):
    """The state vector |digits>, qudit 0 the least significant digit."""
    import numpy as np

    dims = [levels] * len(digits) if isinstance(levels, int) else list(levels)
    index, stride = 0, 1
    for d, digit in zip(dims, digits, strict=False):
        index += digit * stride
        stride *= d
    psi = np.zeros(stride, dtype=complex)
    psi[index] = 1.0
    return psi


def population(rho, levels, **fixed):
    """The total population of the basis states whose digits match *fixed*
    (keys q0, q1, ...)."""
    import itertools

    n = len(levels)
    total = 0.0
    for digits in itertools.product(*[range(d) for d in levels]):
        if all(digits[int(k[1:])] == v for k, v in fixed.items()):
            index, stride = 0, 1
            for d, digit in zip(levels, digits, strict=False):
                index += digit * stride
                stride *= d
            total += rho[index, index].real
    del n
    return total
