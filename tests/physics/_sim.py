# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Devices and runs in the reference conventions: amplitudes are Rabi rates."""

import cudaq_pulse as pulse
from cudaq_pulse.targets import Coupling, Qubit, Target

PASSES = ("verify", "canonicalize", "virtual_z", "fusion")
CLOCK_GHZ = 2.0


def cycles(ns: float) -> int:
    """A duration in clock cycles."""
    value = ns * CLOCK_GHZ
    assert abs(value - round(value)) < 1e-9, ns
    return int(round(value))


def device(freqs_hz, alphas_hz, couplings=(), t1_us=0.0, t2_us=0.0, **qubit):
    target = Target(
        name="reference",
        qubits={
            i: Qubit(
                i,
                f,
                a,
                t1_us,
                t2_us,
                drive_params={"amplitude_scale_rad_per_ns": 1.0},
                **qubit,
            )
            for i, (f, a) in enumerate(zip(freqs_hz, alphas_hz))
        },
        couplings=[Coupling(i, j, g) for i, j, g in couplings],
    )
    return pulse.device({"target": target})


def run(dev, kernel, *args, n=None, backend="cpu", **simulate):
    """Simulate ``kernel`` on the first *n* qudits of *dev*, then *args*."""
    n = len(dev.qudit_handles) if n is None else n
    lowered = pulse.compile(kernel, passes=PASSES).lower(
        *[dev.qudit_handles[i] for i in range(n)], *args
    )
    return pulse.simulate(lowered, backend=backend, **simulate)


def basis(levels, *digits):
    """The state vector |digits>, qudit 0 the least significant digit."""
    import numpy as np

    dims = [levels] * len(digits) if isinstance(levels, int) else list(levels)
    index, stride = 0, 1
    for d, digit in zip(dims, digits):
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
            for d, digit in zip(levels, digits):
                index += digit * stride
                stride *= d
            total += rho[index, index].real
    del n
    return total
