# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Simulation of drives whose waveform is built with waveform arithmetic.

On resonance, with H = e(t) X / 2, a real envelope rotates the qubit about X
by its area, so the excited population is sin^2(area / 2).
"""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError
from cudaq_pulse.targets import Qubit, TargetDescription

from tests._handles import Q
from tests.runtime._lowering import on

CLOCK_GHZ = 2.0


def _target():
    return TargetDescription(
        name="waveform-arithmetic",
        qubits={
            0: Qubit(
                index=0,
                frequency_hz=5.0e9,
                anharmonicity_hz=-200.0e6,
                t1_us=0.0,
                t2_star_us=0.0,
                drive_params={"amplitude_scale_rad_per_ns": 1.0},
            )
        },
    )


def _simulate(kernel):
    result = pulse.simulate(on(_target(), kernel, Q[0]))
    return np.asarray(result.density_matrix)


def _gaussian(t_ns, duration, amplitude, sigma):
    """The envelope of gaussian(duration, amplitude, sigma), in clock cycles."""
    center = duration / CLOCK_GHZ / 2.0
    width = sigma / CLOCK_GHZ
    return amplitude * np.exp(-0.5 * (t_ns - center) ** 2 / width**2)


def _area(envelope, duration):
    t = np.linspace(0.0, duration / CLOCK_GHZ, 400_001)
    return np.trapezoid(envelope(t), t)


def _excited(density):
    return float(np.real(density[1, 1]))


def test_sum_of_gaussians_rotates_by_the_summed_area():
    @pulse.kernel
    def multiplex(q0):
        d0 = get_drive_line(q0)
        drive(d0, wf_add(gaussian(80, 0.05, 20.0), gaussian(80, 0.04, 15.0)))

    area = _area(lambda t: _gaussian(t, 80, 0.05, 20.0) + _gaussian(t, 80, 0.04, 15.0), 80)
    assert _excited(_simulate(multiplex)) == pytest.approx(math.sin(area / 2) ** 2, abs=1e-9)


def test_scale_sub_and_neg_are_exact():
    @pulse.kernel
    def scaled(q0):
        d0 = get_drive_line(q0)
        drive(d0, wf_scale(gaussian(80, 0.03, 20.0), 2.0))

    @pulse.kernel
    def doubled(q0):
        d0 = get_drive_line(q0)
        drive(d0, gaussian(80, 0.06, 20.0))

    @pulse.kernel
    def combined(q0):
        d0 = get_drive_line(q0)
        a = gaussian(80, 0.12, 20.0)
        b = gaussian(80, 0.03, 20.0)
        drive(d0, wf_add(wf_sub(a, b), wf_neg(b)))

    reference = _simulate(doubled)
    np.testing.assert_allclose(_simulate(scaled), reference, atol=1e-12)
    np.testing.assert_allclose(_simulate(combined), reference, atol=1e-12)


def test_a_waveform_minus_itself_does_nothing():
    @pulse.kernel
    def cancel(q0):
        d0 = get_drive_line(q0)
        g = drag(80, 0.08, 20.0, 0.5)
        drive(d0, wf_sub(g, g))
        drive(d0, wf_add(g, wf_scale(g, -1.0)))

    assert _excited(_simulate(cancel)) == pytest.approx(0.0, abs=1e-14)


def test_square_plus_gaussian_rotates_by_the_summed_area():
    @pulse.kernel
    def mixed(q0):
        d0 = get_drive_line(q0)
        drive(d0, wf_add(square(80, 0.02), gaussian(80, 0.04, 20.0)))

    area = 0.02 * 80 / CLOCK_GHZ + _area(lambda t: _gaussian(t, 80, 0.04, 20.0), 80)
    assert _excited(_simulate(mixed)) == pytest.approx(math.sin(area / 2) ** 2, abs=1e-9)


def test_products_are_sampled_once_per_clock_cycle():
    def samples(n):
        t = (np.arange(n) + 0.5) / CLOCK_GHZ
        return list(0.5 * _gaussian(t, n, 0.08, 20.0))

    @pulse.kernel
    def product(q0):
        d0 = get_drive_line(q0)
        drive(d0, wf_mul(square(80, 0.5), gaussian(80, 0.08, 20.0)))

    sampled = samples(80)

    @pulse.kernel
    def by_samples(q0):
        d0 = get_drive_line(q0)
        drive(d0, custom_samples(sampled))

    density = _simulate(product)
    np.testing.assert_allclose(density, _simulate(by_samples), atol=1e-12)
    # Midpoint sampling approximates the continuous product's area closely.
    area = _area(lambda t: 0.5 * _gaussian(t, 80, 0.08, 20.0), 80)
    assert _excited(density) == pytest.approx(math.sin(area / 2) ** 2, abs=1e-4)


def test_retained_loop_of_sums_lowers_like_its_unrolling():
    @pulse.kernel
    def rolled(q0):
        d0 = get_drive_line(q0)
        for _ in range(4):
            drive(d0, wf_add(gaussian(40, 0.02, 10.0), gaussian(40, 0.03, 8.0)))
            wait(d0, 20)

    @pulse.kernel
    def unrolled(q0):
        d0 = get_drive_line(q0)
        for i in range(4):
            drive(d0, wf_add(gaussian(40, 0.02, 10.0), gaussian(40, 0.03, 8.0)))
            wait(d0, 20 + 0 * i)

    assert "scf.for" in rolled.lower(Q[0]).mlir()
    np.testing.assert_allclose(_simulate(rolled), _simulate(unrolled), atol=1e-12)


def test_mismatched_durations_are_rejected_when_tracing():
    @pulse.kernel
    def mismatched(q0):
        d0 = get_drive_line(q0)
        drive(d0, wf_mul(gaussian(40, 0.05, 10.0), square(80, 0.5)))

    with pytest.raises(CompilationError, match="durations 40 and 80"):
        mismatched(Q[0])


def test_scale_takes_the_waveform_first():
    @pulse.kernel
    def swapped(q0):
        d0 = get_drive_line(q0)
        drive(d0, wf_scale(2.0, gaussian(80, 0.03, 20.0)))

    with pytest.raises(CompilationError, match=r"wf_scale\(waveform, scalar\)"):
        swapped(Q[0])


def test_scale_arguments_are_positional_only():
    @pulse.kernel
    def by_keyword(q0):
        d0 = get_drive_line(q0)
        drive(d0, wf_scale(waveform=gaussian(80, 0.03, 20.0), scalar=2.0))

    with pytest.raises(CompilationError, match="wf_scale"):
        by_keyword(Q[0])
