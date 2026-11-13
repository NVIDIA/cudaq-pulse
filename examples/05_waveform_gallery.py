# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Waveform gallery: every envelope type and waveform algebra.

Demonstrates all 8 built-in waveform constructors and the 5 algebraic
combinators (add, sub, mul, scale, neg) available inside a pulse kernel.
"""

import math

import cudaq_pulse as pulse

cal = {
    "duration": 20 * pulse.ns,
    "sigma": 5 * pulse.ns,
    "drag_beta": 0.25 * pulse.ns,
    "square_duration": 50 * pulse.ns,
    "cosine_duration": 30 * pulse.ns,
    "tanh_duration": 40 * pulse.ns,
    "tanh_sigma": 2.5 * pulse.ns,
    "flat_top_duration": 100 * pulse.ns,
    "flat_top_width": 80 * pulse.ns,
}

# ── 1. Every waveform constructor ────────────────────────────────────────────


def sine_envelope(t):
    """Half a sine period over 40 clock cycles (20 ns at 2 GHz), for custom():
    *t* is in clock cycles from the start of the pulse."""
    return 0.3 * math.sin(math.pi * t / 40)


@pulse.kernel
def waveform_showcase(qubit):
    """Drive a qubit with each built-in waveform type sequentially."""
    drive_line = get_drive_line(qubit)
    duration, sigma = cal["duration"], cal["sigma"]

    # Gaussian — the workhorse of single-qubit gates
    drive(drive_line, gaussian(duration, 0.3, sigma))

    # Square — constant amplitude, simplest envelope
    drive(drive_line, square(cal["square_duration"], 0.5))

    # DRAG — derivative removal by adiabatic gate, reduces leakage
    drive(drive_line, drag(duration, 0.25, sigma, cal["drag_beta"]))

    # Cosine — smooth rise/fall for flux pulses
    drive(drive_line, cosine(cal["cosine_duration"], 0.4))

    # Tanh ramp — sigmoidal edges for adiabatic state transfer
    drive(drive_line, tanh_ramp(cal["tanh_duration"], 0.35, cal["tanh_sigma"]))

    # Gaussian square — flat top with Gaussian rise/fall, used in echoed-CR
    flat_top = gaussian_square(cal["flat_top_duration"], 0.1, sigma, cal["flat_top_width"])
    drive(drive_line, flat_top)

    # Custom — a Python function of time, sampled once per clock cycle
    drive(drive_line, custom(duration, sine_envelope))

    # Custom samples — an arbitrary real envelope from pre-computed data
    drive(drive_line, custom_samples([0.1, 0.3, 0.5, 0.3, 0.1]))


# ── 2. Waveform algebra ──────────────────────────────────────────────────────


@pulse.kernel
def waveform_algebra(qubit):
    """Combine waveforms using the built-in algebraic operations."""
    drive_line = get_drive_line(qubit)

    gaussian_envelope = gaussian(cal["duration"], 0.3, cal["sigma"])
    square_envelope = square(cal["duration"], 0.1)

    # Addition — superpose two envelopes (e.g. DRAG = Gaussian + derivative)
    combined = wf_add(gaussian_envelope, square_envelope)
    drive(drive_line, combined)

    # Subtraction — difference of envelopes
    diff = wf_sub(gaussian_envelope, square_envelope)
    drive(drive_line, diff)

    # Multiplication — element-wise product (amplitude modulation)
    modulated = wf_mul(gaussian_envelope, square_envelope)
    drive(drive_line, modulated)

    # Scale — multiply envelope by a scalar
    boosted = wf_scale(gaussian_envelope, 2.0)
    drive(drive_line, boosted)

    # Negation — flip sign (180° phase flip in IQ)
    flipped = wf_neg(gaussian_envelope)
    drive(drive_line, flipped)


# ── 3. Composing complex envelopes ───────────────────────────────────────────


derivative_duration = 20 * pulse.ns
derivative_sigma = 5 * pulse.ns
derivative_amplitude = 0.3
clock_ghz = 2.0  # the transmon device's program clock; one sample per cycle


def gaussian_derivative_samples(duration, amplitude, sigma):
    """d/dt of a Gaussian centered in *duration*, one sample per clock cycle."""
    sample_count = round(duration / pulse.ns * clock_ghz)
    sigma = sigma / pulse.ns * clock_ghz
    center = (sample_count - 1) / 2
    return [
        -amplitude * (t - center) / sigma**2 * math.exp(-((t - center) ** 2) / (2 * sigma**2))
        for t in range(sample_count)
    ]


derivative_samples = gaussian_derivative_samples(
    derivative_duration, derivative_amplitude, derivative_sigma
)


@pulse.kernel
def derivative_pulse(qubit, beta: float):
    """A Gaussian plus beta times its time derivative, built by hand.

    The derivative is sampled in Python and enters the kernel through
    ``custom_samples``; waveform algebra adds it to the built-in Gaussian.
    The sampled correction is in phase with the Gaussian; ``drag()`` puts
    the same correction on the quadrature, as the DRAG scheme needs.
    """
    drive_line = get_drive_line(qubit)

    base = gaussian(derivative_duration, derivative_amplitude, derivative_sigma)
    correction = wf_scale(custom_samples(derivative_samples), beta)
    final_envelope = wf_add(base, correction)

    drive(drive_line, final_envelope)


if __name__ == "__main__":
    dev = pulse.device("transmon_17q")
    print("=== Waveform Showcase ===")
    compiled_kernel = pulse.compile(waveform_showcase, device=dev).lower(dev.qudits[0])
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Waveform Algebra ===")
    compiled_kernel = pulse.compile(waveform_algebra, device=dev).lower(dev.qudits[0])
    print(compiled_kernel.mlir())

    print("\n=== Derivative Pulse ===")
    compiled_kernel = pulse.compile(derivative_pulse, device=dev).lower(dev.qudits[0], 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
