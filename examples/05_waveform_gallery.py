# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Waveform gallery: every envelope type and waveform algebra.

Demonstrates all 8 built-in waveform constructors and the 5 algebraic
combinators (add, sub, mul, scale, neg) available inside a pulse kernel.
"""

import math

import cudaq_pulse as pulse

# ── 1. Every waveform constructor ────────────────────────────────────────────


def sine_envelope(t):
    """Half a sine period over 40 clock cycles, for custom()."""
    return 0.3 * math.sin(math.pi * t / 40)


@pulse.kernel
def waveform_showcase(qubit):
    """Drive a qubit with each built-in waveform type sequentially."""
    drive_line, tone = get_drive_line(qubit)

    # Gaussian — the workhorse of single-qubit gates
    drive(drive_line, gaussian(40, 0.3, 10.0), tone)

    # Square — constant amplitude, simplest envelope
    drive(drive_line, square(100, 0.5), tone)

    # DRAG — derivative removal by adiabatic gate, reduces leakage
    drive(drive_line, drag(40, 0.25, 10.0, 0.5), tone)

    # Cosine — smooth rise/fall for flux pulses
    drive(drive_line, cosine(60, 0.4), tone)

    # Tanh ramp — sigmoidal edges for adiabatic state transfer
    drive(drive_line, tanh_ramp(80, 0.35, 5.0), tone)

    # Gaussian square — flat top with Gaussian rise/fall, used in echoed-CR
    drive(drive_line, gaussian_square(200, 0.1, 10.0, 160), tone)

    # Custom — a Python function of time, sampled once per clock cycle
    drive(drive_line, custom(40, sine_envelope), tone)

    # Custom samples — an arbitrary real envelope from pre-computed data
    drive(drive_line, custom_samples([0.1, 0.3, 0.5, 0.3, 0.1]), tone)


# ── 2. Waveform algebra ──────────────────────────────────────────────────────


@pulse.kernel
def waveform_algebra(qubit):
    """Combine waveforms using the built-in algebraic operations."""
    drive_line, tone = get_drive_line(qubit)

    gaussian_envelope = gaussian(40, 0.3, 10.0)
    square_envelope = square(40, 0.1)

    # Addition — superpose two envelopes (e.g. DRAG = Gaussian + derivative)
    combined = wf_add(gaussian_envelope, square_envelope)
    drive(drive_line, combined, tone)

    # Subtraction — difference of envelopes
    diff = wf_sub(gaussian_envelope, square_envelope)
    drive(drive_line, diff, tone)

    # Multiplication — element-wise product (amplitude modulation)
    modulated = wf_mul(gaussian_envelope, square_envelope)
    drive(drive_line, modulated, tone)

    # Scale — multiply envelope by a scalar
    boosted = wf_scale(gaussian_envelope, 2.0)
    drive(drive_line, boosted, tone)

    # Negation — flip sign (180° phase flip in IQ)
    flipped = wf_neg(gaussian_envelope)
    drive(drive_line, flipped, tone)


# ── 3. Composing complex envelopes ───────────────────────────────────────────


DERIVATIVE_DURATION = 40
DERIVATIVE_SIGMA = 10.0
DERIVATIVE_AMPLITUDE = 0.3


def gaussian_derivative_samples(duration, amplitude, sigma):
    """d/dt of a Gaussian centered in *duration*, one sample per clock cycle."""
    center = (duration - 1) / 2
    return [
        -amplitude * (t - center) / sigma**2 * math.exp(-((t - center) ** 2) / (2 * sigma**2))
        for t in range(duration)
    ]


DERIVATIVE = gaussian_derivative_samples(
    DERIVATIVE_DURATION, DERIVATIVE_AMPLITUDE, DERIVATIVE_SIGMA
)


@pulse.kernel
def derivative_pulse(qubit, beta: float):
    """A Gaussian plus beta times its time derivative, built by hand.

    The derivative is sampled in Python and enters the kernel through
    ``custom_samples``; waveform algebra adds it to the built-in Gaussian.
    The sampled correction is in phase with the Gaussian; ``drag()`` puts
    the same correction on the quadrature, as the DRAG scheme needs.
    """
    drive_line, tone = get_drive_line(qubit)

    base = gaussian(DERIVATIVE_DURATION, DERIVATIVE_AMPLITUDE, DERIVATIVE_SIGMA)
    correction = wf_scale(custom_samples(DERIVATIVE), beta)
    final_envelope = wf_add(base, correction)

    drive(drive_line, final_envelope, tone)


if __name__ == "__main__":
    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    print("=== Waveform Showcase ===")
    compiled_kernel = pulse.compile(waveform_showcase, device=dev).lower(dev.qudit_handles[0])
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")

    print("\n=== Waveform Algebra ===")
    compiled_kernel = pulse.compile(waveform_algebra, device=dev).lower(dev.qudit_handles[0])
    print(compiled_kernel.mlir())

    print("\n=== Derivative Pulse ===")
    compiled_kernel = pulse.compile(derivative_pulse, device=dev).lower(dev.qudit_handles[0], 0.5)
    print(compiled_kernel.mlir())
    print(f"  Compile metrics: {compiled_kernel.metrics}")
