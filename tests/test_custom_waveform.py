# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""custom(duration, f) samples f while tracing, once per clock cycle."""

import math
import re

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import CompilationError

DURATION, AMPLITUDE, SIGMA = 64, 0.05, 16.0


def gaussian_envelope(t):
    return AMPLITUDE * math.exp(-((t - DURATION / 2) ** 2) / (2 * SIGMA**2))


def cosine_envelope(t):
    return AMPLITUDE * 0.5 * (1 - math.cos(2 * math.pi * t / DURATION))


@pulse.kernel
def with_custom(q):
    line = get_drive_line(q)
    drive(line, custom(DURATION, gaussian_envelope))


@pulse.kernel
def with_gaussian(q):
    line = get_drive_line(q)
    drive(line, gaussian(DURATION, AMPLITUDE, SIGMA))


@pulse.kernel
def with_custom_cosine(q):
    line = get_drive_line(q)
    drive(line, custom(DURATION, cosine_envelope))


@pulse.kernel
def with_cosine(q):
    line = get_drive_line(q)
    drive(line, cosine(DURATION, AMPLITUDE))


@pytest.fixture
def qudit():
    return pulse.quantum_system("transmon_surface17").qudits[0]


def _ops(mlir):
    return re.findall(r"\b(pulse\.[a-z_]+)\b(?! =)", mlir.split("func.func", 1)[1])


def _samples(mlir):
    match = re.search(r"pulse\.custom_samples\b.*?\[([^\]]*)\]", mlir)
    assert match, mlir
    return [float(v) for v in match.group(1).split(",")]


def test_the_envelope_is_sampled_at_the_middle_of_each_cycle(qudit):
    mlir = pulse.compile(with_custom, passes=()).lower(qudit).mlir()
    samples = _samples(mlir)
    assert len(samples) == DURATION
    assert samples == pytest.approx([gaussian_envelope(k + 0.5) for k in range(DURATION)])
    assert "pulse.custom " not in mlir and "@gaussian_envelope" not in mlir


def test_it_compiles_to_as_many_ops_as_a_built_in_envelope(qudit):
    custom_ops = _ops(with_custom.lower(qudit).mlir())
    builtin_ops = _ops(with_gaussian.lower(qudit).mlir())
    assert custom_ops.count("pulse.custom_samples") == 1
    assert len(custom_ops) == len(builtin_ops)
    assert [o for o in custom_ops if o != "pulse.custom_samples"] == [
        o for o in builtin_ops if o != "pulse.gaussian"
    ]
    assert with_custom.lower(qudit).schedule == with_gaussian.lower(qudit).schedule


@pytest.mark.parametrize(
    "sampled, builtin", [(with_custom, with_gaussian), (with_custom_cosine, with_cosine)]
)
def test_it_simulates_like_the_equivalent_built_in_envelope(qudit, sampled, builtin):
    rho_sampled = pulse.simulate(sampled.lower(qudit), levels=3).density_matrix
    rho_builtin = pulse.simulate(builtin.lower(qudit), levels=3).density_matrix
    assert rho_builtin[1, 1].real > 0.1
    np.testing.assert_allclose(rho_sampled, rho_builtin, atol=1e-3)


def test_a_parameter_scales_a_sampled_envelope_without_recompiling(qudit):
    @pulse.kernel
    def scaled(q, amplitude: float):
        line = get_drive_line(q)
        drive(line, wf_scale(custom(DURATION, gaussian_envelope), amplitude))

    compiled = pulse.compile(scaled)
    small = pulse.simulate(compiled.lower(qudit, 0.5)).density_matrix[1, 1].real
    large = pulse.simulate(compiled.lower(qudit, 1.0)).density_matrix[1, 1].real
    assert compiled.cache_info().compiled == 1
    assert 0 < small < large


def test_an_int_argument_sets_the_sampled_duration(qudit):
    @pulse.kernel
    def sized(q, n: int):
        line = get_drive_line(q)
        drive(line, custom(n, cosine_envelope))

    assert len(_samples(sized.lower(qudit, 12).mlir())) == 12
    assert sized.lower(qudit, 12).schedule[0].duration_cycles == 12


def test_the_longest_envelope_is_252_cycles(qudit):
    @pulse.kernel
    def sized(q, n: int):
        line = get_drive_line(q)
        drive(line, custom(n, cosine_envelope))

    assert len(_samples(sized.lower(qudit, 252).mlir())) == 252
    with pytest.raises(CompilationError, match="253 clock cycles is over the limit of 252"):
        sized.lower(qudit, 253)
    with pytest.raises(CompilationError, match="at most 252 samples"):
        pulse.compile(samples_253, passes=()).lower(qudit)


@pulse.kernel
def samples_253(q):
    line = get_drive_line(q)
    drive(line, custom_samples([0.01] * 253))


def _complex_envelope(t):
    return 0.01j


def _symbol_envelope(t):
    return "high"


def _infinite_envelope(t):
    return math.inf


@pytest.mark.parametrize(
    "envelope, message",
    [
        ("calibrated_envelope", "got the name 'calibrated_envelope'; pass a callable"),
        (3.0, "argument 2 must be a callable"),
        (_complex_envelope, "complex envelopes are not supported yet"),
        (_symbol_envelope, "not a number"),
        (_infinite_envelope, "returned inf at t=0.5"),
    ],
)
def test_an_unusable_envelope_is_rejected_while_tracing(qudit, envelope, message):
    @pulse.kernel
    def bad(q):
        line = get_drive_line(q)
        drive(line, custom(8, envelope))

    with pytest.raises(CompilationError, match=re.escape(message)):
        bad.lower(qudit)


@pytest.mark.parametrize(
    "duration, message",
    [
        (0, "duration is 0 cycles; a waveform lasts at least 1 cycle"),
        (2.5, r"duration is a float \(2.5\)"),
    ],
)
def test_an_unusable_duration_is_rejected(qudit, duration, message):
    @pulse.kernel
    def bad(q):
        line = get_drive_line(q)
        drive(line, custom(duration, cosine_envelope))

    with pytest.raises(CompilationError, match=message):
        bad.lower(qudit)
