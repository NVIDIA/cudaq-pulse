# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the documentation states about gaussian(), checked against its samples.

The docstring of ``gaussian()`` and the operations guide's table of waveforms
give the formula ``amplitude * exp(-(t - duration / 2)**2 / (2 * sigma**2))``
and say that with ``duration = 4 * sigma`` it starts and ends at about 14% of
the amplitude."""

from __future__ import annotations

import math

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.controller import sample_waveform


@pulse.kernel
def one_gaussian(qudit, duration, amplitude, sigma):
    drive(get_drive_line(qudit), gaussian(duration, amplitude, sigma))


def _samples(duration, amplitude, sigma):
    """The samples of the drive that one_gaussian plays, one per clock cycle."""
    system = pulse.quantum_system("transmon_surface17")
    lowered = pulse.compile(one_gaussian, quantum_system=system).lower(
        system.qudits[0], duration, amplitude, sigma
    )
    events = lowered.export_to_controller()["events"]
    drives = [event for event in events if event["kind"] == "drive"]
    assert len(drives) == 1
    return sample_waveform(drives[0]["waveform"], drives[0]["duration_cycles"])


@pytest.mark.level0
def test_a_gaussian_four_sigma_long_plays_the_documented_formula():
    samples = _samples(40, 0.2, 10)
    assert len(samples) == 40
    # Sample k is the envelope at t = k cycles from the start; duration / 2
    # is 20 and 2 * sigma**2 is 200.
    for k, sample in enumerate(samples):
        assert sample == pytest.approx(0.2 * math.exp(-((k - 20) ** 2) / 200)), f"sample {k}"
    # The first sample is exp(-2) of the amplitude, about 13.5%. The envelope
    # is back at that value at t = 40, one cycle after the last sample.
    assert 0.13 < samples[0].real / 0.2 < 0.14
    assert samples[20] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("duration", "amplitude", "sigma"),
    [
        # The peak, at 20.5 cycles, falls between two samples.
        (41, 0.2, 10.25),
        (60, 0.05, 10),
        (40, 0.2, 7.5),
        (40, -0.2, 10),
        # The shortest waveform: its one sample is the start of the envelope.
        (1, 0.2, 0.25),
    ],
    ids=["odd_duration", "six_sigma", "fractional_sigma", "negative_amplitude", "one_cycle"],
)
def test_the_formula_holds_for_other_durations_amplitudes_and_sigmas(duration, amplitude, sigma):
    samples = _samples(duration, amplitude, sigma)
    assert len(samples) == duration
    for k, sample in enumerate(samples):
        expected = amplitude * math.exp(-((k - duration / 2) ** 2) / (2 * sigma**2))
        assert sample == pytest.approx(expected), f"sample {k}"
