# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""controller.resample_samples: samples made at one rate, made again at
another for custom_samples, explicitly and lasting as long."""

from __future__ import annotations

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import Controller, ControlSystem
from cudaq_pulse._errors import InvalidTypeError, InvalidValueError
from cudaq_pulse.controller import resample_samples

from tests._handles import Q

HALF_CLOCK = ControlSystem(controllers=[Controller(clock_ghz=1.0)])


def ramp(t):
    return 0.1 + 0.01 * t


# custom(8, ramp)'s samples at the 2 GHz program clock, at the middle of each cycle.
AT_CLOCK = [ramp(k + 0.5) for k in range(8)]


@pulse.kernel
def traced(q):
    d = get_drive_line(q)  # noqa: F821
    drive(d, custom(8, ramp))  # noqa: F821


@pulse.kernel
def resampled(q):
    d = get_drive_line(q)  # noqa: F821
    drive(d, custom_samples(resample_samples(AT_CLOCK, 2.0, 1.0), sample_rate_ghz=1.0))  # noqa: F821


@pulse.kernel
def as_made(q):
    d = get_drive_line(q)  # noqa: F821
    drive(d, custom_samples(AT_CLOCK))  # noqa: F821


def _played(lowered):
    return next(e for e in lowered.export_to_controller()["events"] if e["kind"] == "drive")


def test_samples_made_at_the_clock_play_on_a_slower_line_once_resampled():
    with pytest.raises(pulse.CompilationError, match="would change how long they last"):
        pulse.compile(as_made, control_system=HALF_CLOCK).lower(Q[0])
    lowered = pulse.compile(resampled, control_system=HALF_CLOCK).lower(Q[0])
    played = _played(lowered)
    assert played["duration_cycles"] == 4
    assert _played(pulse.compile(as_made).lower(Q[0]))["end_cycles"] == 8
    # A linear envelope interpolates exactly: the samples custom() takes at the line's rate.
    expected = _played(pulse.compile(traced, control_system=HALF_CLOCK).lower(Q[0]))
    assert played["waveform"]["samples"] == pytest.approx(expected["waveform"]["samples"])
    assert played["waveform"]["samples"] == pytest.approx([ramp(t) for t in (1, 3, 5, 7)])


def test_holding_at_a_whole_multiple_plays_the_same_waveform():
    half = [0.2, 0.4, 0.6, 0.8]
    assert resample_samples(half, 1.0, 2.0, method="hold") == [
        0.2,
        0.2,
        0.4,
        0.4,
        0.6,
        0.6,
        0.8,
        0.8,
    ]
    assert resample_samples(half, 1.0, 3.0, method="hold") == [x for x in half for _ in range(3)]

    @pulse.kernel
    def slow(q):
        d = get_drive_line(q)  # noqa: F821
        drive(d, custom_samples(half, sample_rate_ghz=1.0))  # noqa: F821

    @pulse.kernel
    def held(q):
        d = get_drive_line(q)  # noqa: F821
        drive(d, custom_samples(resample_samples(half, 1.0, 2.0, method="hold")))  # noqa: F821

    a = pulse.simulate(pulse.compile(slow, control_system=HALF_CLOCK).lower(Q[0])).density_matrix
    b = pulse.simulate(pulse.compile(held).lower(Q[0])).density_matrix
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_the_same_rate_gives_the_samples_back():
    for method in ("linear", "hold"):
        assert resample_samples(AT_CLOCK, 2.0, 2.0, method=method) == AT_CLOCK


def test_linear_holds_the_first_and_last_values_beyond_their_middles():
    # Two samples made four times: the new middles at 1/8, 3/8, ... of a sample.
    assert resample_samples([0.0, 1.0], 1.0, 4.0) == pytest.approx(
        [0.0, 0.0, 0.125, 0.375, 0.625, 0.875, 1.0, 1.0]
    )
    assert resample_samples([0.0, 1.0, 0.0, 1.0], 2.0, 1.0) == pytest.approx([0.5, 0.5])
    assert resample_samples([0.0, 1.0, 0.0, 1.0], 2.0, 1.0, method="hold") == [1.0, 1.0]
    # Two thirds of the rate, from a NumPy array.
    assert resample_samples(np.arange(3.0), 3.0, 2.0) == pytest.approx([0.25, 1.75])


def test_the_samples_must_last_a_whole_number_of_new_samples():
    with pytest.raises(InvalidValueError, match=r"3 samples at 2 GHz last 1.5 samples at 1 GHz"):
        resample_samples([0.1, 0.2, 0.3], 2.0, 1.0)


@pytest.mark.parametrize(
    ("args", "kwargs", "error", "match"),
    [
        (([0.1], 0.0, 1.0), {}, InvalidValueError, "from_ghz must be a positive number"),
        (([0.1], 1.0, float("nan")), {}, InvalidValueError, "to_ghz must be a positive number"),
        (([0.1], 1.0, True), {}, InvalidValueError, "to_ghz must be a positive number"),
        (([0.1], 1.0, 2.0**0.5), {}, InvalidValueError, "no whole ratio"),
        (([0.1], 1.0, 2.0), {"method": "cubic"}, InvalidValueError, "'linear' or 'hold'"),
        (([], 1.0, 2.0), {}, InvalidValueError, "at least one finite"),
        (([0.1, float("inf")], 1.0, 2.0), {}, InvalidValueError, "at least one finite"),
        ((["a"], 1.0, 2.0), {}, InvalidTypeError, "real numbers"),
        (([0.1j], 1.0, 2.0), {}, InvalidTypeError, "real numbers"),
    ],
)
def test_invalid_input_is_refused(args, kwargs, error, match):
    with pytest.raises(error, match=match):
        resample_samples(*args, **kwargs)
