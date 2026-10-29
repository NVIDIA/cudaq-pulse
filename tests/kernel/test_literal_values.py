# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A value no op can take is a CompilationError at the kernel line that wrote it."""

from __future__ import annotations

import inspect

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import CompilationError

from tests._handles import Q


def _kernel(body: str, signature: str = "q"):
    """A kernel whose line 4 is *body*, compiled from the file ``<kernel>``."""
    source = (
        f"def k({signature}):\n"
        "    d, t = get_drive_line(q)\n"
        "    drive(d, square(8, 0.1), t)\n"
        f"    {body}\n"
    )
    namespace: dict = {}
    exec(compile(source, "<kernel>", "exec"), namespace)
    return pulse.kernel(namespace["k"])


def _note(error: BaseException) -> str:
    return "\n".join(getattr(error, "__notes__", []))


def _line_of(fn, text: str) -> int:
    lines, first = inspect.getsourcelines(fn)
    return first + next(i for i, line in enumerate(lines) if text in line)


NAN, INF = float("nan"), float("inf")

REJECTED = [
    # Waveform floats must be finite.
    ("drive(d, square(8, nan), t)", "square amplitude must be finite, got nan"),
    ("drive(d, square(8, [0.1, inf]), t)", "square amplitude's imaginary part must be finite"),
    ("drive(d, gaussian(40, inf, 10.0), t)", "gaussian amplitude must be finite, got inf"),
    ("drive(d, gaussian(40, -inf, 10.0), t)", "gaussian amplitude must be finite, got -inf"),
    ("drive(d, gaussian(40, 0.1, nan), t)", "gaussian sigma must be positive, got nan"),
    ("drive(d, drag(40, nan, 10.0, 0.5), t)", "drag amplitude must be finite, got nan"),
    ("drive(d, drag(40, 0.1, 10.0, -inf), t)", "drag beta must be finite, got -inf"),
    ("drive(d, cosine(40, nan), t)", "cosine amplitude must be finite, got nan"),
    ("drive(d, tanh_ramp(40, inf, 4.0), t)", "tanh_ramp amplitude must be finite, got inf"),
    ("drive(d, gaussian_square(64, nan, 4.0, 32), t)", "gaussian_square amplitude must be"),
    ("drive(d, custom_samples([0.1, nan]), t)", r"custom_samples\(\) sample 1 must be finite"),
    ("shift_phase(t, nan)", "shift_phase phase must be finite, got nan"),
    ("set_phase(t, inf)", "set_phase phase must be finite, got inf"),
    ("shift_frequency(t, nan)", "shift_frequency frequency must be finite, got nan"),
    ("set_frequency(t, -inf)", "set_frequency frequency must be finite, got -inf"),
    # A sigma is positive (durations: tests/test_units.py).
    ("drive(d, gaussian(40, 0.1, 0.0), t)", "gaussian sigma must be positive, got 0.0"),
    ("drive(d, drag(40, 0.1, -1.0, 0.5), t)", "drag sigma must be positive, got -1.0"),
    ("drive(d, tanh_ramp(40, 0.1, 0), t)", "tanh_ramp sigma must be positive, got 0.0"),
    ("drive(d, gaussian_square(64, 0.1, 0.0, 32), t)", "gaussian_square sigma must be positive"),
    # Values that are not numbers.
    ("drive(d, gaussian(40, None, 10.0), t)", "gaussian amplitude must be a real number, got None"),
    ("drive(d, drag(40, 0.1, 10.0, 'x'), t)", "drag beta must be a real number, got 'x'"),
    ("drive(d, square(8, None), t)", r"square amplitude must be a number or \[re, im\], got None"),
    ("shift_phase(t, None)", "shift_phase phase must be a real number, got None"),
    ("set_frequency(t, [5.0])", r"set_frequency frequency must be a real number, got \[5.0\]"),
]


@pytest.mark.parametrize(("body", "message"), REJECTED, ids=[body for body, _ in REJECTED])
def test_a_rejected_value_names_the_line(body, message):
    kernel = _kernel(body)
    kernel.__wrapped__.__globals__.update(nan=NAN, inf=INF)
    with pytest.raises(CompilationError, match=message) as info:
        kernel.lower(Q[0])
    assert 'File "<kernel>", line 4, in k' in _note(info.value)


@pytest.mark.parametrize("passes", [None, ()])
def test_the_line_is_named_with_or_without_passes(passes):
    kernel = _kernel("drive(d, gaussian(40, 0.1, 0.0), t)")
    with pytest.raises(CompilationError, match="gaussian sigma must be positive") as info:
        pulse.compile(kernel, passes=passes).lower(Q[0])
    assert "line 4, in k" in _note(info.value)


@pytest.mark.parametrize(
    "body",
    [
        "wait(d, 0)",
        "drive(d, square(2**53, 0.1), t)",
        "drive(d, gaussian(1, 0.1, 1e-300), t)",
        "drive(d, square(1, -1.0), t)",
        "shift_phase(t, -1e300)",
        "drive(d, gaussian_square(64, 0.1, 4.0, 0), t)",
    ],
)
def test_values_at_the_edge_of_each_rule_still_compile(body):
    assert _kernel(body).lower(Q[0]) is not None


def _bad_helper(d, t):
    drive(d, gaussian(40, 0.1, 0.0), t)


def test_a_value_rejected_inside_a_helper_names_both_lines():
    @pulse.kernel
    def calls_the_helper(q):
        d, t = get_drive_line(q)
        _bad_helper(d, t)

    with pytest.raises(CompilationError, match="gaussian sigma must be positive") as info:
        calls_the_helper.lower(Q[0])
    note = _note(info.value)
    assert f"line {_line_of(calls_the_helper.__wrapped__, '_bad_helper(d, t)')}" in note
    assert f"line {_line_of(_bad_helper, 'gaussian(40, 0.1, 0.0)')}, in _bad_helper" in note
    assert note.index("calls_the_helper") < note.index("_bad_helper")


def test_a_kernel_that_traces_differently_the_second_time_still_says_what_is_wrong():
    sigmas = iter([0.0, 4.0])

    def next_sigma():
        return next(sigmas)

    @pulse.kernel
    def changes_between_traces(q):
        d, t = get_drive_line(q)
        drive(d, gaussian(40, 0.1, next_sigma()), t)

    with pytest.raises(CompilationError, match="gaussian sigma must be positive") as info:
        changes_between_traces.lower(Q[0])
    assert "line" not in _note(info.value)


def test_the_native_rejection_is_a_value_error():
    from cudaq_pulse._native import PackedValueError

    assert issubclass(PackedValueError, ValueError)
    assert not issubclass(PackedValueError, CompilationError)
