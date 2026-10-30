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
from decimal import Decimal
from fractions import Fraction

import cudaq_pulse as pulse
import numpy as np
import pytest
from cudaq_pulse import CompilationError
from cudaq_pulse.controller import export

from tests._handles import Q


def _kernel(body: str, signature: str = "q"):
    """A kernel whose line 4 is *body*, compiled from the file ``<kernel>``."""
    source = (
        f"def k({signature}):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(8, 0.1))\n"
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


def string_envelope(t):
    return "0.5"


def numpy_envelope(t):
    return np.float64(0.1)


REJECTED = [
    # Waveform floats must be finite.
    ("drive(d, square(8, nan))", "square amplitude must be finite, got nan"),
    ("drive(d, square(8, [0.1, inf]))", "square amplitude's imaginary part must be finite"),
    ("drive(d, gaussian(40, inf, 10.0))", "gaussian amplitude must be finite, got inf"),
    ("drive(d, gaussian(40, -inf, 10.0))", "gaussian amplitude must be finite, got -inf"),
    ("drive(d, gaussian(40, 0.1, nan))", "gaussian sigma must be positive, got nan"),
    ("drive(d, drag(40, nan, 10.0, 0.5))", "drag amplitude must be finite, got nan"),
    ("drive(d, drag(40, 0.1, 10.0, -inf))", "drag beta must be finite, got -inf"),
    ("drive(d, cosine(40, nan))", "cosine amplitude must be finite, got nan"),
    ("drive(d, tanh_ramp(40, inf, 4.0))", "tanh_ramp amplitude must be finite, got inf"),
    ("drive(d, gaussian_square(64, nan, 4.0, 32))", "gaussian_square amplitude must be"),
    ("drive(d, custom_samples([0.1, nan]))", r"custom_samples\(\) sample 1 must be finite"),
    ("drive(d, wf_scale(square(8, 0.1), inf))", "wf_scale scale must be finite, got inf"),
    ("drive(d, wf_scale(square(8, 0.1), nan))", "wf_scale scale must be finite, got nan"),
    ("shift_phase(d.tone, nan)", "shift_phase phase must be finite, got nan"),
    ("set_phase(d.tone, inf)", "set_phase phase must be finite, got inf"),
    ("shift_frequency(d.tone, nan)", "shift_frequency frequency must be finite, got nan"),
    ("set_frequency(d.tone, -inf)", "set_frequency frequency must be finite, got -inf"),
    # A sigma is positive (durations: tests/test_units.py).
    ("drive(d, gaussian(40, 0.1, 0.0))", "gaussian sigma must be positive, got 0.0"),
    ("drive(d, drag(40, 0.1, -1.0, 0.5))", "drag sigma must be positive, got -1.0"),
    ("drive(d, tanh_ramp(40, 0.1, 0))", "tanh_ramp sigma must be positive, got 0.0"),
    ("drive(d, gaussian_square(64, 0.1, 0.0, 32))", "gaussian_square sigma must be positive"),
    # Values that are not numbers.
    ("drive(d, gaussian(40, None, 10.0))", "gaussian amplitude must be a real number, got None"),
    ("drive(d, drag(40, 0.1, 10.0, 'x'))", "drag beta must be a real number, got 'x'"),
    ("drive(d, square(8, None))", r"square amplitude must be a number or \[re, im\], got None"),
    ("shift_phase(d.tone, None)", "shift_phase phase must be a real number, got None"),
    ("set_frequency(d.tone, [5.0])", r"set_frequency frequency must be a real number, got \[5.0\]"),
    # A string is not a number, even one that float() would parse.
    ("drive(d, square(8, '0.5'))", r"square amplitude must be a number or \[re, im\], got '0.5'"),
    ("drive(d, square(8, b'0.5'))", r"square amplitude must be a number or \[re, im\], got b'0.5'"),
    ("drive(d, square(8, ['0.5', 0]))", r"square amplitude must be a number or \[re, im\]"),
    ("drive(d, gaussian(40, '0.1', 10.0))", "gaussian amplitude must be a real number, got '0.1'"),
    ("drive(d, gaussian(40, 0.1, '10'))", "gaussian sigma must be a real number, got '10'"),
    ("drive(d, drag(40, 0.1, 10.0, '0.2'))", "drag beta must be a real number, got '0.2'"),
    ("drive(d, cosine(40, '0.1'))", "cosine amplitude must be a real number, got '0.1'"),
    ("drive(d, custom_samples(['0.5', 0.1]))", r"custom_samples\(\) samples must be real numbers"),
    ("drive(d, custom_samples([None, 0.1]))", r"custom_samples\(\) samples must be real numbers"),
    ("drive(d, custom(8, string_envelope))", r"custom\(\): the envelope returned str '0.5'"),
    ("drive(d, wf_scale(square(8, 0.1), '2'))", "must be a real number, got '2'"),
    ("shift_phase(d.tone, '0.5')", "shift_phase phase must be a real number, got '0.5'"),
    ("set_phase(d.tone, '0.5')", "set_phase phase must be a real number, got '0.5'"),
    ("shift_frequency(d.tone, '1e6')", "shift_frequency frequency must be a real number"),
    ("set_frequency(d.tone, '5e9')", "set_frequency frequency must be a real number, got '5e9'"),
]


@pytest.mark.parametrize(("body", "message"), REJECTED, ids=[body for body, _ in REJECTED])
def test_a_rejected_value_names_the_line(body, message):
    kernel = _kernel(body)
    kernel.__wrapped__.__globals__.update(nan=NAN, inf=INF, string_envelope=string_envelope)
    with pytest.raises(CompilationError, match=message) as info:
        kernel.lower(Q[0])
    assert 'File "<kernel>", line 4, in k' in _note(info.value)


@pytest.mark.parametrize("passes", [None, ()])
def test_the_line_is_named_with_or_without_passes(passes):
    kernel = _kernel("drive(d, gaussian(40, 0.1, 0.0))")
    with pytest.raises(CompilationError, match="gaussian sigma must be positive") as info:
        pulse.compile(kernel, passes=passes).lower(Q[0])
    assert "line 4, in k" in _note(info.value)


@pytest.mark.parametrize(
    "body",
    [
        "wait(d, 0)",
        "drive(d, square(2**53, 0.1))",
        "drive(d, gaussian(1, 0.1, 1e-300))",
        "drive(d, square(1, -1.0))",
        "shift_phase(d.tone, -1e300)",
        "drive(d, gaussian_square(64, 0.1, 4.0, 0))",
    ],
)
def test_values_at_the_edge_of_each_rule_still_compile(body):
    assert _kernel(body).lower(Q[0]) is not None


@pytest.mark.parametrize(
    "body",
    [
        "drive(d, square(8, np.float32(0.25)))",
        "drive(d, square(8, [np.float64(0.1), 1]))",
        "drive(d, square(8, Fraction(1, 4)))",
        "drive(d, gaussian(40, Decimal('0.1'), np.int64(10)))",
        "drive(d, custom_samples([0.5 + 0j, np.float64(0.1), True]))",
        "drive(d, custom(8, numpy_envelope))",
        "shift_phase(d.tone, Fraction(1, 2))",
        "set_frequency(d.tone, np.float64(5e9))",
    ],
)
def test_a_real_number_of_any_type_still_compiles(body):
    kernel = _kernel(body)
    kernel.__wrapped__.__globals__.update(
        np=np, Decimal=Decimal, Fraction=Fraction, numpy_envelope=numpy_envelope
    )
    assert kernel.lower(Q[0]) is not None


@pytest.mark.parametrize(
    "body",
    [
        "drive(d, square(8, x))",
        "drive(d, gaussian(40, 0.1, x))",
        "drive(d, wf_scale(square(8, 0.1), x))",
        "shift_phase(d.tone, x)",
        "set_frequency(d.tone, x)",
    ],
)
@pytest.mark.parametrize("value", [NAN, INF, -INF])
def test_a_float_argument_that_is_not_finite_fails_where_the_program_is_used(body, value):
    compiled = pulse.compile(_kernel(body, "q, x: float"))
    lowered = compiled.lower(Q[0], value)
    message = rf"argument 'x' is {value!r}; a float argument must be a finite number"
    for use in (
        lowered.mlir,
        lambda: export(lowered),
        lambda: pulse.simulate(lowered),
        lambda: pulse.simulate(compiled, Q[0], value),
        lambda: compiled(Q[0], value),
    ):
        with pytest.raises(pulse.InvalidValueError, match=message):
            use()


def _bad_helper(d):
    drive(d, gaussian(40, 0.1, 0.0))


def test_a_value_rejected_inside_a_helper_names_both_lines():
    @pulse.kernel
    def calls_the_helper(q):
        d = get_drive_line(q)
        _bad_helper(d)

    with pytest.raises(CompilationError, match="gaussian sigma must be positive") as info:
        calls_the_helper.lower(Q[0])
    note = _note(info.value)
    assert f"line {_line_of(calls_the_helper.__wrapped__, '_bad_helper(d)')}" in note
    assert f"line {_line_of(_bad_helper, 'gaussian(40, 0.1, 0.0)')}, in _bad_helper" in note
    assert note.index("calls_the_helper") < note.index("_bad_helper")


def test_a_kernel_that_traces_differently_the_second_time_still_says_what_is_wrong():
    sigmas = iter([0.0, 4.0])

    def next_sigma():
        return next(sigmas)

    @pulse.kernel
    def changes_between_traces(q):
        d = get_drive_line(q)
        drive(d, gaussian(40, 0.1, next_sigma()))

    with pytest.raises(CompilationError, match="gaussian sigma must be positive") as info:
        changes_between_traces.lower(Q[0])
    assert "line" not in _note(info.value)


def test_the_native_rejection_is_a_value_error():
    from cudaq_pulse._native import PackedValueError

    assert issubclass(PackedValueError, ValueError)
    assert not issubclass(PackedValueError, CompilationError)
