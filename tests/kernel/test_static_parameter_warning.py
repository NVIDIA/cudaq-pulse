# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A float parameter that tracing makes static warns once."""

from __future__ import annotations

import math
import threading
import warnings

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.ops import square

from tests._handles import Q


def _lower_three(
    kernel, *args: float, step: float = 0.01
) -> tuple[list[warnings.WarningMessage], int]:
    compiled = pulse.compile(kernel, unroll_threshold=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(3):
            compiled.lower(Q[0], *[a + step * i for a in args])
    return _specializations(caught), compiled.cache_info().compiled


def _specializations(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [w for w in caught if issubclass(w.category, RuntimeWarning)]


def _exec_kernel(source: str, **names):
    namespace: dict = {"math": math, **names}
    exec(source, namespace)
    return pulse.kernel(namespace["k"])


def test_a_wait_length_from_the_friction_log_compiles_once_without_a_warning():
    @pulse.kernel
    def t1_shot(q, wait_ns: float):
        d = get_drive_line(q)
        drive(d, square(40, 0.1))
        wait(d, wait_ns)

    caught, compiled = _lower_three(t1_shot, 2000.0, step=40.0)
    assert caught == []
    assert compiled == 1
    assert t1_shot.runtime_parameters == ("q", "wait_ns")


def test_a_waveform_duration_warns_once():
    @pulse.kernel
    def k(q, width: float):
        d = get_drive_line(q)
        drive(d, square(width, 0.1))

    caught, compiled = _lower_three(k, 40.0, step=8.0)
    assert compiled == 3
    (w,) = caught
    message = str(w.message)
    assert "k(): float argument 'width'" in message
    assert "must be an integer, such as a waveform duration" in message
    assert "compiled again for every new value of 'width'" in message
    line = k.__wrapped__.__code__.co_firstlineno + 3
    assert f"such as a waveform duration at line {line}," in message
    assert w.filename == __file__


def test_a_wait_length_through_floor_division_still_warns():
    @pulse.kernel
    def k(q, tau: float):
        d = get_drive_line(q)
        drive(d, square(40, 0.1))
        wait(d, tau // 2)

    caught, compiled = _lower_three(k, 400.0, step=4.0)
    assert compiled == 3
    (w,) = caught
    assert "float argument 'tau'" in str(w.message)


@pytest.mark.parametrize(
    ("expression", "operation"),
    [
        ("abs(amp)", "abs()"),
        ("min(amp, 0.5)", "min()"),
        ("math.sin(amp)", "sin()"),
        ("0.1 if amp < 0.5 else 0.2", "the comparison <"),
    ],
)
def test_nonlinear_uses_name_the_operation_and_line(expression, operation):
    k = _exec_kernel(
        "def k(q, amp: float):\n"
        "    d = get_drive_line(q)\n"
        f"    drive(d, square(10, {expression}))\n"
    )
    caught, compiled = _lower_three(k, 0.1)
    assert compiled == 3
    (w,) = caught
    assert f"float argument 'amp' is used in {operation} at line 3," in str(w.message)


def test_a_structural_use_names_the_line_and_suggests_int_or_a_fixed_value():
    k = _exec_kernel(
        "def k(q, n: float):\n"
        "    d = get_drive_line(q)\n"
        "    for _ in range(int(n)):\n"
        "        drive(d, square(40, 0.1))\n"
    )
    caught, compiled = _lower_three(k, 2.0, step=1.0)
    assert compiled == 3
    (w,) = caught
    message = str(w.message)
    assert "k(): float argument 'n' is used in a range() bound at line 3," in message
    assert "annotate it as int, or give it a fixed value," in message
    assert issubclass(w.category, pulse.PulseWarning)
    assert issubclass(w.category, RuntimeWarning)


def test_the_warning_follows_filters_on_pulse_warnings_and_runtime_warnings():
    source = "def k(q, n: float):\n    d = get_drive_line(q)\n    drive(d, square(int(n), 0.1))\n"
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=pulse.PulseWarning)
        with pytest.raises(pulse.PulseWarning, match="float argument 'n'"):
            pulse.compile(_exec_kernel(source)).lower(Q[0], 8.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pulse.compile(_exec_kernel(source)).lower(Q[0], 8.0)
    assert not _specializations(caught)


def test_a_use_in_a_helper_names_the_kernel_line_and_the_helper_line():
    k = _exec_kernel(
        "def pulse_of(width):\n"
        "    return square(width, 0.1)\n"
        "def k(q, width: float):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, pulse_of(width))\n",
        square=square,
    )
    caught, _ = _lower_three(k, 40.0, step=8.0)
    (w,) = caught
    assert "argument 'width' is used in " in str(w.message)
    assert " at line 5 (in pulse_of(), line 2)," in str(w.message)


def test_the_warning_is_once_per_kernel_and_parameter_not_per_call():
    k = _exec_kernel(
        "def k(q, width: float, amp: float):\n"
        "    d = get_drive_line(q)\n"
        "    drive(d, square(width, 0.1))\n"
        "    drive(d, square(40, abs(amp)))\n"
    )
    first, second = pulse.compile(k), pulse.compile(k)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for compiled in (first, second, first):
            for width in (40.0, 48.0, 40.0):
                compiled.lower(Q[0], width, 0.2)
                compiled.lower(Q[0], width, 0.2)
                k.lower(Q[0], width, 0.3)
    caught = _specializations(caught)
    assert sorted(_parameter(w) for w in caught) == ["'amp'", "'width'"]
    assert first.cache_info().compiled == 2


def _parameter(w: warnings.WarningMessage) -> str:
    return str(w.message).split("float argument ", 1)[1].split(" ", 1)[0]


def test_compiles_tracing_at_once_warn_once():
    barrier = threading.Barrier(2, timeout=30)

    def meet():
        barrier.wait()

    @pulse.kernel
    def k(q, width: float):
        d = get_drive_line(q)
        meet()
        drive(d, square(width, 0.1))

    compiled = [pulse.compile(k), pulse.compile(k)]
    errors: list[BaseException] = []

    def lower(c):
        try:
            c.lower(Q[0], 40.0)
        except BaseException as e:
            errors.append(e)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        threads = [threading.Thread(target=lower, args=(c,)) for c in compiled]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
    assert errors == []
    assert len(_specializations(caught)) == 1


def test_linear_uses_stay_runtime_parameters_and_do_not_warn():
    @pulse.kernel
    def k(q, amp: float, phase: float):
        d = get_drive_line(q)
        shift_phase(d.tone, -phase / 2)
        drive(d, square(10, 2 * amp - 0.1))

    caught, compiled = _lower_three(k, 0.1, 0.2)
    assert caught == []
    assert compiled == 1
