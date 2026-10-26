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
import warnings

import cudaq_pulse as pulse
import pytest

from tests._handles import Q


def _lower_three(
    kernel, *args: float, step: float = 0.01
) -> tuple[list[warnings.WarningMessage], int]:
    compiled = pulse.compile(kernel)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for i in range(3):
            compiled.lower(Q[0], *[a + step * i for a in args])
    return [w for w in caught if w.category is RuntimeWarning], compiled.cache_info().compiled


def test_a_wait_length_from_the_friction_log_warns_once():
    @pulse.kernel
    def t1_shot(q, wait_ns: float):
        d, t = get_drive_line(q)
        drive(d, square(40, 0.1), t)
        wait(d, wait_ns)

    caught, compiled = _lower_three(t1_shot, 2000.0, step=40.0)
    assert compiled == 3
    (w,) = caught
    message = str(w.message)
    assert "t1_shot(): float argument 'wait_ns'" in message
    assert "must be an integer, such as a duration or a wait length" in message
    assert "compiled again for every new value of 'wait_ns'" in message
    assert w.filename == __file__


@pytest.mark.parametrize(
    ("expression", "operation"),
    [
        ("abs(amp)", "abs()"),
        ("min(amp, 0.5)", "min()"),
        ("math.sin(amp)", "sin()"),
        ("0.1 if amp < 0.5 else 0.2", "the comparison <"),
    ],
)
def test_nonlinear_uses_name_the_operation(expression, operation):
    namespace: dict = {"math": math}
    exec(
        "def k(q, amp: float):\n"
        "    d, t = get_drive_line(q)\n"
        f"    drive(d, square(10, {expression}), t)\n",
        namespace,
    )
    caught, compiled = _lower_three(pulse.kernel(namespace["k"]), 0.1)
    assert compiled == 3
    (w,) = caught
    assert f"float argument 'amp' is used in {operation}," in str(w.message)


def test_linear_uses_stay_runtime_parameters_and_do_not_warn():
    @pulse.kernel
    def k(q, amp: float, phase: float):
        d, t = get_drive_line(q)
        shift_phase(t, -phase / 2)
        drive(d, square(10, 2 * amp - 0.1), t)

    caught, compiled = _lower_three(k, 0.1, 0.2)
    assert caught == []
    assert compiled == 1
