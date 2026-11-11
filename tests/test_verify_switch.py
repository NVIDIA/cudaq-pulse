# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDAQ_PULSE_VERIFY: the verifications that only catch a bug of the compiler
itself, on in the test suite and off in a user's process."""

import os
import subprocess
import sys

import pytest
from cudaq_pulse._native import _cudaq_pulse_native as native

VARIABLE = "CUDAQ_PULSE_VERIFY"

_CHILD = """
import cudaq_pulse as pulse
from cudaq_pulse._native import _cudaq_pulse_native as native
from tests._handles import Q

@pulse.kernel
def k(q):
    d = get_drive_line(q)
    drive(d, square(16, 0.1))

try:
    text = pulse.compile(k).lower(Q[0]).mlir()
except ValueError as error:
    print("error", error)
else:
    print(native.verifies_everything(), "pulse.drive" in text)
"""


def _child(value: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != VARIABLE}
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    if value is not None:
        env[VARIABLE] = value
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    run = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert run.returncode == 0, run.stderr[-2000:]
    return run.stdout.strip()


@pytest.mark.skipif(os.environ.get(VARIABLE) == "0", reason="run with CUDAQ_PULSE_VERIFY=0")
def test_the_suite_verifies_everything():
    assert native.verifies_everything()


@pytest.mark.parametrize(("value", "on"), [(None, False), ("", False), ("0", False), ("1", True)])
def test_the_variable_turns_them_on(value, on):
    assert _child(value) == f"{on} True"


def test_another_value_raises():
    assert _child("yes") == "error CUDAQ_PULSE_VERIFY must be 0 or 1, not 'yes'"
