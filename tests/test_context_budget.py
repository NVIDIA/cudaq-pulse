# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The shared MLIR context's retirement limits and their environment variables."""

import cudaq_pulse as pulse
import pytest
from cudaq_pulse._native import _cudaq_pulse_native as native

MODULES = "CUDAQ_PULSE_CONTEXT_MAX_MODULES"
WORDS = "CUDAQ_PULSE_CONTEXT_MAX_WORDS"


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.delenv(MODULES, raising=False)
    monkeypatch.delenv(WORDS, raising=False)


@pulse.kernel
def _pulse(q0):
    d0, t0 = get_drive_line(q0)
    drive(d0, square(16, 0.1), t0)


def _compile():
    return pulse.compile(_pulse, [pulse.qudit_ref()], qubit_freq_hz={0: 5e9})


def _created():
    return native.shared_context_stats()[0]


def test_defaults():
    assert native.shared_context_budget() == (256, 1 << 20)


def test_variables_set_the_limits(monkeypatch):
    monkeypatch.setenv(MODULES, "3")
    monkeypatch.setenv(WORDS, "4096")
    assert native.shared_context_budget() == (3, 4096)


def test_a_module_limit_retires_the_context(monkeypatch):
    monkeypatch.setenv(MODULES, "2")
    _compile()
    before = _created()
    kept = [_compile() for _ in range(6)]
    assert _created() - before == 3
    # Modules built in a retired context stay usable.
    assert all("pulse.drive" in compiled.mlir for compiled in kept)


def test_a_word_limit_retires_the_context(monkeypatch):
    _compile()
    words = native.shared_context_stats()[2]
    monkeypatch.setenv(WORDS, "1")
    before = _created()
    for _ in range(3):
        _compile()
    assert words > 0
    assert _created() - before == 3


def test_one_module_per_context(monkeypatch):
    monkeypatch.setenv(MODULES, "1")
    before = _created()
    for _ in range(4):
        _compile()
    assert _created() - before == 4


@pytest.mark.parametrize("value", ["0", "-5", "12x", "1.5", " 8", "+8", "2000000000000", "lots"])
@pytest.mark.parametrize("name", [MODULES, WORDS])
def test_invalid_values_are_rejected_clearly(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    message = f"{name}={value} is not valid; it must be a whole number from 1 to"
    with pytest.raises(ValueError, match=message.replace("+", r"\+").replace(".", r"\.")):
        native.shared_context_budget()
    with pytest.raises(ValueError, match=name):
        _compile()


def test_an_empty_value_means_the_default(monkeypatch):
    monkeypatch.setenv(MODULES, "")
    assert native.shared_context_budget()[0] == 256
