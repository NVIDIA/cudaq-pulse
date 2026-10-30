# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Ctrl-C during a compile stops it, and the next compile works.

The interrupt is a real SIGINT through Python's default handler. CPython
raises it in the main thread at the next bytecode it runs, so one sent while
a native stage runs surfaces when that stage returns: here, right after the
native trace decoding, passes and scheduling, as the result is wrapped.
"""

from __future__ import annotations

import signal

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import compiler

from tests._handles import Q


@pytest.fixture(autouse=True)
def _default_sigint():
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


def _ctrl_c() -> None:
    signal.raise_signal(signal.SIGINT)


def _drives(n: int, amplitude=lambda: 0.1):
    @pulse.kernel
    def k(q):
        d = get_drive_line(q)
        for i in range(n):
            drive(d, square(8 + i % 5, amplitude()))

    return k


def _programs() -> list[str]:
    return [pulse.compile(_drives(n), unroll_threshold=0).lower(Q[0]).mlir() for n in (1, 7, 64)]


def test_an_interrupt_while_tracing_stops_the_compile():
    expected = _programs()
    interrupt = [True]

    def amplitude():
        if interrupt[0]:
            _ctrl_c()
        return 0.1

    kernel = _drives(16, amplitude)
    with pytest.raises(KeyboardInterrupt):
        kernel.lower(Q[0])
    interrupt[0] = False
    assert kernel.lower(Q[0]).mlir() == _drives(16).lower(Q[0]).mlir()
    assert _programs() == expected


def test_an_interrupt_after_the_native_stages_stops_the_compile(monkeypatch):
    expected = _programs()
    kernel = _drives(200)
    wrap = compiler.Lowered.__init__
    calls = []

    def interrupted(self, *args, **kwargs):
        calls.append(None)
        _ctrl_c()
        wrap(self, *args, **kwargs)

    monkeypatch.setattr(compiler.Lowered, "__init__", interrupted)
    with pytest.raises(KeyboardInterrupt):
        pulse.compile(kernel).lower(Q[0])
    assert calls == [None]
    monkeypatch.undo()
    # The shared MLIR context and the kernel's cache are as before.
    assert _programs() == expected
    assert kernel.lower(Q[0]).mlir() == _drives(200).lower(Q[0]).mlir()
    assert kernel.cache_info().compiled == 1
