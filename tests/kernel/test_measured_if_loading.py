# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tracing of measurement-conditioned ifs (kernel/_measured_if.py) is
imported at a trace's first such if, and only then: a compile without one
does not load it, and two threads started together at their first such if
both trace it as one thread alone does.

Each check runs in a fresh interpreter, where nothing has loaded it yet.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

_KERNELS = """
import cudaq_pulse as pulse
from tests._handles import Q


@pulse.kernel
def plain(a):
    drive(get_drive_line(a), square(40, 0.1))
    return readout(get_readout_line(a), square(40, 0.05))


@pulse.kernel
def measured(a, b):
    da = get_drive_line(a)
    db = get_drive_line(b)
    m = readout(get_readout_line(a), square(40, 0.05))
    if m:
        drive(db, square(40, 0.1))
    else:
        drive(da, square(40, 0.1))
    return readout(get_readout_line(b), square(40, 0.05))
"""


def _run(body: str) -> str:
    source = _KERNELS + textwrap.dedent(body)
    done = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False, cwd=_ROOT
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def test_a_compile_without_a_measured_if_does_not_load_its_tracing():
    out = _run(
        """
        import sys

        pulse.compile(plain).lower(Q[0])
        print("cudaq_pulse.kernel._measured_if" in sys.modules)
        """
    )
    assert out.split() == ["False"]


def test_the_first_measured_if_loads_its_tracing_in_place_of_the_stubs():
    out = _run(
        """
        import sys

        from cudaq_pulse.kernel._emitter import PulseIREmitter

        stub = PulseIREmitter._runtime_branch
        first = pulse.compile(measured).lower(Q[0], Q[1]).mlir()
        module = sys.modules["cudaq_pulse.kernel._measured_if"]
        print(PulseIREmitter._runtime_branch is module._runtime_branch)
        print(PulseIREmitter._trace_arm is module._trace_arm)
        print(PulseIREmitter._runtime_branch is not stub)
        print(pulse.compile(measured).lower(Q[0], Q[1]).mlir() == first)
        """
    )
    assert out.split() == ["True"] * 4


def test_two_threads_reaching_their_first_measured_if_at_once_trace_it_alike():
    out = _run(
        """
        import sys
        import threading

        # Switch threads as often as the interpreter allows, so that the two
        # reach the import of the tracing as close together as they can; the
        # module's import lock is what makes the second wait for the first.
        sys.setswitchinterval(1e-6)
        start = threading.Barrier(2)
        results = [None, None]
        errors = []

        def trace(slot):
            try:
                start.wait()
                results[slot] = pulse.compile(measured).lower(Q[0], Q[1]).mlir()
            except BaseException as error:  # reported below
                errors.append(repr(error))

        threads = [threading.Thread(target=trace, args=(k,)) for k in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        print(errors == [])
        print(results[0] is not None and results[0] == results[1])
        print(results[0] == pulse.compile(measured).lower(Q[0], Q[1]).mlir())
        """
    )
    assert out.split() == ["True"] * 3
