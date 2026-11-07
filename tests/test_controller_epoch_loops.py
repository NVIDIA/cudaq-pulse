# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Rolled loops whose body ends an epoch every iteration: a blocking wait for
an answer, or a wait whose length is one.

Such a loop repeats when every iteration opens as many epochs and moves each
line's epoch as far: the check reasons about it once per body, and what it
gives is what it gives with every iteration walked.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import controller
from cudaq_pulse.controller import check

from tests import _child
from tests._handles import Q
from tests.test_controller_rolled_calls import (
    _CALL_LIMITS,
    OWN_CLOCKS,
    _every,
    _Expansion,
    _outcome,
    _played,
    _summarizing,
)


@pulse.classical(max_latency_ns=400, latency_mode="blocking")
def decode(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


@pulse.classical(max_latency_ns=300, latency_mode="blocking")
def tune(bits: pulse.Bits) -> tuple[float, int]:
    return 0.1 * bits[0], 40 + bits[0]


@pulse.classical(max_latency_ns=200)
def note(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


def _rounds(n: int) -> dict[str, Any]:
    """Loops of rounds that each wait for an answer, blocking."""

    @pulse.kernel
    def decoded(q, p):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            decode.submit([m]).result()
            drive(d, square(20, 0.3))

    @pulse.kernel
    def valued(q, p):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            amp, delay = tune.submit([m]).result()
            drive(d, square(20, amp))

    @pulse.kernel
    def branched(q, p):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(80, 0.05))
            if decode.submit([m]).result()[0]:
                drive(d, square(20, 0.3))

    @pulse.kernel
    def apart(q, p):
        # The other qudit's line is on a timeline no answer holds back.
        d = get_drive_line(q)
        e = get_drive_line(p)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(100, 0.05))
            decode.submit([m]).result()
            drive(d, square(20, 0.3))
            drive(e, square(36, 0.2))

    @pulse.kernel
    def mixed(q, p):
        # A blocking answer beside a bounded one, on lines of two periods.
        d = get_drive_line(q)
        e = get_drive_line(p)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(70, 0.05))
            note.submit([m])
            decode.submit([m]).result()
            drive(e, square(12, 0.2))
            drive(d, square(20, 0.3))

    @pulse.kernel
    def nested(q, p):
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(3):
            for _ in range(n):
                m = readout(ro, square(60, 0.05))
                decode.submit([m]).result()
                drive(d, square(16, 0.3))
            drive(d, square(24, 0.1))

    @pulse.kernel
    def waits(q, p):
        # Each round waits as long as its answer says.
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(n):
            m = readout(ro, square(60, 0.05))
            amp, delay = tune.submit([m]).result()
            wait(d, delay, bounds=(8, 200))
            drive(d, square(20, 0.2))

    return {
        "decoded": decoded,
        "valued": valued,
        "branched": branched,
        "apart": apart,
        "mixed": mixed,
        "nested": nested,
        "waits": waits,
    }


def _lowered(kernel: Any, schedule: str, **options: Any) -> Any:
    return pulse.compile(kernel, schedule=schedule, unroll_threshold=0, **options).lower(Q[0], Q[1])


@pytest.mark.parametrize("name", list(_rounds(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_summarizing_walk_of_epoch_loops_has_every_iteration_s_events(name, schedule):
    lowered = _lowered(_rounds(80)[name], schedule)
    events, repeats = _summarizing(lowered)
    assert repeats
    got = json.dumps(_Expansion(events, repeats).run(), sort_keys=True)
    assert got == json.dumps(_every(lowered), sort_keys=True)


@pytest.mark.parametrize("clock", [None, *range(len(OWN_CLOCKS))])
@pytest.mark.parametrize("name", list(_rounds(1)))
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_an_epoch_loop_is_checked_as_every_iteration(monkeypatch, clock, name, schedule):
    lowered = _lowered(_rounds(100)[name], schedule)
    for limits in _CALL_LIMITS:
        if clock is not None:
            limits = dataclasses.replace(limits, **OWN_CLOCKS[clock])
        found = _outcome(lambda li=limits: check(lowered, li))
        compiled = _outcome(lambda li=limits: controller._compile_check(lowered, li)[0])
        with monkeypatch.context() as patch:
            patch.setattr(controller, "_walk", _played(controller._walk))
            assert found == _outcome(lambda li=limits: check(lowered, li))
            assert compiled == _outcome(lambda li=limits: controller._compile_check(lowered, li)[0])


_PROGRAM = """
import json
import cudaq_pulse as pulse
from cudaq_pulse import ControlSystem, Controller
from tests._handles import Q

@pulse.classical(max_latency_ns=400, latency_mode="blocking")
def decode(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]

@pulse.kernel
def rounds(q):
    d = get_drive_line(q)
    ro = get_readout_line(q)
    for _ in range(10**5):
        m = readout(ro, square(100, 0.05))
        decode.submit([m]).result()
        drive(d, square(20, 0.3))

{step}
"""
_CHECKED = """
system = ControlSystem(controllers=[Controller(instruction_cycles=2, max_amplitude=0.5)])
lowered = pulse.compile(rounds, control_system=system, unroll_threshold=0).lower(Q[0])
print(json.dumps({"ops": lowered.metrics.op_count}))
"""


def test_10_5_rounds_of_blocking_answers_are_checked_in_bounded_memory():
    step, rss_mb = _CHECKED, 150
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    source = _child.limits(30, rss_mb) + _PROGRAM.format(step=step)
    out = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    result = json.loads(out.stdout.splitlines()[-1])
    assert result["ops"] > 0
