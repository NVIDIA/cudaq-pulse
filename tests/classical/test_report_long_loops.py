# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A classical report of a program whose loops play far more events than it
makes calls takes each loop's ops once, and reports what listing every
event reports."""

from __future__ import annotations

import time

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import ControlSystem, _classical_report
from cudaq_pulse.classical import Link

from tests._handles import Q

MACHINE = ControlSystem(links={"host": Link(fixed_ns=4, slots=4)})


def _decide(bits: pulse.Bits) -> pulse.Bits:
    return [bits[0]]


FN = pulse.classical(_decide, max_latency_ns=100, link=0)


def _kernels(trips):
    @pulse.kernel
    def before(q0, q1):
        d = get_drive_line(q0)
        r = get_readout_line(q0)
        r1 = get_readout_line(q1)
        m = readout(r, square(100, 0.05))
        if FN.submit([m]).result()[0]:
            drive(d, square(40, 0.1))
        for _ in range(trips):
            drive(d, square(8, 0.1))
            readout(r1, square(12, 0.05))

    @pulse.kernel
    def after(q0, q1):
        d = get_drive_line(q0)
        r = get_readout_line(q0)
        d1 = get_drive_line(q1)
        for _ in range(trips):
            readout(r, square(12, 0.05))
            drive(d1, square(6, 0.1))
        m = readout(r, square(100, 0.05))
        if FN.submit([m]).result()[0]:
            drive(d, square(40, 0.1))
            drive(d1, square(40, 0.1))

    @pulse.kernel
    def nested(q0, q1):
        d = get_drive_line(q0)
        r = get_readout_line(q0)
        d1 = get_drive_line(q1)
        for _ in range(3):
            m = readout(r, square(100, 0.05))
            pending = FN.submit([m])
            for _ in range(trips):
                drive(d1, square(8, 0.1))
            if pending.result()[0]:
                drive(d, square(40, 0.1))

    @pulse.kernel
    def unsynced(q0, q1):
        d = get_drive_line(q0)
        r = get_readout_line(q0)
        d1 = get_drive_line(q1)
        m = readout(r, square(100, 0.05))
        pending = FN.submit([m])
        for _ in range(trips):
            drive(d, square(5, 0.1))
            drive(d1, square(7, 0.1))
        if pending.result()[0]:
            drive(d, square(40, 0.1))

    return {"before": before, "after": after, "nested": nested, "unsynced": unsynced}


def _report(monkeypatch, lowered, listed):
    monkeypatch.setattr(_classical_report, "_LISTED", 10**18 if listed else 0)
    monkeypatch.setattr(_classical_report, "_EVENTS_PER_CALL", 0)
    looped = []
    real = _classical_report._LoopExposure
    monkeypatch.setattr(_classical_report, "_LoopExposure", lambda *a: looped.append(1) or real(*a))
    entries = lowered.classical_report()
    assert bool(looped) != listed
    return entries


@pytest.mark.parametrize("trips", [1, 2, 7, 64])
@pytest.mark.parametrize("name", ["before", "after", "nested", "unsynced"])
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_the_loops_once_report_what_every_event_does(monkeypatch, trips, name, schedule):
    kernel = _kernels(trips)[name]
    lowered = pulse.compile(kernel, control_system=MACHINE, schedule=schedule, unroll_threshold=0)
    lowered = lowered.lower(Q[0], Q[1])
    every = _report(monkeypatch, lowered, listed=True)
    assert every and any(e.exposed_cycles is not None for e in every)
    assert _report(monkeypatch, lowered, listed=False) == every


def test_a_call_before_a_loop_of_10_7_iterations_is_reported_without_them():
    lowered = pulse.compile(_kernels(10**7)["before"], control_system=MACHINE, unroll_threshold=0)
    lowered = lowered.lower(Q[0], Q[1])
    began = time.process_time()
    (call,) = lowered.classical_report()
    assert time.process_time() - began < 5
    assert call.exposed_cycles is not None
