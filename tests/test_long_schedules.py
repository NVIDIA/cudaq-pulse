# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A schedule of more than 10,000 events builds each event when it is read,
from its op and its loops' iterations, and gives what the list of every
event would; ``duration`` takes the loops' last iterations without them."""

from __future__ import annotations

import pickle

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.compiler import ScheduledEvent

from tests._handles import Q
from tests.test_controller_rolled_loops import _loops


def _every_event(lowered):
    return [ScheduledEvent(*event) for event in lowered._module.events()]


@pytest.mark.parametrize("name", ["close", "branches", "before", "nested"])
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_long_schedule_gives_what_the_list_of_every_event_would(name, schedule):
    trips = 120 if name == "nested" else 6000
    lowered = pulse.compile(_loops(trips)[name], schedule=schedule, unroll_threshold=0)
    lowered = lowered.lower(Q[0], Q[1]) if name in ("branches", "before") else lowered.lower(Q[0])
    every = _every_event(lowered)
    lazy = lowered.schedule
    assert type(lazy) is not list and len(lazy) == len(every) > 10_000
    assert list(lazy) == every
    assert [lazy[k] for k in range(0, len(every), 997)] == every[::997]
    assert lazy[-1] == every[-1] and lazy[-len(every)] == every[0]
    assert lazy[3:40:7] == every[3:40:7]
    with pytest.raises(IndexError):
        lazy[len(every)]
    end = max(e.start_cycles + e.duration_cycles for e in every)
    assert float(lowered.duration) == end / lowered.target.clock_ghz


def test_up_to_10_000_events_the_schedule_is_a_list():
    lowered = pulse.compile(_loops(100)["close"], unroll_threshold=0).lower(Q[0])
    assert type(lowered.schedule) is list and lowered.schedule == _every_event(lowered)


def test_a_loop_of_10_15_iterations_is_read_without_its_iterations():
    @pulse.kernel
    def waits(q):
        d = get_drive_line(q)
        for _ in range(10**15):
            wait(d, 4)

    lowered = pulse.compile(waits, unroll_threshold=0).lower(Q[0])
    schedule = lowered.schedule
    assert len(schedule) == 10**15
    assert schedule[-1] == ScheduledEvent("wait", 0, 0, 4 * (10**15 - 1), 4)
    assert float(lowered.duration) == 4 * 10**15 / lowered.target.clock_ghz
    assert repr(schedule) == f"<{10**15} scheduled events>"
    assert pickle.loads(pickle.dumps(schedule[7])) == schedule[7]


def test_a_long_loop_stays_unlisted_through_a_float_argument_and_a_pickle():
    @pulse.kernel
    def waits(q, tau: float):
        d = get_drive_line(q)
        for _ in range(10**15):
            wait(d, tau, bounds=(4, 8))

    lowered = pulse.compile(waits, unroll_threshold=0).lower(Q[0], 6.0)
    for program in (lowered, pickle.loads(pickle.dumps(lowered))):
        schedule = program.schedule
        assert len(schedule) == 10**15
        assert schedule[-1] == ScheduledEvent("wait", 0, 0, 6 * (10**15 - 1), 6)


def test_a_program_without_loops_lists_its_events_unless_asked_for_its_tree():
    @pulse.kernel
    def once(q):
        d = get_drive_line(q)
        drive(d, square(16, 0.1))
        wait(d, 8)

    module = pulse.compile(once).lower(Q[0])._module
    assert module.event_tree(0) == module.events()
    (nodes,) = module.event_tree(-1)
    assert [node[0] for node in nodes] == ["drive", "wait"]
