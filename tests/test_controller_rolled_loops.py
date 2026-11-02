# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Controller checks of rolled loops.

The checks reason about a rolled loop once per body, with its trip count
applied to what they count, so a loop of 10**15 iterations is checked in
the time and memory of one; those tests run in a child process with a CPU,
wall and memory limit. What a check reports is what it reports with every
iteration played, which the loops past the native walk's cutoff are
compared against.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from typing import Any

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import controller
from cudaq_pulse.controller import ControllerLimitError, ControllerLimits, check

from tests._handles import Q

# The child's limits: a check of one body takes well under a second and a
# few hundred MB.
CPU_S = 30
RSS_MB = 600

# Linux's ru_maxrss carries the parent's peak across fork and exec, which a
# long test run takes past RSS_MB; VmHWM is this process's own.
_LIMITS = """
import os, resource, signal, sys, threading, time
signal.alarm({wall})
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
def _peak():
    if sys.platform == "darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) << 10
    raise RuntimeError("no VmHWM in /proc/self/status")
def _watch():
    while True:
        if (peak := _peak()) > {mb} << 20:
            os.write(2, f"peak RSS {{peak >> 20}} MB, past {mb} MB\\n".encode())
            os._exit(3)
        time.sleep(0.005)
_peak()
threading.Thread(target=_watch, daemon=True).start()
"""

_KERNELS = """
import json
import cudaq_pulse as pulse
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.controller import ControllerLimitError, ControllerLimits
from tests._handles import Q

TRIPS = {trips}

@pulse.kernel
def drives(q):
    d = get_drive_line(q)
    for _ in range(TRIPS):
        drive(d, square(8, 0.1))

@pulse.kernel
def close(q):
    d = get_drive_line(q)
    for _ in range(TRIPS):
        drive(d, square(5, 0.1))
        drive(d, gaussian(8, 0.1, 2.0))

machine = MachineModel(controller=ControllerLimits({limits}))
try:
    lowered = pulse.compile({kernel}, machine=machine).lower(Q[0])
except ControllerLimitError as error:
    print(json.dumps({{"count": len(error.violations), "message": str(error)}}))
else:
    print(json.dumps({{"ops": lowered.metrics.op_count}}))
"""


def _in_a_child(kernel: str, trips: int, limits: str = "") -> dict[str, object]:
    """What compiling *kernel* of *trips* iterations under *limits* gives, in
    a child process that is stopped past CPU_S seconds or RSS_MB MB."""
    code = _LIMITS.format(wall=2 * CPU_S, cpu=CPU_S, mb=RSS_MB) + _KERNELS.format(
        trips=trips, limits=limits, kernel=kernel
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        timeout=4 * CPU_S,
    )
    assert out.returncode == 0, f"the child exited {out.returncode}: {out.stderr[-2000:]}"
    result: dict[str, object] = json.loads(out.stdout.splitlines()[-1])
    return result


@pulse.kernel
def close(q):
    d = get_drive_line(q)
    for _ in range(3000):
        drive(d, square(5, 0.1))
        drive(d, gaussian(8, 0.1, 2.0))


def test_a_rolled_loop_reports_each_iteration_as_it_does_now():
    lowered = pulse.compile(close).lower(Q[0])
    assert lowered.metrics.op_count == 7
    limits = ControllerLimits(issue_cycles=4, min_wait_cycles=4, max_amplitude=0.05, max_events=7)
    over = (
        "peaks at amplitude 0.1; the controller plays at most 0.05 (ControllerLimits.max_amplitude)"
    )
    assert check(lowered, limits) == [
        "6000 drives and readouts; the controller plays at most 7. It does not support "
        "'loop', so rolled loops are unrolled for it: the loop of 3000 iterations from "
        "cycle 0 plays 5998 more",
        *(
            f"line 0: drive at {13 * k + 5} starts 5 cycles after the drive at {13 * k}; "
            "the controller needs exactly 4 or at least 8"
            for k in range(3000)
        ),
        *(
            f"line 0 (drive of qudit 0): the drive at {start} {over}"
            for k in range(3000)
            for start in (13 * k, 13 * k + 5)
        ),
    ]


def test_a_rolled_loop_of_a_few_iterations_compiles_under_a_controller_in_a_child():
    assert _in_a_child("drives", 1000) == {"ops": 5}


def test_a_rolled_loop_of_10_15_iterations_is_checked_in_bounded_time_and_memory():
    assert _in_a_child("drives", 10**15) == {"ops": 5}


def test_a_violation_in_every_iteration_of_10_15_is_counted_not_listed():
    few = _in_a_child("close", 20, "issue_cycles=4, min_wait_cycles=4")
    many = _in_a_child("close", 10**15, "issue_cycles=4, min_wait_cycles=4")
    assert many["count"] == 10**15
    assert many["message"] == str(few["message"]).replace(
        "... and 10 more", f"... and {10**15 - 10} more"
    )


def test_the_event_count_of_a_rolled_loop_is_its_trip_count_times_its_body():
    found = _in_a_child("drives", 10**15 + 1, "max_events=10**15")
    assert found["count"] == 1
    assert found["message"] == (
        "the program exceeds the controller's limits:\n  1000000000000001 drives and readouts; "
        "the controller plays at most 1000000000000000. It does not support 'loop', so rolled "
        "loops are unrolled for it: the loop of 1000000000000001 iterations from cycle 0 plays "
        "1000000000000000 more"
    )


def _every_iteration(monkeypatch: pytest.MonkeyPatch, lowered: Any, limits: Any) -> list[str]:
    """What ``check`` reports with every iteration of every loop played."""
    walk = controller._walk

    def played(*args: Any, repeats: Any = None, **kwargs: Any) -> Any:
        return walk(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(controller, "_walk", played)
        return list(check(lowered, limits))


def _loops(n: int) -> dict[str, Any]:
    @pulse.kernel
    def close(q):
        d = get_drive_line(q)
        for _ in range(n):
            drive(d, square(5, 0.1))
            drive(d, gaussian(8, 0.1, 2.0))

    @pulse.kernel
    def branches(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(q)
        other = get_readout_line(r)
        for _ in range(n):
            drive(line, gaussian(16, 0.1, 4.0))
            sync(line, probe, other)
            m = readout(probe, square(40, 0.05))
            readout(other, square(40, 0.05))
            sync(probe, line, other)
            if m:
                drive(line, square(8, 0.1))
            else:
                drive(line, square(4, 0.2))

    @pulse.kernel
    def before(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(q)
        m = readout(probe, square(40, 0.05))
        sync(probe, line)
        for _ in range(n):
            if m:
                drive(line, square(8, 0.1))
            wait(line, 3)

    @pulse.kernel
    def nested(q):
        d = get_drive_line(q)
        for _ in range(n):
            for _ in range(n):
                drive(d, square(5, 0.1))
                wait(d, 2)
            drive(d, gaussian(8, 0.1, 2.0))

    return {"close": close, "branches": branches, "before": before, "nested": nested}


LIMITS = [
    ControllerLimits(issue_cycles=4, min_wait_cycles=4, max_amplitude=0.05, max_events=7),
    ControllerLimits(
        issue_cycles=2,
        acquirers={0: 0, 1: 0},
        acquirer_hold_cycles=30,
        tones_per_line=1,
        waveform_samples=50,
        feedback_latency_cycles=60,
    ),
    ControllerLimits(feedback_latency_cycles=500, max_events=10),
    ControllerLimits(issue_cycles=2, waveform_samples=10, waveform_memory="shared"),
]


@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize("name", ["close", "branches", "before", "nested"])
@pytest.mark.parametrize("schedule", ["asap", "rcp"])
def test_a_long_rolled_loop_reports_what_every_iteration_played_does(
    monkeypatch, trips, name, schedule
):
    lowered = pulse.compile(_loops(trips)[name], schedule=schedule, unroll_threshold=0)
    lowered = lowered.lower(Q[0], Q[1]) if name in ("branches", "before") else lowered.lower(Q[0])
    repeats: list[Any] = []
    controller._walk(lowered, values=False, repeats=repeats)
    assert repeats, "the native walk summarized the loop"
    for limits in LIMITS:
        found = check(lowered, limits)
        assert (type(found) is list) == (len(found) <= 10_000)
        assert list(found) == _every_iteration(monkeypatch, lowered, limits)


def _timed(event: dict[str, Any]) -> tuple[Any, ...]:
    return event["kind"], event["line"], event["start_cycles"], event["end_cycles"]


def _unsynced(n: int) -> dict[str, Any]:
    """Loops whose lines move by different periods."""

    @pulse.kernel
    def two_lines(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, gaussian(8, 0.1, 2.0))

    @pulse.kernel
    def then(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(5, 0.1))
            drive(b, gaussian(8, 0.1, 2.0))
        sync(a, b)
        drive(a, square(6, 0.2))
        drive(b, square(2, 0.1))

    @pulse.kernel
    def two_readouts(q, r):
        p = get_readout_line(q)
        o = get_readout_line(r)
        for _ in range(n):
            readout(p, square(40, 0.05))
            readout(o, square(48, 0.05))

    @pulse.kernel
    def feedback(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        for _ in range(n):
            m = readout(probe, square(40, 0.05))
            if m:
                drive(line, square(8, 0.1))
            else:
                drive(line, square(4, 0.2))

    @pulse.kernel
    def early(q, r):
        line = get_drive_line(q)
        other = get_drive_line(r)
        probe = get_readout_line(q)
        m = readout(probe, square(40, 0.05))
        sync(probe, line)
        for _ in range(n):
            if m:
                drive(line, square(8, 0.1))
            wait(line, 3)
            drive(other, square(5, 0.1))

    @pulse.kernel
    def early_on_the_slower_line(q, r):
        other = get_drive_line(q)
        line = get_drive_line(r)
        probe = get_readout_line(r)
        m = readout(probe, square(40, 0.05))
        sync(probe, line)
        for _ in range(n):
            drive(other, square(5, 0.1))
            if m:
                drive(line, square(8, 0.1))
            wait(line, 3)

    @pulse.kernel
    def after(q, r):
        line = get_drive_line(q)
        probe = get_readout_line(r)
        other = get_drive_line(r)
        for _ in range(n):
            m = readout(probe, square(40, 0.05))
            drive(line, square(8, 0.1))
            drive(other, square(4, 0.1))
        if m:
            drive(line, square(8, 0.1))

    @pulse.kernel
    def nested(q, r):
        a = get_drive_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            for _ in range(n):
                drive(a, square(5, 0.1))
                wait(a, 2)
            drive(b, gaussian(8, 0.1, 2.0))

    @pulse.kernel
    def partly_synced(q, r):
        a = get_drive_line(q)
        p = get_readout_line(q)
        b = get_drive_line(r)
        for _ in range(n):
            drive(a, square(16, 0.1))
            sync(a, p)
            readout(p, square(40, 0.05))
            drive(b, square(4, 0.1))
            wait(b, 2)

    return {
        "two_lines": two_lines,
        "then": then,
        "two_readouts": two_readouts,
        "feedback": feedback,
        "early": early,
        "early_on_the_slower_line": early_on_the_slower_line,
        "after": after,
        "nested": nested,
        "partly_synced": partly_synced,
    }


_LEFT_TO_THE_WALK = (
    "a long loop whose iterations repeat out of step is checked from its summarizing walk"
)


# The native check walks a loop that does not repeat one period apart at
# every iteration; one of more trips than the summarizing walk keeps rolled
# (rcp peels a few first) it leaves to that walk.
@pytest.mark.parametrize("schedule", ["asap", "rcp"])
def test_the_native_check_leaves_a_long_loop_out_of_step_to_the_summarizing_walk(schedule):
    limits = ControllerLimits(issue_cycles=4, max_amplitude=0.05)
    left = []
    for trips in (64, 200):
        program = pulse.compile(
            _unsynced(trips)["two_lines"], schedule=schedule, unroll_threshold=0
        )
        found = controller._native_check(program.lower(Q[0], Q[1]), limits)
        left.append(found == [_LEFT_TO_THE_WALK])
    assert left == [False, True]


@pytest.mark.parametrize("trips", [65, 100])
@pytest.mark.parametrize(
    "name",
    [
        "two_lines",
        "then",
        "two_readouts",
        "feedback",
        "early",
        "early_on_the_slower_line",
        "after",
        "nested",
        "partly_synced",
    ],
)
@pytest.mark.parametrize("schedule", ["asap", "alap", "rcp"])
def test_a_loop_whose_lines_move_by_different_periods_reports_what_every_iteration_does(
    monkeypatch, trips, name, schedule
):
    from cudaq_pulse import _loops

    lowered = pulse.compile(_unsynced(trips)[name], schedule=schedule, unroll_threshold=0)
    lowered = lowered.lower(Q[0], Q[1])
    repeats: list[Any] = []
    periods: list[Any] = []
    lines, events, *_ = controller._walk(lowered, values=False, repeats=repeats, periods=periods)
    if repeats:
        rest = len(events) - max(r[1] for r in repeats)
        full = controller._walk(lowered, values=False)[1]
        assert [_timed(e) for e in events[len(events) - rest :]] == [
            _timed(e) for e in full[len(full) - rest :]
        ]
    summarized = 0
    for limits in LIMITS:
        found = check(lowered, limits)
        assert list(found) == _every_iteration(monkeypatch, lowered, limits)
        if periods:
            try:
                _loops.violations(lowered, events, repeats, limits, lines, None, None, periods)
            except _loops.UnsyncedError:
                continue
            summarized += 1
    if periods and name != "feedback":
        assert summarized, "some limits are checked on the summarized loop"


def test_past_10_000_violations_they_are_built_when_read(monkeypatch):
    lowered = pulse.compile(_loops(6000)["close"], unroll_threshold=0).lower(Q[0])
    limits = LIMITS[0]
    lazy = check(lowered, limits)
    assert type(lazy) is not list and len(lazy) == 1 + 6000 + 12000
    every = _every_iteration(monkeypatch, lowered, limits)
    assert list(lazy) == every
    assert lazy[0] == every[0] and lazy[-1] == every[-1] and lazy[6001] == every[6001]
    assert lazy[5:9] == every[5:9] and lazy[-3:] == every[-3:]
    with pytest.raises(IndexError):
        lazy[len(every)]
    error = ControllerLimitError(lazy)
    assert error.violations is lazy
    assert str(error) == str(ControllerLimitError(every))
    copy = pickle.loads(pickle.dumps(error))
    assert len(copy.violations) == len(every) and copy.violations[-1] == every[-1]
