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
the time and memory of one. Until then the native walk plays every
iteration, so those tests are strict xfails that run in a child process
with a CPU, wall and memory limit; the reports of loops checked today must
stay exactly as they are.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.controller import ControllerLimits, check

from tests._handles import Q

PENDING = pytest.mark.xfail(
    strict=True, reason="checking a rolled loop once per body lands after multi-controller"
)

# The child's limits: a check of one body takes well under a second and a
# few hundred MB.
CPU_S = 30
RSS_MB = 600

_LIMITS = """
import os, resource, signal, sys, threading, time
signal.alarm({wall})
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
_scale = 1 if sys.platform == "darwin" else 1024
def _watch():
    while True:
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _scale > {mb} << 20:
            os._exit(3)
        time.sleep(0.005)
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


@PENDING
def test_a_rolled_loop_of_10_15_iterations_is_checked_in_bounded_time_and_memory():
    assert _in_a_child("drives", 10**15) == {"ops": 5}


@PENDING
def test_a_violation_in_every_iteration_of_10_15_is_counted_not_listed():
    few = _in_a_child("close", 20, "issue_cycles=4, min_wait_cycles=4")
    many = _in_a_child("close", 10**15, "issue_cycles=4, min_wait_cycles=4")
    assert many["count"] == 10**15
    assert many["message"] == str(few["message"]).replace(
        "... and 10 more", f"... and {10**15 - 10} more"
    )


@PENDING
def test_the_event_count_of_a_rolled_loop_is_its_trip_count_times_its_body():
    found = _in_a_child("drives", 10**15 + 1, "max_events=10**15")
    assert found["count"] == 1
    assert found["message"] == textwrap.dedent(
        """\
        the program exceeds the controller's limits:
          1000000000000001 drives and readouts; the controller plays at most 1000000000000000"""
    )
