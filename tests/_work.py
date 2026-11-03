# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""What a call does in Python, counted rather than timed.

The lines a call runs and the calls it makes, counted with sys.monitoring,
are the same on any machine and at any load, so a test that compares two
sizes of one program can require them equal, or in proportion, where their
times are only close. Work done in native code is not counted: a test that
bounds it as well keeps a timing, the best of many.
"""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable

import pytest


def python_work(fn: Callable[[], object]) -> dict[str, int]:
    """The lines run and the calls made by a call of *fn* after a first,
    uncounted one; the garbage collector is held off while counting."""
    fn()
    monitoring = sys.monitoring
    tool = next((i for i in range(6) if monitoring.get_tool(i) is None), None)
    if tool is None:
        pytest.skip("every sys.monitoring tool id is in use")
    work = {"lines": 0, "calls": 0}

    def line(*_: object) -> None:
        work["lines"] += 1

    def call(*_: object) -> None:
        work["calls"] += 1

    events = monitoring.events
    monitoring.use_tool_id(tool, "cudaq-pulse tests")
    collecting = gc.isenabled()
    try:
        monitoring.register_callback(tool, events.LINE, line)
        monitoring.register_callback(tool, events.CALL, call)
        gc.collect()
        gc.disable()
        monitoring.set_events(tool, events.LINE | events.CALL)
        fn()
    finally:
        monitoring.set_events(tool, events.NO_EVENTS)
        if collecting:
            gc.enable()
        monitoring.register_callback(tool, events.LINE, None)
        monitoring.register_callback(tool, events.CALL, None)
        monitoring.free_tool_id(tool)
    return work
