# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""``Lowered.classical_report()``: one ``ClassicalCall`` per execution of a
call to a classical function. Loaded on the first report."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .compiler import ClassicalCall, Lowered

# report() sets these directly, without __init__.
_CALL_FIELDS = tuple(ClassicalCall.__dataclass_fields__)


class _Exposure:
    """How long a call's first use waits beyond when it could start without
    the call (see ``ClassicalCall.exposed_cycles``), in O(log n) per call."""

    def __init__(self, lowered: Lowered, events: Sequence[Any]) -> None:
        import bisect

        self._bisect = bisect.bisect_right
        clock = lowered.target.clock_ghz
        qubits = lowered.target.target.qubits
        ends: dict[int, list[int]] = {}
        starting: dict[int, list[int]] = {}
        readouts: list[tuple[int, int]] = []
        holds: dict[int, int] = {}
        for kind, line, qubit, start, duration in events:
            if (kind != "drive" and kind != "readout") or start is None or duration is None:
                continue
            end = start + duration
            ends.setdefault(line, []).append(end)
            starting.setdefault(start, []).append(line)
            if kind == "readout":
                hold = holds.get(qubit)
                if hold is None:
                    q = qubits.get(qubit)
                    hold = math.ceil(q.depletion_ns * clock) if q is not None else 0
                    holds[qubit] = hold
                readouts.append((end, end + hold))
        for values in ends.values():
            values.sort()
        readouts.sort()
        self._ends = ends
        self._starting = starting
        self._readout_ends = [end for end, _ in readouts]
        self._holds = [hold for _, hold in readouts]

    def __call__(self, submit_cycles: int, use_cycles: int) -> int | None:
        lines = self._starting.get(use_cycles)
        if not lines:
            return None
        free = 0
        for line in lines:
            ends = self._ends[line]
            k = self._bisect(ends, use_cycles)
            if k:
                free = max(free, ends[k - 1])
        k = self._bisect(self._readout_ends, submit_cycles)
        if k:
            free = max(free, self._holds[k - 1])
        return max(0, use_cycles - max(free, submit_cycles))


def report(lowered: Lowered) -> list[ClassicalCall]:
    """``lowered.classical_report()``."""
    module = lowered._module if lowered.scheduled and not lowered.carriers else lowered._concrete()
    with lowered._lock:
        calls = module.classical_calls()
        if not calls:
            return []
        events = module.events()
    names = lowered.options.machine.link_name
    exposed = _Exposure(lowered, events)
    declared = {function.name: function.base_name for function in lowered.classical}
    keys = lowered._dispatch_keys
    new = object.__new__
    entries: list[ClassicalCall] = []
    for name, link, iteration, start, length, due, taken, use, limit, epoch, owner in calls:
        call = new(ClassicalCall)
        call.__dict__.update(
            zip(
                _CALL_FIELDS,
                (
                    declared.get(name, name),
                    name,
                    keys[name][0],
                    link,
                    tuple(iteration),
                    start,
                    length,
                    due,
                    taken,
                    use,
                    limit,
                    epoch,
                    names(link),
                    None if use is None else exposed(start, use),
                    owner,
                ),
                strict=True,
            )
        )
        entries.append(call)
    return entries
