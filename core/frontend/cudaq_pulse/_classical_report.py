# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""``Lowered.classical_report()``: one ``ClassicalCall`` per execution of a
call to a classical function. Loaded on the first report."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .compiler import _routes

if TYPE_CHECKING:
    from .compiler import Lowered


# classical_report() sets the fields without __init__, so there is no
# __post_init__.
@dataclass(frozen=True)
class ClassicalCall:
    """One execution of a call to a classical function, from ``classical_report()``.

    Times are in clock cycles. The request is sent from
    ``submit_cycles`` for ``submit_duration_cycles``; its answer is due at
    ``due_cycles`` (the budgeted latency after that) and taken at ``taken_cycles``
    (``None`` if the kernel never takes it). ``first_use_cycles`` is the
    earliest start of a pulse whose amplitude, tone or branch comes from the
    answer, and ``deadline_cycles`` the latest due time the function's
    ``deadline_ns`` allows. ``epoch`` is the epoch it is submitted in.
    ``link_name`` is the link's name in the machine model, if it has one.
    ``controller`` is the controller whose instance of a per-controller link
    runs the call (``Link(per_controller=True)``), else ``None``.
    ``function`` is the name the function is declared with and
    ``compiled_name`` the name it is compiled and served under: a
    version's, such as ``"adaptive_t1__1"``, for a function whose list
    lengths come from its calls, else the same name. ``dispatch_key`` is the
    key its requests carry: the hash of ``compiled_name``
    (``classical.function_id``), or the key its controller's firmware fixes
    (the controller's ``dispatch_keys``).

    ``exposed_cycles`` is how long the first use waits for the answer beyond
    when it could start without the call: after its line is free and after
    the readout resonator of the last readout before the call has depleted
    (``Qubit.depletion_ns``), when pulses must wait anyway. The rest of the
    budget, ``hidden_cycles``, overlaps that work. A latency within the
    depletion time is hidden entirely.
    """

    function: str
    compiled_name: str
    dispatch_key: int
    link: int
    iteration: tuple[int, ...]
    submit_cycles: int
    submit_duration_cycles: int
    due_cycles: int
    taken_cycles: int | None
    first_use_cycles: int | None
    deadline_cycles: int | None
    epoch: int
    link_name: str | None = None
    exposed_cycles: int | None = None
    controller: int | None = None

    @property
    def hidden_cycles(self) -> int | None:
        """The part of the budget that overlaps other work."""
        if self.exposed_cycles is None:
            return None
        return max(0, self.due_cycles - self.submit_cycles - self.exposed_cycles)

    @property
    def slack_cycles(self) -> int | None:
        """How long the answer could arrive later without delaying a pulse."""
        return None if self.first_use_cycles is None else self.first_use_cycles - self.due_cycles

    @property
    def deadline_slack_cycles(self) -> int | None:
        """How far the due time is inside the deadline (negative: missed)."""
        return None if self.deadline_cycles is None else self.deadline_cycles - self.due_cycles


ClassicalCall.__module__ = "cudaq_pulse.compiler"

# report() sets these directly, without __init__.
_CALL_FIELDS = tuple(ClassicalCall.__dataclass_fields__)


class _Exposure:
    """How long a call's first use waits beyond when it could start without
    the call (see ``ClassicalCall.exposed_cycles``), in O(log n) per call."""

    def __init__(self, lowered: "Lowered", events: Sequence[Any]) -> None:
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


def report(lowered: "Lowered") -> list[ClassicalCall]:
    """``lowered.classical_report()``."""
    module = lowered._module if lowered.scheduled and not lowered.carriers else lowered._concrete()
    with lowered._lock:
        calls = module.classical_calls()
        if not calls:
            return []
        events = module.events()
    names = lowered.options.machine.link_name
    exposed = _Exposure(lowered, events)
    # Per compiled name, the fields before the link (the declared name, the
    # compiled name and the dispatch key) and its link's name.
    by_name = {
        function.name: ((function.base_name, function.name, key), names(link))
        for function, (link, key, _) in zip(lowered.classical, _routes(lowered), strict=True)
    }
    new = object.__new__
    entries: list[ClassicalCall] = []
    for name, link, iteration, start, length, due, taken, use, limit, epoch, owner in calls:
        head, label = by_name[name]
        call = new(ClassicalCall)
        call.__dict__.update(
            zip(
                _CALL_FIELDS,
                head
                + (
                    link,
                    tuple(iteration),
                    start,
                    length,
                    due,
                    taken,
                    use,
                    limit,
                    epoch,
                    label,
                    None if use is None else exposed(start, use),
                    owner,
                ),
                strict=True,
            )
        )
        entries.append(call)
    return entries
