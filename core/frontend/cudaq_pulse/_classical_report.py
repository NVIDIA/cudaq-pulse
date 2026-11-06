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
    (the ``dispatch_keys`` of the processor that runs it).

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
# The report lists every event up to this many, or this many per call;
# past that it takes the loops' events once each (_LoopExposure).
_LISTED = 10_000
_EVENTS_PER_CALL = 1000


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


class _LoopExposure:
    """``_Exposure`` from ``PulseModule.event_tree()``, for programs whose
    loops play far more events than they make calls: each drive and readout
    once, with its start's shift per iteration of each loop around it. A
    run of a loop's iterations whose ops all lie on one side of the times a
    call asks about counts from the bounds of those ops; only iterations
    that straddle them are looked at one by one."""

    def __init__(self, lowered: "Lowered", nodes: Sequence[Any]) -> None:
        clock = lowered.target.clock_ghz
        qubits = lowered.target.target.qubits
        holds: dict[int, int] = {}

        def hold(qubit: int) -> int:
            if qubit not in holds:
                q = qubits.get(qubit)
                holds[qubit] = math.ceil(q.depletion_ns * clock) if q is not None else 0
            return holds[qubit]

        def build(items: Sequence[Any], trips: tuple[int, ...]) -> tuple[list[Any], list[Any]]:
            # (the block's items, every op under it). An op is (readout,
            # line, start, duration, shifts, hold, extents) with extents[l]
            # the (lowest, highest) shift of the loops inside level l.
            block: list[Any] = []
            under: list[Any] = []
            for item in items:
                if len(item) == 2:
                    inner, ops = build(item[1], (*trips, item[0]))
                    if ops and item[0] > 0:
                        block.append((item[0], inner, ops))
                        under += ops
                    continue
                kind, line, qubit, start, duration, shifts = item
                if (kind != "drive" and kind != "readout") or start is None or duration is None:
                    continue
                if any(s is None for s in shifts):
                    raise _UnshiftedError
                moves = [
                    (min(0, (t - 1) * s), max(0, (t - 1) * s))
                    for t, s in zip(trips, shifts, strict=False)
                ]
                extents = []
                for level in range(len(trips)):
                    deeper = moves[level + 1 :]
                    extents.append((sum(lo for lo, _ in deeper), sum(hi for _, hi in deeper)))
                op = (
                    kind == "readout",
                    line,
                    start,
                    duration,
                    tuple(shifts),
                    hold(qubit) if kind == "readout" else 0,
                    extents,
                )
                block.append(op)
                under.append(op)
            return block, under

        self._root, _ = build(nodes, ())

    def __call__(self, submit_cycles: int, use_cycles: int) -> int | None:
        found = _Found(submit_cycles, use_cycles)
        found.block(self._root, [])
        if not found.starting:
            return None
        free = max((found.ends.get(line, 0) for line in found.starting), default=0)
        if found.readout is not None:
            free = max(free, found.readout[1])
        return max(0, use_cycles - max(free, submit_cycles))


class _UnshiftedError(Exception):
    """An op whose later iterations have no start: list every event."""


class _Found:
    """What ``_Exposure`` finds for one call: the lines a drive or readout
    starts on at the use, each line's latest end by the use, and the latest
    readout end by the submit with its end plus depletion."""

    def __init__(self, submit: int, use: int) -> None:
        self.submit, self.use = submit, use
        self.starting: set[int] = set()
        self.ends: dict[int, int] = {}
        self.readout: tuple[int, int] | None = None

    def _end(self, line: int, end: int) -> None:
        if end > self.ends.get(line, -1):
            self.ends[line] = end

    def _read(self, end: int, held: int) -> None:
        if self.readout is None or (end, held) > self.readout:
            self.readout = (end, held)

    def block(self, items: Sequence[Any], at: list[int]) -> None:
        for item in items:
            if len(item) == 3:
                self.run(item, at, 0, item[0])
                continue
            readout, line, start, duration, shifts, hold, _ = item
            start += sum(k * s for k, s in zip(at, shifts, strict=False))
            end = start + duration
            if start == self.use:
                self.starting.add(line)
            if end <= self.use:
                self._end(line, end)
            if readout and end <= self.submit:
                self._read(end, end + hold)

    def run(self, loop: tuple[int, list[Any], list[Any]], at: list[int], a: int, b: int) -> None:
        """Iterations a to b - 1 of *loop* inside the iterations *at*."""
        level = len(at)
        spans = []
        for op in loop[2]:
            readout, line, start, duration, shifts, hold, extents = op
            base = start + sum(k * s for k, s in zip(at, shifts, strict=False))
            p = shifts[level]
            lo = base + min(a * p, (b - 1) * p) + extents[level][0]
            hi = base + max(a * p, (b - 1) * p) + extents[level][1]
            spans.append((readout, line, lo, hi, duration, hold))
        use, submit = self.use, self.submit
        mixed = False
        for readout, _, lo, hi, duration, _ in spans:
            if lo <= use <= hi:  # may start at the use
                mixed = True
            elif not (hi + duration <= use or lo + duration > use):
                mixed = True
            elif readout and not (hi + duration <= submit or lo + duration > submit):
                mixed = True
            if mixed:
                break
        if not mixed:
            for readout, line, _, hi, duration, hold in spans:
                if hi + duration <= use:
                    self._end(line, hi + duration)
                if readout and hi + duration <= submit:
                    self._read(hi + duration, hi + duration + hold)
            return
        if b - a > 1:
            mid = (a + b) // 2
            self.run(loop, at, a, mid)
            self.run(loop, at, mid, b)
            return
        at.append(a)
        self.block(loop[1], at)
        at.pop()


def report(lowered: "Lowered") -> list[ClassicalCall]:
    """``lowered.classical_report()``."""
    module = lowered._module if lowered.scheduled and not lowered.carriers else lowered._concrete()
    with lowered._lock:
        calls = module.classical_calls()
        if not calls:
            return []
        events = module.event_tree(max(_LISTED, _EVENTS_PER_CALL * len(calls)))
        exposed: Any = None
        if type(events) is not list:
            try:
                exposed = _LoopExposure(lowered, events[0])
            except _UnshiftedError:
                events = module.events()
    names = lowered.options.machine.link_name
    if exposed is None:
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
