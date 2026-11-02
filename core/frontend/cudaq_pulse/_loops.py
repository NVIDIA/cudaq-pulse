# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The controller checks of a program whose long rolled loops the native
walk summarized (``ControllerWalk::summarizeLoop``): the events of one
iteration stand for those of every later one, so each check works on a
loop's body once and counts its copies. A check finds what it finds with
every iteration played, in the same order; past ``LAZY_AFTER`` violations
they are built when read. So is a schedule of more than ``LAZY_AFTER``
events (``Schedule``). Only a check that met such a loop, or such a
schedule, loads this."""

from __future__ import annotations

import bisect
import math
import operator
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, overload

from .controller import (
    _PLAYED,
    ControllerLimits,
    WaveformMemoryScope,
    _exclusive,
    _native_clock,
    _peak_amplitude,
    _sample_rates,
    _samples_per_own_cycle,
    resources,
    shape_key,
)

if TYPE_CHECKING:
    from fractions import Fraction

    from ._clocks import ClockPlan, ControllerClock
    from ._system import ControlSystem
    from .compiler import Lowered

# Up to this many violations are a list, as without summarized loops.
LAZY_AFTER = 10_000
# The copies of loops a check plays one by one, at most: those whose events
# interleave in time with others on one line, or whose feedback depends on a
# readout before the loop.
_MAX_PLAYED = 1 << 20


class UnsyncedError(Exception):
    """A check pairs events of lines that a loop's copies move by different
    periods: the caller walks the program again with only loops of one
    period summarized."""


class _Rep:
    """Copies of the events of ``body``: copy j starts j * ``period``
    cycles after the first, which the events of ``body`` are; or, with
    ``lines``, j times the period of its event's line."""

    __slots__ = ("begin", "end", "copies", "period", "depth", "body", "orig", "lines")

    def __init__(
        self,
        begin: int,
        end: int,
        copies: int,
        period: int,
        depth: int,
        lines: dict[int, int] | None = None,
    ) -> None:
        self.begin, self.end, self.copies = begin, end, copies
        self.period, self.depth = period, depth
        self.lines = lines
        self.body: list[Any] = []
        self.orig = self

    def keeping(self, body: list[Any]) -> _Rep:
        kept = _Rep(self.begin, self.end, self.copies, self.period, self.depth, self.lines)
        kept.body, kept.orig = body, self.orig
        return kept


def _period(rep: _Rep, lines: Iterable[int]) -> int:
    """The period of *rep*'s copies on *lines*, which they must share."""
    if rep.lines is None:
        return rep.period
    found = {rep.lines[line] for line in lines if line in rep.lines}
    if len(found) > 1:
        raise UnsyncedError
    return found.pop() if found else 0


class _Text:
    """A violation: its pieces joined, an int piece shifted by the cycles
    its copy starts after the first."""

    __slots__ = ("pieces",)

    def __init__(self, *pieces: str | int) -> None:
        self.pieces = pieces

    def render(self, shift: int) -> str:
        return "".join(p if isinstance(p, str) else str(p + shift) for p in self.pieces)


class _Copies:
    """The violations ``parts`` of copies ``first`` to ``count`` - 1 of a
    loop, copy j shifted by ``offset`` + j * ``period`` cycles."""

    __slots__ = ("parts", "first", "count", "period", "offset", "inner", "size")

    def __init__(
        self, parts: list[Any], first: int, count: int, period: int, offset: int = 0
    ) -> None:
        self.parts, self.first, self.count = parts, first, count
        self.period, self.offset = period, offset
        self.inner = _length(parts)
        self.size = (count - first) * self.inner


def _length(parts: Sequence[Any]) -> int:
    return sum(p.size if type(p) is _Copies else 1 for p in parts)


def _render(part: Any, shift: int) -> str:
    return part if type(part) is str else part.render(shift)


def _each(parts: Sequence[Any], shift: int) -> Iterator[str]:
    for part in parts:
        if type(part) is _Copies:
            if not part.size:
                continue
            for j in range(part.first, part.count):
                yield from _each(part.parts, shift + part.offset + j * part.period)
        else:
            yield _render(part, shift)


class Violations(Sequence[str]):
    """The limits a program exceeds, each built when read: the checks of a
    rolled loop's body stand for every iteration it plays. ``len()`` is
    exact; indexing, slicing and iterating give what a list of every
    violation would."""

    __slots__ = ("_parts", "_size")

    def __init__(self, parts: list[Any]) -> None:
        self._parts = parts
        self._size = _length(parts)

    def __len__(self) -> int:
        return self._size

    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> list[str]: ...
    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self[k] for k in range(*index.indices(self._size))]
        k = operator.index(index)
        if k < 0:
            k += self._size
        if not 0 <= k < self._size:
            raise IndexError("violation index out of range")
        parts, shift = self._parts, 0
        while True:
            for part in parts:
                if type(part) is _Copies:
                    if k < part.size:
                        j, k = divmod(k, part.inner)
                        shift += part.offset + (part.first + j) * part.period
                        parts = part.parts
                        break
                    k -= part.size
                elif k == 0:
                    return _render(part, shift)
                else:
                    k -= 1

    def __iter__(self) -> Iterator[str]:
        return _each(self._parts, 0)

    def __repr__(self) -> str:
        return f"<{self._size} controller violations>"


class _Block:
    """Nodes of ``PulseModule.event_tree()``: ops, and loops as (trips,
    their body's block), with where each starts among the events listed."""

    __slots__ = ("items", "offsets", "size")

    def __init__(self, nodes: Sequence[Any]) -> None:
        self.items: list[Any] = []
        self.offsets: list[int] = []
        self.size = 0
        for node in nodes:
            if len(node) == 2:
                body = _Block(node[1])
                size = node[0] * body.size
                if not size:
                    continue
                node = (node[0], body)
            else:
                size = 1
            self.items.append(node)
            self.offsets.append(self.size)
            self.size += size


class Schedule(Sequence[Any]):
    """``Lowered.schedule`` past ``LAZY_AFTER`` events: each event built when
    read from its op and the iteration of each loop around it, whose start
    is the op's first plus each iteration's shift. ``len()`` is exact;
    indexing, slicing and iterating give what a list of every event would."""

    __slots__ = ("_event", "_root")

    def __init__(self, nodes: Sequence[Any], event: Callable[..., Any]) -> None:
        self._root = _Block(nodes)
        self._event = event

    def __len__(self) -> int:
        return self._root.size

    @overload
    def __getitem__(self, index: int) -> Any: ...
    @overload
    def __getitem__(self, index: slice) -> list[Any]: ...
    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[k] for k in range(*index.indices(self._root.size))]
        k = operator.index(index)
        if k < 0:
            k += self._root.size
        if not 0 <= k < self._root.size:
            raise IndexError("schedule index out of range")
        block, at = self._root, []
        while True:
            j = bisect.bisect_right(block.offsets, k) - 1
            item = block.items[j]
            k -= block.offsets[j]
            if len(item) == 2:
                trips, block = item
                i, k = divmod(k, block.size)
                at.append(i)
            else:
                return self._made(item, at)

    def __iter__(self) -> Iterator[Any]:
        return self._each(self._root, [])

    def _each(self, block: _Block, at: list[int]) -> Iterator[Any]:
        for item in block.items:
            if len(item) == 2:
                trips, body = item
                at.append(0)
                for i in range(trips):
                    at[-1] = i
                    yield from self._each(body, at)
                at.pop()
            else:
                yield self._made(item, at)

    def _made(self, op: tuple[Any, ...], at: Sequence[int]) -> Any:
        kind, line, qubit, start, duration, shifts = op
        if start is not None:
            start += sum(i * s for i, s in zip(at, shifts, strict=True) if s is not None)
        return self._event(kind, line, qubit, start, duration)

    def latest_end(self) -> int | None:
        """The latest end of an event, or ``None`` when some event has no
        start or duration yet."""
        end = 0
        work: list[tuple[_Block, tuple[int, ...]]] = [(self._root, ())]
        while work:
            block, trips = work.pop()
            for item in block.items:
                if len(item) == 2:
                    work.append((item[1], (*trips, item[0])))
                    continue
                _, _, _, start, duration, shifts = item
                if start is None or duration is None:
                    return None
                # Each loop's iterations shift the op alike: the latest takes
                # each loop's last iteration that moves it later.
                late = sum(
                    (t - 1) * s
                    for t, s in zip(trips, shifts, strict=True)
                    if s is not None and s > 0
                )
                end = max(end, start + late + duration)
        return end

    def __repr__(self) -> str:
        return f"<{self._root.size} scheduled events>"


def _tree(
    count: int,
    repeats: Sequence[Sequence[int]],
    periods: Sequence[Sequence[int]] = (),
) -> tuple[list[Any], list[tuple[_Rep, ...]]]:
    """The events 0 to *count* - 1 with their repeats as nested nodes, and
    the repeats each event lies in, outermost first; *periods* the
    (repeat, line, period) of repeats whose lines have periods of their own."""
    lines: dict[int, dict[int, int]] = {}
    for index, line, period in periods:
        lines.setdefault(index, {})[line] = period
    reps = sorted(
        (
            _Rep(r[0], r[1], r[2], r[3], r[4], lines.get(k, {}) if r[3] < 0 else None)
            for k, r in enumerate(repeats)
            if r[1] > r[0]
        ),
        key=lambda r: (r.begin, -r.end),
    )
    chains: list[tuple[_Rep, ...]] = [()] * count
    at = 0

    def build(lo: int, hi: int, chain: tuple[_Rep, ...]) -> list[Any]:
        nonlocal at
        nodes: list[Any] = []
        i = lo
        while i < hi:
            if at < len(reps) and reps[at].begin == i and reps[at].end <= hi:
                rep = reps[at]
                at += 1
                rep.body = build(rep.begin, rep.end, chain + (rep,))
                nodes.append(rep)
                i = rep.end
            else:
                chains[i] = chain
                nodes.append(i)
                i += 1
        return nodes

    return build(0, count, ()), chains


def _keep(nodes: list[Any], keep: Callable[[int], bool]) -> list[Any]:
    kept: list[Any] = []
    for node in nodes:
        if type(node) is int:
            if keep(node):
                kept.append(node)
        else:
            body = _keep(node.body, keep)
            if body:
                kept.append(node.keeping(body))
    return kept


def _count(nodes: list[Any]) -> int:
    return sum(1 if type(n) is int else n.copies * _count(n.body) for n in nodes)


def _events_of(nodes: list[Any]) -> Iterator[int]:
    for node in nodes:
        if type(node) is int:
            yield node
        else:
            yield from _events_of(node.body)


def _each_event(
    nodes: list[Any], make: Callable[[int], Any], lines: Iterable[int] | None = None
) -> list[Any]:
    """The violation *make* gives for each event of *nodes* (or ``None``),
    in program order; their events are on *lines*, else on any line."""
    parts: list[Any] = []
    for node in nodes:
        if type(node) is int:
            part = make(node)
            if part is not None:
                parts.append(part)
        else:
            body = _each_event(node.body, make, lines)
            if body:
                on = lines if lines is not None else (node.lines or ())
                parts.append(_Copies(body, 0, node.copies, _period(node, on)))
    return parts


class _Budget:
    def __init__(self) -> None:
        self.left = _MAX_PLAYED

    def spend(self, copies: int) -> None:
        self.left -= copies
        if self.left < 0:
            raise ValueError(
                f"the controller check plays more than {_MAX_PLAYED} copies of rolled loops "
                "one by one, whose events interleave with others on one line or depend on a "
                "readout before the loop; give the loops fewer iterations"
            )


# An event of a check: (index, shift in cycles, the copies it is in).
_At = tuple[int, int, tuple[Any, ...]]


class _Pairs:
    """Each event of a line or acquirer with the latest earlier one it can
    follow on one path (``controller._pairs``), a loop's copies alike."""

    def __init__(
        self,
        events: Sequence[Mapping[str, Any]],
        pair: Callable[[Mapping[str, Any], int, Mapping[str, Any], int], Any],
        budget: _Budget,
        lines: Iterable[int],
    ) -> None:
        self.events, self.pair, self.budget = events, pair, budget
        self.lines = tuple(lines)
        self.windows: dict[int, tuple[int, int]] = {}

    def period(self, rep: _Rep) -> int:
        return _period(rep, self.lines)

    def window(self, node: Any) -> tuple[int, int]:
        """The first and last start of *node* (all its copies)."""
        if type(node) is int:
            start = self.events[node]["start_cycles"]
            return start, start
        found = self.windows.get(id(node))
        if found is None:
            lo, hi = self.body_window(node)
            found = self.windows[id(node)] = (lo, hi + (node.copies - 1) * self.period(node))
        return found

    def body_window(self, rep: _Rep) -> tuple[int, int]:
        spans = [self.window(child) for child in rep.body]
        return min(s[0] for s in spans), max(s[1] for s in spans)

    def ordered(self, items: list[tuple[Any, int, tuple[Any, ...]]]) -> list[Any]:
        """*items* in start order, a repeat a block of copies that each come
        whole; one that does not is played copy by copy."""
        while True:
            spans = [self.window(node) for node, _, _ in items]
            order = sorted(range(len(items)), key=lambda k: (spans[k][0] + items[k][1], k))
            bad = None
            for k in order:
                node = items[k][0]
                if type(node) is not int:
                    lo, hi = self.body_window(node)
                    if hi - lo > self.period(node):
                        bad = k
                        break
            if bad is None:
                for a, b in zip(order, order[1:], strict=False):
                    hi = spans[a][1] + items[a][1]
                    lo = spans[b][0] + items[b][1]
                    if hi > lo or (hi == lo and a > b):
                        bad = a if type(items[a][0]) is not int else b
                        break
            if bad is None:
                return [items[k] for k in order]
            node, dt, path = items[bad]
            self.budget.spend(node.copies)
            items = (
                items[:bad]
                + [
                    (child, dt + j * self.period(node), path + ((id(node.orig), j),))
                    for j in range(node.copies)
                    for child in node.body
                ]
                + items[bad + 1 :]
            )

    def walk(
        self, items: list[tuple[Any, int, tuple[Any, ...]]], before: _At | None
    ) -> tuple[list[Any], _At | None]:
        """The pairs of *items* after *before*, and the last of them."""
        events = self.events
        parts: list[Any] = []
        run: list[_At] = []
        anchor = last = before
        for node, dt, path in self.ordered(items):
            if type(node) is int:
                event = events[node]
                mate = anchor
                for other in reversed(run):
                    if other[2] != path or not _exclusive(events[other[0]], event):
                        mate = other
                        break
                if mate is not None:
                    part = self.pair(events[mate[0]], mate[1], event, dt)
                    if part is not None:
                        parts.append(part)
                last = (node, dt, path)
                run.append(last)
                continue
            inner = path + (id(node.orig),)
            first, last = self.walk([(child, dt, inner) for child in node.body], last)
            parts.extend(first)
            assert last is not None
            if node.copies > 1:
                period = self.period(node)
                again, end = self.walk(
                    [(child, 0, inner) for child in node.body],
                    (last[0], last[1] - dt - period, ("copy",)),
                )
                if again:
                    parts.append(_Copies(again, 1, node.copies, period, dt))
                assert end is not None
                last = (end[0], dt + (node.copies - 1) * period + end[1], ("copy",))
            anchor = last
            run = []
        return parts, last


def _rolled_loops(
    nodes: list[Any], events: Sequence[Mapping[str, Any]]
) -> list[tuple[int, int, int]]:
    """``controller._rolled_loops`` of the events of *nodes*, every copy."""
    loops: list[list[int]] = []
    previous = -1

    def scan(nodes: list[Any]) -> None:
        nonlocal previous
        for node in nodes:
            if type(node) is int:
                e = events[node]
                iteration = e["iteration"]
                if not iteration:
                    previous = -1
                    continue
                if previous < 0 or iteration[0] < previous:
                    loops.append([0, e["start_cycles"], 0])
                loop = loops[-1]
                loop[0] = max(loop[0], iteration[0] + 1)
                loop[2] += any(iteration)
                previous = iteration[0]
                continue
            scan(node.body)
            rest = node.copies - 1
            if rest:
                loop = loops[-1]
                loop[2] += rest * _count(node.body)
                if node.depth == 0:
                    previous += rest
                    loop[0] = max(loop[0], previous + 1)

    scan(nodes)
    return [(trips, start, extra) for trips, start, extra in loops if extra]


def violations(
    lowered: Lowered,
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    limits: ControllerLimits,
    lines: Sequence[Mapping[str, Any]],
    unresolved: list[str] | None,
    sample_rate: tuple[float, float] | None = None,
    periods: Sequence[Sequence[int]] = (),
    own: ControllerClock | None = None,
    off: Sequence[tuple[int, int, int]] = (),
    natives: Sequence[int] | None = None,
) -> list[str] | Violations:
    """``controller._violations`` of the events of a summarizing walk
    (``parts``)."""
    return finished(
        parts(
            lowered,
            events,
            repeats,
            limits,
            lines,
            unresolved,
            sample_rate,
            periods,
            own,
            off,
            natives,
        )
    )


def finished(found: list[Any]) -> list[str] | Violations:
    """The violations of *found*, parts of ``parts``: a list, or past
    ``LAZY_AFTER`` built when read."""
    if _length(found) <= LAZY_AFTER:
        return list(_each(found, 0))
    return Violations(found)


class _Labeled:
    """A violation of one controller of several: ``label``, then ``part``."""

    __slots__ = ("label", "part")

    def __init__(self, label: str, part: Any) -> None:
        self.label, self.part = label, part

    def render(self, shift: int) -> str:
        return self.label + _render(self.part, shift)


def labeled(found: list[Any], label: str) -> list[Any]:
    """The parts *found* of ``parts``, each violation after *label*."""
    out: list[Any] = []
    for part in found:
        if type(part) is _Copies:
            out.append(
                _Copies(
                    labeled(part.parts, label), part.first, part.count, part.period, part.offset
                )
            )
        elif type(part) is str:
            out.append(label + part)
        else:
            out.append(_Labeled(label, part))
    return out


def parts(
    lowered: Lowered,
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    limits: ControllerLimits,
    lines: Sequence[Mapping[str, Any]],
    unresolved: list[str] | None,
    sample_rate: tuple[float, float] | None = None,
    periods: Sequence[Sequence[int]] = (),
    own: ControllerClock | None = None,
    off: Sequence[tuple[int, int, int]] = (),
    natives: Sequence[int] | None = None,
) -> list[Any]:
    """``controller._violations`` of the events of a summarizing walk, as
    violations and the copies of a loop's (``finished`` lists them), with
    *sample_rate* (made, plays) when a line plays samples made for another
    rate (``controller._sample_rate_violations``), and *periods* the line
    periods of its unsynced loops; in the cycles of *own*, when the walk
    counted them, with *off* its events off the grid, and *natives* by event
    the runtime qudit whose own frequency its tone plays, or -1.
    ``UnsyncedError`` when a check pairs events of lines of different
    periods."""
    tree, chains = _tree(len(events), repeats, periods)
    budget = _Budget()
    found: list[Any] = []
    if off:
        assert own is not None
        found.extend(_off_grid(tree, events, lines, own, off))
    timed = _keep(
        tree, lambda i: events[i]["kind"] in _PLAYED and events[i]["end_cycles"] is not None
    )
    most = limits.max_events
    if most is not None:
        total = _count(timed)
        if limits._runs_loops:
            held = sum(1 for i in timed if type(i) is int and not any(events[i]["iteration"]))
            if held > most:
                found.append(
                    f"{held} drives and readouts, a rolled loop's body counted once; the "
                    f"controller holds at most {most}"
                )
        elif total > most:
            text = f"{total} drives and readouts; the controller plays at most {most}"
            loops = _rolled_loops(timed, events)
            if loops:
                grown = "; ".join(
                    f"the loop of {trips} iterations from cycle {start} plays {extra} more"
                    for trips, start, extra in loops
                )
                text += (
                    ". It does not support 'loop', so rolled loops are unrolled for it: " + grown
                )
            found.append(text)
    issue, gap = limits.issue_cycles, limits.issue_cycles + limits.min_wait_cycles
    num, den = _samples_per_own_cycle(limits, lowered.target.clock_ghz)
    on_lines = sorted({events[i]["line"] for i in _events_of(timed)})
    shared = limits.waveform_memory is WaveformMemoryScope.SHARED
    stored: dict[Any, int] = {}
    stored_each = 0
    unknown_stored = 0

    def spacing(
        previous: Mapping[str, Any], dp: int, event: Mapping[str, Any], de: int
    ) -> _Text | None:
        distance = event["start_cycles"] + de - previous["start_cycles"] - dp
        if distance == issue or distance >= gap:
            return None
        return _Text(
            f"line {event['line']}: {event['kind']} at ",
            event["start_cycles"] + de,
            f" starts {distance} cycles after the {previous['kind']} at ",
            previous["start_cycles"] + dp,
            f"; the controller needs exactly {issue} or at least {gap}",
        )

    for line in on_lines:
        items = _keep(timed, lambda i: events[i]["line"] == line)  # noqa: B023
        found.extend(
            _Pairs(events, spacing, budget, (line,)).walk([(n, 0, ()) for n in items], None)[0]
        )
        templates = list(_events_of(items))
        if limits.tones_per_line is not None:
            tones = {
                events[i]["frequency_hz"]
                for i in templates
                if events[i].get("frequency_hz") is not None
            }
            if len(tones) > limits.tones_per_line:
                found.append(
                    f"line {line}: {len(tones)} tone frequencies; the controller holds "
                    f"{limits.tones_per_line} per line"
                )
            elif unresolved is not None:
                unknown = _keep(items, lambda i: events[i].get("frequency_hz") is None)
                if natives is None:
                    count = _count(unknown)
                else:
                    count = len({natives[i] for i in _events_of(unknown) if natives[i] >= 0})
                    count += _count(_keep(unknown, lambda i: natives[i] < 0))
                if len(tones) + count > limits.tones_per_line:
                    unresolved.append(f"line {line}: tone frequencies")
        if limits.max_amplitude is not None:
            found.extend(_amplitudes(lines[line], items, events, limits.max_amplitude, unresolved))
        if limits.waveform_samples is not None:
            shapes = stored if shared else {}
            unknown_samples = 0
            each = 0
            for i in templates:
                e = events[i]
                key = shape_key(e.get("waveform"))
                samples = -(-e["duration_cycles"] * num // den)
                if key is not None:
                    shapes[(key, e["duration_cycles"])] = samples
                else:
                    copies = _count(_keep(items, lambda k: k == i))  # noqa: B023
                    if unresolved is not None:
                        unknown_samples += copies * samples
                    else:
                        each += copies * samples
            used = sum(shapes.values()) + each
            if not shared and used > limits.waveform_samples:
                found.append(
                    f"line {line}: its waveforms take {used} samples; the controller's "
                    f"waveform memory holds {limits.waveform_samples}"
                )
            elif not shared and unresolved is not None:
                if used + unknown_samples > limits.waveform_samples:
                    unresolved.append(f"line {line}: waveform memory")
            stored_each += each
            unknown_stored += unknown_samples
    if shared and limits.waveform_samples is not None:
        used = sum(stored.values()) + stored_each
        if used > limits.waveform_samples:
            found.append(
                f"its waveforms take {used} samples; the controller's shared waveform "
                f"memory holds {limits.waveform_samples}"
            )
        elif unresolved is not None and used + unknown_stored > limits.waveform_samples:
            unresolved.append("waveform memory")
    if limits.acquirers:
        acquirers = limits.acquirers
        hold = limits.acquirer_hold_cycles

        def acquired(
            previous: Mapping[str, Any], dp: int, event: Mapping[str, Any], de: int
        ) -> _Text | None:
            free = previous["end_cycles"] + dp + hold
            if event["start_cycles"] + de >= free:
                return None
            return _Text(
                f"acquirer {acquirers[event['qudit']]}: the readout of qudit {event['qudit']} at ",
                event["start_cycles"] + de,
                " starts before the acquirer is free at ",
                free,
                f" (the readout of qudit {previous['qudit']} at ",
                previous["start_cycles"] + dp,
                ")",
            )

        readouts = _keep(
            timed, lambda i: events[i]["kind"] == "readout" and events[i]["qudit"] in acquirers
        )
        for acquirer in sorted({acquirers[events[i]["qudit"]] for i in _events_of(readouts)}):
            items = _keep(readouts, lambda i: acquirers[events[i]["qudit"]] == acquirer)  # noqa: B023
            joint = {events[i]["line"] for i in _events_of(items)}
            found.extend(
                _Pairs(events, acquired, budget, joint).walk([(n, 0, ()) for n in items], None)[0]
            )
    if limits.feedback_latency_cycles:
        found.extend(
            _Feedback(events, chains, limits.feedback_latency_cycles, budget).walk(timed, {})
        )
    if not limits.answer_timeouts:
        functions = {fn.name: fn for fn in lowered.classical}

        def timeout(i: int) -> _Text | None:
            event = events[i]
            if event["kind"] != "answer":
                return None
            fn = functions.get(event["function"])
            if event["wait"] != "timeout" and (fn is None or fn.latency_mode != "timeout"):
                return None
            return _Text(
                f"the wait for {event['function']} at ",
                event["start_cycles"],
                " has a timeout; the controller cannot stop waiting for an answer, so take it "
                "with result() (no timeout)",
            )

        found.extend(_each_event(tree, timeout))
    if limits.computes:
        found.extend(resources(lowered, limits).violations)
    if sample_rate is not None:
        from .controller import _has_samples

        made_ghz, plays_ghz = sample_rate

        def made_for(i: int) -> _Text | None:
            e = events[i]
            if e["kind"] not in _PLAYED or not _has_samples(e.get("waveform")):
                return None
            return _Text(
                f"line {e['line']}: the {e['kind']} at ",
                e["start_cycles"],
                f" plays samples made for {made_ghz:g} GHz, and the controller's lines play "
                f"{plays_ghz:g} GHz",
            )

        found.extend(_each_event(tree, made_for))
    return found


class _OnProgramClock:
    """A violation stated in program cycles of an event whose copies are
    shifted in a controller's own cycles: ``make`` of its start, ``at``,
    shifted."""

    __slots__ = ("make", "at", "num", "den")

    def __init__(self, make: Callable[[int], str], at: int, ratio: Fraction) -> None:
        self.make, self.at = make, at
        self.num, self.den = ratio.numerator, ratio.denominator

    def render(self, shift: int) -> str:
        # The periods are whole own cycles of whole timesteps, so a shift
        # is whole program cycles.
        return self.make(self.at + shift * self.den // self.num)


def _off_grid(
    tree: list[Any],
    events: Sequence[Mapping[str, Any]],
    lines: Sequence[Mapping[str, Any]],
    own: ControllerClock,
    off: Sequence[tuple[int, int, int]],
) -> list[Any]:
    """``_clocks.off_grid_messages`` of every copy of the events of *tree*
    that *off* has (index, start, duration) in program cycles."""
    from ._clocks import _off_grid as message

    at = {index: (start, duration) for index, start, duration in off}

    def make(i: int) -> _OnProgramClock | None:
        found = at.get(i)
        if found is None:
            return None
        start, duration = found
        line, kind = events[i]["line"], events[i]["kind"]
        return _OnProgramClock(
            lambda s: message(own, lines[line], line, kind, s, duration), start, own.ratio
        )

    return _each_event(tree, make)


def _amplitudes(
    line: Mapping[str, Any],
    items: list[Any],
    events: Sequence[Mapping[str, Any]],
    most: float,
    unresolved: list[str] | None,
) -> list[Any]:
    """``controller._over_amplitude`` of the events of *items*."""
    noted = unresolved is None

    def over(i: int) -> _Text | None:
        nonlocal noted
        event = events[i]
        tree = event.get("waveform")
        peak = _peak_amplitude(tree)
        if peak is None and tree is not None and not noted:
            assert unresolved is not None
            unresolved.append(f"line {line['line']}: amplitude")
            noted = True
        if peak is None or peak <= most:
            return None
        return _Text(
            f"line {line['line']} ({line['role']} of qudit {line['qudit']}): the "
            f"{event['kind']} at ",
            event["start_cycles"],
            f" peaks at amplitude {peak:.6g}; the controller plays at most {most:g} "
            "(ControllerLimits.max_amplitude)",
        )

    return _each_event(items, over, (line["line"],))


class _Feedback:
    """The events that start before the readouts they depend on are ready
    (``ControllerLimits.feedback_latency_cycles``). A copy of a loop depends
    on the copies of the readouts in it, or in the copy before it, as the
    first copy does; on a readout before the loop only a few first copies
    can be too early, and those are checked one by one."""

    def __init__(
        self,
        events: Sequence[Mapping[str, Any]],
        chains: Sequence[tuple[_Rep, ...]],
        latency: int,
        budget: _Budget,
    ) -> None:
        self.events, self.chains, self.latency, self.budget = events, chains, latency, budget

    def apart(self, i: int, ref: int) -> tuple[list[_Rep], int]:
        """The loops event *i* is in and readout *ref* is not, whose copy
        decides how far apart the two are, and the cycles the readout's
        copy (the last of each loop it is in alone) starts after its first."""
        mine, theirs = self.chains[i], self.chains[ref]
        line, readout = self.events[i]["line"], self.events[ref]["line"]
        separate = []
        for rep in mine:
            if rep.lines is not None:
                _period(rep, (line, readout))
            if rep in theirs:
                continue
            if rep.begin - (rep.end - rep.begin) <= ref < rep.begin:
                continue
            separate.append(rep)
        later = sum(
            (rep.copies - 1) * _period(rep, (readout,)) for rep in theirs if rep not in mine
        )
        return separate, later

    def period(self, rep: _Rep, i: int) -> int:
        """The period of *rep*'s copies of event *i*."""
        return _period(rep, (self.events[i]["line"],))

    def refs(self, i: int) -> Iterator[tuple[int, Mapping[str, Any]]]:
        for condition in self.events[i]["conditions"]:
            for ref in condition["readouts"]:
                readout = self.events[ref]
                if readout["end_cycles"] is not None:
                    yield ref, readout

    def walk(self, nodes: list[Any], fixed: dict[_Rep, int | None]) -> list[Any]:
        parts: list[Any] = []
        for node in nodes:
            if type(node) is int:
                parts.extend(self.late(node, fixed))
                continue
            rep = node.orig
            head = min(rep.copies, self.head(node, rep, fixed))
            if head:
                self.budget.spend(head)
            for j in range(head):
                body = self.walk(node.body, {**fixed, rep: j})
                if body:
                    parts.append(_Copies(body, j, j + 1, self.body_period(node, rep)))
            if head < rep.copies:
                body = self.walk(node.body, {**fixed, rep: None})
                if body:
                    parts.append(_Copies(body, head, rep.copies, self.body_period(node, rep)))
        return parts

    def body_period(self, node: _Rep, rep: _Rep) -> int:
        """The period of *rep*'s copies of the events of *node* that
        depend on readouts."""
        if rep.lines is None:
            return rep.period
        lines = {
            self.events[i]["line"] for i in _events_of(node.body) if self.events[i]["conditions"]
        }
        return _period(rep, lines)

    def head(self, node: _Rep, rep: _Rep, fixed: Mapping[_Rep, int | None]) -> int:
        """The first copies of *rep* that may start too early after a readout
        outside it."""
        copies = 0
        for i in _events_of(node.body):
            start = self.events[i]["start_cycles"]
            for ref, readout in self.refs(i):
                separate, later = self.apart(i, ref)
                if rep not in separate:
                    continue
                shift = 0
                for other in separate:
                    if other in fixed:
                        j = fixed[other]
                        if j is None:
                            break
                        shift += j * self.period(other, i)
                else:
                    gap = readout["end_cycles"] + later + self.latency - start - shift
                    if gap > 0:
                        period = self.period(rep, i)
                        copies = max(
                            copies,
                            math.ceil(gap / period) if period > 0 else rep.copies,
                        )
        return copies

    def late(self, i: int, fixed: Mapping[_Rep, int | None]) -> list[_Text]:
        event = self.events[i]
        start = event["start_cycles"]
        found = []
        for ref, readout in self.refs(i):
            separate, later = self.apart(i, ref)
            shift = 0
            for rep in separate:
                j = fixed.get(rep)
                if j is None:
                    break
                shift += j * self.period(rep, i)
            else:
                end = readout["end_cycles"] - shift + later
                ready = end + self.latency
                if start < ready:
                    found.append(
                        _Text(
                            f"line {event['line']}: the {event['kind']} at ",
                            start,
                            " depends on the readout ending at ",
                            end,
                            f"; the controller needs {self.latency} cycles of feedback "
                            "latency (start at ",
                            ready,
                            " or later; schedule with rcp and "
                            "readout_latency_cycles >= feedback_latency_cycles)",
                        )
                    )
        return found


def loops_of(
    repeats: Sequence[Sequence[int]], periods: Sequence[Sequence[int]] | None
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """The loops a summarizing walk's *repeats* and line *periods* keep
    rolled, whatever clock it counts their periods in."""
    return (
        [(*r[:3], r[3] < 0, *r[4:]) for r in repeats],
        [tuple(p[:2]) for p in periods or ()],
    )


def summarized_check(
    lowered: Lowered,
    limits: ControllerLimits | None,
    system: ControlSystem | None,
    clocks: ClockPlan | None,
    slots: bool,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[dict[str, Any]],
    repeats: list[tuple[int, ...]],
    periods: list[tuple[int, int, int]] | None,
    off: Sequence[tuple[int, int, int]],
) -> tuple[Sequence[str], bool] | None:
    """``controller._compile_check`` of the events of a summarizing walk with
    *repeats* (and line *periods*); ``None`` when the loops must be walked
    with fewer summarized."""
    machine = lowered.options.machine
    natives = None
    if slots and limits is not None and limits.tones_per_line is not None:
        with lowered._lock:
            natives, again, moved = lowered._module.controller_native_tones(
                True, periods is not None
            )
        if loops_of(repeats, periods) != loops_of(again, moved):
            return None
    rates = (
        None if limits is None or limits is machine.controller else _sample_rates(lowered, limits)
    )
    pending: list[str] | None = [] if lowered.parameters or slots else None
    if system is not None:
        parts = system._summarized(
            lowered, lines, events, repeats, clocks, off, pending, natives, rates
        )
        if parts is None:
            return None
        found: Sequence[str] = finished(parts)
    else:
        assert limits is not None
        try:
            found = violations(
                lowered,
                events,
                repeats,
                limits,
                lines,
                pending,
                rates,
                periods or (),
                _native_clock(clocks),
                off,
                natives,
            )
        except UnsyncedError:
            return None
    if slots and limits is not None and limits.acquirers:
        assert pending is not None
        # Acquirers are given per carrier, which each call binds.
        pending.append("acquirers")
    return found, not pending
