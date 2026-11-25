# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How many pulses can play at once when parts of the program slide.

A wait whose length is computed at run time is scheduled at its lowest
length, and what follows it on its lines (a segment, see
``compiler.Segment``) starts later at run time by up to the wait's range.
Lines the wait does not hold back keep their times, so two pulses the
schedule places apart can meet.

The segments are a tree: each follows the one its wait ended. Pulses of one
segment keep their scheduled distances, and the segments that follow a
segment slide against it by 0 to their wait's range, each on its own. So
with ``own(s, t)`` the pulses of segment ``s`` playing at cycle ``t`` as
scheduled, the most that can play at once at ``t``, in and after ``s``, is

    most(s, t) = own(s, t) + sum over the c that follow s of
                 max over 0 <= d <= range(c) of most(c, t - d)

and ``most(0, t)`` is the most over every run, at the cycle ``t`` of a run
whose segments on the way all start as scheduled. The lengths of two waits
are taken to be independent; two that are computed from one answer move
together, and are counted as if they did not.

A count over time is a step function: ``[(cycle, count), ...]`` by cycle,
each count holding up to the next cycle, 0 before the first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from .compiler import ScheduledKernel

# The waits a message names, at most.
NAMED = 3

# The cycle a count that holds for good ends at.
_NEVER = float("inf")

# A pulse: (start, end, event).
Pulse = tuple[int, int, Mapping[str, Any]]
Steps = list[tuple[int, int]]


def own(pulses: Sequence[Pulse], together: Callable[[list[Pulse]], int] | None = None) -> Steps:
    """The pulses of *pulses* (sorted by start, of one segment) playing at
    once as scheduled, over time; a pulse ending at a cycle is over for one
    starting then. *together* counts those of a set that play in one run,
    where it is not all of them (arms of a branch)."""
    steps: Steps = []
    active: list[tuple[int, int]] = []
    k = 0
    while k < len(pulses) or active:
        if active and (k == len(pulses) or active[0][0] <= pulses[k][0]):
            at = active[0][0]
        else:
            at = pulses[k][0]
        while active and active[0][0] <= at:
            heappop(active)
        while k < len(pulses) and pulses[k][0] == at:
            heappush(active, (pulses[k][1], k))
            k += 1
        count = len(active)
        if together is not None and count > 1:
            count = together([pulses[i] for _, i in sorted(active, key=_place)])
        if not steps or steps[-1][1] != count:
            steps.append((at, count))
    return steps


def _place(item: tuple[int, int]) -> int:
    return item[1]


def by_start(pulse: Pulse) -> tuple[int, int]:
    """The order pulses are counted in: by start, then end."""
    return pulse[0], pulse[1]


def slid(steps: Steps, most: int | None) -> Steps:
    """The most *steps* reaches at each cycle when it starts 0 to *most*
    cycles later (``None``: any number)."""
    if not steps or most == 0:
        return steps
    if most is None:
        out: Steps = []
        highest = 0
        for at, count in steps:
            if count > highest:
                highest = count
                out.append((at, count))
        return out
    # Each count holds from its cycle to the next one plus *most*; the last
    # one, when it is not 0 (after a slide by any number), for good.
    spans = [
        (at, steps[k + 1][0] + most if k + 1 < len(steps) else _NEVER, count)
        for k, (at, count) in enumerate(steps)
        if count
    ]
    edges = sorted(
        {int(edge) for start, end, _ in spans for edge in (start, end) if edge != _NEVER}
    )
    out = []
    held: list[tuple[int, float]] = []
    k = 0
    for cycle in edges:
        at = cycle
        while k < len(spans) and spans[k][0] <= at:
            heappush(held, (-spans[k][2], spans[k][1]))
            k += 1
        while held and held[0][1] <= at:
            heappop(held)
        count = -held[0][0] if held else 0
        if not out or out[-1][1] != count:
            out.append((at, count))
    return out


def added(parts: Sequence[Steps]) -> Steps:
    """The sum of the step functions *parts*."""
    parts = [part for part in parts if part]
    if len(parts) == 1:
        return parts[0]
    changes: dict[int, int] = {}
    for part in parts:
        before = 0
        for at, count in part:
            changes[at] = changes.get(at, 0) + count - before
            before = count
    out: Steps = []
    total = 0
    for at in sorted(changes):
        if changes[at]:
            total += changes[at]
            out.append((at, total))
    return out


def most_at_once(
    pulses: Mapping[int, Sequence[Pulse]],
    after: Sequence[int | None],
    ranges: Sequence[int | None],
    together: Callable[[list[Pulse]], int] | None = None,
    blocks: bool = True,
) -> Steps:
    """The most of *pulses* (by segment, each sorted by start) that can
    play at once over time (see the module). Segment ``k`` follows segment
    ``after[k]`` (``None`` for segment 0) and starts 0 to ``ranges[k]``
    cycles later than scheduled against it (``None``: any number).

    With *blocks*, segments that follow one another alone, each one's
    pulses after the one before's (the trips of a loop with a wait each),
    are counted as one that slides by the sum of their ranges: every trip
    as far as the last can. That never counts fewer, and it keeps the work
    in proportion to the pulses however many such segments there are."""
    # The segments that hold pulses, and those they follow.
    needed = set()
    for segment in pulses:
        at: int | None = segment
        while at is not None and at not in needed:
            needed.add(at)
            at = after[at]
    # A segment that cannot slide plays with the one it follows.
    home: dict[int, int] = {}
    following: dict[int, list[int]] = {}
    held: dict[int, Sequence[Pulse]] = {}
    joined: set[int] = set()
    for segment in sorted(needed):
        before = after[segment]
        if before is None:
            home[segment] = segment
        elif ranges[segment] == 0:
            home[segment] = home[before]
        else:
            home[segment] = segment
            following.setdefault(home[before], []).append(segment)
        if segment in pulses:
            at = home[segment]
            if at in held:
                held[at] = [*held[at], *pulses[segment]]
                joined.add(at)
            else:
                held[at] = pulses[segment]
    for at in joined:
        held[at] = sorted(held[at], key=by_start)
    needed = {segment for segment in needed if home[segment] == segment}
    slides: dict[int, int | None] = {segment: ranges[segment] for segment in needed}
    if blocks:
        _merge(needed, following, held, slides)
    # A segment follows one of a lower number: the last first.
    found: dict[int, Steps] = {}
    for segment in sorted(needed, reverse=True):
        parts = [own(held[segment], together)] if held.get(segment) else []
        for later in following.get(segment, ()):
            parts.append(slid(found.pop(later), slides[later]))
        found[segment] = added(parts)
    return found.get(0, [])


def _merge(
    needed: set[int],
    following: dict[int, list[int]],
    held: dict[int, Sequence[Pulse]],
    slides: dict[int, int | None],
) -> None:
    """Make each run of segments that follow one another alone one segment
    (see ``most_at_once``), in place: the first of the run holds the pulses
    of all and slides by the sum of their ranges. A segment joins the run
    only when its pulses start at or after the end of those before it in
    the run, which sliding later keeps so; what follows the run slides
    against all of it as far as against its first segment."""
    for head in sorted(needed):
        if head not in needed or len(following.get(head, ())) != 1:
            continue
        first = last = following[head][0]
        block = [*held.get(first, ())]
        end = max((pulse[1] for pulse in block), default=None)
        total = slides[first]
        # How far the last of the run can slide against its first.
        within: int | None = 0
        while len(following.get(last, ())) == 1:
            follower = following[last][0]
            theirs = held.get(follower, ())
            if theirs:
                if end is not None and theirs[0][0] < end:
                    break
                end = max(pulse[1] for pulse in theirs)
                block.extend(theirs)
            more = slides[follower]
            total = None if total is None or more is None else total + more
            within = None if within is None or more is None else within + more
            following.pop(last)
            needed.discard(follower)
            last = follower
        if last != first:
            held[first] = block
            slides[first] = total
            rest = following.pop(last, None)
            if rest is not None:
                following[first] = rest
                # What follows the run follows its last segment: against a
                # pulse earlier in the run it slides that much further.
                for later in rest:
                    more = slides[later]
                    slides[later] = None if within is None or more is None else more + within


def first_over(steps: Steps, limit: int) -> tuple[int, int, int] | None:
    """Where *steps* first passes *limit*: the cycle, the count then, and
    the most it reaches; ``None`` when it never does."""
    first = None
    for at, count in steps:
        if count > limit and first is None:
            first = (at, count)
    if first is None:
        return None
    return first[0], first[1], max(count for _, count in steps)


def one_timeline(events: Sequence[Mapping[str, Any]]) -> bool:
    """Whether *events* (those one limit counts) are on lines that wait
    together, as far as their segments tell: every segment but the first
    is of one timeline, so lines with events in one such segment wait
    together, and what waits together plays as it is placed. A line with
    events only in the first segment is taken to wait with no other."""
    group: dict[int, int] = {}

    def of(line: int) -> int:
        while group.setdefault(line, line) != line:
            line = group[line]
        return line

    lines: dict[int, int] = {}
    for event in events:
        line = of(event["line"])
        segment = event["segment"]
        if segment:
            first = lines.setdefault(segment, line)
            if first != line:
                group[line] = of(first)
    return len({of(line) for line in group}) <= 1


class Over(NamedTuple):
    """Pulses that can play at once past a limit: the first cycle they can,
    how many then and at most, and the waits that let them (a clause for a
    message)."""

    first: int
    at_first: int
    most: int
    cause: str


def over(
    lowered: ScheduledKernel,
    played: Sequence[Pulse],
    limit: int,
    lines: Sequence[Mapping[str, Any]],
    together: Callable[[list[Pulse]], int] | None = None,
) -> Over | None:
    """Where *played* (pulses of *lowered* that one limit counts, sorted by
    start) can be more than *limit* at once in some run."""
    segments, after = lowered._segments()
    ranges: list[int | None] = [0]
    for segment, before in zip(segments[1:], after[1:], strict=True):
        most, least = segment.max_shift_cycles, segments[before or 0].max_shift_cycles
        ranges.append(None if most is None or least is None else most - least)
    pulses: dict[int, list[Pulse]] = {}
    for pulse in played:
        pulses.setdefault(pulse[2]["segment"], []).append(pulse)

    def wait(k: int) -> str:
        segment = segments[k]
        if segment.opened_by == "answer":
            on = f"the blocking wait on link {segment.line}"
        else:
            line = lines[segment.line] if segment.line >= 0 else None
            on = f"the wait on {line['role'][0]}{line['qudit']}" if line else "the wait"
        return f"{on} that ends at cycle {segment.start_cycles}"

    # The waits whose range moves a pulse counted: those on the way to it.
    seen: set[int] = set()
    for k in pulses:
        at: int | None = k
        while at and at not in seen:
            seen.add(at)
            at = after[at]
    slid = sorted(k for k in seen if ranges[k] != 0)
    found = first_over(most_at_once(pulses, after, ranges, together), limit)
    if found is None:
        return None
    named = []
    for k in slid[:NAMED]:
        later = "any number of" if ranges[k] is None else f"up to {ranges[k]}"
        named.append(f"{wait(k)} ({later} cycles later)")
    if len(slid) > NAMED:
        named.append(f"{len(slid) - NAMED} more such waits")
    cause = (
        f"what follows {', '.join(named)} starts later in a run than it is placed, each "
        "wait's length counted as independent of the others'"
    )
    return Over(*found, cause)
