# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared lines: a ``DriveLine`` or ``ReadoutLine`` that reaches several
qudits and plays up to ``max_concurrent_tones`` tones at once.

A kernel takes one of its tones with ``get_drive_line(q)`` or
``get_readout_line(q)`` of any qudit it reaches, so the program, its IR and
the simulator see one line per qudit (a handle), as on dedicated lines;
the lines of one controller that share a physical line are grouped here,
for the controller checks (tones at once, tone frequencies and waveform
memory per physical line) and the exports. Only a control system with a
shared line loads this."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from ._errors import InvalidTypeError, InvalidValueError, UnsupportedError, _ranges

if TYPE_CHECKING:
    from ._errors import Problem
    from ._system import Controller, ControlSystem, _Line
    from .compiler import ScheduledKernel


def tone_count(name: str, qudits: Any, given: Any, most: int = 1) -> int:
    """*given*, the ``max_concurrent_tones`` of a line of *name* reaching
    *qudits* (*most* of them), checked."""
    if isinstance(given, bool) or not isinstance(given, int):
        raise InvalidTypeError(f"{name}.max_concurrent_tones is an int, got {given!r}")
    if not 1 <= given <= most:
        if most == 1:
            raise InvalidValueError(
                f"{name}({qudits}) reaches one qudit, so it plays one tone at once; got "
                f"max_concurrent_tones={given}"
            )
        raise InvalidValueError(
            f"{name}({list(qudits)}) reaches {most} qudits, so it plays 1 to {most} tones "
            f"at once; got max_concurrent_tones={given}"
        )
    return int(given)


def line(cls: type[_Line], qudits: Iterable[Any], delay_ns: Any, tones: Any) -> Any:
    """A *cls* line reaching *qudits* (a sequence of carrier indices), with
    its delay and ``max_concurrent_tones`` checked: a dedicated line when
    it reaches one."""
    from ._system import _delay_ns

    name = cls.__name__
    given = list(qudits)
    for qudit in given:
        if isinstance(qudit, bool) or not isinstance(qudit, int):
            raise InvalidTypeError(
                f"{name} reaches qudits by carrier index, got {qudit!r} in {given!r}"
            )
        if qudit < 0:
            raise InvalidValueError(f"{name} has a negative carrier index {qudit}")
    if not given:
        raise InvalidValueError(f"{name} reaches no qudit; give it at least one")
    reached = tuple(map(int, given))
    if len(set(reached)) != len(reached):
        raise InvalidValueError(f"{name}({given}) lists a qudit twice")
    if len(reached) == 1:
        return cls(reached[0], delay_ns, max_concurrent_tones=tones)
    tones = tone_count(name, reached, tones, len(reached))
    if delay_ns.__class__ is not float or not 0.0 <= delay_ns < math.inf:
        delay_ns = _delay_ns(delay_ns, f"{name}({given}).delay_ns")
    # Item 1 is the tones, never 0: _lines_data and the delay filters see it.
    return tuple.__new__(cls, (reached, tones, delay_ns))


def text(line: _Line) -> str:
    """The repr of a shared *line*."""
    qudits, tones, delay = line
    return (
        f"{type(line).__name__}({list(qudits)}{f', delay_ns={delay!r}' if delay else ''}"
        f"{f', max_concurrent_tones={tones}' if tones != 1 else ''})"
    )


def data(line: _Line) -> dict[str, Any]:
    """A shared *line* as a system file holds it, without the defaults."""
    qudits, tones, delay = line
    found: dict[str, Any] = {"qudits": list(qudits)}
    if tones != 1:
        found["max_concurrent_tones"] = tones
    if delay:
        found["delay_ns"] = delay
    return found


def reached_once(lines: Sequence[_Line], cls: type[_Line]) -> None:
    """Raises when two of one controller's *lines* reach one qudit."""
    first: dict[int, _Line] = {}
    for found in lines:
        for qudit in found.qudits:
            seen = first.setdefault(qudit, found)
            if seen is not found:
                if len(seen) == 2 and len(found) == 2:
                    raise InvalidValueError(
                        f"Controller.{cls._role}_lines lists a carrier twice: {qudit}"
                    )
                raise InvalidValueError(
                    f"qudit {qudit} is reached by two {cls._role} lines, {seen!r} and "
                    f"{found!r}; choosing between lines is not supported yet: reach each "
                    f"qudit by one {cls._role} line"
                )


# ── A control system with a shared line ─────────────────────────────────────


def _role_lines(controller: Controller) -> Iterator[tuple[str, int, _Line]]:
    """Each line of *controller*: its role, its position among the lines of
    that role, and the line."""
    for role, lines in (("drive", controller.drive_lines), ("readout", controller.readout_lines)):
        for position, each in enumerate(cast("Sequence[_Line]", lines)):
            yield role, position, each


def tables(
    controllers: Sequence[Controller],
) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], float]]:
    """``ControlSystem._line_owners`` and the delays of its delayed lines, by
    (role, qudit), of *controllers*: every qudit of a shared line gets its
    controller and its delay."""
    owners: dict[tuple[str, int], int] = {}
    delays: dict[tuple[str, int], float] = {}
    for index, controller in enumerate(controllers):
        for role, _, each in _role_lines(controller):
            for qudit in each.qudits:
                owners[(role, qudit)] = index
                if each.delay_ns:
                    delays[(role, qudit)] = each.delay_ns
    return owners, delays


def reach_problems(system: ControlSystem, prefix: str) -> list[Problem]:
    """The qudits of *system* that two lines of a kind reach: on two
    controllers, or where one of the lines is shared (choosing between lines
    is not supported yet)."""
    from ._errors import Problem

    problems = []
    first: dict[tuple[str, int], tuple[int, _Line]] = {}
    for index, controller in enumerate(system.controllers):
        for role, position, each in _role_lines(controller):
            for carrier in each.qudits:
                seen = first.setdefault((role, carrier), (index, each))
                if seen[1] is each:
                    continue
                other, line = seen
                path = f"{prefix}controllers[{index}].{role}_lines[{position}]"
                if len(line) > 2 or len(each) > 2:
                    problems.append(
                        Problem(
                            f"qudit {carrier} is reached by two {role} lines, {line!r} on "
                            f"controllers[{other}] and {each!r}; choosing between lines is "
                            "not supported yet",
                            path=path,
                            value=carrier,
                            fix=f"reach each qudit by one {role} line",
                        )
                    )
                else:
                    problems.append(
                        Problem(
                            f"qudit {carrier}'s {role} line is also on controllers[{other}]",
                            path=path,
                            value=carrier,
                            fix="a line is on one controller",
                        )
                    )
    return problems


def wiring_problems(controller: Controller, path: str, carriers: set[int]) -> list[Problem]:
    """The lines of *controller* (at *path*) reaching qudits not in
    *carriers*, the quantum system's."""
    from ._errors import Problem

    return [
        Problem(
            f"qudit {carrier} is not in the quantum system",
            path=f"{path}.{role}_lines[{position}]",
            value=carrier,
            fix=f"it has qudits {_ranges(sorted(carriers))}",
        )
        for role, position, each in _role_lines(controller)
        for carrier in each.qudits
        if carrier not in carriers
    ]


def undelayed(controllers: Sequence[Controller]) -> list[Controller]:
    """*controllers* with every line undelayed (``ControlSystem._undelayed``)."""
    import dataclasses

    def plain(lines: Sequence[Any]) -> list[Any]:
        return [
            each[0]
            if len(each) == 2
            else type(each)(each[0], max_concurrent_tones=each.max_concurrent_tones)
            for each in lines
        ]

    return [
        dataclasses.replace(
            c, drive_lines=plain(c.drive_lines), readout_lines=plain(c.readout_lines)
        )
        for c in controllers
    ]


# ── The physical lines of a control system ──────────────────────────────────


class Physical:
    """A shared line of a control system: its line, its controller's index
    and the label checks name it by."""

    __slots__ = ("line", "owner", "label", "role", "qudits", "tones")

    def __init__(self, found: _Line, owner: int, position: int) -> None:
        self.line = found
        self.owner = owner
        self.role = found._role
        self.qudits = found.qudits
        self.tones = found.max_concurrent_tones
        self.label = (
            f"the {self.role} line of qudits {_ranges(list(self.qudits))} "
            f"(controllers[{owner}].{self.role}_lines[{position}])"
        )


def physical(system: ControlSystem) -> Mapping[tuple[str, int], Physical] | None:
    """By (role, qudit), the shared line of *system* reaching it; ``None``
    when every line is dedicated. Made once per system."""
    cache = system.__dict__
    try:
        return cache["_shared_lines"]  # type: ignore[no-any-return]
    except KeyError:
        pass
    found: dict[tuple[str, int], Physical] = {}
    for owner, controller in enumerate(system.controllers):
        for lines in (controller.drive_lines, controller.readout_lines):
            for position, each in enumerate(cast("Sequence[_Line]", lines)):
                if len(each) > 2:
                    shared = Physical(each, owner, position)
                    for qudit in shared.qudits:
                        found[(shared.role, qudit)] = shared
    result = found or None
    object.__setattr__(system, "_shared_lines", result)
    return result


# ── Checks ──────────────────────────────────────────────────────────────────

# The format /1 kinds of events that play a tone.
_PLAYED = frozenset({"drive", "readout"})


def _groups(
    shared: Mapping[tuple[str, int], Physical], lines: Sequence[Mapping[str, Any]]
) -> list[tuple[Physical, frozenset[int]]]:
    """Each shared line the program's format /1 *lines* (its handles) play,
    with those handles' indices, in order of first use."""
    found: dict[Physical, set[int]] = {}
    for entry in lines:
        line = shared.get((entry["role"], entry["qudit"]))
        if line is not None:
            found.setdefault(line, set()).add(entry["line"])
    return [(line, frozenset(handles)) for line, handles in found.items()]


def _timed(event: Mapping[str, Any], handles: frozenset[int]) -> bool:
    return (
        event["kind"] in _PLAYED
        and event["line"] in handles
        and event["start_cycles"] is not None
        and event["end_cycles"] is not None
    )


class _AtOnce:
    """The tones a shared line plays at once past its ``max_concurrent_tones``:
    how often a tone starts with the line full, and the first time."""

    __slots__ = ("line", "count", "first", "playing")

    def __init__(self, line: Physical) -> None:
        self.line = line
        self.count = 0
        self.first: int | None = None
        self.playing: list[Mapping[str, Any]] = []

    def sweep(self, played: Sequence[tuple[int, int, Mapping[str, Any], Any]]) -> None:
        """Count *played* ((start, end, event, copy)), sorted by start; a tone
        ending at a cycle frees the line for one starting then. Events of one
        copy in different arms of a branch never play together."""
        from .controller import _exclusive

        active: list[tuple[int, int, Mapping[str, Any], Any]] = []
        for item in played:
            start, _, event, copy = item
            active = [a for a in active if a[1] > start]
            together = [a for a in active if a[3] != copy or not _exclusive(a[2], event)]
            if len(together) >= self.line.tones:
                if self.first is None or start < self.first:
                    self.first = start
                    self.playing = [a[2] for a in together] + [event]
                self.count += 1
            active.append(item)

    def message(self, lines: Sequence[Mapping[str, Any]]) -> str | None:
        if not self.count:
            return None
        playing = self.playing
        kinds = sorted({e["kind"] for e in playing})
        qudits = ", ".join(str(lines[e["line"]]["qudit"]) for e in playing)
        more = self.count - 1
        again = f", and does so {more} more time{'s' if more > 1 else ''}" if more else ""
        return (
            f"{self.line.label} plays {len(playing)} tones at once from cycle {self.first} "
            f"(the {' and '.join(k + 's' for k in kinds)} of qudits {qudits}){again}; it plays "
            f"at most {self.line.tones} (max_concurrent_tones): play some after the others "
            "end (a sync of their lines between them), or raise its max_concurrent_tones"
        )


def _per_line(
    lowered: ScheduledKernel,
    line: Physical,
    controller: Controller,
    played: Sequence[Mapping[str, Any]],
    unresolved: list[str] | None,
) -> list[str]:
    """What the handles of *line* exceed together, as one physical line: the
    controller's distinct tone frequencies per line and, per line, its
    waveform memory (*played*: their tones, each kept event once)."""
    from .controller import WaveformMemoryScope, _samples_per_own_cycle, shape_key

    found = []
    most = controller.max_frequencies_per_line
    if most is not None:
        tones = {e["frequency_hz"] for e in played if e.get("frequency_hz") is not None}
        if len(tones) > most:
            found.append(
                f"{line.label}: {len(tones)} tone frequencies; the controller holds {most} per line"
            )
        elif unresolved is not None:
            unknown = sum(1 for e in played if e.get("frequency_hz") is None)
            if len(tones) + unknown > most:
                unresolved.append(f"{line.label}: tone frequencies")
    memory = controller.waveform_memory_samples
    if memory is not None and controller.waveform_memory is WaveformMemoryScope.PER_LINE:
        num, den = _samples_per_own_cycle(controller, lowered.quantum_system.clock_ghz)
        shapes: dict[Any, int] = {}
        unknown_samples = 0
        for e in played:
            key = shape_key(e.get("waveform"))
            samples = -(-e["duration_cycles"] * num // den)
            if key is not None:
                shapes[(key, e["duration_cycles"])] = samples
            elif unresolved is not None:
                unknown_samples += samples
            else:
                shapes[("event", e["index"])] = samples
        used = sum(shapes.values())
        if used > memory:
            found.append(
                f"{line.label}: its waveforms take {used} samples; the controller's "
                f"waveform memory holds {memory} per line"
            )
        elif unresolved is not None and used + unknown_samples > memory:
            unresolved.append(f"{line.label}: waveform memory")
    return found


def violations(
    system: ControlSystem,
    lowered: ScheduledKernel,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    unresolved: list[str] | None,
) -> list[str]:
    """What the shared lines of *system* that the program plays (format /1
    *lines* and *events*, every iteration) exceed: their tones at once,
    and per physical line the limits its handles share."""
    shared = physical(system)
    assert shared is not None
    found: list[str] = []
    for line, handles in _groups(shared, lines):
        played = [e for e in events if _timed(e, handles)]
        if len(handles) < 2:
            continue
        at_once = _AtOnce(line)
        at_once.sweep(
            sorted(((e["start_cycles"], e["end_cycles"], e, 0) for e in played), key=_first)
        )
        message = at_once.message(lines)
        if message is not None:
            found.append(message)
        found.extend(_per_line(lowered, line, system.controllers[line.owner], played, unresolved))
    return found


def against(lowered: ScheduledKernel, controller: Controller) -> list[str]:
    """``violations`` of *lowered* on a system of *controller* alone, for
    ``controller.check(lowered, controller)``: every iteration walked."""
    from ._system import _one_controller
    from .controller import _walk

    lines, events, _, _, _ = _walk(lowered, values=False)
    return violations(_one_controller(controller), lowered, lines, events, None)


def _first(item: tuple[int, int, Mapping[str, Any], Any]) -> tuple[int, int]:
    return item[0], item[2]["index"]


def summarized(
    system: ControlSystem,
    lowered: ScheduledKernel,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    repeats: Sequence[Sequence[int]],
    unresolved: list[str] | None,
) -> list[str] | None:
    """``violations`` of the events of a summarizing walk with *repeats* (loops
    of one period): a loop whose copies on a shared line follow each other,
    with nothing else on the line meanwhile, is counted from its first copy;
    any other is played copy by copy. ``None`` past the copies a check plays
    one by one (``_loops._MAX_PLAYED``): walk the loops in full."""
    from . import _loops

    shared = physical(system)
    assert shared is not None
    tree, _ = _loops._tree(events, repeats)
    budget = _loops._Budget()
    found: list[str] = []
    for line, handles in _groups(shared, lines):
        if len(handles) < 2:
            continue
        nodes = _loops._keep(tree, lambda i: _timed(events[i], handles))  # noqa: B023
        at_once = _AtOnce(line)
        try:
            _count_copies(at_once, events, nodes, 0, (), 1, budget)
        except ValueError:
            return None
        message = at_once.message(lines)
        if message is not None:
            found.append(message)
        kept = [events[i] for i in _loops._events_of(nodes)]
        found.extend(_per_line(lowered, line, system.controllers[line.owner], kept, unresolved))
    return found


def _window(node: Any, events: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """The first start and last end of *node*'s events, all its copies."""
    if type(node) is int:
        return events[node]["start_cycles"], events[node]["end_cycles"]
    spans = [_window(child, events) for child in node.body]
    lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    return lo, hi + (node.copies - 1) * node.period


def _count_copies(
    at_once: _AtOnce,
    events: Sequence[Mapping[str, Any]],
    nodes: list[Any],
    shift: int,
    copy: tuple[Any, ...],
    times: int,
    budget: Any,
) -> None:
    """Sweep *nodes* (``_loops._tree`` nodes, shifted by *shift* cycles, in
    the *copy* of the loops around them), each violation counted *times*."""
    items: list[tuple[Any, int, tuple[Any, ...]]] = [(node, shift, copy) for node in nodes]
    while True:
        windows = [_window(node, events) for node, _, _ in items]
        expand = None
        for k, (node, dt, _) in enumerate(items):
            if type(node) is int:
                continue
            body = [_window(child, events) for child in node.body]
            lo, hi = windows[k]
            follows = max(s[1] for s in body) - min(s[0] for s in body) <= node.period
            alone = all(
                w[1] + items[j][1] <= lo + dt or w[0] + items[j][1] >= hi + dt
                for j, w in enumerate(windows)
                if j != k
            )
            if not (follows and alone):
                expand = k
                break
        if expand is None:
            break
        node, dt, path = items[expand]
        budget.spend(node.copies)
        items[expand : expand + 1] = [
            (child, dt + j * node.period, (*path, (id(node.orig), j)))
            for j in range(node.copies)
            for child in node.body
        ]
    played = []
    for node, dt, path in items:
        if type(node) is int:
            event = events[node]
            played.append((event["start_cycles"] + dt, event["end_cycles"] + dt, event, path))
        else:
            # Its copies follow each other with the line otherwise free: each
            # plays what the first does, node.copies times.
            _count_copies(
                at_once, events, node.body, dt, (*path, id(node.orig)), times * node.copies, budget
            )
    before = at_once.count
    at_once.sweep(sorted(played, key=_first))
    at_once.count = before + (at_once.count - before) * times


def refuse_format_1(
    lowered: ScheduledKernel, system: ControlSystem, lines: Sequence[Mapping[str, Any]]
) -> None:
    """Raises when *lowered* on *system* plays a shared line (format /1
    *lines*): format /1 has one qudit per line."""
    shared = physical(system)
    if shared is None:
        return
    for entry in lines:
        found = shared.get((entry["role"], entry["qudit"]))
        if found is not None:
            text = f"format /1 has one qudit per line, and {found.label} reaches several"
            if (lowered._generic or lowered).fine_delays is not None:
                raise UnsupportedError(
                    f"{text}; format /2, which has shared lines, does not take line delays "
                    "yet: a program on delayed shared lines does not export yet"
                )
            raise UnsupportedError(f"{text}: export version 2 (export_to_controller(version=2))")


# ── Format /2 ───────────────────────────────────────────────────────────────

# The /2 event kinds on a line (controller._ON_LINE), and those playing a tone.
_ON_LINE = frozenset({"play", "readout", "wait", "frame_update"})
_TONED = frozenset({"play", "readout"})


class _Placed:
    """An event of format /2 on a handle of a shared line: its handle (the
    line's index in the walk), its first start and end, and whether a rolled
    loop repeats it."""

    __slots__ = ("event", "handle", "start", "end", "last", "looped", "order")

    def __init__(
        self, event: dict[str, Any], handle: int, last: int, looped: bool, order: int
    ) -> None:
        self.event, self.handle, self.looped, self.order = event, handle, looped, order
        self.start = event["start_cycles"]
        self.end = self.start + event["duration_cycles"]
        self.last = last


def _placed(
    events: list[dict[str, Any]], shared: Sequence[Physical | None]
) -> dict[Physical, list[_Placed]]:
    """By shared line, the events of format /2 *events* on its handles
    (*shared*: by handle, its shared line), in program order."""
    found: dict[Physical, list[_Placed]] = {}

    def visit(items: list[dict[str, Any]], extra: int, looped: bool) -> None:
        # extra: the cycles the rolled loops around *items* move their last copy by.
        for event in items:
            if event["kind"] == "loop":
                more = (event["trips"] - 1) * event["period_cycles"]
                visit(event["body"], extra + more, True)
                continue
            line = event.get("line")
            if event["kind"] not in _ON_LINE or line is None:
                continue
            physical_line = shared[line]
            if physical_line is not None:
                last = event["start_cycles"] + event["duration_cycles"] + extra
                placed = found.setdefault(physical_line, [])
                placed.append(_Placed(event, line, last, looped, len(placed)))

    visit(events, 0, False)
    return found


def _lowest_free(busy: list[tuple[int, int]], start: int, slots: int) -> int | None:
    """The lowest slot no interval of *busy* ((end, slot)) holds at *start*,
    those ending by then dropped; ``None`` when all *slots* are held."""
    busy[:] = [(end, slot) for end, slot in busy if end > start]
    held = {slot for _, slot in busy}
    return next((slot for slot in range(slots) if slot not in held), None)


def _slots(line: Physical, placed: list[_Placed]) -> dict[int, int]:
    """By handle, the tone slot of each handle of *line* the program uses,
    when each can keep one for the whole program: handles whose events span
    overlapping stretches get different slots, lowest free first, in the
    order of their first event; ``{}`` when that takes more slots than the
    line has."""
    spans: dict[int, list[int]] = {}
    for p in placed:
        if p.event["kind"] not in _TONED:
            continue
        span = spans.setdefault(p.handle, [p.start, p.last])
        span[0] = min(span[0], p.start)
        span[1] = max(span[1], p.last)
    busy: list[tuple[int, int]] = []
    given: dict[int, int] = {}
    for handle, (start, end) in sorted(spans.items(), key=lambda item: (item[1][0], item[0])):
        slot = _lowest_free(busy, start, line.tones)
        if slot is None:
            return {}
        given[handle] = slot
        busy.append((end, slot))
    return given


def _slots_by_event(line: Physical, placed: list[_Placed]) -> dict[int, int]:
    """By ``id`` of event, the tone slot of each event of *line*, when its
    handles take turns at its slots: greedy interval colouring of its tones
    in start order (then handle order), lowest free slot first."""
    if any(p.looped for p in placed):
        raise UnsupportedError(
            f"format /2 gives each event of a rolled loop one tone slot in every "
            f"iteration, and the handles of {line.label} take turns at its "
            f"{line.tones} tones: give each handle a tone of its own "
            "(max_concurrent_tones of at least the qudits it plays at), or unroll "
            "the loop (unroll_threshold)"
        )
    toned = sorted(
        (p for p in placed if p.event["kind"] in _TONED), key=lambda p: (p.start, p.handle)
    )
    busy: list[tuple[int, int]] = []
    given: dict[int, int] = {}
    for p in toned:
        slot = _lowest_free(busy, p.start, line.tones)
        # The compiler's tone check keeps the tones at once within the line's.
        assert slot is not None
        given[id(p.event)] = slot
        busy.append((p.end, slot))
    # A wait or frame update takes the slot of its handle's tone around it.
    previous: dict[int, int] = {}
    pending: dict[int, list[_Placed]] = {}
    for p in sorted(placed, key=lambda p: (p.start, p.order)):
        if id(p.event) in given:
            slot = previous[p.handle] = given[id(p.event)]
            for waiting in pending.pop(p.handle, ()):
                given[id(waiting.event)] = slot
        elif p.handle in previous:
            given[id(p.event)] = previous[p.handle]
        else:
            pending.setdefault(p.handle, []).append(p)
    for waiting in (w for items in pending.values() for w in items):
        given[id(waiting.event)] = 0
    return given


def structured(program: dict[str, Any], system: ControlSystem) -> dict[str, Any]:
    """Format /2 *program* on *system*, written with one line per handle
    (qudit), its shared lines as one line each: ``lines`` lists each
    physical line once, in order of first use, a shared one with its
    ``max_concurrent_tones``; each event on one of its handles has the
    line's index, its ``tone_slot`` and ``qudits`` (the handle's qudit).
    Unchanged when the program plays no shared line."""
    shared = physical(system)
    if shared is None:
        return program
    walked = [(entry["role"], entry["qudits"][0]) for entry in program["lines"]]
    on = [shared.get(handle) for handle in walked]
    if not any(on):
        return program
    index_of: dict[Any, int] = {}
    renumbered: list[int] = []
    lines: list[dict[str, Any]] = []
    for handle, ((role, qudit), line) in enumerate(zip(walked, on, strict=True)):
        same = handle if line is None else line
        if same not in index_of:
            index_of[same] = len(lines)
            entry: dict[str, Any] = {"line": len(lines), "role": role}
            if line is None:
                entry["qudits"] = [qudit]
            else:
                entry["qudits"] = list(line.qudits)
                entry["max_concurrent_tones"] = line.tones
            lines.append(entry)
        renumbered.append(index_of[same])
    slot_of: dict[int, int] = {}
    handle_slot: dict[int, int] = {}
    for line, placed in _placed(program["events"], on).items():
        # One tone at once: every tone takes the line's one slot in turn.
        per_handle = (
            dict.fromkeys((p.handle for p in placed), 0)
            if line.tones == 1
            else _slots(line, placed)
        )
        if per_handle:
            handle_slot.update(per_handle)
            for p in placed:
                # A handle that only waits or changes frames takes slot 0.
                slot_of[id(p.event)] = per_handle.get(p.handle, 0)
        else:
            slot_of.update(_slots_by_event(line, placed))

    def renumber(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for event in items:
            kind = event["kind"]
            if kind == "loop":
                event = {**event, "body": renumber(event["body"])}
            elif kind == "sync":
                joined = dict.fromkeys(renumbered[line] for line in event["lines"])
                event = {**event, "lines": list(joined)}
            elif kind in _ON_LINE and event.get("line") is not None:
                handle = event["line"]
                pairs = list(event.items())
                at = [key for key, _ in pairs].index("line") + 1
                added: list[tuple[str, Any]] = []
                if on[handle] is not None:
                    added = [
                        ("tone_slot", slot_of[id(event)]),
                        ("qudits", [walked[handle][1]]),
                    ]
                event = dict([*pairs[: at - 1], ("line", renumbered[handle]), *added, *pairs[at:]])
            out.append(event)
        return out

    program = dict(program)
    program["lines"] = lines
    program["events"] = renumber(program["events"])
    for key in ("frame_state", "runtime_tones"):
        if key in program:
            program[key] = _relined(program[key], renumbered, on, walked, handle_slot)
    program["segments"] = [
        {**s, "line": renumbered[s["line"]]}
        if s.get("opened_by") == "wait" and 0 <= s["line"] < len(renumbered)
        else s
        for s in program["segments"]
    ]
    return program


def _relined(
    found: Any,
    renumbered: Sequence[int],
    on: Sequence[Physical | None],
    walked: Sequence[tuple[str, int]],
    handle_slot: Mapping[int, int],
) -> Any:
    """``frame_state`` or ``runtime_tones`` with each entry's ``line`` the
    physical line's, and on a shared line its handle's ``qudits`` (and
    ``tone_slot``, when the handle keeps one)."""

    def entry(item: Mapping[str, Any]) -> dict[str, Any]:
        handle = item.get("line")
        if handle is None or not 0 <= handle < len(renumbered):
            return dict(item)
        pairs = [(k, renumbered[handle] if k == "line" else v) for k, v in item.items()]
        if on[handle] is not None:
            at = [k for k, _ in pairs].index("line") + 1
            added: list[tuple[str, Any]] = []
            if handle in handle_slot:
                added.append(("tone_slot", handle_slot[handle]))
            added.append(("qudits", [walked[handle][1]]))
            pairs[at:at] = added
        return dict(pairs)

    if isinstance(found, Mapping):
        return {key: entry(item) for key, item in found.items()}
    return [entry(item) for item in found]
