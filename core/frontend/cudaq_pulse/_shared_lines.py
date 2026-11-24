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
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
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


def _sharing(
    shared: Mapping[tuple[str, int], Physical], lines: Sequence[Mapping[str, Any]]
) -> tuple[list[Physical], dict[int, int]]:
    """The shared lines of ``_groups`` that two or more handles play, and the
    position in that list of each of their handles."""
    found: list[Physical] = []
    of: dict[int, int] = {}
    for line, handles in _groups(shared, lines):
        if len(handles) > 1:
            of.update(dict.fromkeys(handles, len(found)))
            found.append(line)
    return found, of


def _timed_on(event: Mapping[str, Any], of: Mapping[int, int]) -> int | None:
    """The position (``_sharing``) of the shared line a drive or readout
    *event* with known times plays on; ``None`` for any other event."""
    if (
        event["kind"] in _PLAYED
        and event["start_cycles"] is not None
        and event["end_cycles"] is not None
    ):
        return of.get(event["line"])
    return None


def _apart(a: Mapping[Any, Any], b: Mapping[Any, Any]) -> bool:
    """Whether arms *a* and *b* (by branch) differ on a branch execution:
    ``controller._exclusive`` of two events' arms, which a walk tells apart
    by branch id."""
    return any(b.get(branch, arm) != arm for branch, arm in a.items())


# A node of the branches tones lie in: the tones in no branch below it, and
# by branch below it, by arm, the node of that arm.
_Node = tuple[list[int], dict[Any, dict[Any, Any]]]
# The (branch, arm) of each branch execution around a tone, outermost first.
_Path = tuple[tuple[Any, Any], ...]


def _nested(nodes: list[int], path: Callable[[int], _Path]) -> list[int] | None:
    """The most of the tones *nodes* (of one copy; *path*: of a tone, the
    (branch, arm) of each branch execution around it, outermost first) that
    can all play in one execution, in order; ``None`` when the branches do
    not nest (a branch under two arms).

    Each branch execution lies in one arm of the one around it, or in none,
    so the tones form a tree: at a node, the tones there play together with
    the most of each branch below it, and of a branch only the most of one
    arm plays. Two tones are in different arms of one branch execution just
    when their paths part at one branch: below two branches they share none.
    O(m d) for m tones in d branches around each at most."""
    root: _Node = ([], {})
    under: dict[Any, int] = {}
    for k in nodes:
        node = root
        for branch, arm in path(k):
            if under.setdefault(branch, id(node)) != id(node):
                return None
            node = node[1].setdefault(branch, {}).setdefault(arm, ([], {}))
        node[0].append(k)

    def most(node: _Node) -> list[int]:
        found = list(node[0])
        for arms in node[1].values():
            best: list[int] = []
            for child in arms.values():
                below = most(child)
                if len(below) > len(best):
                    best = below
            found += best
        return found

    return sorted(most(root))


def _parts(nodes: list[int], near: Callable[[int], set[int]]) -> list[list[int]]:
    """The connected parts of *nodes*, *near* giving each node's neighbours,
    each part in order."""
    left = set(nodes)
    parts = []
    for k in nodes:
        if k not in left:
            continue
        left.discard(k)
        part, queue = [k], [k]
        while queue:
            found = near(queue.pop()) & left
            left -= found
            part.extend(found)
            queue.extend(found)
        parts.append(sorted(part))
    return parts


def _split(nodes: list[int], apart: Mapping[int, set[int]]) -> list[int]:
    """``_nested`` from *apart* (by tone, the tones apart from it) alone, for
    branches that do not nest: the tones split into parts no tone of which
    is apart from any of another part's, whose mosts add up, or else into
    parts every tone of which is apart from every one of another part's, of
    whose mosts the largest plays, and so on down to single tones; each
    split takes O(m^2) for m tones and leaves fewer, so the whole O(m^3).
    Tones that split neither way all count: never fewer than play
    together."""
    if len(nodes) < 2:
        return nodes
    within = set(nodes)
    parts = _parts(nodes, lambda k: within - apart[k])
    if len(parts) > 1:
        return max((_split(part, apart) for part in parts), key=len)
    parts = _parts(nodes, lambda k: apart[k] & within)
    if len(parts) > 1:
        return sorted(k for part in parts for k in _split(part, apart))
    return nodes


def _most_together(
    nodes: list[int], path: Callable[[int], _Path], copies: Sequence[Any], nested: bool
) -> list[int]:
    """The most of the tones *nodes* that can all play in one execution: no
    two of one copy (*copies*, by tone) in different arms of one branch
    execution (*path*, as ``_nested``), in order; unless *nested*, by
    ``_split`` alone."""
    by_copy: dict[Any, list[int]] = {}
    for k in nodes:
        by_copy.setdefault(copies[k], []).append(k)
    found: list[int] = []
    for each in by_copy.values():
        most = _nested(each, path) if nested else None
        if most is None:
            arms = {k: dict(path(k)) for k in each}
            apart: dict[int, set[int]] = {
                k: {j for j in each if _apart(arms[k], arms[j])} for k in each
            }
            most = _split(each, apart)
        found += most
    return sorted(found)


class _TangledError(Exception):
    """Branches that do not nest: a branch under two arms."""


class _Arm:
    """A node of the tree of branch executions the tones playing on a line
    lie in (each in one arm of the one around it, or in none): how many
    tones lie in no branch below it, by branch below it and arm the node of
    that arm, and the most tones below it that can play together (``most``;
    ``best``: of each branch below it, its arm's with the most)."""

    __slots__ = ("parent", "via", "here", "arms", "best", "most")

    def __init__(self, parent: _Arm | None, via: Any) -> None:
        self.parent, self.via = parent, via
        self.here = self.most = 0
        self.arms: dict[Any, dict[Any, _Arm]] = {}
        self.best: dict[Any, int] = {}


class _Playing:
    """The tones playing on a line in one copy of the loops around them, as
    a tree of ``_Arm``: of a branch only one arm plays, and of the tones in
    an arm, those of different branches below it all can. Adding, taking
    away and counting a tone take time in the branches around it, not in
    the tones playing."""

    __slots__ = ("root", "under")

    def __init__(self) -> None:
        self.root = _Arm(None, None)
        # By branch, the arm it lies in, the same for every tone.
        self.under: dict[Any, _Arm] = {}

    def add(self, path: _Path, count: int) -> int:
        """Add *count* tones (or take away) of *path* (``_Path``); how many
        more of them all can play together."""
        before = self.root.most
        node = self.root
        for branch, arm in path:
            if self.under.setdefault(branch, node) is not node:
                raise _TangledError
            arms = node.arms.setdefault(branch, {})
            node = arms.get(arm) or arms.setdefault(arm, _Arm(node, branch))
        node.here += count
        node.most += count
        while node.parent is not None:
            parent = node.parent
            old = parent.best.get(node.via, 0)
            new = max(child.most for child in parent.arms[node.via].values())
            if new == old:
                break
            parent.best[node.via] = new
            parent.most += new - old
            node = parent
        return self.root.most - before

    def beside(self, path: _Path) -> int:
        """The most tones playing that can play together with one of *path*
        (``add``), in no branch execution around it in another arm."""
        node: _Arm | None = self.root
        found = 0
        for branch, arm in path:
            if self.under.get(branch, node) is not node:
                raise _TangledError
            if node is not None:
                found += node.most - node.best.get(branch, 0)
                node = node.arms.get(branch, {}).get(arm)
        return found + (0 if node is None else node.most)


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
        copy in different arms of a branch never play together (a walk tells
        branch executions apart by id, as ``controller._exclusive``), so a
        tone starts with the line full when the most tones playing then that
        can all play together with it in one execution fill it: those of each
        copy found in the tree of branches they lie in (``_Playing``), else
        (branches that do not nest) ``_split``."""
        count, first, playing = self.count, self.first, self.playing
        try:
            self._sweep(played, True)
        except _TangledError:
            self.count, self.first, self.playing = count, first, playing
            self._sweep(played, False)

    def _sweep(
        self, played: Sequence[tuple[int, int, Mapping[str, Any], Any]], nested: bool
    ) -> None:
        tones = self.line.tones
        # Without branches every tone playing plays together.
        nested = nested and any(item[2]["conditions"] for item in played)
        paths: list[_Path | None] = [None] * len(played)

        def path(k: int) -> _Path:
            found = paths[k]
            if found is None:
                found = paths[k] = tuple(
                    (c["branch"], c["arm"]) for c in played[k][2]["conditions"]
                )
            return found

        copies = [item[3] for item in played]
        # While the line is full, the tones playing as trees by copy, and the
        # most of them that can play together.
        trees: dict[Any, _Playing] | None = None
        playing = 0
        active: list[int] = []
        for k, (start, _, event, copy) in enumerate(played):
            ended = [a for a in active if played[a][1] <= start]
            if ended:
                active = [a for a in active if played[a][1] > start]
            if len(active) < tones:
                # Not full, whatever plays together.
                trees = None
                active.append(k)
                continue
            if not nested:
                most = len(active)
            else:
                if trees is None:
                    trees, playing, ended = {}, 0, []
                    for a in active:
                        tree = trees.setdefault(copies[a], _Playing())
                        playing += tree.add(path(a), 1)
                for a in ended:
                    playing += trees[copies[a]].add(path(a), -1)
                tree = trees.setdefault(copy, _Playing())
                most = playing - tree.root.most + tree.beside(path(k))
            if most >= tones and nested and self.first is not None and start >= self.first:
                self.count += 1
            elif most >= tones:
                # Which tones, for the message (and how many, without the tree).
                mine = dict(path(k))
                together = [
                    a
                    for a in active
                    if not (mine and copies[a] == copy and _apart(dict(path(a)), mine))
                ]
                if any(path(a) for a in together):
                    together = _most_together(together, path, copies, nested)
                if len(together) >= tones:
                    if self.first is None or start < self.first:
                        self.first = start
                        self.playing = [played[a][2] for a in together] + [event]
                    self.count += 1
            if trees is not None:
                playing += tree.add(path(k), 1)
            active.append(k)

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
    sharing, of = _sharing(shared, lines)
    each: list[list[Mapping[str, Any]]] = [[] for _ in sharing]
    for e in events:
        k = _timed_on(e, of)
        if k is not None:
            each[k].append(e)
    for line, played in zip(sharing, each, strict=True):
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
    ``controller.check(lowered, controller)``. Rolled loops of one period
    are counted from the check walk's summary, as compiling counts them;
    when they cannot be (``summarized``), or their lines move by periods
    of their own, every iteration is walked, up to the copies a check
    plays one by one (``_loops.refuse_past_budget``)."""
    from . import _loops
    from ._system import _one_controller
    from .controller import _walk

    system = _one_controller(controller)
    repeats: list[tuple[int, ...]] = []
    periods: list[tuple[int, int, int]] = []
    lines, events, _, _, problems = _walk(lowered, values=False, summary=(repeats, periods))
    # What a walk of every iteration plays, while every loop is one still.
    every = _loops.walked_copies(events, repeats, periods) if repeats else 0
    if repeats and periods:
        # Shared lines are counted of loops of one period: a loop whose
        # lines move by periods of their own is played round by round.
        _loops.refuse_past_budget(_loops.walked_copies(events, repeats, periods, unsynced=True))
        repeats = []
        lines, events, _, _, problems = _walk(lowered, values=False, summary=(repeats, None))
    if repeats:
        if not problems:
            found = summarized(system, lowered, lines, events, repeats, None)
            if found is not None:
                return found
    _loops.refuse_past_budget(every)
    lines, events, _, _, _ = _walk(lowered, values=False)
    return violations(system, lowered, lines, events, None)


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
    sharing, of = _sharing(shared, lines)
    split = _loops._split(tree, lambda i: _timed_on(events[i], of))
    for k, line in enumerate(sharing):
        nodes = split.get(k, [])
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


def _insert(entry: dict[str, Any], after: str, added: Sequence[tuple[str, Any]]) -> None:
    """Put *added* in *entry*, in place, right after its key *after*."""
    keys = list(entry)
    tail = keys[keys.index(after) + 1 :]
    entry.update(added)
    for key in tail:
        entry[key] = entry.pop(key)


def flat(
    system: ControlSystem, lines: Sequence[dict[str, Any]], events: Sequence[dict[str, Any]]
) -> dict[int, int]:
    """Format /1 *lines* and *events* (every execution) of a program on
    *system*, written with its shared lines, in place: each line (handle)
    on a shared line gets, after its ``qudit``, the line's ``qudits`` and
    ``max_concurrent_tones``, and each event on it its ``tone_slot``, the
    slot format /2 gives it. Returns, by handle, the slot of each handle
    that keeps one for the whole program."""
    shared = physical(system)
    if shared is None:
        return {}
    on = [shared.get((entry["role"], entry["qudit"])) for entry in lines]
    if not any(on):
        return {}
    for entry, line in zip(lines, on, strict=True):
        if line is not None:
            _insert(
                entry,
                "qudit",
                [("qudits", list(line.qudits)), ("max_concurrent_tones", line.tones)],
            )
    slot_of, handle_slot = _tone_slots(_placed(events, on))
    for event in events:
        slot = slot_of.get(id(event))
        if slot is not None:
            _insert(event, "qudit", [("tone_slot", slot)])
    return handle_slot


def flat_frames(found: Any, handle_slot: Mapping[int, int]) -> Any:
    """``frame_state`` or ``runtime_tones`` of format /1 with, on each entry
    whose line (handle) keeps one tone slot (*handle_slot*), its
    ``tone_slot`` after its ``line``."""

    def entry(item: Mapping[str, Any]) -> dict[str, Any]:
        out = dict(item)
        slot = handle_slot.get(out.get("line"))  # type: ignore[arg-type]
        if slot is not None:
            _insert(out, "line", [("tone_slot", slot)])
        return out

    if isinstance(found, Mapping):
        return {key: entry(item) for key, item in found.items()}
    return [entry(item) for item in found]


# ── Format /2 ───────────────────────────────────────────────────────────────

# The event kinds on a line (controller._ON_LINE), and those playing a tone,
# in either format.
_ON_LINE = frozenset({"play", "drive", "readout", "wait", "frame_update"})
_TONED = frozenset({"play", "drive", "readout"})


class _Placed:
    """An event of format /2 on a handle of a shared line: its handle (the
    line's index in the walk), its first start and end, whether a rolled
    loop repeats it, and by branch the arm it takes of each branch
    execution around it (none when a rolled loop repeats it and it reaches
    past one iteration, as its copies may then play beside others')."""

    __slots__ = ("event", "handle", "start", "end", "last", "looped", "order", "arms")

    def __init__(
        self,
        event: dict[str, Any],
        handle: int,
        last: int,
        looped: bool,
        order: int,
        arms: dict[Any, Any],
    ) -> None:
        self.event, self.handle, self.looped, self.order = event, handle, looped, order
        self.start = event["start_cycles"]
        self.end = self.start + event["duration_cycles"]
        self.last = last
        self.arms = arms


def _placed(
    events: Sequence[dict[str, Any]], shared: Sequence[Physical | None]
) -> dict[Physical, list[_Placed]]:
    """By shared line, the events of format /2 *events* on its handles
    (*shared*: by handle, its shared line), in program order."""
    found: dict[Physical, list[_Placed]] = {}

    def visit(
        items: Sequence[dict[str, Any]], extra: int, loops: tuple[tuple[int, int], ...]
    ) -> None:
        # extra: the cycles the rolled loops around *items* move their last copy by;
        # loops: the first iteration (start, end) of each.
        for event in items:
            if event["kind"] == "loop":
                more = (event["trips"] - 1) * event["period_cycles"]
                first = event["start_cycles"], event["start_cycles"] + event["period_cycles"]
                visit(event["body"], extra + more, (*loops, first))
                continue
            line = event.get("line")
            if event["kind"] not in _ON_LINE or line is None:
                continue
            physical_line = shared[line]
            if physical_line is not None:
                start = event["start_cycles"]
                end = start + event["duration_cycles"]
                arms: dict[Any, Any] = {}
                if all(lo <= start and end <= hi for lo, hi in loops):
                    arms = {c["branch"]: c["arm"] for c in event.get("conditions") or ()}
                placed = found.setdefault(physical_line, [])
                placed.append(_Placed(event, line, end + extra, bool(loops), len(placed), arms))

    visit(events, 0, ())
    return found


def _free(
    busy: list[tuple[int, int, Mapping[Any, Any]]], start: int, arms: Any, slots: int
) -> int | None:
    """The lowest slot no interval of *busy* ((end, slot, its arms)) holds
    at *start*, those ending by then dropped, an interval with arms
    ``_apart`` from *arms* holding none (``None`` for *arms*: every one
    holds its slot); ``None`` when all *slots* are held."""
    busy[:] = [held for held in busy if held[0] > start]
    taken = {slot for _, slot, other in busy if arms is None or not _apart(arms, other)}
    return next((slot for slot in range(slots) if slot not in taken), None)


def _slots(line: Physical, placed: list[_Placed], apart: bool = False) -> dict[int, int]:
    """By handle, the tone slot of each handle of *line* the program uses,
    when each can keep one for the whole program: handles whose events span
    overlapping stretches get different slots, lowest free first, in the
    order of their first event; ``{}`` when that takes more slots than the
    line has. With *apart*, two handles that never play together (in other
    arms of one branch execution, every tone of both) may share a slot."""
    spans: dict[int, list[int]] = {}
    arms: dict[int, dict[Any, Any]] = {}
    for p in placed:
        if p.event["kind"] not in _TONED:
            continue
        span = spans.setdefault(p.handle, [p.start, p.last])
        span[0] = min(span[0], p.start)
        span[1] = max(span[1], p.last)
        common = arms.setdefault(p.handle, p.arms)
        arms[p.handle] = {b: arm for b, arm in common.items() if p.arms.get(b) == arm}
    busy: list[tuple[int, int, Mapping[Any, Any]]] = []
    given: dict[int, int] = {}
    for handle, (start, end) in sorted(spans.items(), key=lambda item: (item[1][0], item[0])):
        slot = _free(busy, start, arms[handle] if apart else None, line.tones)
        if slot is None:
            return {}
        given[handle] = slot
        busy.append((end, slot, arms[handle]))
    return given


# The steps (tones given a slot, first or again) the search of
# ``_slots_by_event`` takes per tone before it gives up.
_TRIES = 64


def _slots_by_event(line: Physical, placed: list[_Placed], apart: bool) -> dict[int, int] | None:
    """By ``id`` of event, the tone slot of each event of *line*, when its
    handles take turns at its slots: its tones in start order (then handle
    order), each the lowest slot no tone playing beside it holds; ``None``
    when that takes more slots than the line has. With *apart*, a tone in
    another arm of a branch execution than one playing does not hold its
    slot, and where no slot is left the search goes back to the latest of
    the tones that kept it from one, to try that one's next slot (one slot
    for every path need not follow from the tones playing at once on each),
    giving up past ``_TRIES`` steps per tone. Not for events a rolled loop
    repeats."""
    toned = sorted(
        (p for p in placed if p.event["kind"] in _TONED), key=lambda p: (p.start, p.handle)
    )
    # By tone, the earlier tones that play beside it.
    beside: list[list[int]] = []
    active: list[int] = []
    for k, p in enumerate(toned):
        active = [j for j in active if toned[j].end > p.start]
        beside.append([j for j in active if not (apart and _apart(p.arms, toned[j].arms))])
        active.append(k)
    chosen = [-1] * len(toned)
    # By tone, the earlier tones whose slots left it none (conflict-directed
    # backjumping: going back past a tone that is not one of them changes
    # nothing that kept it from a slot).
    why: list[set[int]] = [set() for _ in toned]
    budget = _TRIES * len(toned) if apart else len(toned)
    k = 0
    while k < len(toned):
        budget -= 1
        if budget < 0:
            return None
        held: dict[int, int] = {}
        for j in beside[k]:
            held.setdefault(chosen[j], j)
        nxt = next((s for s in range(chosen[k] + 1, line.tones) if s not in held), None)
        if nxt is not None:
            chosen[k] = nxt
            k += 1
            continue
        why[k].update(held.values())
        if not why[k]:
            return None
        back = max(why[k])
        why[back] |= why[k] - {back}
        for i in range(back + 1, k + 1):
            chosen[i] = -1
            why[i] = set()
        k = back
    given = {id(p.event): s for p, s in zip(toned, chosen, strict=True)}
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


def _tone_slots(
    placed_by_line: Mapping[Physical, list[_Placed]],
) -> tuple[dict[int, int], dict[int, int]]:
    """By ``id`` of event, the tone slot of each event *placed_by_line*
    gives, and by handle, the slot of each handle that keeps one. A handle
    keeps one slot when the line has slots enough, else its tones take
    turns at them; only where neither fits do tones in other arms of a
    branch execution share slots, which the tones at once of the check
    (most on any path) allow, and else not: one slot serves every path."""
    slot_of: dict[int, int] = {}
    handle_slot: dict[int, int] = {}
    for line, placed in placed_by_line.items():
        # One tone at once: every tone takes the line's one slot in turn.
        per_handle = (
            dict.fromkeys((p.handle for p in placed), 0)
            if line.tones == 1
            else _slots(line, placed)
        )
        looped = any(p.looped for p in placed)
        by_event = None
        if not per_handle and not looped:
            by_event = _slots_by_event(line, placed, False)
        if not per_handle and by_event is None:
            per_handle = _slots(line, placed, True)
        if per_handle:
            handle_slot.update(per_handle)
            for p in placed:
                # A handle that only waits or changes frames takes slot 0.
                slot_of[id(p.event)] = per_handle.get(p.handle, 0)
            continue
        if looped:
            raise UnsupportedError(
                f"format /2 gives each event of a rolled loop one tone slot in every "
                f"iteration, and the handles of {line.label} take turns at its "
                f"{line.tones} tones: give each handle a tone of its own "
                "(max_concurrent_tones of at least the qudits it plays at), or unroll "
                "the loop (unroll_threshold)"
            )
        if by_event is None:
            by_event = _slots_by_event(line, placed, True)
        if by_event is None:
            raise UnsupportedError(
                f"an export gives each tone of {line.label} one tone slot for every "
                f"path, and found no such slots among its {line.tones} for its tones in "
                "the arms of branches, though no path plays more at once: play some "
                "after the others end (a sync of their lines between them), or raise "
                "its max_concurrent_tones"
            )
        slot_of.update(by_event)
    return slot_of, handle_slot


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
    slot_of, handle_slot = _tone_slots(_placed(program["events"], on))

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
