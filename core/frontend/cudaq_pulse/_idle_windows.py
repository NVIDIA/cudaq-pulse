# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The idle windows of a scheduled program, for idle replacement.

A trace keeps a record of the drives, readouts, waits, syncs and frame
changes it emits (``Sites``), and the module built from it carries each
one's number, so the times the schedule gave them can be read back per
traced operation (``read``). From those, ``gaps`` finds where each qudit's
drive line has nothing to do and which parts of that the qudit itself
idles through, and ``report`` lists the windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._errors import CompilationError, InternalError, UnsupportedError

if TYPE_CHECKING:
    from .kernel._emitter import PulseIREmitter

__all__ = ["IdleWindow", "IdleWindows"]

# The letter number_sites() returns for each kind of numbered op.
_LETTERS = {
    "pulse.drive": "d",
    "pulse.readout": "r",
    "pulse.wait": "w",
    "pulse.sync": "s",
    "pulse.shift_phase": "f",
    "pulse.set_phase": "f",
    "pulse.shift_frequency": "f",
    "pulse.set_frequency": "f",
}


# The pulse calls that emit a numbered op.
_SITE_CALLS = frozenset(kind.removeprefix("pulse.") for kind in _LETTERS)


class Site:
    """One numbered op of a trace: its kind (the op's name), where its
    record starts in the packed buffer, the regions open when it was
    emitted, the kernel instructions being traced then (innermost first),
    its length in cycles when the trace knows it, and whether the tracer
    was between two kernel statements right before it (``enter``) and right
    after it (``leave``), in its own region: the points where something can
    be written next to it. ``number`` is its place among the kernel's own
    ops, or, for an op written into an idle window, a negative number."""

    __slots__ = ("kind", "mark", "scope", "frames", "cycles", "enter", "leave", "number")

    def __init__(
        self,
        kind: str,
        mark: int,
        scope: tuple[int, ...],
        frames: Any,
        cycles: int | None,
        enter: bool,
        number: int,
    ) -> None:
        self.kind = kind
        self.mark = mark
        self.scope = scope
        self.frames = frames
        self.cycles = cycles
        self.enter = enter
        self.leave = False
        self.number = number

    def describe(self) -> str:
        """``drive() at line 12 of cycle()``."""
        name = f"{self.kind.removeprefix('pulse.')}()"
        if not self.frames:
            return name
        prog, index = self.frames[0]
        line = prog.location(index)[1]
        where = f"{prog.code.co_name}()"
        return f"{name} at line {line} of {where}" if line else f"{name} in {where}"


class Sites:
    """The record one trace keeps of its numbered ops, in program order.

    With a *plan*, the trace also writes it: at the points the plan names,
    the waits and fills of its tiles (``rule`` gives the fill)."""

    def __init__(self, plan: Plan | None = None, rule: Any = None) -> None:
        self.log: list[Site] = []
        # The kernel's own ops recorded so far (all of them without a plan).
        self.hosts = 0
        # Whether a pulse call is being emitted (calls nest: a calibrated
        # gate emits its ops through the same entry), or a tile written.
        self.busy = False
        # (number of the next op of the kernel, regions open) when the
        # outermost pulse call began: an op recorded with both has a point
        # before it.
        self.entered: tuple[int, tuple[int, ...]] | None = None
        self.plan = plan
        self.rule = rule
        # The tile being written (its number), and the fill being traced.
        self.inserting: int | None = None
        self.filling: _Filling | None = None
        # The points of the plan already written, with the buffer's cursor
        # then: a loop body traced again writes its points again.
        self.done: dict[tuple[str, int], int] = {}
        # The trace's waveform lengths, by value id (the tracer's own dict).
        self.durations: dict[int, Any] = {}
        # The fill's length by (qudit, window), once it has been traced.
        self.lengths: dict[tuple[int, int], int] = {}

    def attach(self, builder: Any, tracer: type[PulseIREmitter] | None) -> type[PulseIREmitter]:
        """Record what *builder* is given from here on; returns the tracer
        to trace with, which is *tracer* (or the one *builder*'s sample
        rates take) and notes the points between statements."""
        from .kernel._emitter import tracing_frames

        log = self.log
        emit = builder.emit
        rollback = builder.rollback

        def recorded(
            kind: str,
            operands: tuple[Any, ...] = (),
            result_types: tuple[str, ...] = (),
            attrs: dict[str, Any] | None = None,
        ) -> tuple[Any, ...]:
            filling = self.filling
            if filling is not None:
                filling.take(kind, operands, attrs or {}, builder._cur, self.durations)
            if kind not in _LETTERS:
                return emit(kind, operands, result_types, attrs)  # type: ignore[no-any-return]
            mark = builder._cur
            scope = tuple(builder._scope)
            results = emit(kind, operands, result_types, attrs)
            cycles = None
            if kind == "pulse.wait":
                cycles = (attrs or {}).get("duration")
            elif kind in ("pulse.drive", "pulse.readout"):
                cycles = self.durations.get(operands[1].vid)
            if cycles.__class__ is not int:
                cycles = None
            if self.inserting is None:
                number = self.hosts
                self.hosts += 1
                enter = self.entered == (number, scope)
            else:
                number = -1 - self.inserting
                enter = False
            log.append(Site(kind, mark, scope, tracing_frames(), cycles, enter, number))
            return results  # type: ignore[no-any-return]

        def rolled_back(checkpoint: tuple[Any, ...]) -> None:
            # A loop body traced again: what it emitted is gone.
            rollback(checkpoint)
            cursor = checkpoint[0]
            while log and log[-1].mark >= cursor:
                if log.pop().number >= 0:
                    self.hosts -= 1
            # A point written at the cursor was written before what is undone.
            for point in [point for point, mark in self.done.items() if mark > cursor]:
                del self.done[point]
            if self.filling is not None:
                self.filling.undo(cursor)

        builder.emit = recorded
        builder.rollback = rolled_back
        builder._sites = self
        base: type[Any]
        if tracer is not None:
            base = tracer
        elif builder.sample_rate_of is None:
            from .kernel._emitter import PulseIREmitter

            base = PulseIREmitter
        else:
            from .kernel._mixed_rates import MixedRates

            base = MixedRates
        sited = _SITED.get(base)
        if sited is None:
            sited = _SITED[base] = type(
                f"Sited{base.__name__}", (_SiteTracer, base), {"__slots__": ()}
            )
        return sited  # type: ignore[no-any-return]

    def number(self, module: Any) -> None:
        """Give each numbered op of *module*, just built from the trace,
        its number."""
        kinds = module.number_sites([site.number for site in self.log])
        if kinds != "".join(_LETTERS[site.kind] for site in self.log):
            raise InternalError(
                "idle replacement: the module's operations are not the ones the trace recorded"
            )
        plan = self.plan
        if plan is not None and len(self.done) != len(plan.before) + len(plan.after):
            raise InternalError(
                "idle replacement: the program traced differently the second time, so "
                "not every idle window was written"
            )

    def hosted(self) -> list[Site]:
        """The kernel's own ops, by number."""
        return [site for site in self.log if site.number >= 0]


class _Filling:
    """The fill being traced into a window: what it may do (drive and wait
    on its own qudit's drive line, at that line's tone, and shift the
    tone's phase by constants that add up to zero) and how long it is."""

    __slots__ = ("name", "line", "tone", "ops")

    def __init__(self, name: str, line: Any, tone: Any) -> None:
        self.name = name
        self.line = line
        self.tone = tone
        # (cursor of the buffer, what, value): "length" in cycles, "phase"
        # in radians, "for" with a trip count, "end".
        self.ops: list[tuple[int, str, Any]] = []

    def refuse(self, what: str) -> CompilationError:
        return CompilationError(
            f"idle replacement: fill {self.name}() {what}; in this version a fill drives and "
            "waits on its own qudit's drive line, at that line's tone, and nothing else",
            code="unsupported_construct",
        )

    def take(
        self,
        kind: str,
        operands: tuple[Any, ...],
        attrs: dict[str, Any],
        mark: int,
        durations: dict[int, Any],
    ) -> None:
        """Check the op the fill emits, and note what it adds."""
        if kind == "pulse.drive":
            if operands[0].chan is not self.line:
                raise self.refuse("drives another line than its qudit's drive line")
            if operands[2].chan is not self.tone:
                raise self.refuse("plays a tone that is not its drive line's own")
            cycles = durations.get(operands[1].vid)
            if cycles.__class__ is not int:
                raise self.refuse("plays a waveform whose length is not known when it compiles")
            self.ops.append((mark, "length", cycles))
        elif kind == "pulse.wait":
            if operands[0].chan is not self.line:
                raise self.refuse("waits on another line than its qudit's drive line")
            cycles = attrs.get("duration", 0)
            if cycles.__class__ is not int or attrs.get("bounds") is not None:
                raise self.refuse("waits a length that is not a whole number of cycles")
            self.ops.append((mark, "length", cycles))
        elif kind == "pulse.shift_phase":
            phase = attrs.get("phase_rad", 0.0)
            if operands[0].chan is not self.tone or not isinstance(phase, (int, float)):
                raise self.refuse("shifts a phase that is not a number, or of another tone")
            self.ops.append((mark, "phase", float(phase)))
        elif kind == "scf.for":
            trips = max(0, -(-(attrs.get("ub", 0) - attrs.get("lb", 0)) // attrs.get("step", 1)))
            self.ops.append((mark, "for", trips))
        elif kind == "scf.for_end":
            self.ops.append((mark, "end", None))
        elif kind == "scf.yield":
            pass
        elif kind in _LETTERS or kind.startswith("scf.") or kind in _ACQUIRING:
            raise self.refuse(f"uses {kind.removeprefix('pulse.').removeprefix('scf.')}")

    def undo(self, cursor: int) -> None:
        ops = self.ops
        while ops and ops[-1][0] >= cursor:
            ops.pop()

    def totals(self) -> tuple[int, float]:
        """Its length in cycles and the phase it leaves its tone shifted
        by, a loop's body counted once per iteration."""
        stack = [[1, 0, 0.0]]
        for _, what, value in self.ops:
            if what == "for":
                stack.append([value, 0, 0.0])
            elif what == "end":
                trips, length, phase = stack.pop()
                stack[-1][1] += trips * length
                stack[-1][2] += trips * phase
            elif what == "length":
                stack[-1][1] += value
            else:
                stack[-1][2] += value
        return stack[0][1], stack[0][2]


# What a fill may not make: a line or a tone it does not have yet.
_ACQUIRING = ("pulse.get_drive_line", "pulse.get_readout_line", "pulse.tone")


class _SiteTracer:
    """Notes, for the record of its builder, the points between kernel
    statements, and writes the record's plan at them; emits as the tracer
    it is mixed into."""

    __slots__ = ()

    def _emit_pulse_call(self, fname: str, call_args: list[Any]) -> tuple[Any, ...]:
        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        if sites.busy:
            return super()._emit_pulse_call(fname, call_args)  # type: ignore[misc,no-any-return]
        sites.durations = self._durations  # type: ignore[attr-defined]
        before = sites.hosts
        scope = tuple(builder._scope)
        plan = sites.plan
        if plan is not None and fname in _SITE_CALLS:
            tiles = plan.before.get(before)
            if tiles is not None and ("before", before) not in sites.done:
                self._write(("before", before), tiles, scope)
        sites.entered = (before, scope)
        sites.busy = True
        try:
            results: tuple[Any, ...] = super()._emit_pulse_call(fname, call_args)  # type: ignore[misc]
        finally:
            sites.busy = False
            sites.entered = None
        if sites.hosts > before:
            last = sites.hosts - 1
            if plan is None:
                if sites.log[-1].scope == tuple(builder._scope):
                    sites.log[-1].leave = True
            else:
                tiles = plan.after.get(last)
                if tiles is not None and ("after", last) not in sites.done:
                    self._write(("after", last), tiles, tuple(builder._scope))
        return results

    def _write(self, point: tuple[str, int], tiles: list[Tile], scope: tuple[int, ...]) -> None:
        """Write *tiles* here, the point *point* of the plan."""
        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        sites.done[point] = builder._cur
        sites.busy = True
        try:
            for tile in tiles:
                if tile.scope != scope:
                    raise InternalError(
                        "idle replacement: the program traced differently the second time, "
                        "so an idle window is not where it was found"
                    )
                held = self._drive_channel(tile.qudit)
                sites.inserting = tile.number
                for item in tile.items:
                    if item[0] == "wait":
                        self._emit_pulse_call("wait", [held.cells[0].value, item[1]])
                    else:
                        self._fill(held, item[1])
        finally:
            sites.busy = False
            sites.inserting = None
            sites.filling = None
        self._refresh_lines()  # type: ignore[attr-defined]

    def _drive_channel(self, qudit: int) -> Any:
        """The drive channel the trace holds of *qudit*."""
        qudits = self.b._qref_to_qudit  # type: ignore[attr-defined]
        for key, held in self._lines.held.items():  # type: ignore[attr-defined]
            if len(key) == 2 and key[1] == "get_drive_line" and qudits.get(key[0]) == qudit:
                return held
        raise InternalError(f"idle replacement: the trace holds no drive line of qudit {qudit}")

    def _fill(self, held: Any, window: int) -> None:
        """Trace the rule's fill into a window of *window* cycles on the
        channel *held*, as a call ``fill(qudit, window)`` written here, and
        complete it with waits to the window's length."""
        from .kernel._emitter import _channel_values, _restore_channels

        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        rule = sites.rule
        line = held.cells[0]
        if sites.plan is not None and sites.plan.pads:
            self._emit_pulse_call("wait", [line.value, window])
            return
        qudit = builder._qref_to_qudit.get(held.key[0])
        length = None
        if rule.align != "start":
            # The wait before the fill is as long as the fill leaves over:
            # the fill is traced once to learn its length, and undone.
            length = sites.lengths.get((qudit, window))
            if length is None:
                channels = self._lines.held  # type: ignore[attr-defined]
                count, values = len(channels), _channel_values(channels)
                checkpoint = builder.checkpoint()
                length = self._traced_fill(held, window, qudit)
                builder.rollback(checkpoint)
                _restore_channels(channels, count, values)
                self._refresh_lines()  # type: ignore[attr-defined]
                sites.lengths[qudit, window] = length
            spare = window - length
            lead = spare if rule.align == "end" else spare // 2
            if lead:
                self._emit_pulse_call("wait", [line.value, lead])
        else:
            lead = 0
        traced = self._traced_fill(held, window, qudit)
        if length is not None and traced != length:
            raise InternalError(
                f"idle replacement: fill {rule.fill.__name__}() has another length each time "
                "it is traced"
            )
        if window - traced - lead:
            self._emit_pulse_call("wait", [line.value, window - traced - lead])

    def _traced_fill(self, held: Any, window: int, qudit: int) -> int:
        """Trace the fill here; its length in cycles."""
        from .kernel.ir_builder import IRValue

        builder: Any = self.b  # type: ignore[attr-defined]
        sites: Sites = builder._sites
        fill = sites.rule.fill
        name = fill.__name__
        line, tone = held.cells
        calls, readouts = len(builder.classical), builder._n_readouts
        filling = sites.filling = _Filling(name, line, tone)
        try:
            if getattr(fill, "schedule", None) is not None:
                raise filling.refuse("has a schedule= of its own")
            self._dispatch_call(fill, [IRValue(held.key[0], "qref"), window], None)  # type: ignore[attr-defined]
            self.stack.pop()  # type: ignore[attr-defined]
        finally:
            sites.filling = None
        if len(builder.classical) != calls or builder._n_readouts != readouts:
            raise filling.refuse("calls a classical function or reads out")
        length, phase = filling.totals()
        if abs(phase) > 1e-9:
            raise filling.refuse(
                f"leaves its tone's phase shifted by {phase:g} rad (its shifts must add up to zero)"
            )
        if length > window:
            raise CompilationError(
                f"idle replacement: fill {name}() takes {length} cycles, {length - window} more "
                f"than its window of {window} cycles on qudit {qudit}",
                fix="shorten the fill, or raise threshold= so that this window is left alone",
                code="invalid_duration",
            )
        return length


_SITED: dict[type[Any], type[Any]] = {}


# ---------------------------------------------------------------- reading the schedule


class _Event:
    """One execution of a numbered op: its site, kind, line and the line
    whose tone it plays, its start and end in cycles, its start's shift per
    iteration of each loop around it, and where it lies: ``path`` holds
    (loop, copy, trips) per loop around it, outermost first, ``arms`` the
    (branch, arm) pairs around it. A sync is an instant where its lines
    meet; ``lines`` holds them."""

    __slots__ = ("site", "kind", "line", "tone", "start", "end", "shifts", "path", "arms", "lines")

    def __init__(
        self,
        site: int | None,
        kind: str,
        line: int,
        tone: int,
        start: int,
        end: int,
        shifts: tuple[int, ...],
        path: tuple[tuple[int, int, int], ...],
        arms: tuple[tuple[int, int], ...],
        lines: tuple[int, ...] = (),
    ) -> None:
        self.site = site
        self.kind = kind
        self.line = line
        self.tone = tone
        self.start = start
        self.end = end
        self.shifts = shifts
        self.path = path
        self.arms = arms
        self.lines = lines

    def advance(self, depth: int, loop: int, iterations: int) -> int:
        """How much later this event runs *iterations* iterations of *loop*
        (at *depth*) on, 0 if it is not in that loop's second copy."""
        path = self.path
        if len(path) > depth and path[depth][0] == loop and path[depth][1] == 1:
            return iterations * self.shifts[depth]
        return 0


def read(module: Any) -> tuple[list[tuple[str, int]], list[Any]]:
    """The lines of scheduled *module*, (role, qudit) by index, and its
    numbered ops in program order, each loop's body listed for its first
    iteration and, when it has more, its second: items are an ``_Event``,
    ``("barrier",)`` where a wait of run-time length is, and ``("advance",
    loop, depth, iterations)`` after the second copy of a loop of more than
    two iterations, which stands for the iterations that follow it."""
    lines, tree = module.site_tree()
    out: list[Any] = []
    counter = [0]

    def walk(
        nodes: list[Any],
        copies: tuple[int, ...],
        path: tuple[tuple[int, int, int], ...],
        arms: tuple[tuple[int, int], ...],
    ) -> None:
        for node in nodes:
            kind = node[0]
            if kind == "for":
                trips, body = node[1], node[2]
                counter[0] += 1
                loop = counter[0]
                for copy in range(min(trips, 2)):
                    walk(body, (*copies, copy), (*path, (loop, copy, trips)), arms)
                if trips > 2:
                    out.append(("advance", loop, len(path), trips - 2))
            elif kind == "if":
                counter[0] += 1
                branch = counter[0]
                walk(node[1], copies, path, (*arms, (branch, 0)))
                walk(node[2], copies, path, (*arms, (branch, 1)))
            elif kind == "sync":
                out.append(_Event(node[1], "sync", -1, -1, 0, 0, (), path, arms, tuple(node[2])))
            elif kind == "answer":
                out.append(("barrier",))
            else:
                _, site, line, tone, start, duration, shifts, variable = node
                if start is None or duration is None or None in shifts:
                    raise UnsupportedError(
                        "idle replacement: the program is not scheduled without its "
                        "arguments' values"
                    )
                if variable:
                    out.append(("barrier",))
                at = start + sum(k * shift for k, shift in zip(copies, shifts, strict=True))
                out.append(
                    _Event(site, kind, line, tone, at, at + duration, tuple(shifts), path, arms)
                )

    walk(tree, (), (), ())
    return [(str(role), int(qudit)) for role, qudit in lines], out


# ---------------------------------------------------------------- gaps and their idle parts

_PLAYS = ("drive", "readout")
# What uses the qudit's tone in a gap: the program orders it with whatever
# else uses that tone.
_CHAINED = ("tone", "frame", "frequency")
# What plays on the qudit in a gap.
_ON_QUDIT = ("tone", "busy", "readout")
_FREQUENCY = ("shift_frequency", "set_frequency")


class _Timed:
    """A time that moves with the loop iterations of the event it comes
    from."""

    __slots__ = ("at", "event")

    def __init__(self, at: int, event: _Event | None) -> None:
        self.at = at
        self.event = event

    def advance(self, depth: int, loop: int, iterations: int) -> None:
        if self.event is not None:
            self.at += self.event.advance(depth, loop, iterations)


class Gap:
    """A stretch in which a qudit's drive line has no operation: from the
    end of ``lo`` (an op or sync on the line; None at the program's start)
    to the start of ``hi`` (None at its end), with the events on the qudit
    in between, each (event, start, end, what): ``"tone"`` while its tone
    plays from another line, ``"readout"`` while its own readout runs,
    ``"frame"`` where the kernel changes the tone's phase,
    ``"frequency"`` where it changes its frequency, and ``"busy"`` for a
    pulse at its tone that began before the gap.

    ``first`` and ``last`` say whether the qudit has no operation before
    the gap and none after it; ``barrier`` whether a wait of run-time
    length comes before its end; ``detuned`` whether the kernel changed the
    tone's frequency before it. ``rounds`` counts the executions the gap
    stands for."""

    __slots__ = (
        "qudit",
        "lo",
        "hi",
        "start",
        "end",
        "inner",
        "first",
        "last",
        "barrier",
        "detuned",
        "rounds",
        "varies",
    )

    def __init__(
        self,
        qudit: int,
        lo: _Event | None,
        hi: _Event | None,
        start: int,
        end: int,
        inner: list[tuple[_Event, int, int, str]],
        first: bool,
        barrier: bool,
        detuned: bool,
    ) -> None:
        self.qudit = qudit
        self.lo = lo
        self.hi = hi
        self.start = start
        self.end = end
        self.inner = inner
        self.first = first
        self.last = False
        self.barrier = barrier
        self.detuned = detuned
        self.rounds = _rounds(lo, hi)
        # Whether the executions it stands for differ in length or in what
        # happens on the qudit meanwhile.
        self.varies = False

    def key(self) -> tuple[Any, ...]:
        """What the executions of one gap of the kernel share."""
        return (
            self.qudit,
            None if self.lo is None else self.lo.site,
            None if self.hi is None else self.hi.site,
            tuple(event.site for event, _, _, _ in self.inner),
        )

    def shape(self) -> tuple[Any, ...]:
        """Its length and what happens when, from its start."""
        return (
            self.end - self.start,
            tuple((a - self.start, b - self.start, what) for _, a, b, what in self.inner),
            self.first,
            self.last,
            self.barrier,
            self.detuned,
        )

    def branched(self) -> bool:
        """Whether one of its events lies in an arm of a branch."""
        events = [self.lo, self.hi, *(event for event, _, _, _ in self.inner)]
        return any(event is not None and event.arms for event in events)

    def ordered(self) -> bool:
        """Whether the pulses at the qudit's tone and the changes of its
        frame in the gap come in the order the program writes them: what
        is written into the gap is written between them."""
        at = self.start
        for _, start, _, what in self.inner:
            if what in _CHAINED:
                if start < at:
                    return False
                at = start
        return at <= self.end

    def idle(self) -> list[tuple[int, int]]:
        """The parts of it the qudit idles through, as (start, end) in
        order: not while its tone plays or its readout runs, and cut where
        its frame changes."""
        parts: list[tuple[int, int]] = []
        at = self.start
        for _, start, end, _ in sorted(self.inner, key=lambda item: item[1]):
            start = min(max(start, self.start), self.end)
            end = min(max(end, self.start), self.end)
            if start > at:
                parts.append((at, start))
            at = max(at, end)
        if self.end > at:
            parts.append((at, self.end))
        return parts


def _rounds(lo: _Event | None, hi: _Event | None) -> int:
    """How many executions a gap from *lo* to *hi* stands for: a loop's
    second copy stands for every iteration after the first, and a gap from
    the first copy into the second for every pass from one iteration to the
    next."""
    a = () if lo is None else lo.path
    b = () if hi is None else hi.path
    rounds = 1
    for depth in range(min(len(a), len(b))):
        if a[depth] == b[depth]:
            _, copy, trips = a[depth]
            if copy:
                rounds *= trips - 1
            continue
        if a[depth][0] == b[depth][0]:
            rounds *= a[depth][2] - 1
        break
    return rounds


class _Line:
    """What a walk knows of one qudit's drive line: where its open gap
    starts and after which event, and what happened on the qudit since."""

    __slots__ = ("since", "inner", "started", "detuned", "gaps")

    def __init__(self) -> None:
        self.since = _Timed(0, None)
        self.inner: list[tuple[_Event, _Timed, _Timed, str]] = []
        self.started = False
        self.detuned = False
        self.gaps: list[Gap] = []


def gaps(lines: list[tuple[str, int]], items: list[Any]) -> dict[int, list[Gap]]:
    """Per qudit with a drive line, the gaps of that line in program order;
    the last one ends where the last line does. Executions of one gap of the kernel
    (the iterations of a loop, and the copies the schedule made of one) are
    one ``Gap``, of the first; ``varies`` says they differ."""
    state: dict[int, _Line] = {}
    drive_of: dict[int, int] = {}
    for index, (role, qudit) in enumerate(lines):
        if role == "drive" and qudit >= 0 and qudit not in drive_of:
            drive_of[qudit] = index
            state[qudit] = _Line()
    # Where each line's latest op ends: a sync meets where the last of its
    # lines is free.
    free: dict[int, _Timed] = {}
    barrier = False
    # Each qudit's own readouts, as (event, start, end). The program orders
    # a readout with nothing on the drive line, so one written after a gap
    # may run during it: they are laid over the gaps at the end.
    readouts: dict[int, list[tuple[_Event, int, int]]] = {qudit: [] for qudit in state}

    def close(qudit: int, hi: _Event | None, end: int, restart: int) -> None:
        """End the open gap of *qudit* at *hi*, at *end*; the next one
        starts at *restart*, and what still runs then runs on in it."""
        line = state[qudit]
        inner = [(event, a.at, b.at, what) for event, a, b, what in line.inner]
        line.gaps.append(
            Gap(
                qudit,
                line.since.event,
                hi,
                line.since.at,
                end,
                inner,
                not line.started,
                barrier,
                line.detuned,
            )
        )
        line.inner = [(event, a, b, "busy") for event, a, b, _ in line.inner if b.at > restart]

    for item in items:
        if item.__class__ is tuple:
            if item[0] == "barrier":
                barrier = True
            else:
                _, loop, depth, iterations = item
                for timed in free.values():
                    timed.advance(depth, loop, iterations)
                for line in state.values():
                    line.since.advance(depth, loop, iterations)
                    for _, a, b, _ in line.inner:
                        a.advance(depth, loop, iterations)
                        b.advance(depth, loop, iterations)
                # The readouts of the iteration after the second, which may
                # run during the second's gaps, and of the last.
                for listed in readouts.values():
                    for event, start, end in list(listed):
                        for more in {1, iterations}:
                            shift = event.advance(depth, loop, more)
                            if shift:
                                listed.append((event, start + shift, end + shift))
            continue
        event: _Event = item
        if event.kind == "sync":
            meet = max((free[index].at for index in event.lines if index in free), default=0)
            latest = max(
                (free[index] for index in event.lines if index in free),
                key=lambda timed: timed.at,
                default=None,
            )
            event.start = event.end = meet
            if latest is not None and latest.event is not None:
                shifts = latest.event.shifts
                event.shifts = (*shifts, *(0,) * len(event.path))[: len(event.path)]
            else:
                event.shifts = (0,) * len(event.path)
            for index in event.lines:
                role, qudit = lines[index] if 0 <= index < len(lines) else ("", -1)
                if role == "drive" and drive_of.get(qudit) == index:
                    close(qudit, event, meet, meet)
                    state[qudit].since = _Timed(meet, event)
            continue
        role, qudit = lines[event.line] if event.line >= 0 else ("", -1)
        if event.line >= 0 and event.kind in ("drive", "readout", "wait"):
            known = free.get(event.line)
            if known is None or event.end >= known.at:
                free[event.line] = _Timed(event.end, event)
            if role == "drive" and drive_of.get(qudit) == event.line:
                close(qudit, event, event.start, event.end)
                line = state[qudit]
                line.since = _Timed(event.end, event)
                line.started = True
            elif role == "readout" and event.kind == "readout" and qudit in state:
                readouts[qudit].append((event, event.start, event.end))
        if event.tone < 0 or event.tone == event.line:
            continue
        tone_role, owner = lines[event.tone]
        if tone_role != "drive" or owner not in state:
            continue
        if event.kind in _PLAYS:
            what = "tone"
            state[owner].started = True
        else:
            what = "frequency" if event.kind in _FREQUENCY else "frame"
        state[owner].inner.append(
            (event, _Timed(event.start, event), _Timed(event.end, event), what)
        )
        if what == "frequency":
            state[owner].detuned = True

    found: dict[int, list[Gap]] = {}
    length = max((timed.at for timed in free.values()), default=0)
    for qudit, line in state.items():
        end = max(length, line.since.at)
        close(qudit, None, end, end)
        # Nothing plays on the qudit after the last gap, nor in the gaps
        # before it that no pulse at its tone follows.
        for gap in line.gaps:
            gap.inner += sorted(
                (
                    (event, start, end, "readout")
                    for event, start, end in readouts[qudit]
                    if start < gap.end and end > gap.start
                ),
                key=lambda item: item[1],
            )
        for gap in reversed(line.gaps):
            gap.last = True
            if gap.lo is not None and gap.lo.kind != "sync":
                break
            if any(what == "tone" for _, _, _, what in gap.inner):
                break
        found[qudit] = _merged(line.gaps)
    return found


def _merged(executions: list[Gap]) -> list[Gap]:
    """One gap per gap of the kernel, in the order of their first
    executions."""
    merged: dict[tuple[Any, ...], Gap] = {}
    for gap in executions:
        first = merged.setdefault(gap.key(), gap)
        if first is gap:
            continue
        first.rounds += gap.rounds
        if gap.shape() != first.shape():
            first.varies = True
            if gap.end - gap.start < first.end - first.start:
                # The shortest execution is the one a fill must fit.
                gap.rounds, gap.varies = first.rounds, True
                merged[gap.key()] = gap
    return list(merged.values())


# ---------------------------------------------------------------- the report


@dataclass(frozen=True)
class IdleWindow:
    """One idle window of a compiled program: a stretch of at least the
    threshold in which a qudit's drive line has no operation, none of its
    tones plays, and its readout does not run.

    ``start`` and ``end`` are in clock cycles, of the window's first
    execution; ``rounds`` counts its executions (the iterations of a loop
    it lies in), and when they differ the shortest is given. ``before``
    names the operation the window ends at, as ``"drive() at line 12 of
    cycle()"``, or is ``None`` for a window that lasts to the program's
    end. ``filled`` says whether the window was replaced, and ``result``
    what was done: ``"filled by echo()"``, or why it was left alone.
    """

    qudit: int
    start: int
    end: int
    rounds: int
    before: str | None
    filled: bool
    result: str

    @property
    def cycles(self) -> int:
        """The window's length in clock cycles."""
        return self.end - self.start


class IdleWindows(tuple[IdleWindow, ...]):
    """The idle windows of a compiled program, in the order of their
    qudits and, per qudit, of the program. ``print()`` shows them as a
    table."""

    __slots__ = ()

    def __str__(self) -> str:
        if not self:
            return "no idle windows"
        rows = [("qudit", "start", "end", "cycles", "rounds", "before", "result")]
        for w in self:
            rows.append(
                (
                    str(w.qudit),
                    str(w.start),
                    str(w.end),
                    str(w.cycles),
                    str(w.rounds),
                    w.before or "the end of the program",
                    w.result,
                )
            )
        widths = [max(len(row[i]) for row in rows) for i in range(7)]
        return "\n".join(
            "  ".join(
                cell.rjust(widths[i]) if i < 5 else cell.ljust(widths[i])
                for i, cell in enumerate(row)
            ).rstrip()
            for row in rows
        )


def kept(gap: Gap, part: tuple[int, int], rule: Any, log: list[Site]) -> str | None:
    """Why the idle part *part* of *gap* is left alone under *rule*
    whatever the fill, or None."""
    lo, hi = gap.lo, gap.hi
    if gap.first and not rule.leading:
        # Before the qudit's first operation; a pulse at its tone from
        # another line and its own readout are operations too.
        if not any(what in _ON_QUDIT and a < part[0] for _, a, _, what in gap.inner):
            return "left alone: before the qudit's first operation (leading=True fills it)"
    if gap.last and not rule.trailing:
        if not any(what in _ON_QUDIT and a >= part[1] for _, a, _, what in gap.inner):
            return "left alone: after the qudit's last operation (trailing=True fills it)"
    if gap.branched():
        return "left alone: in or next to an arm of a branch"
    if gap.barrier:
        return "left alone: after a wait whose length is known only when the program runs"
    if gap.detuned:
        return "left alone: the kernel changed the frequency of the qudit's tone"
    if gap.varies:
        return "left alone: its rounds differ"
    if not gap.ordered():
        return "left alone: the program writes what surrounds it in another order than it runs"
    if lo is not None and hi is not None and lo.site is not None and hi.site is not None:
        if hi.site <= lo.site:
            return "left alone: across the end of a loop iteration (not built yet)"
    return None


# ---------------------------------------------------------------- what is written where


class Tile:
    """What is written at one point of the program on one qudit's drive
    line: waits and fills back to back, as ``("wait", cycles)`` and
    ``("fill", cycles)``, in the region *scope*."""

    __slots__ = ("qudit", "items", "scope", "number")

    def __init__(
        self, qudit: int, items: list[tuple[str, int]], scope: tuple[int, ...], number: int
    ) -> None:
        self.qudit = qudit
        self.items = items
        self.scope = scope
        self.number = number


class Plan:
    """The tiles to write right before an op of the kernel and right after
    one, by the op's number. With *pads*, each fill is written as a wait
    of its window's length."""

    def __init__(self, pads: bool = False) -> None:
        self.before: dict[int, list[Tile]] = {}
        self.after: dict[int, list[Tile]] = {}
        self.pads = pads
        self.tiles = 0

    def add(
        self, where: str, site: int, qudit: int, items: list[tuple[str, int]], scope: Any
    ) -> None:
        points = self.before if where == "before" else self.after
        points.setdefault(site, []).append(Tile(qudit, items, scope, self.tiles))
        self.tiles += 1


def _site(event: _Event | None, log: list[Site]) -> Site | None:
    if event is None or event.site is None or not 0 <= event.site < len(log):
        return None
    return log[event.site]


def _region(gap: Gap, log: list[Site]) -> tuple[int, ...]:
    """The region a gap lies in: the innermost that holds both the op it
    starts after and the op it ends at."""
    lo, hi = _site(gap.lo, log), _site(gap.hi, log)
    if lo is None or hi is None:
        return ()
    region = []
    for a, b in zip(lo.scope, hi.scope, strict=False):
        if a != b:
            break
        region.append(a)
    return tuple(region)


def _point(
    a: _Event | None, b: _Event | None, region: tuple[int, ...], log: list[Site]
) -> tuple[str, int] | None:
    """Where the stretch from *a* to *b* is written: right before *b*, else
    right after *a*, if the tracer is between two statements of *region*
    there. An op the passes made of several traced ones has no point after
    it: that point is inside it."""
    site = _site(b, log)
    if site is not None and site.enter and site.scope == region:
        assert b is not None and b.site is not None
        return "before", b.site
    site = _site(a, log)
    if site is not None and site.leave and site.scope == region:
        assert a is not None and a.site is not None
        if a.kind in ("drive", "readout", "wait") and site.cycles != a.end - a.start:
            return None
        return "after", a.site
    return None


def _tiles(
    gap: Gap, fills: set[tuple[int, int]], log: list[Site]
) -> list[tuple[str, int, list[tuple[str, int]], tuple[int, ...]]] | None:
    """The whole of *gap* as waits and fills (its idle parts in *fills*),
    cut where the kernel plays or changes the qudit's tone: each piece with
    the point it is written at, or None if one has none. The whole gap is
    written so that its line has no slack left: every piece then stays
    where it is planned under every schedule policy."""
    region = _region(gap, log)
    bounds = [(event, start) for event, start, _, what in gap.inner if what in _CHAINED]
    edges = [gap.start, *(start for _, start in bounds), gap.end]
    points = [gap.lo, *(event for event, _ in bounds), gap.hi]
    parts = gap.idle()
    out = []
    for k in range(len(edges) - 1):
        a, b = edges[k], edges[k + 1]
        if b <= a:
            continue
        items: list[tuple[str, int]] = []

        def wait(cycles: int, items: list[tuple[str, int]] = items) -> None:
            if items and items[-1][0] == "wait":
                items[-1] = ("wait", items[-1][1] + cycles)
            else:
                items.append(("wait", cycles))

        at = a
        for u, v in parts:
            if v <= a or u >= b:
                continue
            if u > at:
                wait(u - at)
            if (u, v) in fills:
                items.append(("fill", v - u))
            else:
                wait(v - u)
            at = v
        if b > at:
            wait(b - at)
        where = _point(points[k], points[k + 1], region, log)
        if where is None:
            return None
        out.append((*where, items, region))
    return out


_NO_POINT = "left alone: the program has no statement to write it before or after"


def plan(
    found: dict[int, list[Gap]],
    rule: Any,
    log: list[Site],
    threshold: int,
    chosen: frozenset[int] | None,
    pads: bool = False,
) -> tuple[Plan, IdleWindows]:
    """What to write for *rule* into the gaps *found*, and the windows with
    what is done in each."""
    planned = Plan(pads)
    name = "a wait" if pads else f"{rule.fill.__name__}()"
    rows = []
    for qudit in sorted(found):
        if chosen is not None and qudit not in chosen:
            continue
        for gap in found[qudit]:
            parts = [part for part in gap.idle() if part[1] - part[0] >= threshold]
            if not parts:
                continue
            reasons = {part: kept(gap, part, rule, log) for part in parts}
            fills = {part for part in parts if reasons[part] is None}
            if fills:
                tiles = _tiles(gap, fills, log)
                if tiles is None:
                    for part in fills:
                        reasons[part] = _NO_POINT
                else:
                    for where, site, items, region in tiles:
                        planned.add(where, site, qudit, items, region)
            for part in parts:
                reason = reasons[part]
                rows.append(
                    IdleWindow(
                        qudit,
                        part[0],
                        part[1],
                        gap.rounds,
                        _before(gap, part, log),
                        reason is None,
                        reason or f"filled by {name}",
                    )
                )
    return planned, IdleWindows(rows)


# ---------------------------------------------------------------- the proof


def _shape(item: Any) -> Any:
    """What an item of a program read back must share with the same item
    of the program before its windows were filled."""
    if item.__class__ is tuple:
        # Loops are numbered as they are met; a fill may hold some.
        return (item[0], *item[2:])
    where = (tuple((copy, trips) for _, copy, trips in item.path), len(item.arms))
    if item.kind == "sync":
        return (item.site, "sync", item.lines, where)
    return (item.site, item.kind, item.line, item.tone, item.start, item.end, item.shifts, where)


def prove(
    lines: list[tuple[str, int]], base: list[Any], module: Any, log: list[Site], name: str
) -> None:
    """Check that *module*, the program of *base* with its windows filled,
    is that program with something written into its gaps: every op of the
    kernel is where it was, and what was written lies back to back from the
    op before it to the op after it. A ``TimingError`` names the first op
    that moved."""
    filled_lines, final = read(module)
    if filled_lines != lines:
        raise InternalError("idle replacement: the filled program has other lines")
    # The kernel's own ops, and of the loops only those that hold some.
    loops = {
        loop
        for item in final
        if item.__class__ is not tuple and (item.site is None or item.site >= 0)
        for loop, _, _ in item.path
    }
    own = [
        item
        for item in final
        if (
            (item[0] != "advance" or item[1] in loops)
            if item.__class__ is tuple
            else (item.site is None or item.site >= 0)
        )
    ]
    for was, now in zip(base, own, strict=False):
        a, b = _shape(was), _shape(now)
        if a == b:
            continue
        site = _site(was if was.__class__ is not tuple else None, log)
        if site is None or a[:2] != b[:2] or a[1] == "sync":
            break
        if a[4] != b[4]:
            moved = f"starts at cycle {b[4]} instead of {a[4]}"
        elif a[6] != b[6]:
            moved = (
                f"repeats every {', '.join(map(str, b[6]))} cycles instead of every "
                f"{', '.join(map(str, a[6]))}"
            )
        else:
            moved = f"ends at cycle {b[5]} instead of {a[5]}"
        raise CompilationError(
            f"idle replacement: with the idle windows of {name}() filled, "
            f"{site.describe()} {moved}",
            code="timing",
        )
    else:
        if len(base) == len(own):
            _back_to_back(final, name)
            return
    raise InternalError(
        f"idle replacement: {name}() traced differently the second time, so its filled "
        "program is not the program with its windows filled"
    )


def _back_to_back(final: list[Any], name: str) -> None:
    """Check that each op written into a gap starts where the op before it
    on its line ends, and that the kernel's next op on the line starts
    where the last written one ends."""
    ends: dict[int, int] = {}
    latest: dict[int, _Event] = {}
    tiled: set[int] = set()
    for item in final:
        if item.__class__ is tuple:
            if item[0] == "advance":
                # The iterations of a loop that are not listed: its lines
                # are that much further on.
                _, loop, depth, iterations = item
                for line, event in latest.items():
                    ends[line] += event.advance(depth, loop, iterations)
            continue
        if item.kind == "sync":
            # Its lines go on from where the last of them is free; what was
            # written before it on a line ends there.
            meet = max((ends.get(line, 0) for line in item.lines), default=0)
            last = max(
                (latest[line] for line in item.lines if line in latest),
                key=lambda event: event.end,
                default=None,
            )
            for line in item.lines:
                if line in tiled and ends[line] != meet:
                    raise CompilationError(
                        f"idle replacement: what was written into an idle window of {name}() "
                        f"does not stay in place: line {line} is free at cycle {ends[line]}, "
                        f"before the sync it was written up to, at {meet}",
                        code="timing",
                    )
                ends[line] = meet
                if last is not None:
                    latest[line] = last
            tiled.difference_update(item.lines)
            continue
        if item.kind not in ("drive", "readout", "wait"):
            continue
        line = item.line
        written = item.site is not None and item.site < 0
        expected = ends.get(line, 0)
        if (written or line in tiled) and item.start != expected:
            raise CompilationError(
                f"idle replacement: what was written into an idle window of {name}() does "
                f"not stay in place: a {item.kind} on line {line} starts at cycle "
                f"{item.start}, not at {expected} where the operation before it ends",
                code="timing",
            )
        if written:
            tiled.add(line)
        else:
            tiled.discard(line)
        ends[line] = item.end
        latest[line] = item


def report(
    found: dict[int, list[Gap]],
    rule: Any,
    log: list[Site],
    threshold: int,
    chosen: frozenset[int] | None,
) -> IdleWindows:
    """The windows of *found* under *rule*, none of them filled: the idle
    parts of at least *threshold* cycles on the qudits *chosen* (None:
    every qudit)."""
    rows = []
    for qudit in sorted(found):
        if chosen is not None and qudit not in chosen:
            continue
        for gap in found[qudit]:
            for part in gap.idle():
                if part[1] - part[0] < threshold:
                    continue
                reason = kept(gap, part, rule, log)
                rows.append(
                    IdleWindow(
                        qudit,
                        part[0],
                        part[1],
                        gap.rounds,
                        _before(gap, part, log),
                        False,
                        reason or "not filled: the rule has no fill",
                    )
                )
    return IdleWindows(rows)


def _before(gap: Gap, part: tuple[int, int], log: list[Site]) -> str | None:
    """What the idle part *part* of *gap* ends at."""
    ending = gap.hi
    for event, start, _, _ in gap.inner:
        if start == part[1]:
            ending = event
            break
    if ending is None:
        return None
    if ending.site is None or not 0 <= ending.site < len(log):
        return f"{ending.kind}()"
    return log[ending.site].describe()
