# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pulse-verify's check of a schedule's times, against a walk of every corner.

The verifier walks a scheduled function in execution order, each loop at
its first two iterations and its last, and checks every drive, readout and
wait: not before time zero, not before the measurements its branches wait
for, not before the operation before it on its physical line ends. The
reference here is that walk written plainly, loop inside loop, every corner
visited: what the verifier reports (which operation, which message, with
which times, each operation once and in which order) is held to it on
random nests of loops and branches."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from cudaq_pulse._native._cudaq_pulse_native import MLIRPipeline

_DRIVE = "!pulse.drive_line, !pulse.tone"
_READ = "!pulse.readout_line, !pulse.tone"


@dataclass
class Timed:
    """A drive on drive line *line*, or the readout (line -1)."""

    line: int
    start: int
    duration: int
    periods: list[int]
    measures: int | None = None  # the measurement a readout makes
    at: int = 0  # the line of the module's text it is written on


@dataclass
class Loop:
    trips: int
    body: list


@dataclass
class Branch:
    """A branch on the measurements *measurements*, all of them set."""

    measurements: list[int]
    then: list
    other: list


@dataclass
class Nest:
    lines: int
    body: list
    text: str = ""
    readouts: dict[int, tuple[Timed, int]] = field(default_factory=dict)


# ── The module's text ────────────────────────────────────────────────────────


class _Writer:
    def __init__(self, nest: Nest):
        self.nest = nest
        self.out: list[str] = []
        self.names = 0
        self.durations: set[int] = set()

    def name(self) -> str:
        self.names += 1
        return f"%v{self.names}"

    def emit(self, line: str) -> int:
        self.out.append(line)
        return len(self.out)

    def types(self) -> str:
        return ", ".join([_DRIVE] * self.nest.lines + [_READ])

    def values(self, held) -> str:
        return ", ".join(f"{line}, {tone}" for line, tone in held)

    def block(self, body, held, depth, pad):
        """Writes *body* with the lines *held* (line, tone per drive line,
        then the readout line); returns what holds them after it."""
        held = list(held)
        for node in body:
            if isinstance(node, Timed):
                self.durations.add(node.duration)
                period = (
                    f"period_cycles = array<i64: {', '.join(map(str, node.periods))}>, "
                    if node.periods
                    else ""
                )
                timing = (
                    f"<{{duration_cycles = {node.duration} : i64, {period}"
                    f"start_cycles = {node.start} : i64}}>"
                )
                line, tone = held[node.line]
                new_line, new_tone = self.name(), self.name()
                if node.line >= 0:
                    node.at = self.emit(
                        f"{pad}{new_line}, {new_tone} = pulse.drive {line}, %w{node.duration}, "
                        f"{tone} {timing} : !pulse.drive_line, !pulse.waveform, !pulse.tone "
                        f"-> {_DRIVE}"
                    )
                else:
                    node.at = self.emit(
                        f"{pad}{new_line}, {new_tone}, %m{node.measures} = pulse.readout {line}, "
                        f"%w{node.duration}, {tone}, iq {timing} : !pulse.readout_line, "
                        f"!pulse.waveform, !pulse.tone -> {_READ}, !pulse.measurement"
                    )
                    self.nest.readouts[node.measures] = (node, depth)
                held[node.line] = (new_line, new_tone)
            elif isinstance(node, Loop):
                inner = [(self.name(), self.name()) for _ in held]
                result = self.name()
                args = ", ".join(
                    f"{a} = {x}, {b} = {y}" for (a, b), (x, y) in zip(inner, held, strict=True)
                )
                bound = self.name()
                self.emit(f"{pad}{bound} = arith.constant {node.trips} : index")
                self.emit(
                    f"{pad}{result}:{2 * len(held)} = scf.for %i{self.names} = %lb to {bound} "
                    f"step %step iter_args({args}) -> ({self.types()}) {{"
                )
                after = self.block(node.body, inner, depth + 1, pad + "  ")
                self.emit(f"{pad}  scf.yield {self.values(after)} : {self.types()}")
                self.emit(f"{pad}}}")
                held = [(f"{result}#{2 * i}", f"{result}#{2 * i + 1}") for i in range(len(held))]
            else:
                result = self.name()
                bit = ""
                for measurement in node.measurements:
                    one = self.name()
                    self.emit(
                        f"{pad}{one} = pulse.is_set %m{measurement} : !pulse.measurement -> i1"
                    )
                    if bit:
                        both = self.name()
                        self.emit(f"{pad}{both} = arith.andi {bit}, {one} : i1")
                        one = both
                    bit = one
                self.emit(f"{pad}{result}:{2 * len(held)} = scf.if {bit} -> ({self.types()}) {{")
                for arm, last in ((node.then, False), (node.other, True)):
                    after = self.block(arm, held, depth, pad + "  ")
                    self.emit(f"{pad}  scf.yield {self.values(after)} : {self.types()}")
                    self.emit(f"{pad}}}" if last else f"{pad}}} else {{")
                held = [(f"{result}#{2 * i}", f"{result}#{2 * i + 1}") for i in range(len(held))]
        return held


def write(nest: Nest) -> str:
    """The module of *nest*; every Timed gets the line it is written on."""
    writer = _Writer(nest)
    head = ["module {", "  func.func @shot() {"]
    held = []
    for line in range(nest.lines + 1):
        kind = "drive" if line < nest.lines else "readout"
        head.append(f"    %q{line} = pulse.qudit_alloc {{qudit = {line} : i64}} : !pulse.qudit")
        head.append(
            f"    %l{line}, %t{line} = pulse.get_{kind}_line %q{line} : (!pulse.qudit) -> "
            f"({_DRIVE if line < nest.lines else _READ})"
        )
        held.append((f"%l{line}", f"%t{line}"))
    head += [
        "    %lb = arith.constant 0 : index",
        "    %step = arith.constant 1 : index",
        "    %amp = arith.constant 0.1 : f64",
        "    %zero = arith.constant 0.0 : f64",
    ]
    # The waveforms come after the head, so the body is written first to
    # learn its durations, and its line numbers are moved down by them.
    writer.block(nest.body, held, 0, "    ")
    waves = []
    for duration in sorted(writer.durations):
        waves.append(f"    %n{duration} = arith.constant {duration} : i64")
        waves.append(
            f"    %w{duration} = pulse.square %n{duration}, %amp, %zero : i64, f64, f64 "
            "-> !pulse.waveform"
        )
    moved = len(head) + len(waves)

    def renumber(body):
        for node in body:
            if isinstance(node, Timed):
                node.at += moved
            elif isinstance(node, Loop):
                renumber(node.body)
            else:
                renumber(node.then)
                renumber(node.other)

    renumber(nest.body)
    nest.text = "\n".join(head + waves + writer.out + ["    return", "  }", "}", ""])
    return nest.text


# ── The walk of every corner ─────────────────────────────────────────────────


class Walk:
    """The verifier's walk as it was: every loop at iterations 0, 1 and its
    last, inside one another."""

    def __init__(self, nest: Nest):
        self.nest = nest
        self.errors: list[tuple[int, str]] = []
        self.reported: set[int] = set()
        self.last_end: dict[int, int] = {}
        self.iteration: list[int] = []
        self.measured: int | None = None

    def error(self, op: Timed, message: str) -> None:
        if op.at not in self.reported:
            self.reported.add(op.at)
            self.errors.append((op.at, message))

    def start(self, op: Timed, iteration) -> int:
        return op.start + sum(k * p for k, p in zip(iteration, op.periods, strict=False))

    def block(self, body) -> None:
        for node in body:
            if isinstance(node, Timed):
                self.timed(node)
            elif isinstance(node, Loop):
                self.loop(node)
            else:
                self.branch(node)

    def timed(self, op: Timed) -> None:
        start = self.start(op, self.iteration)
        if len(op.periods) != len(self.iteration):
            self.error(
                op, "scheduled operation needs one period_cycles entry per enclosing scf.for"
            )
            return
        if start < 0:
            self.error(op, f"operation is scheduled before time zero: start {start}")
        if self.measured is not None and start < self.measured:
            self.error(
                op,
                f"operation in a measurement-conditioned branch starts at {start}, before its "
                f"measurement completes at {self.measured}",
            )
        end = self.last_end.get(op.line)
        if end is not None and start < end:
            self.error(
                op,
                "operation overlaps or precedes its physical-line predecessor: start "
                f"{start} < prior end {end}",
            )
        self.last_end[op.line] = max(self.last_end.get(op.line, 0), start + op.duration)

    def periods(self, loop: Loop) -> bool:
        level, ok, period_of = len(self.iteration), True, {}

        def each(body):
            nonlocal ok
            for node in body:
                if isinstance(node, Timed):
                    if len(node.periods) <= level:
                        continue
                    period = node.periods[level]
                    if period < 0:
                        ok = False
                        self.error(node, "period_cycles must not be negative")
                        continue
                    if period_of.setdefault(node.line, period) != period:
                        ok = False
                        self.error(node, "operations on one line in a loop must share its period")
                elif isinstance(node, Loop):
                    each(node.body)
                else:
                    each(node.then)
                    each(node.other)

        each(loop.body)
        return ok

    def loop(self, loop: Loop) -> None:
        if not self.periods(loop):
            return
        self.iteration.append(0)
        k = 0
        while k < loop.trips:
            if k == 2 and loop.trips > 3:
                k = loop.trips - 1
            self.iteration[-1] = k
            self.block(loop.body)
            k += 1
        self.iteration.pop()

    def branch(self, branch: Branch) -> None:
        saved = self.measured
        for measurement in branch.measurements:
            readout, depth = self.nest.readouts[measurement]
            end = self.start(readout, self.iteration[:depth]) + readout.duration
            self.measured = end if self.measured is None else max(self.measured, end)
        before = dict(self.last_end)
        self.block(branch.then)
        after_then = self.last_end
        self.last_end = before
        self.block(branch.other)
        for line, end in after_then.items():
            self.last_end[line] = max(self.last_end.get(line, 0), end)
        self.measured = saved


def walked(nest: Nest) -> list[tuple[int, str]]:
    walk = Walk(nest)
    walk.block(nest.body)
    return walk.errors


# ── What the verifier reports ────────────────────────────────────────────────

_TIMES = (
    "operation is scheduled before time zero",
    "operation in a measurement-conditioned branch starts",
    "operation overlaps or precedes its physical-line predecessor",
    "period_cycles must not be negative",
    "operations on one line in a loop must share its period",
    "scheduled operation needs one period_cycles entry",
)


def verified(text: str) -> list[tuple[int, str]]:
    module = MLIRPipeline().parse_to_module(text)
    try:
        module.run_passes(["pulse-verify"])
    except RuntimeError as error:
        found = []
        for line in str(error).splitlines():
            match = re.match(r'\s*loc\("-":(\d+):\d+\): (.*)', line)
            if match and match.group(2).startswith(_TIMES):
                found.append((int(match.group(1)), match.group(2)))
        return found
    return []


# ── Random nests ─────────────────────────────────────────────────────────────


def random_nest(rng: random.Random, deepest: int = 3) -> Nest:
    """Up to four lines shared between the levels of loops nested up to
    *deepest*, of zero to nine trips, branches on one to three
    measurements with both arms, starts sometimes before what they follow,
    periods sometimes too
    short, zero, negative, off the line's own, or one too few."""
    lines = rng.randrange(1, 5)
    nest = Nest(lines, [])
    measurements = 0
    clock = [0]

    def body(depth, periods, measured, size):
        nonlocal measurements
        nodes = []
        for _ in range(size):
            kind = rng.random()
            if kind < 0.5:
                line = rng.randrange(lines)
                duration = rng.choice([4, 8, 16])
                # Mostly after what came before, sometimes too early.
                clock[0] += rng.choice([0, duration, duration, 2 * duration, -6, -40])
                start = clock[0] + rng.choice([0, 0, 0, -3, 5, -30])
                own = list(periods)
                if own and rng.random() < 0.03:
                    own[rng.randrange(len(own))] += rng.choice([1, -1, 7])
                if rng.random() < 0.02:
                    own = own[:-1] if own else [5]
                nodes.append(Timed(line, start, duration, own))
                clock[0] += duration
            elif kind < 0.65:
                measurements += 1
                clock[0] += rng.choice([0, 20])
                nodes.append(Timed(-1, clock[0], 20, list(periods), measures=measurements))
                clock[0] += 20
                measured = [*measured, measurements]
            elif kind < 0.85 and depth < deepest:
                trips = rng.choice([0, 1, 2, 3, 4, 5, 9])
                period = rng.choice([40, 100, 100, 250, 600, 3, 0, 1, 17])
                if rng.random() < 0.05:
                    period = -5
                inner = body(depth + 1, [*periods, period], measured, rng.randrange(1, 4))
                nodes.append(Loop(trips, inner))
            elif measured:
                which = rng.sample(measured, min(len(measured), rng.choice([1, 1, 2, 3])))
                saved = clock[0]
                then = body(depth, periods, measured, rng.randrange(0, 3))
                reached = clock[0]
                clock[0] = saved
                other = body(depth, periods, measured, rng.randrange(0, 3))
                clock[0] = max(clock[0], reached)
                nodes.append(Branch(which, then, other))
        return nodes

    nest.body = body(0, [], [], rng.randrange(2, 7))
    return nest


def test_the_reference_walk_reads_a_small_nest():
    # One line: a drive, then a loop of three whose drive starts where the
    # one before it ends in the first iteration and four cycles early in
    # the next.
    nest = Nest(1, [Timed(0, 0, 8, []), Loop(3, [Timed(0, 8, 8, [4])])])
    write(nest)
    assert walked(nest) == [
        (
            nest.body[1].body[0].at,
            "operation overlaps or precedes its physical-line predecessor: start 12 < prior end 16",
        )
    ]
    assert verified(nest.text) == walked(nest)


def test_the_verifier_reports_what_a_walk_of_every_corner_reports():
    wrong = 0
    for deepest, cases, seed in ((3, 600, 3), (5, 300, 4)):
        rng = random.Random(seed)
        for case in range(cases):
            nest = random_nest(rng, deepest)
            expected = walked_and_written(nest)
            assert verified(nest.text) == expected, (deepest, case)
            wrong += bool(expected)
    # Schedules with something wrong and schedules with nothing wrong.
    assert 300 < wrong < 850


def _column(depth: int, trips: int, inner: int) -> Nest:
    """One drive of 8 cycles inside *depth* loops of *trips*, each loop's
    period the whole of the loop inside it; the innermost *inner*."""
    periods = [inner * trips ** (depth - 1 - level) for level in range(depth)]
    body: list = [Timed(0, 0, 8, periods)]
    for _ in range(depth):
        body = [Loop(trips, body)]
    return Nest(1, body)


def test_loops_nested_deeper_than_every_corner_can_be_walked():
    # Forty loops of two iterations: two to the fortieth corners. In order,
    # nothing is wrong; with the innermost period two cycles short of the
    # drive, its second iteration starts before the first ends, and that is
    # the one fault, found at the first corner that has it.
    nest = _column(40, 2, 8)
    assert verified(write(nest)) == []
    nest = _column(40, 2, 6)
    text = write(nest)
    drive = nest.body[0]
    while isinstance(drive, Loop):
        drive = drive.body[0]
    assert verified(text) == [
        (
            drive.at,
            "operation overlaps or precedes its physical-line predecessor: start 6 < prior end 8",
        )
    ]


def walked_and_written(nest: Nest) -> list[tuple[int, str]]:
    write(nest)
    return walked(nest)
