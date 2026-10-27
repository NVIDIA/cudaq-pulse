# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""What ``backends.validate`` runs: the export test corpus and its checks.

Imported by ``validate`` alone, so nothing here costs anything until a
backend is validated.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import cudaq_pulse as pulse

from ._device import Qudit
from .backends import Backend, IncompatibleBackendError
from .compiler import MachineModel
from .controller import (
    _FORMATS,
    _KINDS,
    FORMAT,
    ControllerLimitError,
    ControllerLimits,
    Event,
    unroll,
)
from .kernel._kernel import Kernel
from .ops import (
    drag,
    drive,
    gaussian,
    get_drive_line,
    get_readout_line,
    readout,
    shift_phase,
    square,
    sync,
    wait,
)
from .targets import Qubit, TargetDescription

if TYPE_CHECKING:
    from .compiler import Lowered

# The kinds whose timing and order a backend reports and is checked on: the
# others (waits, syncs, loops) follow from them.
_TIMED = frozenset({Event.PLAY, Event.ACQUIRE, Event.CALL, Event.ANSWER})
_QUDITS = 2


# ── Corpus ───────────────────────────────────────────────────────────────────


def _build() -> dict[str, Kernel[[Qudit, Qudit]]]:
    """One program per kind of event, named for it, and one with a branch."""

    @pulse.classical(max_latency_ns=200, name="echo")
    def echo(bits: pulse.Bits) -> pulse.Bits:
        return [bits[0]]

    @pulse.kernel
    def plays(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        e = get_drive_line(r)
        drive(d, gaussian(32, 0.4, 8.0))
        drive(e, square(16, 0.1))
        shift_phase(d.tone, 0.5)
        drive(d, drag(32, 0.2, 8.0, 0.3))

    @pulse.kernel
    def acquires(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        drive(d, gaussian(32, 0.5, 8.0))
        sync(d, ro)
        readout(ro, square(120, 0.05))

    @pulse.kernel
    def waits(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        drive(d, square(16, 0.1))
        wait(d, 100)
        drive(d, square(16, 0.1))

    @pulse.kernel
    def syncs(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        e = get_drive_line(r)
        ro = get_readout_line(q)
        drive(d, gaussian(48, 0.3, 12.0))
        drive(e, square(16, 0.1))
        sync(d, e, ro)
        drive(e, square(16, 0.1))
        readout(ro, square(80, 0.05))

    @pulse.kernel
    def loops(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        for _ in range(6):
            drive(d, gaussian(16, 0.2, 4.0))
            sync(d, ro)
            readout(ro, square(40, 0.05))
            sync(d, ro)

    @pulse.kernel
    def branches(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        drive(d, gaussian(32, 0.5, 8.0))
        sync(d, ro)
        m = readout(ro, square(80, 0.05))
        sync(d, ro)
        if m:
            drive(d, gaussian(32, 0.5, 8.0))

    @pulse.kernel
    def calls(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        m = readout(ro, square(80, 0.05))
        echo.submit([m])
        drive(d, square(16, 0.1))

    @pulse.kernel
    def answers(q: Qudit, r: Qudit) -> None:
        d = get_drive_line(q)
        ro = get_readout_line(q)
        m = readout(ro, square(80, 0.05))
        if echo.submit([m]).result()[0]:
            drive(d, gaussian(32, 0.5, 8.0))

    return {
        Event.PLAY: plays,
        Event.ACQUIRE: acquires,
        Event.WAIT: waits,
        Event.SYNC: syncs,
        Event.LOOP: loops,
        Event.CALL: calls,
        Event.ANSWER: answers,
        "branch": branches,
    }


_built: dict[str, Kernel[[Qudit, Qudit]]] = {}


def _corpus() -> Mapping[str, Kernel[[Qudit, Qudit]]]:
    """The corpus, built on first use."""
    if not _built:
        _built.update(_build())
    return _built


def _unsupported_event(program: dict[str, Any]) -> dict[str, Any]:
    """*program* with an event of a kind no format has."""
    probe = copy.deepcopy(program)
    last = probe["events"][-1]
    probe["events"].append({**last, "kind": "forward_outcome"})
    return probe


def _newer_format(program: dict[str, Any]) -> dict[str, Any]:
    """*program* in a format this cudaq-pulse does not write."""
    probe = copy.deepcopy(program)
    probe["format"] = f"cudaq-pulse.controller/{len(_FORMATS) + 1}"
    return probe


# Exports a backend must refuse with IncompatibleBackendError, and what each
# is refused for.
_PROBES: Mapping[str, tuple[Callable[[dict[str, Any]], dict[str, Any]], str]] = {
    "unsupported_event": (_unsupported_event, "'forward_outcome'"),
    "newer_format": (_newer_format, "a format it does not read"),
}


# ── Report ───────────────────────────────────────────────────────────────────


_TALLIED = {"PASS": "passed", "FAIL": "failed", "REFUSED": "refused", "SKIP": "skipped"}


class _Report:
    """What ``validate`` found: one line per program, ``PASS``, ``FAIL``,
    ``REFUSED`` (the backend raised ``IncompatibleBackendError``) or
    ``SKIP`` (the program exceeds the controller's limits)."""

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.lines: list[tuple[str, str, str]] = []

    @property
    def ok(self) -> bool:
        """Whether no program failed."""
        return all(status != "FAIL" for status, _, _ in self.lines)

    def _add(self, status: str, name: str, detail: str = "") -> None:
        self.lines.append((status, name, detail))

    def __str__(self) -> str:
        rows = [f"{s} {n}: {d}" if d else f"{s} {n}" for s, n, d in self.lines]
        counts = Counter(s for s, _, _ in self.lines)
        tally = ", ".join(f"{n} {_TALLIED[s]}" for s, n in counts.items()) or "nothing run"
        rows.append(f"backend {self.backend!r}: {tally}")
        return "\n".join(rows)

    def __repr__(self) -> str:
        return f"<validation of {self.backend!r}: {'ok' if self.ok else 'failed'}>"


# ── Checks ───────────────────────────────────────────────────────────────────


def _machine(backend: Backend, controller: ControllerLimits | None) -> MachineModel:
    machine = backend.machine_model()
    if controller is not None:
        machine = dataclasses.replace(machine, controller=controller)
    return machine


def _device(machine: MachineModel) -> Any:
    limits = machine.controller
    clock = limits.clock_ghz if limits is not None and limits.clock_ghz else 1.0
    target = TargetDescription(
        name="validation",
        clock_ghz=clock,
        qubits={i: Qubit(i, 5.0e9 + 1.5e8 * i, -2.5e8, 50.0, 30.0) for i in range(_QUDITS)},
    )
    return pulse.device({"target": target})


def _formats(backend: Backend) -> list[str]:
    read = getattr(backend, "formats", None)
    return [f for f in _FORMATS if read is None and f == FORMAT or read and f in read]


def _expected(program: Mapping[str, Any]) -> tuple[int, dict[Any, list[tuple[Any, ...]]]]:
    """The programs *program* is for, and its timed events by line, each
    ``(kind, start_cycles, duration_cycles)`` in program order."""
    flat = unroll(program)
    by_line: dict[Any, list[tuple[Any, ...]]] = {}
    for event in flat["events"]:
        kind = _KINDS.get(event["kind"], event["kind"])
        if kind in _TIMED and event["start_cycles"] is not None:
            by_line.setdefault(event["line"], []).append(
                (kind, event["start_cycles"], event["duration_cycles"])
            )
    return max(1, len(program.get("programs") or ())), by_line


def _emitted(programs: Sequence[Iterable[Mapping[str, Any]]]) -> dict[Any, list[tuple[Any, ...]]]:
    by_line: dict[Any, list[tuple[Any, ...]]] = {}
    for events in programs:
        for event in events:
            kind = _KINDS.get(str(event["kind"]), str(event["kind"]))
            if kind in _TIMED:
                by_line.setdefault(event["line"], []).append(
                    (kind, event["start_cycles"], event.get("duration_cycles"))
                )
    return by_line


def _timing(
    expected: Mapping[Any, list[tuple[Any, ...]]], emitted: Mapping[Any, list[tuple[Any, ...]]]
) -> str | None:
    """The first difference in timing or order, or None."""
    for line in sorted(set(expected) | set(emitted), key=repr):
        want, got = expected.get(line, []), emitted.get(line, [])
        for k, (a, b) in enumerate(zip(want, got, strict=False)):
            same = a[0] == b[0] and a[1] == b[1] and (b[2] is None or a[2] == b[2])
            if not same:
                return (
                    f"line {line}: event {k} is a {b[0]} at {b[1]} cycles; "
                    f"the controller program has a {a[0]} at {a[1]}"
                )
        if len(want) != len(got):
            return f"line {line}: emitted {len(got)} events, the controller program has {len(want)}"
    return None


def _limits(
    limits: ControllerLimits | None, emitted: Mapping[Any, list[tuple[Any, ...]]]
) -> str | None:
    """The first declared limit the emitted events exceed, or None."""
    if limits is None:
        return None
    played = sum(
        1 for events in emitted.values() for e in events if e[0] in (Event.PLAY, Event.ACQUIRE)
    )
    if limits.max_events is not None and played > limits.max_events:
        return f"emitted {played} pulses; the controller plays at most {limits.max_events}"
    for line, events in emitted.items():
        starts = sorted(e[1] for e in events if e[0] in (Event.PLAY, Event.ACQUIRE))
        for a, b in zip(starts, starts[1:], strict=False):
            if b - a < limits.issue_cycles:
                return (
                    f"line {line}: two pulses start {b - a} cycles apart; the controller "
                    f"issues one per {limits.issue_cycles}"
                )
    return None


def _simulated(lowered: Lowered) -> list[float]:
    """The probability of outcome 1 of each measurement, in end-time order."""
    result = pulse.simulate(lowered)
    p = [0.0] * len(result.measurements)
    for branch in result.branches:
        for k, level in enumerate(branch.levels):
            if level == 1:
                p[k] += branch.probability
    return p


def _emulation(
    backend: Backend, artifact: Any, lowered: Lowered, shots: int, tolerance: float
) -> str | None:
    """How the backend's emulator differs from the simulator, or None."""
    rows = [list(row) for row in backend.emulate(artifact, shots=shots)]  # type: ignore[attr-defined]
    expected = _simulated(lowered)
    if len(rows) != shots:
        return f"the emulator returned {len(rows)} shots, expected {shots}"
    if any(len(row) != len(expected) for row in rows):
        return f"the emulator reports a shot without {len(expected)} measurements"
    for k, p in enumerate(expected):
        mean = sum(1 for row in rows if row[k] == 1) / shots
        # The tolerance, widened by three standard errors of the shots.
        allowed = tolerance + 3.0 * math.sqrt(p * (1.0 - p) / shots)
        if abs(mean - p) > allowed:
            return (
                f"measurement {k}: the emulator gives 1 in {mean:.3f} of shots, "
                f"the simulator {p:.3f} (tolerance {allowed:.3f})"
            )
    return None


def _emit(
    report: _Report,
    backend: Backend,
    name: str,
    program: dict[str, Any],
    lowered: Lowered,
    limits: ControllerLimits | None,
    shots: int,
    tolerance: float,
) -> bool:
    """Emit one program and check it; whether the backend emitted it."""
    try:
        artifact = backend.emit(program)
    except IncompatibleBackendError as e:
        report._add("REFUSED", name, str(e))
        return False
    except Exception as e:
        report._add(
            "FAIL",
            name,
            f"emit() raised {type(e).__name__}: {e}; a program it cannot emit is refused "
            "with IncompatibleBackendError",
        )
        return False
    decode = getattr(backend, "events", None)
    if decode is not None:
        programs = list(decode(artifact))
        count, expected = _expected(program)
        if len(programs) != count:
            s = "" if len(programs) == 1 else "s"
            report._add("FAIL", name, f"emitted {len(programs)} program{s}, expected {count}")
            return True
        emitted = _emitted(programs)
        problem = _timing(expected, emitted) or _limits(limits, emitted)
        if problem is not None:
            report._add("FAIL", name, problem)
            return True
    if getattr(backend, "emulate", None) is not None:
        problem = _emulation(backend, artifact, lowered, shots, tolerance)
        if problem is not None:
            report._add("FAIL", name, problem)
            return True
    report._add("PASS", name)
    return True


def validate(
    backend: Backend,
    controller: ControllerLimits | None = None,
    shots: int = 1000,
    tolerance: float = 0.05,
) -> _Report:
    """See ``cudaq_pulse.backends.validate``."""
    report = _Report(backend.name)
    machine = _machine(backend, controller)
    dev = _device(machine)
    formats = _formats(backend)
    tagged = len(formats) > 1
    emitted = 0
    first: dict[str, Any] | None = None
    for kind, kernel in _corpus().items():
        try:
            compiled = pulse.compile(kernel, device=dev, machine=machine)
            lowered = compiled.lower(*dev.qudit_handles[:_QUDITS])
        except ControllerLimitError as e:
            report._add("SKIP", str(kind), f"exceeds the controller's limits: {e.violations[0]}")
            continue
        for fmt in formats:
            version = 1 if fmt == FORMAT else 2
            name = f"{kind} (/{version})" if tagged else str(kind)
            program = lowered.export_controller(version=version)
            first = first or program
            emitted += _emit(
                report, backend, name, program, lowered, machine.controller, shots, tolerance
            )
    if first is not None:
        for name, (make, what) in _PROBES.items():
            try:
                backend.emit(make(first))
            except IncompatibleBackendError:
                report._add("PASS", name)
            except Exception as e:
                report._add(
                    "FAIL",
                    name,
                    f"emit() raised {type(e).__name__} for {what}; it should raise "
                    "IncompatibleBackendError",
                )
            else:
                report._add(
                    "FAIL",
                    name,
                    f"emit() accepted {what}; it should raise IncompatibleBackendError",
                )
    if not emitted and any(s == "REFUSED" for s, _, _ in report.lines):
        report._add("FAIL", "corpus", "the backend refused every program")
    return report


def opted_in(
    descriptions: Iterable[tuple[str, Mapping[str, Any], ControllerLimits]],
    get: Callable[[str], Backend],
    shots: int = 1000,
    tolerance: float = 0.05,
) -> list[tuple[str, _Report]]:
    """The CI hook: for each controller description, its name, its parsed
    table and the controller it describes, the report of validating the
    backend its table opts in with ``backend = "..."``; descriptions
    without one are left out."""
    out = []
    for name, table, controller in descriptions:
        chosen = table.get("backend")
        if chosen:
            out.append((name, validate(get(chosen), controller, shots, tolerance)))
    return out
