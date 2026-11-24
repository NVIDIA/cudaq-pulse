# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coupling lines while a kernel is traced: ``get_coupling_line(*qudits)``
and ``drive`` on the line it gives.

A coupling line reaches a control of the quantum system, one qudit's own
tuning or the control of several qudits, and plays a real waveform at
baseband: it has no tone. The tracer hands both calls over from where it
already handles a call that is not (line, waveform, tone), so a kernel
without coupling lines runs nothing of this, and the module loads with the
first kernel that fetches one."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._errors import CompilationError, UnsupportedError
from .ir_builder import IRValue
from .packed_ir_builder import (
    _ALLOC_COUPLING,
    _DRIVE_COUPLING,
    _MAX_PAYLOAD,
    _UNSCHEDULED,
    PackedIRBuilder,
    _header,
)

if TYPE_CHECKING:
    from ._emitter import PulseIREmitter

_FETCH = "get_coupling_line"
_LINE = "coupling_line"


def get_coupling_line(emitter: PulseIREmitter, call_args: list[Any]) -> tuple[Any, ...]:
    """``get_coupling_line(*qudits)``: the line of the control of *qudits*,
    fetched once for the whole program, as the other lines are."""
    if not call_args:
        raise CompilationError(
            "get_coupling_line() takes the qudits the line reaches (one qudit: its own "
            "tuning; two: the control of their coupling), got none",
            code="wrong_argument",
        )
    for place, qudit in enumerate(call_args):
        if qudit.__class__ is not IRValue or qudit.vtype != "qref":
            from ._emitter import _describe

            raise CompilationError(
                f"get_coupling_line(): argument {place + 1} must be a qudit, got "
                f"{_describe(qudit)}",
                code="wrong_argument",
                fix="pass the qudits themselves, as get_coupling_line(a) or "
                "get_coupling_line(a, d)",
            )
    b = emitter.b
    indices: dict[int, int] = getattr(b, "_qref_to_qudit", {})
    # The line is named by the set of its qudits: in the system's order where
    # they are known, so that any order of the call fetches one line.
    qudits = sorted(call_args, key=lambda q: (indices.get(q.vid, q.vid), q.vid))
    named = [indices.get(q.vid, ("value", q.vid)) for q in qudits]
    if len(set(named)) != len(named) or len({q.vid for q in qudits}) != len(qudits):
        raise CompilationError(
            "get_coupling_line() got the same qudit twice; a coupling line reaches each of "
            "its qudits once",
            code="wrong_argument",
        )
    lines = emitter._lines
    key = ("_".join(str(q.vid) for q in qudits), _FETCH)
    held = lines.held.get(key)
    if held is not None:
        lines.share(held)
        return (held.cells[0].value,)
    _require_scheduled(b)
    _require_declared(b, [indices.get(q.vid) for q in qudits])
    # Counted as an acquisition and not listed among those a rolled loop
    # lifts out: a loop whose body first fetches a coupling line is unrolled.
    emitter._acquired[0] += 1
    results: tuple[IRValue, ...]
    if isinstance(b, PackedIRBuilder):
        results = (_packed_line(b, qudits, indices),)
    else:
        results = b.emit("pulse.get_coupling_line", tuple(qudits), (_LINE,), {})
    lines.add(key, results)
    return results


def _require_scheduled(b: Any) -> None:
    """Refuses what the scheduler does not do for a coupling line yet: its
    controller among several, and its delay."""
    machine: Any = getattr(b, "machine", None)
    system = getattr(machine, "_system", None)
    if system is not None:
        controllers = system.controllers
    elif getattr(machine, "controller", None) is not None:
        controllers = (machine.controller,)
    else:
        return
    if len(controllers) > 1:
        raise UnsupportedError(
            "coupling lines are not scheduled on a control system of several controllers "
            "yet; compile the kernel for one controller"
        )
    from .._coupling_lines import lines_of

    for controller in controllers:
        for line in lines_of(controller) if controller.coupling_lines else ():
            if line.delay_ns:
                raise UnsupportedError(
                    f"the delay of {line!r} is not applied when a program is scheduled yet; "
                    "declare the coupling line without delay_ns"
                )


def _require_declared(b: Any, reached: list[int | None]) -> None:
    """Refuses a line the quantum system declares no control for, or that no
    coupling line of the control system reaches: *reached* are the qudits'
    indices in the system (None, or negative, for a qudit a call binds at
    run time, which is checked when it is bound)."""
    target = getattr(getattr(b, "line_tones", None), "__self__", None)
    if target is None or any(q is None or q < 0 for q in reached):
        return
    from .._coupling_lines import _and, lines_of, reach_problem

    acts_on = tuple(sorted(q for q in reached if q is not None))
    # The control system the kernel is compiled for, else the quantum system's.
    machine: Any = getattr(b, "machine", None)
    system = getattr(machine, "_system", None)
    if system is not None:
        controllers: Any = system.controllers
    elif getattr(machine, "controller", None) is not None:
        controllers = (machine.controller,)
    else:
        system = target.control_system
        controllers = None if system is None else system.controllers
    if controllers is not None:
        declared = [
            line
            for controller in controllers
            if controller.coupling_lines
            for line in lines_of(controller)
        ]
        if not any(line.qudits == acts_on for line in declared):
            near = [repr(line) for line in declared if not set(acts_on).isdisjoint(line.qudits)]
            many = len(acts_on) > 1
            if near:
                fix = f"the coupling lines that reach {'them' if many else 'it'} are " + ", ".join(
                    near
                )
            elif declared:
                fix = "the control system's coupling lines are " + ", ".join(map(repr, declared))
            else:
                fix = (
                    "the control system has no coupling line; give its controller "
                    f"coupling_lines=[CouplingLine({acts_on[0] if not many else acts_on!r})]"
                )
            raise CompilationError(
                f"get_coupling_line(): no coupling line of the control system acts on exactly "
                f"qudit{'s' if many else ''} {_and(acts_on)}",
                code="wrong_argument",
                fix=fix,
            )
    problem = reach_problem(acts_on, target._description, target.calibration, system)
    if problem is not None:
        raise CompilationError(
            f"get_coupling_line(): {problem[0]}", code="wrong_argument", fix=problem[1]
        )


def _packed_line(b: Any, qudits: list[IRValue], indices: dict[int, int]) -> IRValue:
    """The packed record of a coupling line: its result takes one value id
    and each further qudit one more, unused, so that the builder's table
    of the qudits that lines are on lists every qudit of the line and
    forgets them with the line when a traced loop body is undone."""
    count = len(qudits)
    if count + 2 >= _MAX_PAYLOAD:
        raise CompilationError(
            f"get_coupling_line() of {count} qudits: a coupling line reaches at most "
            f"{_MAX_PAYLOAD - 3}",
            code="limit_exceeded",
        )
    vid = b._next_id
    b._next_id = vid + count
    b._op_count += 1
    reached = [indices.get(q.vid, 0) for q in qudits]
    for place, qudit in enumerate(reached):
        b._line_qudit[vid + place] = qudit
    b._put((_header(_ALLOC_COUPLING, count + 2), count, vid, *reached))
    return IRValue(vid, _LINE)


def drive(emitter: PulseIREmitter, fname: str, call_args: list[Any]) -> tuple[Any, ...]:
    """``drive(line, waveform)`` on a coupling line: the waveform itself, at
    baseband, with no tone."""
    line = call_args[0]
    if fname != "drive":
        raise CompilationError(
            f"{fname}() takes a readout line, got a coupling line; a coupling line plays "
            "with drive(line, waveform)",
            code="wrong_argument",
        )
    if len(call_args) > 2 and call_args[2] is not None:
        raise CompilationError(
            "drive() on a coupling line takes no tone: a coupling line plays the waveform "
            "itself, at baseband",
            code="wrong_argument",
            fix="call it as drive(line, waveform)",
        )
    waveform = call_args[1]
    if waveform.__class__ is not IRValue or waveform.vtype != "waveform":
        from ..quantum_systems.base import _Component
        from ._emitter import _describe

        if isinstance(waveform, _Component):
            raise CompilationError(
                f"drive(): {waveform!r} is a calibrated pulse; the calibrated gates of a "
                "coupling line are not available yet",
                code="calibration",
                fix="drive the coupling line with a waveform, such as square(duration, amplitude)",
            )
        raise CompilationError(
            f"drive(): argument 2 must be a waveform, got {_describe(waveform)}",
            code="wrong_argument",
            fix="pass a real waveform such as square(duration, amplitude)",
        )
    chan = line.chan
    if chan is not None and chan.value is not line:
        line = chan.value
    b = emitter.b
    results: tuple[IRValue, ...]
    if isinstance(b, PackedIRBuilder):
        vid = b._next_id
        b._next_id = vid + 1
        b._op_count += 1
        # start_cycles and duration_cycles are set by the scheduler.
        b._put(
            (_header(_DRIVE_COUPLING, 5), line.vid, waveform.vid, vid, _UNSCHEDULED, _UNSCHEDULED)
        )
        results = (IRValue(vid, _LINE),)
    else:
        results = b.emit("pulse.drive", (line, waveform), (_LINE,), {})
    # Through the channels, which note the move for an open if.
    emitter._lines.rebind_linear(emitter, fname, [line, waveform], results)
    return results


def attribute_error(name: str) -> CompilationError:
    """The error for ``line.name`` on a coupling line."""
    if name == "get_calibration":
        return CompilationError(
            "the calibrated gates of a coupling line are not available yet",
            code="calibration",
            fix="drive the coupling line with a waveform, such as square(duration, amplitude)",
        )
    return CompilationError(
        f"a coupling line has no attribute {name!r}",
        code="unknown_attribute",
        fix="a coupling line plays a baseband waveform: it has no tone and no frame"
        if name in ("tone", "tones")
        else "a coupling line is driven with drive(line, waveform)",
    )
