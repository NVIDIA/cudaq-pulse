# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coupling lines while a kernel is traced: ``get_coupling_line(*qudits)``
and ``drive`` on the line it gives.

A coupling line reaches a control of the quantum system, one qudit's own
tuning or the control of several qudits, and plays a real waveform at
baseband: it has no tone. The tracer hands ``get_coupling_line`` over from
where it already handles a call it has no emitter for; from the first
coupling line on, the tracers of the trace are of a class that sends a
``drive`` on a coupling line here (``_DrivesCouplingLines``). A kernel
without coupling lines runs nothing of this, and the module loads with the
first kernel that fetches one."""

from __future__ import annotations

import struct
import sys
from typing import TYPE_CHECKING, Any

from .._errors import CompilationError, UnsupportedError
from .ir_builder import IRValue
from .packed_ir_builder import (
    _ALLOC_COUPLING,
    _DRIVE_COUPLING,
    _MAX_PAYLOAD,
    _UNSCHEDULED,
    _WF_ADD,
    _WF_COSINE,
    _WF_CUSTOM_SAMPLES,
    _WF_DRAG,
    _WF_GAUSS_SQUARE,
    _WF_GAUSSIAN,
    _WF_MUL,
    _WF_NEG,
    _WF_SCALE,
    _WF_SQUARE,
    _WF_SUB,
    _WF_TANH_RAMP,
    PackedIRBuilder,
    _header,
)

if TYPE_CHECKING:
    from ._emitter import PulseIREmitter

_FETCH = "get_coupling_line"
_F64, _I64 = struct.Struct("=d"), struct.Struct("=q")
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
    _watch_drives(emitter)
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


class _DrivesCouplingLines:
    """A tracer that has fetched a coupling line: its pulse calls come here
    first. A ``drive`` or ``readout`` whose line is a coupling line is this
    module's (it takes no tone, and no calibrated pulse yet); every other
    call goes on to the tracer."""

    __slots__ = ()

    def _emit_pulse_call(self, fname: str, call_args: list[Any]) -> Any:
        if (fname == "drive" or fname == "readout") and call_args:
            line = call_args[0]
            if line.__class__ is IRValue and line.vtype == _LINE:
                return drive(self, fname, call_args)  # type: ignore[arg-type]
        return super()._emit_pulse_call(fname, call_args)  # type: ignore[misc]


# The class of a tracer that has fetched a coupling line, by the class it had.
_WATCHING: dict[type, type] = {}


def _watch_drives(emitter: Any) -> None:
    """From the first coupling line of a trace on, the tracer is of a class
    that looks at its pulse calls first (``_DrivesCouplingLines``), and so
    are the tracers of the kernels and helpers it is called from, which
    the line may go back to: they are the tracers of this trace on the call
    stack, since a helper is traced inside its caller. A tracer made later
    takes its maker's class. A trace without coupling lines keeps the
    tracer's class."""
    if isinstance(emitter, _DrivesCouplingLines):
        return
    _watch(emitter)
    from ._emitter import PulseIREmitter as Tracer

    builder = emitter.b
    frame: Any = sys._getframe(1)
    while frame is not None:
        owner = frame.f_locals.get("self")
        # Only a tracer of this trace: nothing else on the stack is read.
        if (
            isinstance(owner, Tracer)
            and getattr(owner, "b", None) is builder
            and not isinstance(owner, _DrivesCouplingLines)
        ):
            _watch(owner)
        frame = frame.f_back


def _watch(emitter: Any) -> None:
    cls = emitter.__class__
    watching = _WATCHING.get(cls)
    if watching is None:
        watching = _WATCHING[cls] = type(
            f"Coupled{cls.__name__}", (_DrivesCouplingLines, cls), {"__slots__": ()}
        )
    emitter.__class__ = watching


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
    for controller in controllers:
        for line in controller.coupling_lines:
            if line.delay_ns:
                raise UnsupportedError(
                    f"the delay of {line!r} is not applied when a program is scheduled yet; "
                    "declare the coupling line without delay_ns"
                )


def _require_declared(b: Any, reached: list[int | None]) -> None:
    """Refuses a line the quantum system declares no control for, or that no
    coupling line of the control system reaches: *reached* are the qudits'
    indices in the system (None, or negative, for a qudit a call binds at
    run time: its line is checked when it is bound, by the control system's
    check of each call, so it is refused without a control system)."""
    target = getattr(getattr(b, "line_tones", None), "__self__", None)
    if target is None:
        return
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
    if any(q is None or q < 0 for q in reached):
        if controllers is None:
            raise CompilationError(
                "get_coupling_line(): the line of a qudit that a call binds (a parameter "
                "annotated Qudit) is checked when the qudit is bound, which a kernel with no "
                "control system does not do",
                code="control_system",
                fix="compile the kernel for a control system, or pass the qudit as a static "
                "argument (leave its parameter unannotated)",
            )
        return
    from .._coupling_lines import _and, reach_problem

    acts_on = tuple(sorted(q for q in reached if q is not None))
    if controllers is not None:
        declared = [line for controller in controllers for line in controller.coupling_lines]
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
                from .._coupling_lines import METHOD

                fix = (
                    "the control system has no coupling line; give one to its controller with "
                    f".{METHOD}([CouplingLine({acts_on[0] if not many else acts_on!r})])"
                )
            raise CompilationError(
                f"get_coupling_line(): no coupling line of the control system acts on exactly "
                f"qudit{'s' if many else ''} {_and(acts_on)}",
                code="wrong_argument",
                fix=fix,
            )
    from ..quantum_systems.components import sources_of

    problem = reach_problem(acts_on, target._description, sources_of(target), system)
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
    if len(call_args) < 2:
        from ._emit_errors import play_arity_error

        raise play_arity_error(fname, len(call_args))
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
            raise calibrated_error(waveform, fname)
        raise CompilationError(
            f"drive(): argument 2 must be a waveform, got {_describe(waveform)}",
            code="wrong_argument",
            fix="pass a real waveform such as square(duration, amplitude)",
        )
    chan = line.chan
    if chan is not None and chan.value is not line:
        line = chan.value
    b = emitter.b
    if isinstance(b, PackedIRBuilder):
        complex_by = _complex_by(b, waveform.vid)
        if complex_by:
            raise CompilationError(
                "drive() on a coupling line plays a real waveform; this one is made from "
                f"{complex_by}",
                code="invalid_waveform",
                fix="a coupling line plays at baseband: give it a real amplitude, and a "
                "drag() only with beta 0",
            )
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
        results = b.emit("pulse.drive_baseband", (line, waveform), (_LINE,), {})
    # Through the channels, which note the move for an open if.
    emitter._lines.rebind_linear(emitter, fname, [line, waveform], results)
    return results


def calibrated_error(pulse: Any, fname: str = "drive") -> CompilationError:
    """The error for a calibrated pulse driven on a coupling line."""
    return CompilationError(
        f"{fname}(): {pulse!r} is a calibrated pulse; the calibrated gates of a coupling line "
        "are not available yet",
        code="calibration",
        fix="drive the coupling line with a waveform, such as square(duration, amplitude)",
    )


# The waveform records pulse-verify knows to be real, and those made from
# other waveforms.
_REAL_SHAPES = frozenset(
    {_WF_GAUSSIAN, _WF_COSINE, _WF_TANH_RAMP, _WF_GAUSS_SQUARE, _WF_CUSTOM_SAMPLES}
)
_COMBINED = frozenset({_WF_ADD, _WF_SUB, _WF_MUL})


def _complex_by(b: Any, vid: int) -> str | None:
    """What the waveform *vid* of the packed program is made from that has
    an imaginary part, where the program says: a square with an imaginary
    amplitude, a DRAG pulse with a beta other than 0, or a sum, difference,
    product, multiple or negation of one. None for a real waveform and for
    one the program does not decide (a beta that is a kernel argument, a
    waveform chosen at run time), which pulse-verify decides. The records
    are read once: each call reads those written since the last."""
    scan = b.__dict__.get("_coupling_waveforms")
    if scan is None:
        scan = b._coupling_waveforms = [0, {}]
        undo = b.rollback

        def rollback(checkpoint: tuple[Any, ...]) -> None:
            # A traced body that is undone gives its records and value ids
            # back: read from the start again.
            scan[0] = 0
            scan[1].clear()
            undo(checkpoint)

        b.rollback = rollback
    found: dict[int, str | None] = scan[1]
    buf, cur, end = b._buf, scan[0], b._cur
    while cur < end:
        word = buf[cur]
        opcode, length, mask = word & 0xFF, (word >> 8) & 0xFF, word >> 16
        start = cur + 1
        if length == _MAX_PAYLOAD:
            length, start = buf[cur + 1], cur + 2
        if opcode == _WF_SQUARE:
            imaginary = _F64.unpack(_I64.pack(buf[start + 3]))[0]
            literal = not mask & (1 << 3)
            found[buf[start]] = (
                f"square() with the amplitude's imaginary part {imaginary!r}"
                if literal and imaginary != 0.0
                else None
            )
        elif opcode == _WF_DRAG:
            beta = _F64.unpack(_I64.pack(buf[start + 4]))[0]
            if mask & (1 << 4):
                found.pop(buf[start], None)
            else:
                found[buf[start]] = f"drag() with beta {beta!r}" if beta != 0.0 else None
        elif opcode in _REAL_SHAPES:
            found[buf[start]] = None
        elif opcode in _COMBINED or opcode == _WF_NEG or opcode == _WF_SCALE:
            operands = buf[start + 1 : start + (2 if opcode == _WF_SCALE else length)]
            reasons = [found.get(v, "") for v in operands]
            by = next((r for r in reasons if r), None)
            if by:
                found[buf[start]] = by
            elif all(r is None for r in reasons):
                found[buf[start]] = None
            else:
                found.pop(buf[start], None)
        cur = start + length
    scan[0] = end
    return found.get(vid)


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
