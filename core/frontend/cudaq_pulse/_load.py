# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cudaq_pulse.load``: a scheduled function of the Pulse dialect that
another compiler built, as a compiled program."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._device import Qudit
    from ._system import Controller, ControlSystem
    from .compiler import ScheduledKernel


class _NoTrace:
    """What the clock plan sets on a tracing builder: a loaded function is
    already traced, so nothing reads it."""

    line_grid: Any = None
    wait_grid: Any = None
    wait_grids: Any = None
    sample_rate: Any = None
    sample_rate_of: Any = None

    def limit_instructions(self, count: int) -> None:
        pass

    def limit_cycles(self, count: int) -> None:
        pass


def load(
    module: str | bytes,
    /,
    *,
    entry: str,
    qudits: Sequence[Qudit],
    control_system: ControlSystem | Controller | None = None,
    classical: Sequence[Any] = (),
) -> ScheduledKernel:
    """The compiled program of *entry*, a scheduled function of *module*.

    *module* is the assembly or the bytecode of an MLIR module of the Pulse
    dialect, written by any compiler. *entry* names a function in it that
    takes one ``!pulse.qudit`` per qudit it plays on and nothing else, gets
    its lines from them (``pulse.get_drive_line``, ``pulse.get_readout_line``)
    and is scheduled: every timed operation has its ``start_cycles`` and
    ``duration_cycles``, in cycles of the quantum system's clock, and the
    function is marked ``pulse.schedule_state = #pulse.schedule_state<timed>``.
    A call in it states its start (``pulse.start_cycles``) and is written in
    place. A module that states its clock
    (``pulse.control_system`` with ``clock_period_fs``) must state the
    quantum system's.

    *qudits* gives, for each argument in order, the qudit of the quantum
    system it plays on; their quantum system is the program's. The lines get
    that system's frequencies. *control_system* is as for ``compile``; without
    it, the quantum system's own control system is used.

    The classical functions the function calls are the module's
    ``classical.callable`` declarations: their types, latency, link and
    dispatch key. *classical* gives functions declared here
    (``pulse.classical``) for them, by name: each must take and give what the
    module states, and its latency and implementation replace the module's,
    so a schedule that takes an answer before that latency allows is refused.

    The schedule is kept as written: ``load`` does not schedule again. The
    function is verified as every scheduled program is, and checked against
    the controller limits; either raises ``CompilationError``. What the
    function returns is dropped: read a record from its readout in the
    export. Attributes another compiler put on the function or on its
    operations are not part of the program: an operation keeps its own
    attributes and those named ``pulse.``.

    The result is what ``kernel.lower(...)`` returns: ``schedule()``,
    ``timeline()``, ``export_to_controller()`` and the backends take it.
    """
    from ._device import Qudit
    from ._errors import CompilationError, InvalidTypeError, InvalidValueError, UnsupportedError
    from ._native import _cudaq_pulse_native as native
    from .compiler import (
        CompileMetrics,
        CompileOptions,
        _check_controller,
        _ClassicalFunctions,
        _clocked_builder,
        _clocked_module,
        _lowered,
    )

    if isinstance(module, str):
        text = module.encode()
    elif isinstance(module, (bytes, bytearray, memoryview)):
        text = bytes(module)
    else:
        raise InvalidTypeError(
            "load() takes a module's assembly (str) or bytecode (bytes), "
            f"got {type(module).__name__}"
        )
    if not isinstance(entry, str) or not entry:
        raise InvalidTypeError(f"load(entry=) takes the function's name, got {entry!r}")
    placed = tuple(qudits)
    for qudit in placed:
        if not isinstance(qudit, Qudit):
            raise InvalidTypeError(
                f"load(qudits=) takes qudits of a quantum system, got {type(qudit).__name__}"
            )
    if not placed:
        raise InvalidValueError("load(qudits=) is empty: a loaded function plays on a qudit")
    target = placed[0].quantum_system
    if any(qudit.quantum_system is not target for qudit in placed):
        raise InvalidValueError("load(qudits=) names qudits of more than one quantum system")
    carriers = tuple(qudit.index for qudit in placed)
    if len(set(carriers)) != len(carriers):
        raise InvalidValueError("load(qudits=) names a qudit twice")

    if control_system is None:
        control_system = target.control_system
    options = CompileOptions.make(control_system=control_system)
    machine = options.machine
    if machine._line_delays:
        raise UnsupportedError(
            "load() keeps the function's schedule, and this control system has line "
            "delays, which move it; load on a control system without line delays"
        )
    try:
        declared = native.PulseModuleBuilder().load_callables(text)
    except (RuntimeError, ValueError) as error:
        raise CompilationError(str(error), code="wrong_argument") from None
    given = _given_functions(classical, declared)
    try:
        generic: Any = native.PulseModuleBuilder().load_function(
            text,
            entry,
            target.clock_ghz,
            len(placed),
            {
                name: (fn.min_latency_ns, fn.nominal_latency_ns, fn.max_latency_ns)
                for name, fn in given.items()
            },
        )
        # A module that calls classical functions is also checked against
        # their latencies: an answer is taken no earlier than it is due. The
        # links they travel on are the control system's.
        if declared:
            links = [row[2] for row in declared]
            system = machine._system
            generic.set_attributes(
                machine.links_attribute(links)
                if system is None or machine.controller is not None
                else system._links_attribute(machine, links, target.clock_ghz)
            )
        generic.run_passes(
            ["pulse-verify", "pulse-classical-check"] if declared else ["pulse-verify"]
        )
    except ValueError as error:
        # The module or the entry is not what load() takes.
        raise CompilationError(str(error), code="wrong_argument") from None
    except RuntimeError as error:
        # The schedule as written does not verify.
        raise CompilationError(str(error), code="timing") from None
    _, frequencies, readouts = target._frequency_tables(max(carriers) + 1)
    bound = generic.place_carriers(carriers, frequencies, readouts)
    clocks = _clocked_builder(_NoTrace(), machine, target)
    fine_delays = _clocked_module(bound, machine, target, clocks, None)
    metrics = CompileMetrics(quantum_system=target.name)
    program = _lowered(
        bound,
        clocks,
        fine_delays,
        quantum_system=target,
        metrics=metrics,
        parameters=(),
        qudits=tuple(sorted(carriers)),
        options=options,
    )
    if declared:
        # The module states the classical functions it calls: their schema,
        # latency, link and dispatch key.
        called = _ClassicalFunctions(_stated_function(*row, given.get(row[0])) for row in declared)
        called.routes = [(row[2], row[1], None) for row in declared]
        program.classical = called
    program = _check_controller(program)
    program.metrics.op_count = len(program.schedule)
    return program


def _stated_type(descriptor: str, function: str) -> Any:
    """The value type a module spells *descriptor*."""
    import re

    from ._errors import UnsupportedError
    from .classical import _Bits, _Byte, _Float, _Floats, _Integer

    found = re.fullmatch(r"(bits|levels|iq):(\d+)|(u8|i32|i64|f32|f64)(?:\[(\d+)\])?", descriptor)
    if found is not None:
        kind, count, scalar, length = found.groups()
        if kind == "bits":
            return _Bits(int(count))
        if scalar == "u8" and length is None:
            return _Byte()
        if scalar in ("f32", "f64"):
            bits = int(scalar[1:])
            return _Float(bits) if length is None else _Floats(int(length), bits)
        if scalar in ("i32", "i64") and length is None:
            return _Integer(int(scalar[1:]))
    raise UnsupportedError(
        f"load: the classical function {function} takes or gives {descriptor!r}, "
        "which a loaded module cannot state yet"
    )


def _stated_function(
    name: str,
    key: int,
    link: int,
    args: Sequence[str],
    results: Sequence[str],
    latency_ns: Sequence[float],
    mode: str,
    submit: str,
    timeout_ns: float | None,
    deadline_ns: float | None,
    reaction_deadline_ns: float | None,
    given: Any,
) -> Any:
    """The classical function a module declares; with *given*, the function
    declared for it, its implementation and latency."""
    from .classical import ClassicalFunction

    if given is not None:
        latency_ns = (given.min_latency_ns, given.nominal_latency_ns, given.max_latency_ns)
    function: Any = ClassicalFunction(
        None if given is None else given.implementation,
        name=name,
        min_latency_ns=latency_ns[0],
        nominal_latency_ns=latency_ns[1],
        max_latency_ns=latency_ns[2],
        latency_mode=mode,
        timeout_ns=timeout_ns,
        submit=submit,
        deadline_ns=deadline_ns,
        reaction_deadline_ns=reaction_deadline_ns,
        link=link,
        _dispatch_key=key,
    )
    function.args = tuple(_stated_type(spelled, name) for spelled in args)
    function.results = tuple(_stated_type(spelled, name) for spelled in results)
    return function


def _given_functions(given: Sequence[Any], declared: Sequence[Any]) -> dict[str, Any]:
    """Per classical function the module declares, the function of
    ``load(classical=)`` given for it: the one of its name, or the one it is
    a version of (``decode__1`` of ``decode``). Their types must agree."""
    import re

    from ._errors import CompilationError, InvalidTypeError, InvalidValueError
    from .classical import ClassicalFunction, _resolve, _unsized_names

    out: dict[str, Any] = {}
    for function in given:
        if not isinstance(function, ClassicalFunction):
            raise InvalidTypeError(
                "load(classical=) takes classical functions (pulse.classical), "
                f"got {type(function).__name__}"
            )
        rows = [
            row
            for row in declared
            if row[0] == function.name or row[0].startswith(function.name + "__")
        ]
        if not rows:
            raise InvalidValueError(
                f"load(classical=): the module calls no classical function named {function.name}"
            )
        for row in rows:
            # A function with lists of unstated length is declared once for
            # every length: its version for the module's lengths is compared.
            version: Any = function
            _resolve(function)
            if function._shape is not None:
                names = _unsized_names(function)
                counts = [int(n) for spelled in row[3] for n in re.findall(r"[:\[](\d+)", spelled)]
                try:
                    version = function.specialize(**dict(zip(names, counts, strict=True)))
                except (ValueError, InvalidTypeError, InvalidValueError):
                    version = None
            took = None if version is None else [value.descriptor for value in version.args]
            gave = None if version is None else [value.descriptor for value in version.results]
            if (took, gave) != (list(row[3]), list(row[4])):
                raise CompilationError(
                    f"load: the module's {row[0]} takes ({', '.join(row[3])}) and gives "
                    f"({', '.join(row[4])}), and the function given for it "
                    + (
                        "has no version of those lengths"
                        if took is None or gave is None
                        else f"takes ({', '.join(took)}) and gives ({', '.join(gave)})"
                    ),
                    code="wrong_argument",
                )
            out[row[0]] = function
    return out
