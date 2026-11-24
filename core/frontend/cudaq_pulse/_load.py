# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cudaq_pulse.load``: a scheduled function of the Pulse dialect that
another compiler built, as a compiled program."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    links: Mapping[int | str, str] | None = None,
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
    module states, with its latency mode and submit policy, and its latency
    and implementation replace the module's. The links are the control
    system's: the module names each link its calls travel on
    (``classical.links``), and *links* gives the control system's name for a
    link the module names otherwise or only numbers
    (``links={"host": "nvqlink"}``). The dispatch keys are the control
    system's too, as for ``compile``: a key a controller fixes for a function
    replaces the module's, and a reserved key or two functions on one link
    and key are refused. A schedule that takes an
    answer before the latency and the link allow, or gives a submission
    less time than the link takes, is refused.

    The schedule is kept as written: ``load`` does not schedule again. The
    function is verified on its qudits and control system as every scheduled
    program is, and checked against what the controllers can play; either
    raises ``CompilationError``. A control system with line delays, or a
    controller with ``max_instructions`` or ``max_program_duration``, is
    refused: ``load`` does not apply or check those yet. The function and
    the functions it calls play on the qudits the entry takes and allocate
    none. What the function returns is dropped: read a record from its
    readout in the export. Attributes another compiler put on the function
    or on its operations are not part of the program: an operation keeps its
    own attributes and those named ``pulse.``.

    The result is what ``kernel.lower(...)`` returns: ``schedule``,
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
    controllers = (
        (machine.controller,)
        if machine.controller is not None
        else (() if machine._system is None else tuple(machine._system.controllers))
    )
    for controller in controllers:
        if controller.max_instructions is not None or controller.max_program_duration is not None:
            raise UnsupportedError(
                "load() does not check a controller's max_instructions or "
                "max_program_duration yet; load on a control system without them"
            )
    builder = native.PulseModuleBuilder()
    try:
        declared = builder.load_callables(text)
    except (RuntimeError, ValueError) as error:
        raise CompilationError(_first_line(error), code="wrong_argument") from None
    given = _given_functions(classical, declared)
    try:
        devices = _target_links(
            sorted({row[2] for row in declared}), builder.load_link_names(text), links, machine
        )
    except RuntimeError as error:
        if isinstance(error, CompilationError):
            raise
        raise CompilationError(_first_line(error), code="wrong_argument") from None
    declared = [(*row[:2], devices[row[2]], *row[3:]) for row in declared]
    # The module states the classical functions it calls: their schema and
    # latency. Their links and dispatch keys are the control system's.
    stated = [_stated_function(*row, given.get(row[0])) for row in declared]
    routes = _target_keys(stated, machine)
    try:
        generic: Any = builder.load_function(
            text,
            entry,
            target.clock_ghz,
            len(placed),
            {
                name: (fn.min_latency_ns, fn.nominal_latency_ns, fn.max_latency_ns)
                for name, (fn, _) in given.items()
            },
            {function.name: route[1] for function, route in zip(stated, routes, strict=True)},
            devices,
        )
        _, frequencies, readouts = target._frequency_tables(max(carriers) + 1)
        bound = generic.place_carriers(carriers, frequencies, readouts)
    except (RuntimeError, ValueError) as error:
        # The module or the entry is not what load() takes.
        raise CompilationError(_first_line(error), code="wrong_argument") from None
    if declared:
        # The links the calls travel on are the control system's.
        used = [row[2] for row in declared]
        system = machine._system
        bound.set_attributes(
            machine.links_attribute(used)
            if system is None or machine.controller is not None
            else system._links_attribute(machine, used, target.clock_ghz)
        )
    clocks = _clocked_builder(_NoTrace(), machine, target)
    fine_delays = _clocked_module(bound, machine, target, clocks, None)
    # The schedule as written is verified on the bound module, as a compiled
    # one is. A module that calls classical functions is also checked against
    # their latencies and its links: an answer is taken no earlier than it
    # is due.
    try:
        bound.run_passes(["pulse-verify"])
    except (RuntimeError, ValueError) as error:
        raise CompilationError(str(error), code="other") from None
    if declared:
        try:
            bound.run_passes(["pulse-classical-check"])
        except (RuntimeError, ValueError) as error:
            raise CompilationError(str(error), code="timing") from None
        _check_submissions(builder.load_submissions(bound.serialize()[0]))
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
    if stated:
        called = _ClassicalFunctions(stated)
        called.routes = routes
        program.classical = called
    program = _check_controller(program)
    program.metrics.op_count = len(program.schedule)
    return program


def _target_links(
    used: Sequence[int],
    named: Mapping[int, str],
    given: Mapping[int | str, str] | None,
    machine: Any,
) -> dict[int, int]:
    """Per link id the module's classical functions state, the id of that
    link on the control system: the link of the name the module gives it,
    or of the name ``load(links=)`` gives for that name or id."""
    from ._errors import CompilationError, InvalidTypeError, InvalidValueError

    given = dict(given or {})
    for key, name in given.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)) or not isinstance(name, str):
            raise InvalidTypeError(
                "load(links=) maps a link's name or id in the module to the name of a "
                f"link of the control system, got {key!r}: {name!r}"
            )
    known = ", ".join(repr(name) for name in machine.link_names) or "none"
    out: dict[int, int] = {}
    taken: set[int | str] = set()
    for link in used:
        stated = named.get(link)
        found: str | None = stated
        keys: tuple[int | str, ...] = (link,) if stated is None else (stated, link)
        for key in keys:
            if key in given:
                found = given[key]
                taken.add(key)
                break
        if found is None:
            raise CompilationError(
                f"load: the module does not name its link {link}; name it in the module's "
                f"classical.links, or give load(links={{{link}: ...}}) the control system's "
                f"link (its links: {known})",
                code="control_system",
            )
        if found not in machine.link_names:
            raise CompilationError(
                f"load: the module's calls travel on the link {found!r}, and the control "
                f"system has no link of that name (its links: {known})",
                code="control_system",
            )
        out[link] = machine.link_names[found]
    unused = [key for key in given if key not in taken]
    if unused:
        raise InvalidValueError(
            f"load(links=) names {unused[0]!r}, which is not a link the module's calls travel on"
        )
    return out


def _target_keys(functions: Sequence[Any], machine: Any) -> list[tuple[int, int, str | None]]:
    """Per classical function of a loaded module, its link id, dispatch key
    and what fixes the key, by the rules ``compile`` applies: a controller
    that fixes the function's key gives it, a key reserved for decoder
    messages is refused, and so are two functions on one link and key."""
    from ._errors import CompilationError
    from .classical import _RESERVED_KEYS
    from .compiler import _dispatch_tables
    from .kernel.packed_ir_builder import _collision, _fixed_key

    tables = (
        _dispatch_tables(machine)
        if machine.controller is not None or machine._system is not None
        else {}
    )
    routes: list[tuple[int, int, str | None]] = []
    seen: dict[tuple[int, int], Any] = {}
    for function in functions:
        key, fixed_by = function._dispatch_key, None
        table = tables.get(function.link)
        if table is not None:
            key, fixed_by = _fixed_key(function, table)
        if key in _RESERVED_KEYS:
            raise CompilationError(
                f"{function.name} is dispatch key {key:#x}, which {_RESERVED_KEYS[key]} uses"
                " for its decoder messages; rename it",
                code="control_system",
            )
        other = seen.get((function.link, key))
        if other is not None:
            raise CompilationError(_collision(other, function, key), code="control_system")
        seen[function.link, key] = function
        routes.append((function.link, key, fixed_by))
    return routes


def _first_line(error: BaseException) -> str:
    return str(error).splitlines()[0] if str(error) else type(error).__name__


def _check_submissions(submissions: Sequence[tuple[str, int, int]]) -> None:
    """Refuse a submission scheduled for less time than this control
    system's link takes to send it. *submissions* is every submission of the
    module as the native loader walks them: the function, the cycles the
    schedule gives it, and the cycles the classical check wrote that the
    link takes. One the check wrote nothing on is refused too: it was not
    checked."""
    from ._errors import CompilationError

    for name, written, needed in submissions:
        if written < 0 or needed < 0:
            raise CompilationError(
                f"load: a submission to {name} has no "
                + ("duration" if written < 0 else "link time to check it against"),
                code="timing",
            )
        if written < needed:
            raise CompilationError(
                f"load: a submission to {name} is given {written} cycle(s), and on this "
                f"control system's link it takes {needed}",
                code="timing",
            )


def _stated_type(descriptor: str, function: str) -> Any:
    """The value type a module spells *descriptor*."""
    import re

    from ._errors import CompilationError
    from .classical import (
        _IQ,
        _Bits,
        _Byte,
        _Bytes,
        _Float,
        _Floats,
        _Integer,
        _Integers,
        _Levels,
    )

    found = re.fullmatch(r"(bits|levels|iq):(\d+)|(u8|i32|i64|f32|f64)(?:\[(\d+)\])?", descriptor)
    if found is not None:
        kind, count, scalar, length = found.groups()
        if kind is not None:
            return {"bits": _Bits, "levels": _Levels, "iq": _IQ}[kind](int(count))
        if scalar == "u8":
            return _Byte() if length is None else _Bytes(int(length))
        if scalar in ("f32", "f64"):
            bits = int(scalar[1:])
            return _Float(bits) if length is None else _Floats(int(length), bits)
        if length is None:
            return _Integer(int(scalar[1:]))
        if scalar == "i32":
            return _Integers(int(length))
    raise CompilationError(
        f"load: the classical function {function} takes or gives {descriptor!r}, "
        "which is not a type a classical function has",
        code="wrong_argument",
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
    import re

    from ._errors import CompilationError, PulseError
    from .classical import ClassicalFunction

    version = None
    if given is not None:
        given, version = given
        latency_ns = (given.min_latency_ns, given.nominal_latency_ns, given.max_latency_ns)
    try:
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
    except PulseError as error:
        raise CompilationError(
            f"load: the module's classical function {name}: {error}", code="wrong_argument"
        ) from None
    if version is not None:
        # The given function's own types: a ``-> bool`` answers one bool, as
        # its implementation returns it.
        function.args, function.results = tuple(version.args), tuple(version.results)
        function.base_name = given.name
    else:
        function.args = tuple(_stated_type(spelled, name) for spelled in args)
        function.results = tuple(_stated_type(spelled, name) for spelled in results)
        # A version of a function of several lengths is named after it.
        function.base_name = re.sub(r"__i?\d+$", "", name)
    return function


def _given_functions(given: Sequence[Any], declared: Sequence[Any]) -> dict[str, tuple[Any, Any]]:
    """Per classical function the module declares, the function of
    ``load(classical=)`` given for it and its version of the module's
    lengths: the one of its name, or the one it is a version of
    (``decode__1`` of ``decode``). Their types, latency mode and submit
    policy must agree."""
    import itertools
    import re

    from ._errors import CompilationError, InvalidTypeError, InvalidValueError, PulseError
    from .classical import ClassicalFunction, _resolve, _unsized_names

    out: dict[str, tuple[Any, Any]] = {}
    for function in given:
        if not isinstance(function, ClassicalFunction):
            raise InvalidTypeError(
                "load(classical=) takes classical functions (pulse.classical), "
                f"got {type(function).__name__}"
            )
        versions = re.compile(re.escape(function.name) + r"(__i?\d+)?")
        rows = [row for row in declared if versions.fullmatch(row[0])]
        if not rows:
            raise InvalidValueError(
                f"load(classical=): the module calls no classical function named {function.name}"
            )
        _resolve(function)
        for row in rows:
            if row[0] in out:
                raise InvalidValueError(
                    f"load(classical=) gives two functions for the module's {row[0]}"
                )
            stated = (list(row[3]), list(row[4]))
            # A function with lists of unstated length is declared once for
            # every length: its version for the module's lengths is compared.
            # A classical function has at most 8 arguments, so a row states at
            # most 8 lengths: at most C(8, 4) = 70 candidate versions.
            candidates: list[Any] = [function]
            if function._shape is not None:
                names = _unsized_names(function)
                counts = [int(n) for spelled in row[3] for n in re.findall(r"[:\[](\d+)", spelled)]
                candidates = []
                for chosen in itertools.combinations(counts, len(names)):
                    try:
                        candidates.append(
                            function.specialize(**dict(zip(names, chosen, strict=True)))
                        )
                    except (ValueError, PulseError):
                        continue
            version = next(
                (
                    made
                    for made in candidates
                    if (
                        [value.descriptor for value in made.args],
                        [value.descriptor for value in made.results],
                    )
                    == stated
                ),
                None,
            )
            if version is None:
                shown = candidates[0] if len(candidates) == 1 else None
                raise CompilationError(
                    f"load: the module's {row[0]} takes ({', '.join(row[3])}) and gives "
                    f"({', '.join(row[4])}), and the function given for it "
                    + (
                        "has no version of those lengths"
                        if shown is None
                        else f"takes ({', '.join(v.descriptor for v in shown.args)}) and gives "
                        f"({', '.join(v.descriptor for v in shown.results)})"
                    ),
                    code="wrong_argument",
                )
            for what, here, there in (
                ("latency mode", function.latency_mode, row[6]),
                ("submit policy", function.submit_policy, row[7]),
            ):
                if here != there:
                    raise CompilationError(
                        f"load: the module's {row[0]} has the {what} {there!r}, and the "
                        f"function given for it {here!r}",
                        code="wrong_argument",
                    )
            out[row[0]] = (function, version)
    return out
