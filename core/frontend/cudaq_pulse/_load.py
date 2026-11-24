# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``cudaq_pulse.load``: a scheduled function of the Pulse dialect that
another compiler built, as a compiled program."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._device import Qudit
    from .compiler import ScheduledKernel
    from ._system import Controller, ControlSystem


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
) -> ScheduledKernel:
    """The compiled program of *entry*, a scheduled function of *module*.

    *module* is the assembly or the bytecode of an MLIR module of the Pulse
    dialect, written by any compiler. *entry* names a function in it that
    takes one ``!pulse.qudit`` per qudit it plays on and nothing else, gets
    its lines from them (``pulse.get_drive_line``, ``pulse.get_readout_line``)
    and is scheduled: every timed operation has its ``start_cycles`` and
    ``duration_cycles``, in cycles of the quantum system's clock, and the
    function is marked ``pulse.schedule_state = #pulse.schedule_state<timed>``.
    Its calls must be inlined. A module that states its clock
    (``pulse.control_system`` with ``clock_period_fs``) must state the
    quantum system's.

    *qudits* gives, for each argument in order, the qudit of the quantum
    system it plays on; their quantum system is the program's. The lines get
    that system's frequencies. *control_system* is as for ``compile``; without
    it, the quantum system's own control system is used.

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
            f"load() takes a module's assembly (str) or bytecode (bytes), got {type(module).__name__}"
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
        generic: Any = native.PulseModuleBuilder().load_function(
            text, entry, target.clock_ghz, len(placed)
        )
        generic.run_passes(["pulse-verify"])
    except (RuntimeError, ValueError) as error:
        raise CompilationError(str(error), code="load") from None
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
    program = _check_controller(program)
    program.metrics.op_count = len(program.schedule)
    return program
