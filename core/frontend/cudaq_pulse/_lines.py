# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The table of a compiled kernel's lines (``ScheduledKernel.lines``).

Made when the table is asked for, from the module's acquisitions and from
what the kernel's quantum system and control system say then: nothing of it
is kept with the compiled program, and a kernel whose table is never asked
for does not load this module."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ._device import QuantumSystem
    from .quantum_systems.tuning import CouplingControl, Tuning


@dataclass(frozen=True, slots=True, repr=False)
class Line:
    """One line of a compiled kernel.

    ``index`` is the number ``ScheduledEvent.line`` names the line by: the
    kernel's drive, readout and coupling lines in allocation order. ``role``
    is ``"drive"``, ``"readout"`` or ``"coupling"``, and ``qudits`` the
    qudits the line acts on, by index, ascending: one for a drive or readout
    line, one or more for a coupling line. A qudit that a call has yet to
    bind is -1.

    ``controller`` is the index, among the control system's controllers, of
    the one that plays the line, and ``delay_ns`` the line's calibrated
    delay, 0.0 where the control system gives it none. Both are ``None`` for
    a kernel compiled without a control system or for one with no
    controllers, for a line no controller plays (a coupling line is played
    by the controller that declares its ``CouplingLine``), and for the line
    of a qudit not bound yet.

    For a coupling line, ``component`` is what the line reaches in the
    kernel's quantum system: the tuning of its one qudit (``qs.tuning(q)``)
    or the control of its two (``qs.control(a, d)``), and
    ``amplitude_per_volt`` and ``max_volts`` are what the control system's
    ``CouplingLine`` declares. Each is ``None`` where it is not declared,
    and for a drive or readout line.

    ``output_qudits`` are all the qudits that the controller's output
    playing the line reaches, ascending: the line's own ``qudits`` for a
    dedicated output, and every qudit of a drive or readout line that
    several qudits share (``DriveLine([0, 1], ...)``). Two lines of the
    table with the same ``controller`` (not ``None``), ``role`` and
    ``output_qudits`` are one output: a shared line is listed once for each
    qudit the program drives or reads out through it. ``None`` where
    ``controller`` is.

    The repr shows ``index``, ``role`` and ``qudits``, and of the other
    fields those that are not ``None``.
    """

    index: int
    role: str
    qudits: tuple[int, ...]
    controller: int | None = None
    output_qudits: tuple[int, ...] | None = None
    delay_ns: float | None = None
    component: Tuning | CouplingControl | None = None
    amplitude_per_volt: float | None = None
    max_volts: float | None = None

    def __repr__(self) -> str:
        shown = [f"index={self.index!r}", f"role={self.role!r}", f"qudits={self.qudits!r}"]
        for field in fields(self)[3:]:
            value = getattr(self, field.name)
            if value is not None:
                shown.append(f"{field.name}={value!r}")
        return f"Line({', '.join(shown)})"


def table(
    rows: Sequence[tuple[str, tuple[int, ...]]], quantum_system: QuantumSystem, machine: Any
) -> tuple[Line, ...]:
    """The lines of a module, from its *rows* (the native ``lines()``: the
    role and qudits of each acquisition, in order), the *quantum_system* of
    the kernel and the *machine* it was compiled for (``_MachineModel``)."""
    system = machine._system
    one = machine.controller
    if system is not None:
        controllers: tuple[Any, ...] = tuple(system.controllers)
    elif one is not None:
        # One controller that plays every line.
        controllers = (one,)
    else:
        controllers = ()
    everything = bool(controllers) and (system is None or system._plays_everything)
    delays = machine._line_delays
    # (controller, role) -> the qudits of each shared output, by qudit.
    outputs: dict[tuple[int, str], dict[int, tuple[int, ...]]] = {}
    lines: list[Line] = []
    for index, (role, reached) in enumerate(rows):
        qudits = tuple(sorted(reached))
        bound = qudits[0] >= 0
        owner: int | None = None
        delay: float | None = None
        if role != "coupling":
            if bound and controllers:
                owner = 0 if everything else system._line_owners.get((role, qudits[0]))
            output: tuple[int, ...] | None = None
            if owner is not None:
                delay = float(delays.get((role, qudits[0]), 0.0))
                shared = outputs.get((owner, role))
                if shared is None:
                    shared = outputs[owner, role] = _shared(controllers[owner], role)
                output = shared.get(qudits[0], qudits)
            lines.append(Line(index, role, qudits, owner, output, delay))
            continue
        component: Tuning | CouplingControl | None = None
        declared = None
        if bound:
            component = _reached(quantum_system, qudits)
            for place, controller in enumerate(controllers):
                declared = next(
                    (line for line in controller.coupling_lines if line.qudits == qudits), None
                )
                if declared is not None:
                    owner = place
                    break
        if declared is None:
            lines.append(Line(index, role, qudits, component=component))
        else:
            lines.append(
                Line(
                    index,
                    role,
                    qudits,
                    owner,
                    qudits,
                    float(declared.delay_ns),
                    component,
                    declared.amplitude_per_volt,
                    declared.max_volts,
                )
            )
    return tuple(lines)


def _shared(controller: Any, role: str) -> dict[int, tuple[int, ...]]:
    """For each qudit of a *role* line of *controller* that several qudits
    share, all the qudits that line reaches, ascending. A qudit not in it
    has an output of its own."""
    listed = controller.drive_lines if role == "drive" else controller.readout_lines
    shared: dict[int, tuple[int, ...]] = {}
    for line in listed:
        reached: tuple[int, ...] = line.qudits
        if len(reached) > 1:
            together = tuple(sorted(reached))
            for qudit in reached:
                shared[qudit] = together
    return shared


def _reached(
    quantum_system: QuantumSystem, qudits: tuple[int, ...]
) -> Tuning | CouplingControl | None:
    """What a coupling line of *qudits* reaches in *quantum_system*: the
    tuning of one qudit, the control of the coupling of two; None for more,
    and where the system has no such qudit or coupling."""
    from ._errors import NotFoundError

    try:
        if len(qudits) == 1:
            return quantum_system.tuning(qudits[0])
        if len(qudits) == 2:
            return quantum_system.control(qudits[0], qudits[1])
    except NotFoundError:
        pass
    return None
