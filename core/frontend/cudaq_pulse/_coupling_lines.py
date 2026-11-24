# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coupling lines: the outputs of a controller that play a baseband
waveform on a control of the quantum system.

A ``CouplingLine`` is named by the set of qudits it acts on: one qudit (its
own tuning), or several (the control of their coupling). It says how an
amplitude at the device follows from the volts the output plays, the most
volts it plays, and its delay. What the control does to its qudits (the
tuning curve, the idle point, the exchange, the response of the wiring) is
the quantum system's: a qudit's ``tuning``, a coupling's ``control``. Only
a controller that is given coupling lines loads this."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, NoReturn

from ._errors import InvalidTypeError, InvalidValueError, Problem, _ranges

if TYPE_CHECKING:
    from ._system import Controller, ControlSystem, _Reader
    from .quantum_systems.base import _Description

__all__ = ["CouplingLine"]


def _and(qudits: Sequence[int]) -> str:
    listed = list(map(str, qudits))
    return listed[0] if len(listed) == 1 else ", ".join(listed[:-1]) + " and " + listed[-1]


def _what(qudits: tuple[int, ...]) -> str:
    """What a line that acts on *qudits* reaches, in words."""
    if len(qudits) == 1:
        return f"qudit {qudits[0]}'s tuning"
    return f"the control of qudits {_and(qudits)}"


def _number(given: Any, what: str, nonzero: bool) -> float:
    """*given*, a finite number (other than 0, or above 0), as a float."""
    try:
        usable = (
            not isinstance(given, bool)
            and isinstance(given, (int, float))
            and math.isfinite(given)
            and (given != 0 if nonzero else given > 0)
        )
    except OverflowError:
        usable = False
    if not usable:
        raise InvalidValueError(f"{what}, or None; got {given!r}")
    return float(given)


class CouplingLine:
    """A coupling line of a controller (``Controller.coupling_lines``): an
    output that plays a baseband waveform, with no tone, on a control of
    the quantum system.

    ``qudits`` are the qudits it acts on, by carrier index: one index, or
    any iterable of them, in any order. The set names the line:
    ``CouplingLine(0)`` is the line of qudit 0's own tuning
    (``PhysicalQudit.tuning``) and ``CouplingLine((1, 2))`` that of the
    control of the coupling of qudits 1 and 2 (``Coupling.control``); a
    kernel fetches them with ``get_coupling_line(q0)`` and
    ``get_coupling_line(q1, q2)``. A line has one handle, and a pulse on it
    acts on all its qudits at once.

    ``amplitude_per_volt`` converts what the output plays to what the
    device receives: the amplitude at the device, in the unit of the
    component the line reaches (flux quanta for a ``FluxTuning``), per volt
    at the output, linear through the idle point (amplitude 0).
    ``max_volts`` is the most the output plays, either way; checking a
    program against it takes the conversion. ``delay_ns`` is its calibrated
    delay, as a drive line's. Each is ``None`` (0 for the delay) when not
    declared. A line is read-only and compares by content.
    """

    __slots__ = ("_qudits", "_per_volt", "_max_volts", "_delay_ns")
    _role: ClassVar[str] = "coupling"
    _qudits: tuple[int, ...]
    _per_volt: float | None
    _max_volts: float | None
    _delay_ns: float

    def __init__(
        self,
        qudits: int | Iterable[int],
        *,
        amplitude_per_volt: float | None = None,
        max_volts: float | None = None,
        delay_ns: float = 0.0,
    ) -> None:
        if isinstance(qudits, int) and not isinstance(qudits, bool):
            given: list[Any] = [qudits]
        elif isinstance(qudits, (str, bytes, Mapping, bool)) or not isinstance(qudits, Iterable):
            raise InvalidTypeError(
                "CouplingLine acts on a qudit or on several, by carrier index, as "
                f"CouplingLine(0) or CouplingLine((1, 2)); got {qudits!r}"
            )
        else:
            given = list(qudits)
        for qudit in given:
            if isinstance(qudit, bool) or not isinstance(qudit, int):
                raise InvalidTypeError(
                    f"CouplingLine acts on qudits by carrier index, got {qudit!r} in {given!r}"
                )
            if qudit < 0:
                raise InvalidValueError(f"CouplingLine has a negative carrier index {qudit}")
        if not given:
            raise InvalidValueError("CouplingLine acts on no qudit; give it at least one")
        if len(set(given)) != len(given):
            raise InvalidValueError(f"CouplingLine({given!r}) lists a qudit twice")
        acts_on = tuple(sorted(map(int, given)))
        shown = self._shown(acts_on)
        per_volt = amplitude_per_volt
        if per_volt is not None:
            per_volt = _number(
                per_volt,
                f"{shown}.amplitude_per_volt is the amplitude at the device per volt at the "
                "output, a finite number other than 0",
                True,
            )
        if max_volts is not None:
            max_volts = _number(
                max_volts,
                f"{shown}.max_volts is the most volts the output plays, a finite number above 0",
                False,
            )
        if delay_ns.__class__ is not float or not 0.0 <= delay_ns < math.inf:
            from ._system import _delay_ns

            delay_ns = _delay_ns(delay_ns, f"{shown}.delay_ns")
        set_ = object.__setattr__
        set_(self, "_qudits", acts_on)
        set_(self, "_per_volt", per_volt)
        set_(self, "_max_volts", max_volts)
        set_(self, "_delay_ns", delay_ns)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError(f"cannot assign to field {name!r}: a CouplingLine is read-only")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError(f"cannot delete field {name!r}: a CouplingLine is read-only")

    @property
    def qudits(self) -> tuple[int, ...]:
        """The qudits it acts on, by carrier index, ascending."""
        return self._qudits

    @property
    def amplitude_per_volt(self) -> float | None:
        """The amplitude at the device per volt at the output, or ``None``
        when not declared."""
        return self._per_volt

    @property
    def max_volts(self) -> float | None:
        """The most volts the output plays, either way, or ``None`` when
        not declared."""
        return self._max_volts

    @property
    def delay_ns(self) -> float:
        """Its calibrated delay, in ns."""
        return self._delay_ns

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        assert isinstance(other, CouplingLine)
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def _key(self) -> tuple[Any, ...]:
        return (self._qudits, self._per_volt, self._max_volts, self._delay_ns)

    @staticmethod
    def _shown(qudits: tuple[int, ...]) -> str:
        return f"CouplingLine({qudits[0] if len(qudits) == 1 else qudits!r})"

    def __repr__(self) -> str:
        shown = self._shown(self._qudits)[:-1]
        if self._per_volt is not None:
            shown += f", amplitude_per_volt={self._per_volt!r}"
        if self._max_volts is not None:
            shown += f", max_volts={self._max_volts!r}"
        if self._delay_ns:
            shown += f", delay_ns={self._delay_ns!r}"
        return shown + ")"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuilt, self._key())

    def _data(self) -> dict[str, Any]:
        """The line as a system file holds it: ``{"qudits": [1, 2]}`` with
        ``amplitude_per_volt``, ``max_volts`` and ``delay_ns`` where given."""
        data: dict[str, Any] = {"qudits": list(self._qudits)}
        if self._per_volt is not None:
            data["amplitude_per_volt"] = self._per_volt
        if self._max_volts is not None:
            data["max_volts"] = self._max_volts
        if self._delay_ns:
            data["delay_ns"] = self._delay_ns
        return data


def _rebuilt(
    qudits: tuple[int, ...], per_volt: float | None, max_volts: float | None, delay_ns: float
) -> CouplingLine:
    return CouplingLine(qudits, amplitude_per_volt=per_volt, max_volts=max_volts, delay_ns=delay_ns)


def keep_lines(controller: Controller) -> None:
    """Checks the coupling lines *controller* is given, as it is built, and
    keeps them as a tuple: each a ``CouplingLine``, no two acting on one set
    of qudits."""
    given: Any = controller.coupling_lines
    if isinstance(given, (str, bytes, CouplingLine, Mapping)) or not isinstance(given, Iterable):
        raise InvalidTypeError(
            f"Controller.coupling_lines is a collection of CouplingLine, got {given!r}"
        )
    kept = tuple(given)
    first: dict[tuple[int, ...], int] = {}
    for position, line in enumerate(kept):
        if not isinstance(line, CouplingLine):
            raise InvalidTypeError(
                "Controller.coupling_lines holds CouplingLines, as CouplingLine(0) for a qudit's "
                f"tuning or CouplingLine((0, 1)) for a pair's control, got {line!r}"
            )
        seen = first.setdefault(line.qudits, position)
        if seen != position:
            raise InvalidValueError(
                f"{_what(line.qudits)} is reached by two coupling lines, {kept[seen]!r} and "
                f"{line!r}; a control is reached by one coupling line"
            )
    # The lines of a controller are a set: kept in the order of their
    # qudits, so that controllers given the same lines in any order are equal.
    object.__setattr__(
        controller, "coupling_lines", tuple(sorted(kept, key=lambda line: line.qudits))
    )


def list_lines(data: list[dict[str, Any]], controllers: Sequence[Controller]) -> None:
    """Adds to *data*, each controller of a system file, the coupling lines
    of those of *controllers* that have some."""
    for entry, controller in zip(data, controllers, strict=True):
        if controller.coupling_lines:
            entry["coupling_lines"] = [line._data() for line in controller.coupling_lines]


def line_from(reader: _Reader, data: Any, path: str) -> CouplingLine | None:
    """A system file's coupling line at *path*; its problems go to *reader*."""
    entry = reader.mapping(data, path, ("qudits", "amplitude_per_volt", "max_volts", "delay_ns"))
    if entry is None:
        return None
    qudits = entry.get("qudits")
    if not isinstance(qudits, list):
        reader.problem(
            f"{path}.qudits",
            "must be the list of the qudits the line acts on"
            if "qudits" in entry
            else "is missing",
            qudits if "qudits" in entry else Problem.UNSET,
            fix='give the carrier indices, as "qudits": [1, 2]',
        )
        return None
    options = {k: entry[k] for k in ("amplitude_per_volt", "max_volts", "delay_ns") if k in entry}
    return reader.build(path, CouplingLine, qudits=qudits, **options)


def lines_from(reader: _Reader, data: Any, path: str) -> list[CouplingLine] | None:
    """A system file's controller's coupling lines at *path*, checked as a
    controller checks them, each problem at the line's own path."""
    if not isinstance(data, list):
        reader.problem(
            path,
            f"must be a list of coupling lines, got {type(data).__name__}",
            data,
            fix='give each line its qudits, as [{"qudits": [0]}, {"qudits": [1, 2]}]',
        )
        return None
    read = [line_from(reader, item, f"{path}[{i}]") for i, item in enumerate(data)]
    if None in read:
        return None
    first: dict[tuple[int, ...], int] = {}
    for position, line in enumerate(read):
        assert line is not None
        seen = first.setdefault(line.qudits, position)
        if seen != position:
            reader.problem(
                f"{path}[{position}]",
                f"{_what(line.qudits)} is reached by two coupling lines, this one and "
                f"{path}[{seen}]; a control is reached by one coupling line",
                data[position],
            )
            return None
    return [line for line in read if line is not None]


def reaching(system: ControlSystem, qudit: int) -> list[CouplingLine]:
    """The coupling lines of *system* that act on *qudit*, among others or alone."""
    return [
        line
        for controller in system.controllers
        if controller.coupling_lines
        for line in controller.coupling_lines
        if qudit in line.qudits
    ]


def wiring_problems(
    system: ControlSystem, target: _Description, calibration: Any = None
) -> list[Problem]:
    """The coupling lines of *system* that reach what *target*, the quantum
    system, does not have (a qudit it lacks, a qudit with no tuning, qudits
    with no control of their own), or what a line of another controller
    reaches. They are found when a control system meets its quantum
    system. *calibration*, the quantum system's, gives the tunings and
    interactions that are not declared; ``False`` leaves the check to the
    quantum system that is built next."""
    problems: list[Problem] = []
    if calibration is False:
        return problems
    first: dict[tuple[int, ...], str] = {}
    for index, controller in enumerate(system.controllers):
        for position, line in enumerate(controller.coupling_lines):
            where = f"controllers[{index}].coupling_lines[{position}]"
            path = f"control_system.{where}"
            seen = first.setdefault(line.qudits, where)
            if seen != where:
                problems.append(
                    Problem(
                        f"{_what(line.qudits)} is also reached by {seen}",
                        path=path,
                        value=line,
                        fix="a control is reached by one coupling line",
                    )
                )
            problem = reach_problem(line.qudits, target, calibration, system)
            if problem is not None:
                problems.append(Problem(problem[0], path=path, value=line, fix=problem[1]))
    return problems


def reach_problem(
    reached: tuple[int, ...],
    target: _Description,
    calibration: Any,
    system: ControlSystem | None = None,
) -> tuple[str, str] | None:
    """What is wrong with a line acting on *reached* in the quantum system
    *target* with its *calibration* (which gives the tunings and
    interactions that are not declared), with its fix; None when the
    system has the control of exactly those qudits."""
    from .quantum_systems.components import interaction_of, tuning_of

    known = target.qudits
    for qudit in reached:
        if qudit not in known:
            return (
                f"qudit {qudit} is not in the quantum system",
                f"it has qudits {_ranges(sorted(known))}",
            )
    if len(reached) == 1:
        (qudit,) = reached
        if tuning_of(target, calibration, qudit) is None:
            return (
                f"qudit {qudit} has no tuning, so no coupling line reaches it",
                f"give PhysicalQudit({qudit}, ..., tuning=...) the component that tunes it, as "
                "a FluxTuning",
            )
        return None
    if len(reached) > 2:
        return (
            f"acts on qudits {_and(reached)}; a quantum system declares the control of one "
            "qudit (its tuning) or of the coupling of two in this release, so nothing says "
            "what a line of three or more does",
            "reach each qudit's tuning with a line of its own, as CouplingLine(0), or a "
            "pair's control, as CouplingLine((0, 1))",
        )
    a, b = reached
    coupling = next((c for c in target.couplings if set(c.qudits) == set(reached)), None)
    if coupling is None:
        return (
            f"qudits {a} and {b} are not coupled, so no coupling line reaches their control",
            "list the pair in the quantum system's couplings, with its control",
        )
    if coupling.control is not None:
        return None
    if interaction_of(target, calibration, coupling) is not None:
        through = [
            repr(line)
            for q in reached
            for line in ([] if system is None else reaching(system, q))
            if line.qudits != reached
        ]
        if through:
            fix = f"the lines that reach its qudits are {', '.join(dict.fromkeys(through))}"
        else:
            fix = f"reach its qudits' tunings: CouplingLine({a}) or CouplingLine({b})"
        return (
            f"the coupling of qudits {a} and {b} has no control of its own: its interaction "
            "is made through its qudits' tunings",
            fix,
        )
    return (
        f"the coupling of qudits {a} and {b} has no control: it is always on",
        "give Coupling(..., control=...) the component that turns it on, as an ExchangeControl",
    )
