# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a quantum system says of the baseband controls of its qudits and
couplings: the tuning of a qudit, the control of a coupling and the
interaction of a coupling.

They are declared on the system, by qudit and by pair
(``pulse.quantum_system(..., tunings={q: ...}, controls={(a, d): ...},
interactions={(a, d): ...})``, ``qs.replace(tunings=...)``), and asked of
it the same way (``qs.tuning(q)``, ``qs.control(a, d)``,
``qs.interaction(a, d)``). A ``PhysicalQudit`` and a ``Coupling`` hold none
of it. A system that declares some is of a class made from
``QuantumSystem`` (``_Declaring``) and its description of one made from
the description's (``_Declared``): a system that declares nothing is built,
compared, printed and pickled by the code it always was.

What is not declared is answered from a calibration, by value: a qudit that
has the values of the qudit a calibration built has that calibration's
tuning; a qudit changed in any value has none. The calibration is the one
the system holds; for a variant made with ``replace``, the one of the
system it was made from; for a system built from qudits, any shipped one
that builds a qudit of those values. What a calibration built is, for a
shipped one, its own build; for another, the registered quantum system it
names with the calibration applied; a calibration that names no registered
system answers for the qudits of the system that holds it, by label.

The module loads when a system first declares or is asked for one of these.
"""

from __future__ import annotations

import numbers
import operator
import threading
import weakref
from collections.abc import Mapping
from operator import attrgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .._device import QuantumSystem, Qudit, _fingerprint, _Handles
from .._errors import (
    InvalidTypeError,
    NotFoundError,
    Problem,
    QuantumSystemDefinitionError,
    _count,
)
from .base import _DESCRIPTION_FIELDS, _Description

if TYPE_CHECKING:
    from .base import Coupling
    from .calibration import Calibration
    from .tuning import CouplingControl, CouplingInteraction, Tuning

# The keywords a quantum system declares its components with.
COMPONENTS = ("tunings", "controls", "interactions")

# What a fingerprint takes of a qudit and of a coupling.
_QUDIT_VALUES = attrgetter(
    "index",
    "frequency_hz",
    "anharmonicity_hz",
    "t1_us",
    "t2_star_us",
    "t2_echo_us",
    "thermal_population",
    "drive",
    "readout",
    "_calibrations",
)
_COUPLING_VALUES = attrgetter("qudits", "strength_hz", "_calibrations")


def _same(values: Any, built: Any, given: Any) -> bool:
    if built is given:
        return True
    a, b = values(built), values(given)
    return bool(a == b and a[-1]._content == b[-1]._content)


def _canonical(value: Any) -> Any:
    """*value* as a fingerprint takes it: equal components give equal
    content and unequal ones do not. A number is its float where that is
    exact (``1``, ``1.0`` and a numpy 1 are one value; -0.0 is 0.0), a
    sequence is tagged as one, and a component is its class and its values."""
    if value.__class__ is float:
        return value + 0.0
    if value.__class__ is str:
        return value
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            exact = float(value)
        except OverflowError:
            return ("number", repr(value))
        return exact + 0.0 if exact == value else ("number", repr(value))
    if isinstance(value, (tuple, list)):
        return ("sequence", *map(_canonical, value))
    values = getattr(value, "_values", None)
    if values.__class__ is tuple:
        kind = type(value)
        return ("component", kind.__module__, kind.__qualname__, *map(_canonical, values))
    return ("other", repr(value))


# ── What a system declares ──────────────────────────────────────────────────


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _carrier(key: Any) -> int | None:
    """The qudit index a key of a declaration names: a handle's, or an
    integer (not a bool); None for anything else."""
    if key.__class__ is Qudit:
        return int(key.index)
    if key.__class__ is int:
        return int(key)
    if isinstance(key, bool):
        return None
    try:
        return operator.index(key)
    except TypeError:
        return None


class _Declared(_Description):
    """The description of a quantum system that declares components: the
    description's fields, and ``tunings`` by qudit index, ``controls`` and
    ``interactions`` by pair (the lower index first), read-only."""

    __slots__ = (*COMPONENTS, "_through")

    tunings: Mapping[int, Tuning]
    controls: Mapping[tuple[int, int], CouplingControl]
    interactions: Mapping[tuple[int, int], CouplingInteraction]

    def __init__(self, plain: _Description, components: Mapping[str, Any]) -> None:
        set_ = object.__setattr__
        for name in _DESCRIPTION_FIELDS:
            set_(self, name, getattr(plain, name))
        for name in COMPONENTS:
            set_(self, name, MappingProxyType(dict(components.get(name) or {})))
        # The interactions it has answered with, by the pair as asked.
        set_(self, "_through", {})

    def _plain(self) -> _Description:
        """The description without what it declares."""
        return _Description(**{name: getattr(self, name) for name in _DESCRIPTION_FIELDS})

    def _components(self) -> dict[str, Any]:
        return {name: dict(getattr(self, name)) for name in COMPONENTS}

    def _content(self) -> tuple[Any, ...]:
        """What a fingerprint takes of the components."""
        return tuple(
            (
                name,
                tuple(sorted((key, _canonical(part)) for key, part in getattr(self, name).items())),
            )
            for name in COMPONENTS
        )

    def replace(self, **changes: Any) -> _Description:
        """A copy with *changes*. The components of the qudits and the
        couplings that remain stay; ``tunings=``, ``controls=`` and
        ``interactions=`` give them anew."""
        given = {name: changes.pop(name) for name in COMPONENTS if name in changes}
        plain = _Description.replace(self, **changes)
        kept = self._components()
        kept["tunings"] = {q: t for q, t in kept["tunings"].items() if q in plain.qudits}
        pairs = {_pair(*c.qudits) for c in plain.couplings}
        for name in COMPONENTS[1:]:
            kept[name] = {pair: part for pair, part in kept[name].items() if pair in pairs}
        kept.update(given)
        return declared(plain, kept)

    def __repr__(self) -> str:
        shown = _Description.__repr__(self)
        parts = "".join(
            f", {name}={dict(getattr(self, name))!r}" for name in COMPONENTS if getattr(self, name)
        )
        return f"{shown[:-1]}{parts})"

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return bool(
            all(getattr(self, f) == getattr(other, f) for f in _DESCRIPTION_FIELDS)
            and all(getattr(self, name) == getattr(other, name) for name in COMPONENTS)
        )

    def __hash__(self) -> int:
        return _Description.__hash__(self)

    def __reduce__(self) -> tuple[Any, ...]:
        return (declared, (self._plain(), self._components()))


def declared(plain: _Description, components: Mapping[str, Any]) -> _Description:
    """*plain* with the *components* it declares, checked; *plain* itself
    when they are all empty. Every problem is at its path
    (``tunings[3]``, ``controls[(1, 2)]``)."""
    from .tuning import CouplingControl, CouplingInteraction, Tuning

    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise InvalidTypeError(
            f"pulse.quantum_system() got an unexpected keyword argument {unknown[0]!r}; a "
            f"quantum system declares {', '.join(COMPONENTS)}"
        )
    problems: list[Problem] = []
    kept: dict[str, dict[Any, Any]] = {}
    for name in COMPONENTS:
        given = components.get(name)
        kept[name] = {}
        if given is None:
            continue
        if not isinstance(given, Mapping):
            what = "qudit indices" if name == "tunings" else "pairs of qudit indices, as (a, d),"
            problems.append(
                Problem(
                    f"maps {what} to components, got {type(given).__name__}",
                    path=name,
                    value=given,
                )
            )
            continue
        for key, part in given.items():
            where = f"{name}[{key!r}]"
            index: Any
            if name == "tunings":
                index = _carrier(key)
                if index not in plain.qudits:
                    problems.append(
                        Problem(
                            f"{key!r} is not a qudit of the quantum system",
                            path=where,
                            value=key,
                            fix=f"its qudits are {sorted(plain.qudits)}",
                        )
                    )
                    continue
                kind: type = Tuning
                found = part._problems(where) if isinstance(part, Tuning) else []
            else:
                try:
                    a, b = map(_carrier, key) if key.__class__ is tuple else (None, None)
                    qudits: tuple[int, ...] = (
                        () if a is None or b is None else plain.coupling(a, b).qudits
                    )
                except (TypeError, ValueError, NotFoundError):
                    qudits = ()
                if not qudits:
                    problems.append(
                        Problem(
                            f"{key!r} is not a pair of coupled qudits of the quantum system",
                            path=where,
                            value=key,
                            fix="give the two qudits of one of its couplings, in either "
                            f"order; they are {[c.qudits for c in plain.couplings]}",
                        )
                    )
                    continue
                index = _pair(qudits[0], qudits[1])
                kind = CouplingControl if name == "controls" else CouplingInteraction
                found = part._problems(qudits, where) if isinstance(part, kind) else []
            if not isinstance(part, kind):
                found = [
                    Problem(
                        f"must be a {kind.__name__}, got {type(part).__name__}",
                        path=where,
                        value=part,
                    )
                ]
            elif index in kept[name]:
                found = [Problem("is given twice", path=where, value=key)]
            problems += found
            kept[name][index] = part
    for pair in sorted(set(kept["controls"]) & set(kept["interactions"])):
        problems.append(
            Problem(
                "has a control of its own and an interaction made through its qudits; a "
                "coupling is turned on by one of them",
                path=f"interactions[{pair!r}]",
                value=pair,
            )
        )
    if problems:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in the quantum system {plain.name!r}",
            problems=problems,
        )
    if not any(kept.values()):
        return plain._plain() if isinstance(plain, _Declared) else plain
    return _Declared(plain, kept)


class _Declaring(QuantumSystem):
    """A quantum system that declares components: its fingerprint takes
    them in, its variants keep them, and its system file lists them."""

    def _finish(self) -> None:
        target, layout, control_system = self._description, self.layout, self.control_system
        self.frequencies = target.frequencies
        fingerprint = _fingerprint(target, layout, self.calibration)
        content = getattr(target, "_content", None)
        if content is not None:
            import hashlib

            fingerprint = hashlib.blake2b(
                repr((fingerprint, content())).encode(), digest_size=16
            ).hexdigest()
        self.fingerprint = fingerprint
        self._identity = fingerprint if control_system is None else (fingerprint, control_system)
        self.qudits = _Handles(tuple(Qudit(self, p, i) for i, p in enumerate(layout)))

    @property
    def _twin(self) -> QuantumSystem:
        """Itself: the drift code takes a system of any class but
        ``QuantumSystem`` for a drifting one and asks for its twin. It is
        asked where a system is given drift, which ``redeclared`` does for
        a system that declares components; a system given components and
        drift in one ``replace`` comes here without that."""
        if "_drifts" not in self.__dict__:
            raise InvalidTypeError(
                "QuantumSystem.replace() gives a quantum system that declares no component "
                "yet its components or parameter_drift (even False), not both in one call: "
                "give the components first, as "
                "qs.replace(tunings=...).replace(parameter_drift=...)"
            )
        return self

    def replace(self, **changes: Any) -> QuantumSystem:
        return redeclared(self, changes)

    def as_dict(self) -> dict[str, Any]:
        data = QuantumSystem.as_dict(self)
        data.update(components_data(self._description))
        return data


def _missing_tunings(system: QuantumSystem) -> None:
    """Refuses an interaction made through a qudit the system has no tuning for."""
    described = system._description
    sources = sources_of(system)
    problems = []
    for pair, part in getattr(described, "interactions", {}).items():
        for qudit in pair:
            if tuning_of(described, sources, qudit) is None:
                problems.append(
                    Problem(
                        f"is made through its qudits' tunings, and qudit {qudit} has none",
                        path=f"interactions[{pair!r}]",
                        value=part,
                        fix=f"give it one with tunings={{{qudit}: ...}}",
                    )
                )
    if problems:
        raise QuantumSystemDefinitionError(
            f"{_count(len(problems), 'problem')} in the quantum system {system.name!r}",
            problems=problems,
        )


def _built(
    plain: QuantumSystem,
    components: Mapping[str, Any],
    control_system: Any,
    layout: Any,
    clock_ghz: Any,
) -> QuantumSystem:
    """*plain*, a system built without a control system, declaring
    *components* and given *control_system*."""
    described = declared(plain._description, components)
    if described is plain._description and control_system is None:
        return plain
    cls = _Declaring if isinstance(described, _Declared) else QuantumSystem
    made = cls(
        described,
        name=plain.name,
        layout=layout,
        source=plain.source,
        calibration=plain.calibration,
        control_system=control_system,
        clock_ghz=clock_ghz,
    )
    made._spec = plain._spec
    made._base_target = plain._base_target
    _missing_tunings(made)
    return made


def declaring(components: Mapping[str, Any], arguments: dict[str, Any]) -> QuantumSystem:
    """``pulse.quantum_system(**arguments, **components)``: the system the
    arguments build, declaring *components*."""
    from .._device import quantum_system

    control_system = arguments.pop("control_system", None)
    drift = {
        key: arguments.pop(key)
        for key in ("parameter_drift", "parameter_drift_seed")
        if key in arguments
    }
    made = _built(
        quantum_system(**arguments),
        components,
        control_system,
        arguments.get("layout"),
        arguments.get("clock_ghz"),
    )
    if drift.get("parameter_drift") or drift.get("parameter_drift_seed") is not None:
        return made.replace(**drift)
    return made


def declared_in(
    config: Mapping[str, Any],
    problems: list[Problem],
    build: Any,
    control: Any,
    layout: Any,
    clock_ghz: Any,
    name: str = "",
) -> QuantumSystem | None:
    """The quantum system of a system file whose reading left only notes of
    what it declares (``_coupling_lines.Declares``) among *problems*: its
    controllers' coupling lines, its tunings, controls and interactions.
    *build* builds the system (named *name*) without a control system. With
    a real problem among them, the notes are dropped and nothing is built."""
    import re

    from .._coupling_lines import Declares, lines_from, with_lines
    from .._system import _Reader

    notes = [p for p in problems if p.__class__ is Declares]
    problems[:] = [p for p in problems if p.__class__ is not Declares]
    if problems or not notes:
        return None
    reader = _Reader()
    controllers = None if control is None else list(control.controllers)
    for note in notes:
        if not note.path.endswith(".coupling_lines"):
            continue
        at = re.fullmatch(r"control_system\.controllers\[(\d+)\]\.coupling_lines", note.path)
        if controllers is None or at is None or int(at[1]) >= len(controllers):
            reader.problem(note.path, "is not a key here: a controller has coupling lines")
            continue
        lines = lines_from(reader, note.value, note.path)
        if lines is not None:
            reader.build(
                note.path,
                lambda i=int(at[1]), given=lines: controllers.__setitem__(
                    i, with_lines(controllers[i], given)
                ),
            )
    components, where = components_from(reader, config)
    problems.extend(reader.problems)
    if problems:
        return None
    try:
        if controllers is not None and controllers != list(control.controllers):
            control = _with_lines(control, controllers)
    except (TypeError, ValueError) as error:
        # What the control system finds wrong with its controllers as they
        # now are is a problem of the file's control system.
        found = getattr(error, "problems", None)
        if not found or any(p.__class__ is not Problem for p in found):
            found = [Problem(Exception.__str__(error))]
        problems.extend(
            Problem(
                p.message,
                path=p.path
                if p.path.startswith("control_system")
                else f"control_system.{p.path}".rstrip("."),
                value=p.value,
                fix=p.fix,
            )
            for p in found
        )
        return None
    try:
        return _built(build(), components, control, layout, clock_ghz)
    except QuantumSystemDefinitionError as error:
        # A declaration's problem is at its entry in the file.
        for problem in error.problems:
            head, bracket, rest = problem.path.partition("]")
            entry = where.get(head + bracket)
            problems.append(
                problem
                if entry is None
                else Problem(
                    problem.message, path=entry + rest, value=problem.value, fix=problem.fix
                )
            )
    except (KeyError, ValueError) as error:
        problems.append(
            Problem(
                f"does not fit quantum system {name!r}: {Exception.__str__(error)}",
                path="calibration",
            )
        )
    return None


def _with_lines(control: Any, controllers: list[Any]) -> Any:
    """*control* with *controllers*, some of them now with coupling lines,
    in place of its own; its links name the new ones."""
    import dataclasses

    links = {
        label: link
        if link.controllers is None
        else dataclasses.replace(
            link, controllers=[controllers[control._index(c)] for c in link.controllers]
        )
        for label, link in control.links.items()
    }
    return dataclasses.replace(control, controllers=controllers, links=links)


def redeclared(system: QuantumSystem, changes: dict[str, Any]) -> QuantumSystem:
    """``system.replace(**changes)`` for a system that declares components
    or is given some: a variant that keeps or takes them."""
    if "parameter_drift" in changes or "parameter_drift_seed" in changes:
        from .._errors import InvalidValueError
        from .drift import _drifting

        # As drift._replaced, for a twin that declares components: it has
        # its fingerprint before the drifting system copies what it holds.
        changes = dict(changes)
        parameter_drift = changes.pop("parameter_drift", None)
        seed = changes.pop("parameter_drift_seed", None)
        twin = system if system.__class__ is _Declaring else vars(system)["_twin"]
        if changes:
            twin = twin.replace(**changes)
        if seed is not None and (parameter_drift is None or parameter_drift is False):
            raise InvalidValueError(
                "parameter_drift_seed seeds parameter_drift, which is not given"
            )
        if twin.__class__ is _Declaring:
            twin.fingerprint  # noqa: B018
            vars(twin)["_drifts"] = True
        return _keeping(_drifting(twin, parameter_drift, seed))
    described = system._description
    layout = changes.pop(
        "layout", None if system.layout == tuple(sorted(described.qudits)) else system.layout
    )
    control_system = changes.pop("control_system", system.control_system)
    clock_ghz = changes.pop(
        "clock_ghz", None if system.clock_ghz == described.clock_ghz else system.clock_ghz
    )
    unknown = sorted(changes.keys() - {*_DESCRIPTION_FIELDS, *COMPONENTS})
    if unknown:
        raise InvalidTypeError(
            f"QuantumSystem.replace() got an unexpected keyword argument {unknown[0]!r}; "
            f"it takes {', '.join(_DESCRIPTION_FIELDS[:-1])}, {', '.join(COMPONENTS)}, layout, "
            "control_system, clock_ghz, parameter_drift and parameter_drift_seed"
        )
    if isinstance(described, _Declared):
        replaced = described.replace(**changes)
    else:
        given = {name: changes.pop(name) for name in COMPONENTS if name in changes}
        replaced = declared(described.replace(**changes), given)
    cls = _Declaring if isinstance(replaced, _Declared) else QuantumSystem
    made = cls(
        replaced,
        name=changes.get("name", system.name),
        layout=layout,
        source=f"{system.name}.replace()",
        control_system=control_system,
        clock_ghz=clock_ghz,
        _variant=vars(system).get("_made_from", system),
    )
    _missing_tunings(made)
    return made


# The class of a drifting system that declares components, made when the
# first one is.
_DRIFTING: list[type] = []


def _keeping(made: QuantumSystem) -> QuantumSystem:
    """*made*, as the drift code built it; a drifting system whose
    description declares components is given the class made from the
    drifting one, whose variants keep them and whose system file lists
    them, as a ``_Declaring``'s."""
    from .drift import _DriftingQuantumSystem

    if made.__class__ is not _DriftingQuantumSystem or not isinstance(made._description, _Declared):
        return made
    if not _DRIFTING:

        class _DriftingDeclaring(_DriftingQuantumSystem):
            # One rebuilt from a pickle whose twin is a variant has no
            # fingerprint yet: it is a declaring system's.
            _finish = _Declaring._finish

            def replace(self, **changes: Any) -> QuantumSystem:
                return redeclared(self, changes)

            def as_dict(self) -> dict[str, Any]:
                data = _DriftingQuantumSystem.as_dict(self)
                data.update(components_data(self._description))
                return data

            def __reduce__(self) -> tuple[Any, ...]:
                return (_rebuilt_drifting, _DriftingQuantumSystem.__reduce__(self)[1])

        _DriftingDeclaring.__name__ = _DriftingDeclaring.__qualname__ = "_DriftingDeclaring"
        _DRIFTING.append(_DriftingDeclaring)
    made.__class__ = _DRIFTING[0]
    return made


def _rebuilt_drifting(*state: Any) -> QuantumSystem:
    """A drifting system that declares components, from its pickle."""
    from .drift import _rebuilt

    # The twin has its fingerprint before anything copies what it holds.
    twin = state[0]
    if twin.__class__ is _Declaring:
        twin.fingerprint  # noqa: B018
        vars(twin)["_drifts"] = True
    return _keeping(_rebuilt(*state))


# ── System files ────────────────────────────────────────────────────────────

_KINDS = {"tunings": "flux", "controls": "exchange", "interactions": "charge"}


def components_data(described: Any) -> dict[str, Any]:
    """What a system file holds of the components *described* declares:
    ``"tunings"``, ``"controls"`` and ``"interactions"``, each a list of
    entries naming the qudit or the pair."""
    from .tuning import _component_data

    data: dict[str, Any] = {}
    for name in COMPONENTS:
        parts = getattr(described, name, None)
        if not parts:
            continue
        key = "qudit" if name == "tunings" else "qudits"
        data[name] = [
            {key: list(k) if isinstance(k, tuple) else k, **_component_data(_KINDS[name], part)}
            for k, part in sorted(parts.items())
        ]
    return data


def components_from(reader: Any, data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """The components a system file's *data* declares, read by *reader*
    (its problems are the reader's), and where each is in the file: the
    path of its entry (``tunings[0]``) by the path a problem of the
    declaration has (``tunings[3]``, by qudit)."""
    from .tuning import ChargeCoupling, ExchangeControl, FluxTuning, _response_from

    made: dict[str, Any] = {}
    places: dict[str, str] = {}
    classes = {"tunings": FluxTuning, "controls": ExchangeControl, "interactions": ChargeCoupling}
    for name in COMPONENTS:
        if name not in data:
            continue
        entries = data[name]
        if not isinstance(entries, list):
            reader.problem(name, f"must be a list, got {type(entries).__name__}", entries)
            continue
        cls: Any = classes[name]
        key = "qudit" if name == "tunings" else "qudits"
        known = tuple(k for k in cls._fields if k != "tunings")
        found: dict[Any, Any] = {}
        for i, item in enumerate(entries):
            where = f"{name}[{i}]"
            entry = reader.mapping(item, where, (key, "kind", *known))
            if entry is None:
                continue
            if entry.get("kind") != _KINDS[name]:
                reader.problem(
                    f"{where}.kind",
                    f"a system file holds {name} of kind {_KINDS[name]!r}",
                    entry.get("kind", Problem.UNSET),
                )
                continue
            target = entry.get(key)
            if name == "tunings":
                fits = target.__class__ is int
                seen = target
            else:
                fits = (
                    isinstance(target, list)
                    and len(target) == 2
                    and all(q.__class__ is int for q in target)
                )
                if fits:
                    target = tuple(target)
                    seen = _pair(*target)
            if not fits:
                reader.problem(
                    f"{where}.{key}",
                    "is the index of a qudit"
                    if name == "tunings"
                    else "is the two qudits of a coupling, as [1, 2]",
                    entry.get(key, Problem.UNSET),
                )
                continue
            if f"{name}[{seen!r}]" in places:
                reader.problem(where, "is given twice", target)
                continue
            given = {k: entry[k] for k in known if k in entry}
            for k in ("amplitudes", "exchange_hz"):
                if isinstance(given.get(k), list):
                    given[k] = tuple(given[k])
            if "response" in given:
                try:
                    given["response"] = _response_from(given["response"])
                except (TypeError, ValueError) as error:
                    reader.problem(f"{where}.response", str(error), entry["response"])
                    continue
            part = reader.build(where, cls, **given)
            if part is not None:
                found[target] = part
                places[f"{name}[{target!r}]"] = places[f"{name}[{seen!r}]"] = where
        made[name] = found
    return made, places


# ── What a calibration gives ────────────────────────────────────────────────

# What was made of each calibration, kept apart from the calibration (its
# pickle and its values are what they were) and by the calibration itself,
# not by its value (which would read all of it): its tuning values, what it
# built, the tuning of each label, and the interactions made for each pair.
_MADE: dict[int, _Made] = {}
_MADE_LOCK = threading.Lock()
# The interactions kept per calibration before they are made again.
_INTERACTIONS_KEPT = 1024
_UNSET: Any = object()
# In place of a registry entry, for a shipped calibration's own build.
_SHIPPED_BUILD: Any = object()


class _Made:
    __slots__ = ("values", "built", "building", "tunings", "interactions")

    def __init__(self, calibration: Calibration) -> None:
        used, g_c, circuits = _tuning_values(calibration)
        self.values = (used, g_c, dict(circuits))
        self.built: Any = _UNSET
        # The thread that is building what the calibration is compared against.
        self.building: int | None = None
        self.tunings: dict[str, Any] = {}
        self.interactions: dict[Any, Any] = {}


def _tuning_values(
    calibration: Calibration,
) -> tuple[str, float | None, tuple[tuple[str, tuple[float, float, float, float]], ...]]:
    """What the tuning components take of *calibration*: the key of the
    charge coupling used, g_c in Hz, and each qudit whose circuit is
    calibrated with (E_C, E_J,max, d, idle flux), by label; a calibration
    that does not give them gives ``("", None, ())``. An indexed shipped
    calibration's is in ``_shipped_tunings``."""
    from .._source_stamp import CALIBRATIONS
    from .calibration import _SWEET_SPOT_FLUX

    if CALIBRATIONS.get(calibration.name) == calibration.__dict__.get("_shipped", 0):
        from ._shipped_tunings import SHIPPED_TUNINGS

        found = SHIPPED_TUNINGS.get(calibration.name)
        if found is not None:
            return found
    try:
        couplings = calibration._couplings()
        used = couplings["used"]
        g_c = couplings[used]["value"]
    except (KeyError, TypeError):
        used, g_c = "", None
    circuits = []
    try:
        entries = list(calibration._data["qudits"].items())
    except (KeyError, TypeError, AttributeError):
        entries = []
    for label, q in entries:
        try:
            if q.get("flux_quanta_capable"):
                circuits.append(
                    (
                        label,
                        (
                            q["ec"]["value"],
                            q["ej_max"]["value"],
                            q["asymmetry_d"]["value"],
                            _SWEET_SPOT_FLUX[q["sweet_spot_idle"]],
                        ),
                    )
                )
        except (KeyError, TypeError, AttributeError):
            # A qudit whose circuit is not all there has no tuning.
            continue
    return (used, g_c, tuple(circuits))


def _made(calibration: Calibration) -> _Made:
    """What is kept of *calibration*; called under ``_MADE_LOCK``."""
    key = id(calibration)
    made = _MADE.get(key)
    if made is None:
        made = _MADE[key] = _Made(calibration)
        # Forgotten with the calibration, whose id another may then take.
        weakref.finalize(calibration, _MADE.pop, key, None)
    return made


def _derived_tuning(calibration: Calibration, label: str) -> Tuning | None:
    """The ``FluxTuning`` of the qudit labelled *label*, from its circuit in
    *calibration*, or None for a qudit whose circuit the calibration does
    not give. Made once per calibration and label."""
    with _MADE_LOCK:
        made = _made(calibration)
        if label not in made.tunings:
            circuit = made.values[2].get(label)
            if circuit is None:
                made.tunings[label] = None
            else:
                from .tuning import FluxTuning

                ec, ej_max, d, idle = circuit
                made.tunings[label] = FluxTuning(
                    ec_hz=ec,
                    ej_max_hz=ej_max,
                    asymmetry=d,
                    idle_flux=idle,
                    source=f"{calibration.name} calibration, qudits.{label}: ec, ej_max, "
                    "asymmetry_d, sweet_spot_idle",
                )
        found: Tuning | None = made.tunings[label]
        return found


def _derived_interaction(
    calibration: Calibration, labels: tuple[str, str], tunings: tuple[Tuning, Tuning]
) -> CouplingInteraction | None:
    """The ``ChargeCoupling`` of the qudits labelled *labels*, made through
    *tunings* (the ones the system answers with) with the g_c of
    *calibration*; None if the calibration gives no g_c. Made once per pair
    and pair of tunings."""
    with _MADE_LOCK:
        made = _made(calibration)
        used, g_c, _ = made.values
        if g_c is None:
            return None
        # By the tunings themselves (the interaction keeps them, so the key
        # stays theirs): systems of one calibration that answer with
        # different tunings each keep their own.
        key = (labels, id(tunings[0]), id(tunings[1]))
        kept = made.interactions.get(key)
        if kept is None:
            from .tuning import ChargeCoupling

            if len(made.interactions) >= _INTERACTIONS_KEPT:
                made.interactions.clear()
            kept = made.interactions[key] = ChargeCoupling(
                charge_coupling_hz=g_c,
                tunings=tunings,
                source=f"{calibration.name} calibration, coupling.charge_coupling_g_c.{used}",
            )
        found: CouplingInteraction = kept
        return found


# The calibrations a system's answers come from, each with the description
# it built: a qudit of the system that has the values of a qudit of that
# description is that calibration's.
Sources = list[tuple["Calibration", "_Description"]]


def _shipped() -> Sources:
    """Each shipped calibration that gives tunings, with the description it builds."""
    from ._shipped_tunings import SHIPPED_TUNINGS
    from .calibration import shipped_calibration
    from .transmon import _calibrated_surface17

    return [(shipped_calibration(name), _calibrated_surface17(name)) for name in SHIPPED_TUNINGS]


def _built_by(calibration: Calibration) -> _Description | None:
    """What *calibration* built: a shipped one that gives tunings, its own
    build (a measurement run changes no value of it); another, the
    registered quantum system it names with the calibration applied; None
    for one that names no registered system, or that does not fit the one
    it names (it then has nothing to be compared against and answers by
    label). Kept with the registry's entry for the name, so a system
    registered later is found the next time a tuning is asked."""
    with _MADE_LOCK:
        made = _made(calibration)
        kept: Any = made.built
    if kept is not _UNSET and kept[0] is _SHIPPED_BUILD:
        shipped: _Description | None = kept[1]
        return shipped
    from ._shipped_tunings import SHIPPED_TUNINGS

    name = calibration.name
    entry: Any
    if calibration._shipped is not None and name in SHIPPED_TUNINGS:
        entry = _SHIPPED_BUILD
    else:
        from ._registry import _registry

        entry = _registry().get(name.partition(":")[0]) if isinstance(name, str) else None
    if kept is _UNSET or kept[0] is not entry:
        # Built outside the lock: a registered system's factory is anyone's
        # code. One that asks for this calibration's answers while it is
        # being built here is answered by label.
        ident = threading.get_ident()
        if made.building == ident:
            return None
        found: Any = None
        if entry is _SHIPPED_BUILD:
            from .transmon import _calibrated_surface17

            found = _calibrated_surface17(name)
        elif entry is not None:
            from ._registry import make_quantum_system

            made.building = ident
            try:
                found = calibration._apply(make_quantum_system(name))
            except (KeyError, ValueError):
                # Does not fit the system it names, as _device._checked says it.
                found = None
            finally:
                made.building = None
        with _MADE_LOCK:
            if made.built is _UNSET or made.built[0] is not entry:
                made.built = (entry, found)
            kept = made.built
    built: _Description | None = kept[1]
    return built


def sources_for(
    calibration: Calibration | None, description: _Description, made_from: Any = None
) -> Sources:
    """The sources of a system of *description*: its *calibration*, with
    what it built (see ``_built_by``; this description, for a calibration
    that names no registered system); with none, those of the system it was
    *made_from* (a variant made with ``replace``); with neither (a system
    built from qudits), the shipped calibrations."""
    if calibration is not None:
        return [(calibration, _built_by(calibration) or description)]
    if made_from:
        return sources_of(made_from)
    return _shipped()


def sources_of(system: Any) -> Sources:
    """``sources_for`` a quantum system. A variant is made from a system
    that was built (``QuantumSystem.replace`` hands on the system its own
    system was made from) or from what another variant or its pickle
    carried: that system's calibration and description, which it keeps in
    place of the system when it is first asked. What the calibration is
    compared against is found at each ask."""
    if system.calibration is not None:
        return sources_for(system.calibration, system._description)
    made = system.__dict__.get("_made_from")
    if made is None or made is system:
        return _shipped()
    if made.__class__ is not _Carried:
        made = system._made_from = _Carried(made.calibration, made._description)
    if made.calibration is None:
        return _shipped()
    return sources_for(made.calibration, made._description)


class _Carried:
    """What a pickled variant carries of the system its answers come from:
    that system's calibration, one this package does not ship, and its
    description. It stands for that system where the variant's sources are
    found (``sources_of``), which is when the variant is first asked."""

    def __init__(self, calibration: Calibration, description: _Description) -> None:
        self.calibration = calibration
        self._description = description


def sourced_reduce(
    system: Any, key: tuple[Any, ...], parts: tuple[Any, ...]
) -> tuple[Any, ...] | None:
    """How a variant whose answers come from a calibration this package does
    not ship pickles: with that calibration and the description of the
    system that held it, which its constructor takes as the system it is
    made from, and under a key of its own among the systems a process has
    unpickled (a variant of equal values that answers from another
    calibration has the same fingerprint, name and source). None for any
    other variant, which pickles as it did: what a shipped calibration
    gives is found again by value. Nothing is built or compared here."""
    import pickle

    from .. import _pickling

    # Kept with the description it is of: a copy of the system's state under
    # another description (a drifting system's true values) makes its own.
    held = system.__dict__.get("_sourced")
    kept: tuple[Any, ...] | None = (
        None if held is None or held[0] is not system._description else held[1]
    )
    if kept is None:
        if system.calibration is not None:
            # It answers from the calibration it holds, which it pickles.
            return None
        root = system.__dict__["_made_from"]
        if root.calibration is None or root.calibration._shipped is not None:
            return None
        held_by = root._description
        parts[2]["_variant"] = _Carried(root.calibration, held_by)
        # The key among unpickled systems: the calibration and the system
        # that held it, by value and by label (two systems may hold one
        # calibration, which then answers by label for each).
        order = tuple(sorted(held_by.qudits))
        extension = (
            root.calibration.fingerprint,
            _fingerprint(held_by, order, None),
            tuple(held_by.qudits[q].label for q in order),
        )
        kept = (
            rebuilt_sourced,
            (key, extension, pickle.dumps(parts, pickle.HIGHEST_PROTOCOL)),
        )
        system._sourced = (system._description, kept)
    with _pickling._devices_lock:
        if _pickling._devices.get(key) is system:
            del _pickling._devices[key]
    _pickling._keep((key, kept[1][1]), system)
    return kept


def rebuilt_sourced(key: tuple[Any, ...], extension: tuple[Any, ...], description: bytes) -> Any:
    """``_pickling.rebuilt_device`` for what ``sourced_reduce`` pickled."""
    import pickle

    from .. import _pickling

    found = _pickling._devices.get((key, extension))
    if found is not None:
        return found
    cls, target, inputs, *state = pickle.loads(description)
    if len(key) > 3:
        inputs["control_system"] = key[3]
    system = cls(target, **inputs)
    for attributes in state:
        vars(system).update(attributes)
    _pickling._keep((key, extension), system)
    return system


def _origin(sources: Sources, qudit: Any) -> tuple[Calibration, str, _Description] | None:
    """The calibration among *sources* that built *qudit* (one of its values,
    at its index), the label the qudit has there, and what it built."""
    for calibration, description in sources:
        built = description.qudits.get(qudit.index)
        if built is not None and _same(_QUDIT_VALUES, built, qudit):
            return (calibration, built.label, description)
    return None


# ── The answers ─────────────────────────────────────────────────────────────


def tuning_of(description: _Description, sources: Sources, index: int) -> Tuning | None:
    """The tuning of qudit *index* of *description*: declared, else what the
    calibration among *sources* (the system's, ``sources_of``) that built
    the qudit gives it, else None."""
    qudit = description.qudits.get(index)
    if qudit is None:
        raise NotFoundError(f"qudit {index!r} is not in quantum system {description.name!r}")
    tunings = getattr(description, "tunings", None)
    if tunings:
        found: Tuning | None = tunings.get(index)
        if found is not None:
            return found
    origin = _origin(sources, qudit)
    if origin is None:
        return None
    return _derived_tuning(origin[0], origin[1])


def control_of(description: _Description, coupling: Coupling) -> CouplingControl | None:
    """The control *coupling* of *description* is declared, or None."""
    controls = getattr(description, "controls", None)
    if not controls:
        return None
    found: CouplingControl | None = controls.get(_pair(*coupling.qudits))
    return found


def interaction_of(
    description: _Description,
    sources: Sources,
    coupling: Coupling,
    first: int | None = None,
) -> CouplingInteraction | None:
    """The interaction of *coupling* of *description*, made through the
    tunings the system answers with for its two qudits: the one declared
    for the pair, else, for a coupling with no control whose qudits a
    calibration built, a charge coupling of that calibration's g_c; else
    None. Its tunings are in the order of the coupling's qudits, or, with
    *first* (the index of one of them), that qudit's first."""
    if control_of(description, coupling) is not None:
        return None
    order = coupling.qudits
    if first is not None and first != order[0]:
        order = order[::-1]
    qudits = description.qudits
    a, b = (qudits[q] for q in order)
    through_a = tuning_of(description, sources, a.index)
    through_b = tuning_of(description, sources, b.index)
    interactions = getattr(description, "interactions", None)
    declared_for = interactions.get(_pair(*order)) if interactions else None
    if declared_for is not None:
        if through_a is None or through_b is None:
            return declared_for  # type: ignore[no-any-return]
        kept = description._through  # type: ignore[attr-defined]
        made = kept.get(order)
        if made is None or made.tunings[0] is not through_a or made.tunings[1] is not through_b:
            made = kept[order] = declared_for._through((through_a, through_b))
        return made  # type: ignore[no-any-return]
    one, other = _origin(sources, a), _origin(sources, b)
    known = one or other
    if known is None or (one and other and one[0] is not other[0]):
        return None
    source, _, built = known
    # Only a coupling that the calibration built.
    if built is not description and not any(
        _same(_COUPLING_VALUES, c, coupling) for c in built.couplings
    ):
        return None
    if through_a is None or through_b is None:
        return None
    from .tuning import FluxTuning

    # A charge coupling is read through transmon circuits.
    if not (isinstance(through_a, FluxTuning) and isinstance(through_b, FluxTuning)):
        return None
    return _derived_interaction(source, (a.label, b.label), (through_a, through_b))


def answer(system: Any, name: str) -> Any:
    """``system.tuning``, ``system.control`` or ``system.interaction``: the
    question *name* asks, as a function of the qudits it is asked about (a
    handle of the system, or a qudit's index)."""

    def index(qudit: Any) -> int:
        if isinstance(qudit, Qudit):
            if qudit.quantum_system is not system and qudit.quantum_system != system:
                from .._device import QuantumSystemMismatchError

                other = qudit.quantum_system.name
                raise QuantumSystemMismatchError(
                    f"{qudit!r} is a qudit of "
                    + (
                        f"another quantum system named {other!r}"
                        if other == system.name
                        else f"{other!r}, not of {system.name!r}"
                    )
                    + "; pass a handle of this quantum system, or the qudit's index"
                )
            return qudit.index
        import numbers

        if isinstance(qudit, bool) or not isinstance(qudit, numbers.Integral):
            raise InvalidTypeError(
                f"QuantumSystem.{name} takes qudit handles, as qs.qudits[i], or their "
                f"indices; got {qudit!r}"
            )
        return int(qudit)

    described = system._description
    if name == "tuning":

        def tuning(qudit: Any) -> Tuning | None:
            """How the frequency of *qudit* is tuned: the tuning the system
            declares for it, else the one its calibration gives it, else None."""
            return tuning_of(described, sources_of(system), index(qudit))

        return tuning
    if name == "control":

        def control(a: Any, b: Any) -> CouplingControl | None:
            """The control of the coupling of *a* and *b*, which a coupling
            line reaches: what the system declares for the pair, or None."""
            return control_of(described, described.coupling(index(a), index(b)))

        return control

    def interaction(a: Any, b: Any) -> CouplingInteraction | None:
        """How *a* and *b* interact through a coupling with no line of its
        own, made through their tunings in the order asked."""
        first = index(a)
        return interaction_of(
            described, sources_of(system), described.coupling(first, index(b)), first
        )

    return interaction
