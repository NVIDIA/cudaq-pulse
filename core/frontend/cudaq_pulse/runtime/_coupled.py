# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simulation of a program that holds a coupling line.

What a pulse on a coupling line does is the quantum system's to say: the
tuning of the qudit the line tunes (``qs.tuning(q)``), or the control of the
coupling it reaches (``qs.control(a, d)``). The compiled program holds
neither, so it is the same program on every system, and the lowering to
operators writes each such pulse as terms that hold the pulse's amplitude at
every cycle (``kMappedSamples`` in PulseToQOp.cpp). This module maps them
with the system the program is simulated on:

- a pulse on the line of one qudit, of amplitude ``a(t)``, adds
  ``2 pi (f(a) - f(0)) n + pi (alpha(a) - alpha(0)) n (n - 1)`` in the
  qudit's frame, which turns at its idle frequency;
- a pulse on the line of a pair adds the change of the pair's exchange,
  ``2 pi (J(a) - J(0)) (a^dagger b + a b^dagger)``, turning with the two
  frames as the static exchange does.

``simulate`` comes here only after the lowering refused a program for
holding a coupling line: a program without one is simulated by the code it
always was. Loaded by the first such simulation.
"""

from __future__ import annotations

import functools
import math
import types
from typing import Any

import numpy as np

from .._errors import InvalidValueError, NotFoundError, UnsupportedError
from . import simulate as _simulate

# How the refusal of a program that holds a coupling line starts
# (bindings.cpp, refuseSimulatedCouplingLines; PulseToQOp.cpp).
_REFUSAL = "this program holds a coupling line"
# The callback kind of an unmapped pulse, and of the samples it becomes.
_UNMAPPED = 7 | 0x400
_SAMPLES = 7
# A callback that adds callbacks (WaveformCallback.h, kSumKind): the number
# of them, then each one's kind, size and parameters.
_SUM = 0x200
_HEADER = 10
# Hz to rad/ns.
_RADIANS = 2.0 * math.pi * 1.0e-9
# The most a mapped coefficient may turn in one step of the integrator, in
# radians: the step is shortened to it in the intervals a pulse plays in.
_TURN = 0.05
_MOST_STEPS = 20_000_000


def simulate_coupled(
    refused: UnsupportedError,
    diverged: Any,
    system: Any,
    lowered: Any,
    target: Any,
    **options: Any,
) -> Any:
    """``simulate`` of *lowered* on the quantum system *system* (its
    description: *target*) after the lowering *refused* it for holding a
    coupling line; any other refusal is raised again. *diverged* makes the
    error of an integrator that diverged."""
    if not str(refused).startswith(_REFUSAL):
        raise refused
    from ..quantum_systems.models import TransmonModel

    if getattr(type(target.model), "_simulate", None) is not TransmonModel._simulate:
        raise refused
    if options["backend"] != "cpu":
        raise UnsupportedError(
            "a program with a coupling line is simulated by the CPU backend "
            '(backend="cpu"); the GPU backend does not simulate one yet'
        ) from None
    if getattr(lowered, "carriers", None):
        raise UnsupportedError(
            "a program with a coupling line and a qudit that a call binds (a parameter "
            "annotated Qudit) is not simulated yet; pass the qudit as a static argument"
        ) from None
    try:
        return _compiled(system, lowered)(lowered, target, **options)
    except _simulate._DivergedError as error:
        raise diverged(error) from None


def _compiled(system: Any, lowered: Any) -> Any:
    """``simulate._simulate_compiled`` with a CPU backend that maps the
    pulses on coupling lines of *lowered* with *system*: the same code,
    reading the backend's name from its own globals."""
    plain = _simulate._simulate_compiled
    names = dict(plain.__globals__)
    names["_CpuBackend"] = functools.partial(_MappingBackend, system, lowered)
    mapping = types.FunctionType(
        plain.__code__, names, plain.__name__, plain.__defaults__, plain.__closure__
    )
    mapping.__kwdefaults__ = plain.__kwdefaults__
    return mapping


class _MappingBackend(_simulate._CpuBackend):
    """The CPU backend on a module whose pulses on coupling lines are
    mapped; its step is the given one, shortened in the intervals where a
    mapped coefficient turns faster than ``_TURN`` per step."""

    def __init__(
        self,
        target: Any,
        compiled: Any,
        module: Any,
        dims: tuple[int, ...],
        dt_ns: float,
        base: str,
        lock: Any,
        arguments: list[float],
    ) -> None:
        if arguments:
            # A pulse's samples are read from its waveform's values: the
            # program with its float arguments substituted.
            module, arguments = compiled._concrete(), []
        super().__init__(module, dims, dt_ns, base, lock, arguments)
        self.module = _Mapping(module, target)
        self._given = dt_ns
        self._cycle_ns = 1.0 / float(compiled.quantum_system.clock_ghz)

    def evolve(self, state: Any, entries: list[list[int]], t0: float, t1: float) -> Any:
        with self.lock:
            self.module.set_attributes(
                self.base, _simulate._interval_attributes(entries, t0, t1, 1)
            )
            self.module.operator_program(self.arguments)
            fastest = self.module.fastest
        step = self._given
        if fastest * step > _TURN:
            # Whole steps per program cycle: a pulse's samples change at the
            # cycles' edges, which a step must not straddle.
            cycle = self._cycle_ns
            step = cycle / math.ceil(cycle * fastest / _TURN)
            if (t1 - t0) / step > _MOST_STEPS:
                raise InvalidValueError(
                    f"simulate(): a pulse on a coupling line turns a level at "
                    f"{fastest:.3g} rad/ns between {t0:g} and {t1:g} ns, which takes more "
                    f"than {_MOST_STEPS} steps of the integrator"
                )
        self.dt_ns = step
        return super().evolve(state, entries, t0, t1)


class _Mapping:
    """A compiled module whose operator program has the pulses on coupling
    lines mapped by the quantum system *target*; everything else is the
    module's."""

    __slots__ = ("_module", "_target", "_kept", "fastest")

    def __init__(self, module: Any, target: Any) -> None:
        self._module = module
        self._target = target
        # The shifts of the pulses mapped so far, by qudits and samples.
        self._kept: dict[Any, Any] = {}
        self.fastest = 0.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def set_attributes(self, base: str, interval: str) -> None:
        # The lowering writes the pulses for whoever asks it to.
        self._module.set_attributes(base, interval[:-1] + ", pulse.simulate_coupling_lines = true}")

    def operator_program(self, arguments: Any) -> Any:
        try:
            program = self._module.operator_program(arguments)
        finally:
            # The module is shared: whoever lowers it next has not asked.
            self._module.set_attributes("{pulse.simulate_coupling_lines = false}")
        self.fastest = 0.0
        hamiltonian = []
        for term in program[1]:
            kind = term[2]
            if kind == _UNMAPPED:
                term = (*term[:2], _SAMPLES, self._mapped(term[3]), term[4])
            elif kind == _SUM:
                term = (*term[:3], self._summed(term[3]), term[4])
            hamiltonian.append(term)
        return (program[0], hamiltonian, *program[2:])

    def _summed(self, params: Any) -> Any:
        """The parameters of a sum of callbacks with its unmapped ones mapped."""
        out = [params[0]]
        at = 1
        changed = False
        for _ in range(int(params[0])):
            kind, size = int(params[at]), int(params[at + 1])
            member = params[at + 2 : at + 2 + size]
            at += 2 + size
            if kind == _UNMAPPED:
                member = self._mapped(member)
                kind = _SAMPLES
                changed = True
            out += [float(kind), float(len(member)), *member]
        if not changed:
            return params
        return out if params.__class__ is list else tuple(out)

    def _mapped(self, params: Any) -> Any:
        """The parameters of an unmapped pulse as samples: its header and
        the coefficient at each cycle."""
        what, sign, count = int(params[_HEADER]), params[_HEADER + 1], int(params[_HEADER + 2])
        first = _HEADER + 3 + count
        qudits = tuple(int(q) for q in params[_HEADER + 3 : first])
        amplitudes = np.asarray(params[first:], dtype=float)
        key = (what, qudits, amplitudes.tobytes())
        shift = self._kept.get(key)
        if shift is None:
            shift = self._kept[key] = self._shift(what, qudits, amplitudes)
        # A callback's value is half its sample (WaveformCallback.h): the
        # number operator's coefficient is the whole shift, the exchange's
        # half of it on each of X X and Y Y.
        values = (_RADIANS if what else 2.0 * _RADIANS) * sign * shift
        self.fastest = max(self.fastest, float(np.max(np.abs(values), initial=0.0)))
        mapped = [*params[:_HEADER], *values.tolist()]
        return mapped if params.__class__ is list else tuple(mapped)

    def _shift(self, what: int, qudits: tuple[int, ...], amplitudes: Any) -> Any:
        """In Hz, at each of *amplitudes*: the frequency shift of the qudit
        (*what* 0), its anharmonicity's (1), or the change of the pair's
        exchange (2)."""
        target = self._target
        if what == 2:
            control = target.control(*qudits)
            if control is None:
                raise NotFoundError(
                    f"simulate(): quantum system {target.name!r} declares no control of the "
                    f"coupling of qudits {qudits[0]} and {qudits[1]}, so nothing says what the "
                    "pulse on its coupling line does there; declare it with "
                    f"controls={{{qudits!r}: ...}}"
                )
            _no_response(control, f"the control of qudits {qudits[0]} and {qudits[1]}")
            at = getattr(control, "at", None)
            if at is None:
                raise UnsupportedError(
                    f"simulate(): {type(control).__name__} gives no at(amplitude), the "
                    "exchange at an amplitude, which the simulator reads"
                )
            return np.asarray(at(amplitudes).exchange_hz) - at(0.0).exchange_hz
        (qudit,) = qudits
        tuning = target.tuning(qudit)
        if tuning is None:
            raise NotFoundError(
                f"simulate(): qudit {qudit} has no tuning in quantum system {target.name!r}, so "
                "nothing says what the pulse on its coupling line does there; declare it "
                f"with tunings={{{qudit}: ...}}"
            )
        _no_response(tuning, f"the tuning of qudit {qudit}")
        read = tuning.anharmonicity_at if what else tuning.frequency_at
        return np.asarray(read(amplitudes)) - read(0.0)


def _no_response(component: Any, what: str) -> None:
    if getattr(component, "response", ()):
        raise UnsupportedError(
            f"simulate(): {what} declares a response of its wiring "
            f"({', '.join(map(repr, component.response))}), which the simulator does not "
            "apply yet; simulate it without response="
        )
