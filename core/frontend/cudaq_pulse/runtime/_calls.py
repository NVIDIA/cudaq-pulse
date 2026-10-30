# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The classical calls of a simulated or emulated program: the record of
a call, a function's latency, the values that go in and come back, and
the IQ points of readouts.

The simulation of programs that call classical functions, or that read out
through a readout model, is in ``_call_tree``, which needs numpy;
``ReadoutModel`` and ``simulate_tree`` are reached through this module.
Every export imports this module, so it loads neither numpy nor
``_call_tree`` until a simulation asks for one of them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidTypeError, InvalidValueError
from ..classical import ClassicalFunction

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class CallRecord:
    """One call a path made: what went in, what came back, in time."""

    function: str
    start_ns: float
    inputs: tuple[Any, ...]
    answer: Any
    in_time: bool


def latency_in_time_probability(function: ClassicalFunction, timeout_ns: float) -> float:
    """P(latency <= timeout) for the function's latency, triangular on
    [min, max] with its mode at the nominal latency."""
    lo, mode, hi = function.min_latency_ns, function.nominal_latency_ns, function.max_latency_ns
    if timeout_ns >= hi:
        return 1.0
    if timeout_ns < lo:
        return 0.0
    if hi == lo:
        return 1.0
    x = timeout_ns
    if x <= mode:
        return (x - lo) ** 2 / ((hi - lo) * (mode - lo)) if mode > lo else 0.0
    return 1.0 - (hi - x) ** 2 / ((hi - lo) * (hi - mode))


def takes_iq(function: ClassicalFunction) -> bool:
    return any(spec.descriptor.startswith("iq") for spec in function.args)


def _call_inputs(
    function: ClassicalFunction,
    arguments: Sequence[tuple[int, int, Sequence[int], float]],
    outcomes: Mapping[tuple[int, tuple[int, ...]], int],
    iq: Callable[[tuple[int, tuple[int, ...]]], tuple[float, float]] | None = None,
) -> tuple[Any, ...]:
    """The handler's arguments, grouped by the function's argument types.

    *iq* gives the IQ point of a measurement, for IQ arguments."""
    values: list[Any] = []
    keys: list[tuple[int, tuple[int, ...]]] = []
    for kind, readout, iteration, number in arguments:
        key = (readout, tuple(iteration))
        keys.append(key)
        values.append(outcomes[key] if kind == 0 else number)
    grouped: list[Any] = []
    cursor = 0
    for spec in function.args:
        part = values[cursor : cursor + spec.count]
        part_keys = keys[cursor : cursor + spec.count]
        cursor += spec.count
        if spec.descriptor.startswith("bits"):
            part = [int(v != 0) for v in part]
        elif spec.descriptor.startswith("iq"):
            if iq is None:
                raise InvalidValueError(
                    f"{function.name}: IQ points are drawn per shot; simulate with shots"
                )
            part = [iq(key) for key in part_keys]
        elif spec.source == "integer":
            part = [int(v) for v in part]
        grouped.append(part[0] if spec.is_scalar else list(part))
    return tuple(grouped)


def _answer_values(function: ClassicalFunction, answer: Any) -> list[float]:
    """The handler's answer as the flat list of SSA values the wait gives."""
    results = function.results
    parts = answer if len(results) > 1 else (answer,)
    if len(results) > 1 and (not isinstance(answer, (tuple, list)) or len(answer) != len(results)):
        raise InvalidTypeError(
            f"{function.name} must return {len(results)} results, got {answer!r}"
        )
    flat: list[float] = []
    for spec, part in zip(results, parts):
        if spec.is_scalar:
            items = [part]
        else:
            items = list(part)
            if len(items) != spec.answer_count:
                raise InvalidTypeError(
                    f"{function.name}: result {spec.descriptor} has {spec.answer_count} "
                    f"values, got {len(items)}"
                )
        for item in items:
            flat.append(float(bool(item)) if spec.answer == "i1" else float(item))
    return flat


class IQModel:
    """Synthesized IQ points of readouts (dispersive readout, per qudit).

    The resonator, at ``ro_freq`` (else at the readout tone, so the tone sits
    midway between the pulled frequencies), is pulled by
    ``chi_k = (2k - 1) chi`` for qudit level ``k`` (``chi_hz``, default
    1 MHz). Driven at the readout tone's frequency ``f`` with field rate
    ``epsilon`` it settles at

        alpha_k = epsilon / (kappa / 2 - i 2 pi (f - f_r - chi_k)),

    (``kappa_eff_hz``, default 2 MHz). ``epsilon`` scales with the readout
    pulse's amplitude: at ``ro_amp`` (default: the amplitude played) driving
    midway holds ``ro_photons`` photons (default 4). The integrated point
    over ``ro_integration`` ns (else the readout's length) ``tau`` is

        sqrt(eta kappa tau) alpha_k + noise,

    with vacuum noise of variance 1/2 per quadrature and efficiency
    ``ro_efficiency`` eta (default 0.5). With ``decay``, a qubit in ``|1>``
    that relaxes at a time ``t`` drawn from its T1 within the window
    integrates ``alpha_1`` for ``t`` and ``alpha_0`` after.
    """

    def __init__(self, target: Any, qudits: Sequence[int], decay: bool) -> None:
        self.decay = decay
        self._params: dict[int, dict[str, float]] = {}
        for qudit in qudits:
            q = target.qubits[qudit]
            p = q.readout_params
            self._params[qudit] = {
                "chi": float(p.get("chi_hz", 1e6)),
                "kappa": 2 * math.pi * float(p.get("kappa_eff_hz", 2e6)) * 1e-9,
                "f_r": float(p.get("ro_freq", math.nan)),
                "amp": float(p.get("ro_amp", math.nan)),
                "photons": float(p.get("ro_photons", 4.0)),
                "eta": float(p.get("ro_efficiency", 0.5)),
                "tau": float(p.get("ro_integration", math.nan)),
                "t1": float(q.t1_us) * 1e3,
                "depletion": float(p.get("depletion_ns", math.nan)),
            }

    def depletion_ns(self, qudit: int) -> float:
        """The time the resonator takes to empty to 1% of its photons
        (``readout_params["depletion_ns"]`` if given)."""
        explicit = self._params[qudit].get("depletion")
        if explicit is not None and math.isfinite(explicit):
            return explicit
        return math.log(100.0) / self._params[qudit]["kappa"]

    def center(
        self, qudit: int, level: int, frequency_hz: float, amplitude: float, duration_ns: float
    ) -> complex:
        """The noiseless integrated IQ point of *level*."""
        p = self._params[qudit]
        kappa, chi = p["kappa"], p["chi"]
        f_r = p["f_r"] if math.isfinite(p["f_r"]) else frequency_hz
        if not math.isfinite(frequency_hz):
            frequency_hz = f_r if math.isfinite(f_r) else 0.0
            f_r = frequency_hz
        reference = p["amp"] if math.isfinite(p["amp"]) else amplitude
        scale = amplitude / reference if math.isfinite(amplitude) and reference else 1.0
        epsilon = math.sqrt(p["photons"]) * abs(complex(kappa / 2, -2 * math.pi * chi * 1e-9))
        tau = p["tau"] if math.isfinite(p["tau"]) else duration_ns
        detuning = 2 * math.pi * (frequency_hz - f_r - (2 * level - 1) * chi) * 1e-9
        alpha = epsilon * scale / complex(kappa / 2, -detuning)
        return complex(math.sqrt(p["eta"] * kappa * tau) * alpha)

    def sample(
        self,
        rng: np.random.Generator,
        qudit: int,
        level: int,
        frequency_hz: float,
        amplitude: float,
        duration_ns: float,
    ) -> tuple[float, float]:
        point = self.center(qudit, level, frequency_hz, amplitude, duration_ns)
        p = self._params[qudit]
        if self.decay and level == 1 and p["t1"] > 0:
            tau = p["tau"] if math.isfinite(p["tau"]) else duration_ns
            t = rng.exponential(p["t1"])
            if t < tau:
                ground = self.center(qudit, 0, frequency_hz, amplitude, duration_ns)
                point = (t * point + (tau - t) * ground) / tau
        noise = rng.normal(0.0, math.sqrt(0.5), 2)
        return (point.real + noise[0], point.imag + noise[1])


def __getattr__(name: str) -> Any:
    # Removed once it has bound both names: the import system asks a module
    # for __path__ on every ``from ... import``, and simulate() imports from
    # this module on every call.
    if name in ("ReadoutModel", "simulate_tree"):
        from . import _call_tree

        found = globals()
        found["ReadoutModel"] = _call_tree.ReadoutModel
        found["simulate_tree"] = _call_tree.simulate_tree
        found.pop("__getattr__", None)
        return found[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
