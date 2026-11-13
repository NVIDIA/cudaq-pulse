# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidTypeError, InvalidValueError
from ..classical import ClassicalFunction

if TYPE_CHECKING:
    import numpy as np

    from ._call_tree import ReadoutModel as ReadoutModel
    from ._call_tree import simulate_tree as simulate_tree


@dataclass(frozen=True)
class CallRecord:
    """One call a path made: what went in, what came back, in time.

    ``function`` is the name it is declared with and ``compiled_name`` the
    name it is compiled and served under: a version's, such as
    ``"flip__3"``, for a function whose list lengths come from its calls.
    """

    function: str
    compiled_name: str
    start_ns: float
    inputs: tuple[Any, ...]
    answer: Any
    in_time: bool


def latency_in_time_probability(function: ClassicalFunction[..., Any], timeout_ns: float) -> float:
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


def _point(value: Any) -> complex:
    return complex(value) if isinstance(value, complex) else complex(*value)


def takes_iq(function: ClassicalFunction[..., Any]) -> bool:
    return any(spec.descriptor.startswith("iq") for spec in function.args)


def _call_inputs(
    function: ClassicalFunction[..., Any],
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
                    f"{function.name}: IQ points are drawn per shot; simulate with shots_count"
                )
            part = [_point(iq(key)) for key in part_keys]
        elif spec.source == "integer":
            part = [int(v) for v in part]
        grouped.append(part[0] if spec.is_scalar else list(part))
    return tuple(grouped)


def _answer_values(function: ClassicalFunction[..., Any], answer: Any) -> list[float]:
    """The handler's answer as the flat list of SSA values the wait gives."""
    results = function.results
    parts = answer if len(results) > 1 else (answer,)
    if len(results) > 1 and (not isinstance(answer, (tuple, list)) or len(answer) != len(results)):
        raise InvalidTypeError(
            f"{function.name} must return {len(results)} results, got {answer!r}"
        )
    flat: list[float] = []
    for spec, part in zip(results, parts, strict=False):
        if spec.is_scalar:
            items = [part]
        else:
            items = list(part)
            if len(items) != spec.answer_count:
                from ..classical import _wrong_length

                position = [s is spec for s in results].index(True)
                raise InvalidTypeError(_wrong_length(function, position, len(items)))
        for item in items:
            flat.append(float(bool(item)) if spec.answer == "i1" else float(item))
    return flat


class IQModel:
    """Synthesized IQ points of readouts, from each qudit's
    ``DispersiveReadout``, which must give its ``dispersive_shift_hz`` and
    ``linewidth_hz``, and its ``READOUT`` calibration (a ``ReadoutPulse``;
    one with every field left out if it publishes none); a qudit without a
    readout, or with another, is ``UnsupportedError``.

    The resonator, at its ``frequency_hz`` (else at the readout tone, so the
    tone sits midway between the pulled frequencies), is pulled by
    ``chi_k = (2k - 1) chi`` for qudit level ``k`` (``dispersive_shift_hz``).
    Driven at the readout tone's frequency ``f`` with field rate ``epsilon``
    it settles at

        alpha_k = epsilon / (kappa / 2 - i 2 pi (f - f_r - chi_k)),

    (``linewidth_hz``). ``epsilon`` scales with the readout pulse's
    amplitude: at the pulse's ``amplitude`` (default: the amplitude played) driving
    midway holds ``photon_number`` photons. The integrated point over
    ``integration_ns`` (else the readout's length) ``tau`` is

        sqrt(eta kappa tau) alpha_k + noise,

    with vacuum noise of variance 1/2 per quadrature and efficiency
    ``efficiency`` eta. With ``decay``, a qubit in ``|1>``
    that relaxes at a time ``t`` drawn from its T1 within the window
    integrates ``alpha_1`` for ``t`` and ``alpha_0`` after.

    A readout whose tone frequency (with a ``frequency_hz``) or pulse
    amplitude (with an ``amplitude``) is not a number where it plays is
    ``UnsupportedError``.
    """

    def __init__(self, target: Any, qudits: Sequence[int], decay: bool) -> None:
        from cudaq_pulse.targets.base import DispersiveReadout, ReadoutPulse

        self.decay = decay
        self._params: dict[int, dict[str, float]] = {}
        nan = math.nan
        for qudit in qudits:
            q = target.qudits[qudit]
            r = q.readout
            if not isinstance(r, DispersiveReadout):
                from .._errors import UnsupportedError

                problem = (
                    f"qudit {qudit} has no readout"
                    if r is None
                    else f"no readout model for {type(r).__name__} (qudit {qudit})"
                )
                raise UnsupportedError(
                    f"{problem}: IQ points are synthesized from a DispersiveReadout"
                )
            # DispersiveReadout._fields and ReadoutPulse._fields, in their order.
            f_r, chi, linewidth, eta, _, _, depletion = r._values
            pulse = q.calibrations.get("readout") or ReadoutPulse()
            _, amp, _, tau, photons = pulse._values
            if chi is None or linewidth is None:
                from .._errors import UnsupportedError

                missing = [
                    n
                    for n, v in (("dispersive_shift_hz", chi), ("linewidth_hz", linewidth))
                    if v is None
                ]
                raise UnsupportedError(
                    f"the IQ points of qudit {qudit} need its DispersiveReadout's "
                    f"{' and '.join(missing)}, which it does not give"
                )
            self._params[qudit] = {
                "chi": float(chi),
                "kappa": 2 * math.pi * float(linewidth) * 1e-9,
                "f_r": nan if f_r is None else float(f_r),
                "amp": nan if amp is None else float(amp),
                "photons": float(photons),
                "eta": float(eta),
                "tau": nan if tau is None else float(tau),
                "t1": float(q.t1_us) * 1e3,
                "depletion": nan if depletion is None else float(depletion),
            }

    def depletion_ns(self, qudit: int) -> float:
        """The time the resonator takes to empty to 1% of its photons
        (its ``depletion_ns`` if given)."""
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
        f_r = p["f_r"]
        if not math.isfinite(frequency_hz):
            if math.isfinite(f_r):
                from .._errors import UnsupportedError

                raise UnsupportedError(
                    f"the IQ points of a readout of qudit {qudit} depend on its tone's "
                    "frequency, which is not a number where the readout plays: the readout "
                    "model computes it from constants, float arguments and the answers of "
                    "earlier calls; its DispersiveReadout's frequency_hz needs it"
                )
            frequency_hz = f_r = 0.0
        elif not math.isfinite(f_r):
            f_r = frequency_hz
        reference = p["amp"]
        if not math.isfinite(amplitude):
            if math.isfinite(reference):
                from .._errors import UnsupportedError

                raise UnsupportedError(
                    f"the IQ points of a readout of qudit {qudit} depend on its pulse's peak "
                    "amplitude, which is not a number where the readout plays: the readout "
                    "model computes it from constants, float arguments and the answers of "
                    "earlier calls, for a built-in waveform; its READOUT calibration's amplitude "
                    "needs it"
                )
            scale = 1.0
        else:
            scale = amplitude / reference if math.isfinite(reference) and reference else 1.0
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


class _FromTree(str):
    """The name of a ``_call_tree`` attribute, imported on first use and then
    kept in this module's namespace, where later reads find it as a plain
    attribute."""

    def __get__(self, module: types.ModuleType | None, owner: type) -> Any:
        if module is None:
            return self
        from . import _call_tree

        name = str(self)
        value = module.__dict__[name] = getattr(_call_tree, name)
        return value


class _CallsModule(types.ModuleType):
    # Not a module __getattr__: the import system asks a module for __path__
    # on every ``from ... import``, and simulate() imports from this module
    # on every call.
    ReadoutModel = _FromTree("ReadoutModel")
    simulate_tree = _FromTree("simulate_tree")


sys.modules[__name__].__class__ = _CallsModule
