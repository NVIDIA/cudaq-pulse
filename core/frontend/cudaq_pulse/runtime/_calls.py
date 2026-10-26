# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Simulation of programs that call classical functions, or that read out
through a readout model.

The program evolves in segments between events, in time order: the end of a
readout and the start of a call. At a readout each outcome continues on its
own path, as in ``simulate``; with a readout model the outcome a branch and a
call see is the *reported* one, drawn from the assignment matrix, while the
state is projected on the true one. At a call the handler runs on the
outcomes and numbers of the path, and its answer is bound to the program as
constants, which is then scheduled again: a path's timeline, variable waits
included, is static once its answers are known. Handlers are deterministic
functions of the path, so a call does not branch, except a timeout, which
branches on whether the answer came in time, with the probability of the
function's latency model. Branches that share a state share the array and
carry a weight instead of copying it.

A call that takes IQ points needs shots: each shot on the path from that
call on is its own trajectory, with IQ points drawn for it from the
``IQModel`` at the true level of each readout (the state stays shared until
the answers differ).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..classical import ClassicalFunction

# Branches whose weight falls below this fraction of the initial state's are
# not followed (as in simulate.py).
_NEGLIGIBLE = 1e-14


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
                raise ValueError(
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
        raise TypeError(f"{function.name} must return {len(results)} results, got {answer!r}")
    flat: list[float] = []
    for spec, part in zip(results, parts):
        if spec.is_scalar:
            items = [part]
        else:
            items = list(part)
            if len(items) != spec.answer_count:
                raise TypeError(
                    f"{function.name}: result {spec.descriptor} has {spec.answer_count} "
                    f"values, got {len(items)}"
                )
        for item in items:
            flat.append(float(bool(item)) if spec.answer == "i1" else float(item))
    return flat


@dataclass
class _Leaf:
    state: Any
    path: list[tuple[tuple[int, tuple[int, ...]], int]]
    count: int
    weight: float
    calls: list[CallRecord]
    end: float = 0.0


class ReadoutModel:
    """Per-qudit assignment matrices: P(report r | true k).

    For a qubit, ``readout_params`` of the target give
    ``error_1to0`` (|1> reported 0) and ``error_0to1`` (|0> reported 1). With
    ``decay=True`` relaxation during the readout window (``ro_dur`` ns, else
    the readout pulse) adds 1 - exp(-t / (2 T1)) to the |1> -> 0 error.
    Higher levels are reported as themselves.
    """

    def __init__(
        self, target: Any, qudits: Sequence[int], dims: Sequence[int], decay: bool
    ) -> None:
        self.matrices: dict[int, np.ndarray] = {}
        for position, qudit in enumerate(qudits):
            q = target.qubits[qudit]
            params = q.readout_params
            d = dims[position]
            m = np.eye(d)
            to0 = float(params.get("error_1to0", 0.0))
            to1 = float(params.get("error_0to1", 0.0))
            if decay and q.t1_us > 0 and "ro_dur" in params:
                to0 = to0 + (1 - to0) * (1 - math.exp(-float(params["ro_dur"]) / (2e3 * q.t1_us)))
            if d >= 2:
                m[0, 0], m[1, 0] = 1 - to1, to1
                m[0, 1], m[1, 1] = to0, 1 - to0
            self.matrices[position] = m

    def reports(self, position: int, true_level: int) -> list[tuple[int, float]]:
        column = self.matrices[position][:, true_level]
        return [(r, float(p)) for r, p in enumerate(column) if p > 0]


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
            }

    def depletion_ns(self, qudit: int) -> float:
        """The time the resonator takes to empty to 1% of its photons."""
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


def _schedule_end(module: Any, clock_ghz: float) -> float:
    """The end of a module's schedule, in ns."""
    ends = [s + d for _, _, _, s, d in module.events() if s is not None and d is not None]
    return max(ends, default=0) / clock_ghz


def simulate_tree(
    module: Any,
    engine: Any,
    dims: tuple[int, ...],
    state: Any,
    t_start: float,
    t_end: float | None,
    shots: int | None,
    rng: np.random.Generator,
    functions: Sequence[ClassicalFunction],
    handlers: Mapping[str, Callable[..., Any]],
    schedule_args: tuple[Any, ...],
    readout: ReadoutModel | None,
    latency_branching: bool,
    clock_ghz: float,
    modes: Mapping[int, int],
    iq_model: IQModel | None = None,
) -> tuple[list[_Leaf], dict[tuple[int, tuple[int, ...]], Any]]:
    """Walk every path; returns the leaves and the measurements seen.

    *modes* gives the state's mode of each device qudit."""
    by_name = {f.name: f for f in functions}
    seen: dict[tuple[int, tuple[int, ...]], Any] = {}
    leaves: list[_Leaf] = []
    placeholder_cache: dict[int, Any] = {}
    norm = engine.weight(state)

    def segment_module(current: Any) -> Any:
        key = id(current)
        cached = placeholder_cache.get(key)
        if cached is None or cached[0] is not current:
            cached = (current, current.placeholders() if functions else current)
            placeholder_cache[key] = cached
        return cached[1]

    def invoke(callee: str, inputs: tuple[Any, ...], calls: list[CallRecord]) -> Any:
        handler = handlers.get(callee)
        if handler is None:
            raise KeyError(
                f"no handler for classical function {callee!r}; pass handlers={{{callee!r}: ...}}"
            )
        if getattr(handler, "takes_history", False):
            return handler(*inputs, history=calls)
        return handler(*inputs)

    def answer_call(
        current: Any,
        path: list[tuple[tuple[int, tuple[int, ...]], int]],
        truth: dict[tuple[int, tuple[int, ...]], int],
        state: Any,
        t: float,
        count: int,
        weight: float,
        calls: list[CallRecord],
        call: Any,
        inputs: tuple[Any, ...],
        answer: Any,
    ) -> None:
        index, start, callee, arguments, answered, timeout = call
        function = by_name[callee]
        if not answered:
            calls_here = [*calls, CallRecord(callee, start, inputs, answer, True)]
            run(
                current.bind_call(index, [], *schedule_args),
                path,
                truth,
                state,
                t,
                count,
                weight,
                calls_here,
            )
            return
        values = _answer_values(function, answer)
        limit = timeout if timeout >= 0 else function.timeout_ns
        if limit is None:
            run(
                current.bind_call(index, values, *schedule_args),
                path,
                truth,
                state,
                t,
                count,
                weight,
                [*calls, CallRecord(callee, start, inputs, answer, True)],
            )
            return
        p_ok = latency_in_time_probability(function, limit)
        if shots is None and not latency_branching:
            p_ok = 1.0 if function.nominal_latency_ns <= limit else 0.0
        outcomes_ok = [(True, p_ok), (False, 1.0 - p_ok)]
        counts = (
            rng.multinomial(count, [p_ok, 1.0 - p_ok]) if count else np.zeros(2, dtype=np.int64)
        )
        for (in_time, p), n in zip(outcomes_ok, counts):
            if p <= 0:
                continue
            bound = [1.0, *values] if in_time else [0.0] * (1 + len(values))
            run(
                current.bind_call(index, bound, *schedule_args),
                path,
                truth,
                state,
                t,
                int(n),
                weight * p,
                [
                    *calls,
                    CallRecord(callee, start, inputs, answer if in_time else None, in_time),
                ],
            )
        return

    def run(
        current: Any,
        path: list[tuple[tuple[int, tuple[int, ...]], int]],
        truth: dict[tuple[int, tuple[int, ...]], int],
        state: Any,
        t: float,
        count: int,
        weight: float,
        calls: list[CallRecord],
    ) -> None:
        entries = [[index, *iteration, k] for (index, iteration), k in path]
        end_here = t_end if t_end is not None else _schedule_end(current, clock_ghz)
        found = current.next_measurement(entries)
        call = current.next_call(entries) if functions else None
        measure_at = math.inf if found is None else found[4]
        call_at = math.inf if call is None else call[1]
        take_call = call is not None and call_at < measure_at - 1e-9 and call_at <= end_here
        boundary = min(end_here, call_at if take_call else measure_at)
        if boundary > t:
            engine.module = segment_module(current)
            state = engine.evolve(state, entries, t, boundary)
            t = boundary
        if take_call:
            assert call is not None
            index, start, callee, arguments, answered, timeout = call
            function = by_name[callee]
            outcomes = {key: k for key, k in path}
            if not takes_iq(function):
                inputs = _call_inputs(function, arguments, outcomes)
                answer = invoke(callee, inputs, calls)
                answer_call(
                    current, path, truth, state, t, count, weight, calls, call, inputs, answer
                )
                return
            if shots is None or iq_model is None:
                raise ValueError(f"{callee}: IQ points are drawn per shot; simulate with shots")
            drives = {(i, tuple(it)): (f, a) for i, it, f, a in current.readout_drives(entries)}

            def point(key: tuple[int, tuple[int, ...]]) -> tuple[float, float]:
                _, _, qudit, begin, finish = seen[key]
                frequency, amplitude = drives.get(key, (math.nan, math.nan))
                return iq_model.sample(rng, qudit, truth[key], frequency, amplitude, finish - begin)

            # Each shot draws its own points; a path that got no shots still
            # draws one, for the ensemble state. What follows depends only on
            # the answer, so shots with equal answers continue as one path
            # (its record holds the first such shot's inputs).
            n = max(count, 1)
            groups: dict[tuple[float, ...], list[Any]] = {}
            for _ in range(n):
                inputs = _call_inputs(function, arguments, outcomes, point)
                answer = invoke(callee, inputs, calls)
                same = tuple(_answer_values(function, answer)) if answered else ()
                group = groups.setdefault(same, [inputs, answer, 0])
                group[2] += 1
            for inputs, answer, k in groups.values():
                answer_call(
                    current,
                    path,
                    truth,
                    state,
                    t,
                    k if count else 0,
                    weight * k / n,
                    calls,
                    call,
                    inputs,
                    answer,
                )
            return
        if found is None or found[4] > end_here:
            leaves.append(_Leaf(state, path, count, weight, calls, max(t, end_here)))
            return
        index, iteration, qudit, start, end = found
        key = (index, tuple(iteration))
        seen[key] = (index, tuple(iteration), qudit, start, end)
        m = modes[qudit]
        projected = [engine.project(state, dims, m, k) for k in range(dims[m])]
        weights = np.array([engine.weight(s) for s in projected]) * weight
        branches: list[tuple[int, int, Any, float]] = []
        for k, (branch, w) in enumerate(zip(projected, weights)):
            if w <= _NEGLIGIBLE * norm:
                continue
            if readout is None:
                branches.append((k, k, branch, 1.0))
            else:
                for r, p in readout.reports(m, k):
                    branches.append((r, k, branch, p))
        probabilities = np.array(
            [engine.weight(b) * p * weight for _, _, b, p in branches], dtype=float
        )
        total = probabilities.sum()
        counts = (
            rng.multinomial(count, probabilities / total)
            if count and total > 0
            else np.zeros(len(branches), dtype=np.int64)
        )
        for (r, k, branch, p), n in zip(branches, counts):
            run(
                current,
                [*path, (key, r)],
                {**truth, key: k},
                branch,
                max(end, t),
                int(n),
                weight * p,
                calls,
            )

    run(module, [], {}, state, t_start, shots or 0, 1.0, [])
    return leaves, seen
