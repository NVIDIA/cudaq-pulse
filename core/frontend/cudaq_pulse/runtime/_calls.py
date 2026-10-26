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


def _call_inputs(
    function: ClassicalFunction,
    arguments: Sequence[tuple[int, int, Sequence[int], float]],
    outcomes: Mapping[tuple[int, tuple[int, ...]], int],
) -> tuple[Any, ...]:
    """The handler's arguments, grouped by the function's argument types."""
    values: list[Any] = []
    for kind, readout, iteration, number in arguments:
        if kind == 0:
            values.append(outcomes[(readout, tuple(iteration))])
        else:
            values.append(number)
    grouped: list[Any] = []
    cursor = 0
    for spec in function.args:
        part = values[cursor : cursor + spec.count]
        cursor += spec.count
        if spec.descriptor.startswith("bits"):
            part = [int(v != 0) for v in part]
        elif spec.descriptor.startswith("iq"):
            raise NotImplementedError(
                f"{function.name}: IQ arguments are drawn per shot; simulate with "
                "trajectories=True and shots"
            )
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
) -> tuple[list[_Leaf], dict[tuple[int, tuple[int, ...]], Any]]:
    """Walk every path; returns the leaves and the measurements seen."""
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

    def run(
        current: Any,
        path: list[tuple[tuple[int, tuple[int, ...]], int]],
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
            inputs = _call_inputs(function, arguments, outcomes)
            handler = handlers.get(callee)
            if handler is None:
                raise KeyError(
                    f"no handler for classical function {callee!r}; "
                    f"pass handlers={{{callee!r}: ...}}"
                )
            answer = handler(*inputs)
            if not answered:
                calls_here = [*calls, CallRecord(callee, start, inputs, answer, True)]
                run(
                    current.bind_call(index, [], *schedule_args),
                    path,
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
        if found is None or found[4] > end_here:
            leaves.append(_Leaf(state, path, count, weight, calls, max(t, end_here)))
            return
        index, iteration, qudit, start, end = found
        key = (index, tuple(iteration))
        seen[key] = (index, tuple(iteration), qudit, start, end)
        projected = [engine.project(state, dims, qudit, k) for k in range(dims[qudit])]
        weights = np.array([engine.weight(s) for s in projected]) * weight
        branches: list[tuple[int, Any, float]] = []
        for k, (branch, w) in enumerate(zip(projected, weights)):
            if w <= _NEGLIGIBLE * norm:
                continue
            if readout is None:
                branches.append((k, branch, 1.0))
            else:
                for r, p in readout.reports(qudit, k):
                    branches.append((r, branch, p))
        probabilities = np.array(
            [engine.weight(b) * p * weight for _, b, p in branches], dtype=float
        )
        total = probabilities.sum()
        counts = (
            rng.multinomial(count, probabilities / total)
            if count and total > 0
            else np.zeros(len(branches), dtype=np.int64)
        )
        for (r, branch, p), n in zip(branches, counts):
            run(current, [*path, (key, r)], branch, max(end, t), int(n), weight * p, calls)

    run(module, [], state, t_start, shots or 0, 1.0, [])
    return leaves, seen
