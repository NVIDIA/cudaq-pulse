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

from .._errors import InvalidValueError, NotFoundError
from ..classical import ClassicalFunction
from ._calls import (
    CallRecord,
    IQModel,
    _answer_values,
    _call_inputs,
    latency_in_time_probability,
    takes_iq,
)

# Branches whose weight falls below this fraction of the initial state's are
# not followed (as in simulate.py).
_NEGLIGIBLE = 1e-14


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
    functions: Sequence[ClassicalFunction[..., Any]],
    handlers: Mapping[str, Callable[..., Any]],
    schedule_args: tuple[Any, ...],
    readout: ReadoutModel | None,
    latency_branching: bool,
    clock_ghz: float,
    modes: Mapping[int, int],
    iq_model: IQModel | None = None,
    sampled: bool = False,
) -> tuple[list[_Leaf], dict[tuple[int, tuple[int, ...]], Any]]:
    """Walk every path; returns the leaves and the measurements seen.

    *modes* gives the state's mode of each device qudit. With *sampled*,
    only the paths some shot takes are walked."""
    by_name = {f.name: f for f in functions}
    with_lists = {f.name for f in functions if not all(s.is_scalar for s in f.results)}
    seen: dict[tuple[int, tuple[int, ...]], Any] = {}
    leaves: list[_Leaf] = []
    placeholder_cache: dict[int, Any] = {}
    end_cache: dict[int, tuple[Any, float]] = {}
    norm = engine.weight(state)

    def schedule_end(current: Any) -> float:
        cached = end_cache.get(id(current))
        if cached is None or cached[0] is not current:
            cached = (current, _schedule_end(current, clock_ghz))
            end_cache[id(current)] = cached
        return cached[1]

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
            raise NotFoundError(
                f"classical function {callee!r} is external and has no handler; pass "
                f"simulate(..., handlers={{{callee!r}: my_{callee}}}), or declare {callee!r} "
                "with a Python body"
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
        name = function.base_name
        if not answered:
            if callee in with_lists:
                _answer_values(function, answer)
            calls_here = [*calls, CallRecord(name, callee, start, inputs, answer, True)]
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
                [*calls, CallRecord(name, callee, start, inputs, answer, True)],
            )
            return
        p_ok = latency_in_time_probability(function, limit)
        if shots is None and not latency_branching:
            p_ok = 1.0 if function.nominal_latency_ns <= limit else 0.0
        outcomes_ok = [(True, p_ok), (False, 1.0 - p_ok)]
        counts = (
            rng.multinomial(count, [p_ok, 1.0 - p_ok]) if count else np.zeros(2, dtype=np.int64)
        )
        for (in_time, p), n in zip(outcomes_ok, counts, strict=False):
            if p <= 0 or (sampled and n == 0):
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
                    CallRecord(name, callee, start, inputs, answer if in_time else None, in_time),
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
        end_here = t_end if t_end is not None else schedule_end(current)
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
            if function.uniform is not None:
                n = function.uniform
                draws = (
                    rng.multinomial(count, [1.0 / n] * n) if count else np.zeros(n, dtype=np.int64)
                )
                # Every value is followed (every value a shot draws, with sampled
                # paths), so the ensemble state stays exact.
                for value, drawn in enumerate(draws):
                    if sampled and drawn == 0:
                        continue
                    answer_call(
                        current,
                        path,
                        truth,
                        state,
                        t,
                        int(drawn),
                        weight / n,
                        calls,
                        call,
                        (),
                        value,
                    )
                return
            if not takes_iq(function):
                inputs = _call_inputs(function, arguments, outcomes)
                answer = invoke(callee, inputs, calls)
                answer_call(
                    current, path, truth, state, t, count, weight, calls, call, inputs, answer
                )
                return
            if shots is None or iq_model is None:
                raise InvalidValueError(
                    f"{callee}: IQ points are drawn per shot; simulate with shots"
                )
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
        floor = _NEGLIGIBLE * (weights.sum() if sampled else norm)
        for k, (branch, w) in enumerate(zip(projected, weights, strict=False)):
            if w <= floor:
                continue
            if readout is None:
                branches.append((k, k, branch, 1.0))
            else:
                for r, p in readout.reports(m, k):
                    branches.append((r, k, branch, p))
        # Exact propagation leaves an empty level at a trace of +-1e-17.
        probabilities = np.maximum([engine.weight(b) * p * weight for _, _, b, p in branches], 0.0)
        total = probabilities.sum()
        counts = (
            rng.multinomial(count, probabilities / total)
            if count and total > 0
            else np.zeros(len(branches), dtype=np.int64)
        )
        for (r, k, branch, p), n in zip(branches, counts, strict=False):
            if sampled:
                if n == 0:
                    continue
                # A path of many measurements is improbable as a whole;
                # each followed outcome is kept at unit weight instead.
                branch = branch / engine.weight(branch)
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
