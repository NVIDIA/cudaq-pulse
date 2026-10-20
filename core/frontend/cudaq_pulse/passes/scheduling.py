# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Dependency-correct scheduling passes for the lightweight pulse IR.

A retained ``FOR_LOOP`` region is scheduled as its trip count of
back-to-back iterations: each iteration's linear resources (lines, tones)
continue from the previous iteration's results, exactly as if the loop had
been unrolled. Iterations are simulated until the per-iteration advance of
every resource repeats, then the rest are extrapolated, so scheduling time
does not grow with the trip count. Events are reported for the first
iteration; metrics cover all of them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .ir_types import (
    Op,
    OpKind,
    Program,
    ValueType,
    duration_of,
    tone_id_of,
    waveform_of,
)


@dataclass()
class ScheduledEvent:
    """A scheduled pulse event with absolute timing.

    Inside a retained loop the event is the op's first iteration;
    ``iterations`` is how many times the op runs.
    """

    op_index: int
    kind: str
    start_vtu: float
    duration_vtu: float
    line_id: int | None = None
    tone_id: int | None = None
    waveform_id: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    iterations: int = 1

    @property
    def end_vtu(self) -> float:
        return self.start_vtu + self.duration_vtu


@dataclass()
class ScheduleMetrics:
    """Summary statistics for a computed schedule."""

    total_length_vtu: float = 0.0
    total_length_ns: float = 0.0
    per_line_length_vtu: dict[int, float] = field(default_factory=dict)
    op_count: int = 0
    compile_time_ms: float = 0.0
    idle_total_vtu: float = 0.0
    idle_fraction: float = 0.0


@dataclass()
class MachineModel:
    """Hardware resource constraints for resource-constrained scheduling."""

    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    readout_latency_vtu: float = 0.0
    line_switch_penalty_vtu: float = 0.0
    qubit_connectivity: dict[int, list[int]] = field(default_factory=dict)


class ScheduleError(ValueError):
    """The program cannot be scheduled to concrete times."""


_TIMED_KINDS = frozenset({OpKind.DRIVE, OpKind.READOUT, OpKind.WAIT})
_LINE_TYPES = frozenset({ValueType.DRIVE_LINE, ValueType.READOUT_LINE})
_LINEAR_TYPES = _LINE_TYPES | {ValueType.TONE}
_REGION_STARTS = frozenset({OpKind.FOR_LOOP, OpKind.IF})
_REGION_ENDS = frozenset({OpKind.END_FOR, OpKind.END_IF})

# Iterations simulated before giving up on finding a steady per-iteration
# advance; past this the loop is simulated in full.
_MAX_TRANSIENT = 64


def _validate_machine(machine: MachineModel) -> None:
    if machine.max_concurrent_drives <= 0:
        raise ValueError("max_concurrent_drives must be positive")
    if machine.max_concurrent_readouts <= 0:
        raise ValueError("max_concurrent_readouts must be positive")
    if machine.readout_latency_vtu < 0:
        raise ValueError("readout_latency_vtu cannot be negative")
    if machine.line_switch_penalty_vtu < 0:
        raise ValueError("line_switch_penalty_vtu cannot be negative")


def _resource_roots(program: Program) -> dict[int, int]:
    """Map every line and tone SSA value to its allocation.

    The k-th linear result of a given type continues the k-th operand of
    that type; values with no such operand are roots themselves.
    """
    roots: dict[int, int] = {}
    for op in program.ops:
        by_type: dict[ValueType, list[int]] = {}
        for value in op.operands:
            if value.vtype in _LINEAR_TYPES:
                by_type.setdefault(value.vtype, []).append(value.vid)
        seen: dict[ValueType, int] = {}
        for result in op.results:
            if result.vtype not in _LINEAR_TYPES:
                continue
            k = seen.get(result.vtype, 0)
            seen[result.vtype] = k + 1
            sources = by_type.get(result.vtype, [])
            if k < len(sources):
                roots[result.vid] = roots.get(sources[k], sources[k])
            else:
                roots[result.vid] = result.vid
    return roots


def _loop_ends(ops: list[Op]) -> dict[int, int]:
    ends: dict[int, int] = {}
    stack: list[int] = []
    for index, op in enumerate(ops):
        if op.kind == OpKind.FOR_LOOP:
            stack.append(index)
        elif op.kind == OpKind.END_FOR:
            if not stack:
                raise ScheduleError(f"END_FOR at op[{index}] without matching FOR_LOOP")
            ends[stack.pop()] = index
    if stack:
        raise ScheduleError(f"FOR_LOOP at op[{stack[-1]}] without matching END_FOR")
    return ends


def _if_bounds(ops: list[Op]) -> dict[int, tuple[int, int]]:
    """Map each IF and END_IF index to the indices of its ELSE and other end."""
    bounds: dict[int, tuple[int, int]] = {}
    stack: list[list[int]] = []
    for index, op in enumerate(ops):
        if op.kind == OpKind.IF:
            stack.append([index, -1])
        elif op.kind == OpKind.ELSE and stack:
            stack[-1][1] = index
        elif op.kind == OpKind.END_IF:
            if not stack or stack[-1][1] < 0:
                raise ScheduleError(f"END_IF at op[{index}] without a matching IF and ELSE")
            start, middle = stack.pop()
            bounds[start] = (middle, index)
            bounds[index] = (start, middle)
    if stack:
        raise ScheduleError(f"IF at op[{stack[-1][0]}] without matching END_IF")
    return bounds


def _trip_count(op: Op, index: int) -> int:
    attrs = op.attrs
    if "count" in attrs and "ub" not in attrs:
        bounds: tuple[Any, ...] = (attrs["count"],)
    else:
        bounds = (attrs.get("lb", 0), attrs.get("ub", 1), attrs.get("step", 1))
    if not all(isinstance(b, int) and not isinstance(b, bool) for b in bounds):
        raise ScheduleError(
            f"FOR_LOOP at op[{index}] has a non-constant trip count {bounds!r}; "
            "bind its parameters to integers before scheduling"
        )
    try:
        return len(range(*bounds))
    except ValueError as exc:
        raise ScheduleError(f"FOR_LOOP at op[{index}]: {exc}") from exc


class _Accounting:
    """Per-line busy time and extent over every executed op instance."""

    def __init__(self) -> None:
        self.active: dict[int, float] = {}
        self.first_start: dict[int, float] = {}
        self.last_end: dict[int, float] = {}
        self.makespan = 0.0

    def add(self, line: int | None, start: float, duration: float) -> None:
        end = start + duration
        if end > self.makespan:
            self.makespan = end
        if line is None or duration <= 0:
            return
        self.active[line] = self.active.get(line, 0.0) + duration
        if start < self.first_start.get(line, start + 1.0):
            self.first_start[line] = start
        if end > self.last_end.get(line, end - 1.0):
            self.last_end[line] = end


class _Scheduler:
    def __init__(self, program: Program, machine: MachineModel | None) -> None:
        self.program = program
        self.ops = program.ops
        self.machine = machine
        self.roots = _resource_roots(program)
        self.ends = _loop_ends(self.ops)
        self.loop_starts = {end: start for start, end in self.ends.items()}
        self.ifs = _if_bounds(self.ops)
        # Earliest start for ops inside a measurement-conditioned arm.
        self.floor = 0.0
        # Earliest begin of the ops in each enclosing arm (ALAP).
        self.arm_begin: list[float] = []
        self.events: list[ScheduledEvent | None] = [None] * len(self.ops)
        self.acct = _Accounting()
        n_drive = machine.max_concurrent_drives if machine is not None else 0
        n_readout = machine.max_concurrent_readouts if machine is not None else 0
        self.drive_lanes = [0.0] * n_drive
        self.readout_lanes = [0.0] * n_readout
        roots = self.roots
        self.line_of: list[int | None] = []
        for op in self.ops:
            line = next((v for v in op.operands if v.vtype in _LINE_TYPES), None)
            self.line_of.append(roots.get(line.vid, line.vid) if line is not None else None)

    def _record(self, index: int, start: float, duration: float, iterations: int) -> None:
        op = self.ops[index]
        self.events[index] = ScheduledEvent(
            op_index=index,
            kind=op.kind,
            start_vtu=start,
            duration_vtu=duration,
            line_id=self.line_of[index],
            tone_id=tone_id_of(op),
            waveform_id=waveform_of(op),
            attrs=dict(op.attrs),
            iterations=iterations,
        )

    def _body_inputs(self, start: int, end: int) -> list[int]:
        """Linear values a loop body reads that are defined before it."""
        defined = {r.vid for op in self.ops[start + 1 : end] for r in op.results}
        inputs: dict[int, None] = {}
        for op in self.ops[start + 1 : end]:
            for v in op.operands:
                if v.vtype in _LINEAR_TYPES and v.vid not in defined:
                    inputs[v.vid] = None
        return list(inputs)

    # ── Forward (ASAP / resource-constrained) ────────────────────────

    def forward(self) -> None:
        self.ready: dict[int, float] = {}
        self._forward_range(0, len(self.ops), True, 1)

    def _forward_range(self, i0: int, i1: int, record: bool, iterations: int) -> None:
        ops = self.ops
        ready = self.ready
        machine = self.machine
        floor = self.floor
        i = i0
        while i < i1:
            op = ops[i]
            if op.kind in _REGION_STARTS:
                if op.kind == OpKind.FOR_LOOP:
                    end = self.ends[i]
                    self._forward_loop(i, end, record, iterations)
                    i = end + 1
                else:
                    i = self._forward_if(i, record, iterations) + 1
                continue
            operand_ready = max((ready.get(v.vid, 0.0) for v in op.operands), default=floor)
            if operand_ready < floor:
                operand_ready = floor
            duration = duration_of(op) if op.kind in _TIMED_KINDS else 0.0
            start = operand_ready
            result_ready = start + duration
            if op.kind == OpKind.SYNC:
                result_ready = operand_ready
            elif machine is not None and op.kind in (OpKind.DRIVE, OpKind.READOUT):
                lanes = self.drive_lanes if op.kind == OpKind.DRIVE else self.readout_lanes
                lane = min(range(len(lanes)), key=lanes.__getitem__)
                start = max(start, lanes[lane])
                result_ready = start + duration
                lanes[lane] = result_ready + machine.line_switch_penalty_vtu
                if op.kind == OpKind.READOUT:
                    result_ready += machine.readout_latency_vtu
            if record:
                self._record(i, start, duration, iterations)
            self.acct.add(self.line_of[i], start, duration)
            for result in op.results:
                ready[result.vid] = result_ready
            i += 1

    def _forward_if(self, start: int, record: bool, iterations: int) -> int:
        """Schedule both arms from the same state; return the END_IF index."""
        middle, end = self.ifs[start]
        ops = self.ops
        cond_ready = max(
            self.floor, max((self.ready.get(v.vid, 0.0) for v in ops[start].operands), default=0.0)
        )
        if record:
            self._record(start, cond_ready, 0.0, iterations)
        entry = dict(self.ready)
        entry_lanes = (list(self.drive_lanes), list(self.readout_lanes))
        outer_floor = self.floor
        self.floor = cond_ready

        self._forward_range(start + 1, middle, record, iterations)
        then_ready = dict(self.ready)
        then_lanes = (self.drive_lanes, self.readout_lanes)
        self.ready.clear()
        self.ready.update(entry)
        self.drive_lanes, self.readout_lanes = (list(lanes) for lanes in entry_lanes)
        self._forward_range(middle + 1, end, record, iterations)
        self.floor = outer_floor

        ready = self.ready
        for vid, value in then_ready.items():
            if value > ready.get(vid, value - 1.0):
                ready[vid] = value
        for mine, theirs in zip((self.drive_lanes, self.readout_lanes), then_lanes):
            mine.sort()
            for j, value in enumerate(sorted(theirs)):
                mine[j] = max(mine[j], value)
        joined = []
        for result, a, b in zip(ops[end].results, ops[middle].operands, ops[end].operands):
            ready[result.vid] = max(ready.get(a.vid, 0.0), ready.get(b.vid, 0.0))
            joined.append(ready[result.vid])
        if record:
            self._record(middle, cond_ready, 0.0, iterations)
            self._record(end, max(joined, default=cond_ready), 0.0, iterations)
        return end

    def _forward_loop(self, start: int, end: int, record: bool, iterations: int) -> None:
        n = _trip_count(self.ops[start], start)
        ready = self.ready
        roots = self.roots
        inputs = self._body_inputs(start, end)
        yields = [v.vid for v in self.ops[end].operands]
        produced: dict[int, int] = {}  # root -> vid of its last value in the body
        for op in self.ops[start + 1 : end]:
            for r in op.results:
                if r.vtype in _LINEAR_TYPES:
                    produced[roots.get(r.vid, r.vid)] = r.vid
        carried = [v for v in inputs if roots.get(v, v) in produced]
        state_roots = sorted(produced)
        if record:
            loop_start = max((ready.get(v, 0.0) for v in carried), default=0.0)
            self._record(start, loop_start, 0.0, iterations)

        def state() -> tuple[float, ...]:
            acct = self.acct
            return (
                *(ready[produced[r]] for r in state_roots),
                *self.drive_lanes,
                *self.readout_lanes,
                *(acct.last_end.get(line, 0.0) for line in lines),
            )

        lines = sorted({ln for ln in self.line_of[start + 1 : end] if ln is not None})
        history: list[tuple[float, ...]] = []
        done = 0
        while done < n:
            if done:
                for v in carried:
                    ready[v] = ready[produced[roots.get(v, v)]]
            busy_before = dict(self.acct.active)
            self._forward_range(start + 1, end, record and done == 0, iterations * n)
            done += 1
            history.append(state())
            per_iter = {k: self.acct.active.get(k, 0.0) - busy_before.get(k, 0.0) for k in lines}
            if done >= 3 and done < n and len(history) <= _MAX_TRANSIENT:
                d1 = _delta(history[-1], history[-2])
                if d1 == _delta(history[-2], history[-3]):
                    self._skip_forward(n - done, d1, state_roots, produced, lines, per_iter)
                    done = n
        if n == 0:
            init = {roots.get(v, v): v for v in carried}
            for result in self.ops[end].results:
                source = init.get(roots.get(result.vid, result.vid))
                ready[result.vid] = ready.get(source, 0.0) if source is not None else 0.0
        else:
            for result, source in zip(self.ops[end].results, yields):
                ready[result.vid] = ready.get(source, 0.0)
        if record:
            self._record(end, max((ready.get(v, 0.0) for v in yields), default=0.0), 0.0, 1)

    def _skip_forward(
        self,
        remaining: int,
        delta: tuple[float, ...],
        state_roots: list[int],
        produced: dict[int, int],
        lines: list[int],
        per_iter: dict[int, float],
    ) -> None:
        ready = self.ready
        k = 0
        for r in state_roots:
            ready[produced[r]] += remaining * delta[k]
            k += 1
        for lanes in (self.drive_lanes, self.readout_lanes):
            for j in range(len(lanes)):
                lanes[j] += remaining * delta[k]
                k += 1
        acct = self.acct
        for line in lines:
            end = acct.last_end.get(line, 0.0) + remaining * delta[k]
            k += 1
            acct.last_end[line] = end
            acct.active[line] = acct.active.get(line, 0.0) + remaining * per_iter[line]
            if end > acct.makespan:
                acct.makespan = end
        for r in state_roots:
            if ready[produced[r]] > acct.makespan:
                acct.makespan = ready[produced[r]]

    # ── Backward (ALAP) ──────────────────────────────────────────────

    def backward(self, makespan: float) -> None:
        self.makespan = makespan
        self.latest: dict[int, float] = {}
        self.acct = _Accounting()
        self._backward_range(0, len(self.ops), True, 1)

    def _backward_range(self, i0: int, i1: int, record: bool, iterations: int) -> None:
        ops = self.ops
        latest = self.latest
        makespan = self.makespan
        arm_begin = self.arm_begin
        i = i1 - 1
        while i >= i0:
            op = ops[i]
            if op.kind in _REGION_ENDS:
                if op.kind == OpKind.END_IF:
                    i = self._backward_if(i, record, iterations) - 1
                else:
                    start = self.loop_starts[i]
                    self._backward_loop(start, i, record, iterations)
                    i = start - 1
                continue
            duration = duration_of(op) if op.kind in _TIMED_KINDS else 0.0
            end = min((latest.get(v.vid, makespan) for v in op.results), default=makespan)
            begin = end - duration
            if arm_begin and begin < arm_begin[-1]:
                arm_begin[-1] = begin
            if record:
                self._record(i, begin, duration, iterations)
            self.acct.add(self.line_of[i], begin, duration)
            for operand in op.operands:
                latest[operand.vid] = min(latest.get(operand.vid, makespan), begin)
            i -= 1

    def _backward_if(self, end: int, record: bool, iterations: int) -> int:
        """Schedule both arms back from the END_IF; return the IF index."""
        start, middle = self.ifs[end]
        ops = self.ops
        latest = self.latest
        makespan = self.makespan
        joined = []
        for result, a, b in zip(ops[end].results, ops[middle].operands, ops[end].operands):
            limit = latest.get(result.vid, makespan)
            latest[a.vid] = min(latest.get(a.vid, makespan), limit)
            latest[b.vid] = min(latest.get(b.vid, makespan), limit)
            joined.append(limit)
        entry = dict(latest)
        self.arm_begin.append(makespan)
        self._backward_range(middle + 1, end, record, iterations)
        else_latest = dict(latest)
        latest.clear()
        latest.update(entry)
        self._backward_range(start + 1, middle, record, iterations)
        earliest = self.arm_begin.pop()
        for vid, value in else_latest.items():
            if value < latest.get(vid, value + 1.0):
                latest[vid] = value
        if self.arm_begin and earliest < self.arm_begin[-1]:
            self.arm_begin[-1] = earliest
        for v in ops[start].operands:
            latest[v.vid] = min(latest.get(v.vid, makespan), earliest)
        if record:
            self._record(end, min(joined, default=earliest), 0.0, iterations)
            self._record(middle, earliest, 0.0, iterations)
            self._record(start, earliest, 0.0, iterations)
        return start

    def _backward_loop(self, start: int, end: int, record: bool, iterations: int) -> None:
        n = _trip_count(self.ops[start], start)
        latest = self.latest
        roots = self.roots
        makespan = self.makespan
        end_op = self.ops[end]
        yields = [v.vid for v in end_op.operands]
        for result, source in zip(end_op.results, yields):
            latest[source] = min(latest.get(source, makespan), latest.get(result.vid, makespan))
        if record:
            self._record(end, min((latest.get(v, makespan) for v in yields), default=0.0), 0.0, 1)
        inputs = self._body_inputs(start, end)
        body_defined = {r.vid for op in self.ops[start + 1 : end] for r in op.results}
        produced: dict[int, int] = {}
        for op in self.ops[start + 1 : end]:
            for r in op.results:
                if r.vtype in _LINEAR_TYPES:
                    produced[roots.get(r.vid, r.vid)] = r.vid
        carried = [v for v in inputs if roots.get(v, v) in produced]
        input_roots = sorted({roots.get(v, v) for v in carried})
        lines = sorted({ln for ln in self.line_of[start + 1 : end] if ln is not None})
        boundary = {v: latest.get(v, makespan) for v in inputs}

        def state() -> tuple[float, ...]:
            per_root: dict[int, float] = {}
            for v in carried:
                r = roots.get(v, v)
                per_root[r] = min(per_root.get(r, makespan), latest.get(v, makespan))
            acct = self.acct
            return (
                *(per_root[r] for r in input_roots),
                *(acct.first_start.get(line, makespan) for line in lines),
            )

        history: list[tuple[float, ...]] = []
        done = 0
        while done < n:
            last = done == n - 1
            if done:
                carried_latest = history[-1]
                for vid in body_defined:
                    latest.pop(vid, None)
                for root, value in zip(input_roots, carried_latest):
                    latest[produced[root]] = min(latest.get(produced[root], makespan), value)
            for v in inputs:
                latest[v] = boundary[v] if last else makespan
            busy_before = dict(self.acct.active)
            self._backward_range(start + 1, end, record and last, iterations * n)
            done += 1
            history.append(state())
            per_iter = {k: self.acct.active.get(k, 0.0) - busy_before.get(k, 0.0) for k in lines}
            if 3 <= done < n - 1 and len(history) <= _MAX_TRANSIENT:
                d1 = _delta(history[-1], history[-2])
                if d1 == _delta(history[-2], history[-3]):
                    skip = n - 1 - done
                    jumped = tuple(h + skip * d for h, d in zip(history[-1], d1))
                    history.append(jumped)
                    acct = self.acct
                    for k, line in enumerate(lines):
                        idx = len(input_roots) + k
                        acct.first_start[line] = jumped[idx]
                        acct.active[line] = acct.active.get(line, 0.0) + skip * per_iter[line]
                    done = n - 1
        for v in inputs:
            latest[v] = min(boundary[v], latest.get(v, makespan))
        if record:
            loop_start = min((latest.get(v, makespan) for v in carried), default=makespan)
            self._record(start, loop_start, 0.0, iterations)

    # ── Results ──────────────────────────────────────────────────────

    def result_events(self) -> list[ScheduledEvent]:
        return [e for e in self.events if e is not None]

    def annotate(self) -> None:
        for event in self.result_events():
            if event.kind in _TIMED_KINDS:
                op = self.ops[event.op_index]
                op.attrs["start_vtu"] = event.start_vtu
                op.attrs["duration_vtu"] = event.duration_vtu

    def metrics(self, makespan: float) -> ScheduleMetrics:
        acct = self.acct
        per_line = dict(acct.last_end)
        available = sum(per_line[line] - acct.first_start[line] for line in per_line)
        idle_total = max(0.0, available - sum(acct.active.values()))
        return ScheduleMetrics(
            total_length_vtu=makespan,
            total_length_ns=makespan * self.program.vtu_to_ns,
            per_line_length_vtu=per_line,
            op_count=self.program.op_count(),
            idle_total_vtu=idle_total,
            idle_fraction=idle_total / available if available else 0.0,
        )


def _delta(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _run(
    program: Program, machine: MachineModel | None, alap: bool
) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    started = time.perf_counter()
    scheduler = _Scheduler(program, machine)
    scheduler.forward()
    makespan = scheduler.acct.makespan
    if alap:
        scheduler.backward(makespan)
    scheduler.annotate()
    metrics = scheduler.metrics(makespan)
    metrics.compile_time_ms = (time.perf_counter() - started) * 1000.0
    return scheduler.result_events(), metrics


def schedule_asap(program: Program) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """Schedule operations at their earliest dependency-ready times."""
    return _run(program, None, alap=False)


def schedule_alap(program: Program) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """Schedule operations as late as dependencies allow at the ASAP makespan."""
    return _run(program, None, alap=True)


def schedule_rcp(
    program: Program,
    machine: MachineModel | None = None,
) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """List-schedule with interval-correct drive and readout resource limits."""
    machine = machine or MachineModel()
    _validate_machine(machine)
    return _run(program, machine, alap=False)
