# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Verification passes for the pulse IR.

Checks linearity, monotone time, drive exclusivity, and cross-resonance
calibration heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ir_types import (
    OpKind,
    Program,
    Value,
    ValueType,
    duration_of,
    is_linear_type,
    line_id_of,
)

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


@dataclass()
class TypeCheckError:
    """Base class for all verification errors."""

    message: str
    op_index: int | None = None
    severity: str = "error"

    def __str__(self) -> str:
        loc = f" at op[{self.op_index}]" if self.op_index is not None else ""
        return f"[{self.severity}]{loc}: {self.message}"


@dataclass()
class LinearityViolation(TypeCheckError):
    """A linear value was produced more than once or consumed != 1 time.

    Unconsumed terminal values are warnings. Duplicate production and
    multiple consumption are errors because line and tone handles are linear.
    """

    value: Value | None = None
    detail: str = ""
    severity: str = "warning"


@dataclass()
class UnintentionalOverlapError(TypeCheckError):
    """Two drive ops on the same line overlap in time."""

    line_vid: int | None = None
    interval_a: tuple[float, float] = (0.0, 0.0)
    interval_b: tuple[float, float] = (0.0, 0.0)


@dataclass()
class BackwardTimeTravelError(TypeCheckError):
    """An operation references a time earlier than a predecessor on the same line."""

    line_vid: int | None = None
    expected_min: float = 0.0
    actual: float = 0.0


@dataclass()
class PhaseBookkeepingError(TypeCheckError):
    """Inconsistency detected in phase tracking for a tone."""

    tone_vid: int | None = None
    detail: str = ""


@dataclass()
class CrossResonanceMiscalibrationError(TypeCheckError):
    """Heuristic: possible cross-resonance tone mismatch."""

    target_qubit: int | None = None
    tone_tag: str = ""
    severity: str = "warning"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_linearity(program: Program) -> list[TypeCheckError]:
    """Every linear value must be produced exactly once and consumed exactly once."""
    errors: list[TypeCheckError] = []
    produced: dict[int, int] = {}  # vid -> op_index that produced it
    consumed: dict[int, int] = {}  # vid -> consume count
    # Inside an if, consumption is counted per arm; a value consumed once in
    # each arm is consumed once on every path.
    arms: list[tuple[dict[int, int], dict[int, int], bool]] = []

    def consume(vid: int) -> None:
        if arms:
            then_counts, else_counts, in_else = arms[-1]
            counts = else_counts if in_else else then_counts
        else:
            counts = consumed
        counts[vid] = counts.get(vid, 0) + 1

    for idx, op in enumerate(program.ops):
        if op.kind == OpKind.ELSE and arms:
            for v in op.operands:
                if is_linear_type(v.vtype):
                    consume(v.vid)
            then_counts, else_counts, _ = arms[-1]
            arms[-1] = (then_counts, else_counts, True)
            continue
        merged_arms = op.kind == OpKind.END_IF and bool(arms)
        if merged_arms:
            for v in op.operands:
                if is_linear_type(v.vtype):
                    consume(v.vid)
            then_counts, else_counts, _ = arms.pop()
            if arms:
                outer_then, outer_else, outer_in_else = arms[-1]
                target = outer_else if outer_in_else else outer_then
            else:
                target = consumed
            for vid in then_counts.keys() | else_counts.keys():
                target[vid] = target.get(vid, 0) + max(
                    then_counts.get(vid, 0), else_counts.get(vid, 0)
                )

        for v in op.results:
            if not is_linear_type(v.vtype):
                continue
            if v.vid in produced:
                errors.append(
                    LinearityViolation(
                        message=f"Linear value {v} produced again (first at op[{produced[v.vid]}])",
                        op_index=idx,
                        value=v,
                        detail="duplicate_production",
                        severity="error",
                    )
                )
            else:
                produced[v.vid] = idx

        for v in () if merged_arms else op.operands:
            if not is_linear_type(v.vtype):
                continue
            consume(v.vid)
        if op.kind == OpKind.IF:
            arms.append(({}, {}, False))

    for vid, prod_idx in produced.items():
        count = consumed.get(vid, 0)
        if count == 0:
            v_repr = f"%{vid}"
            errors.append(
                LinearityViolation(
                    message=f"Linear value {v_repr} produced at op[{prod_idx}] but never consumed",
                    op_index=prod_idx,
                    detail="unconsumed",
                )
            )
        elif count > 1:
            v_repr = f"%{vid}"
            errors.append(
                LinearityViolation(
                    message=f"Linear value {v_repr} consumed {count} times (expected 1)",
                    op_index=prod_idx,
                    detail="multiple_consumption",
                    severity="error",
                )
            )

    return errors


def _line_roots(program: Program) -> dict[int, int]:
    """Map each line SSA value to its physical line allocation."""
    roots: dict[int, int] = {}
    line_types = (ValueType.DRIVE_LINE, ValueType.READOUT_LINE)
    for op in program.ops:
        operands = [value for value in op.operands if value.vtype in line_types]
        results = [value for value in op.results if value.vtype in line_types]
        if op.kind in (OpKind.ALLOC_DRIVE, OpKind.ALLOC_READOUT):
            for result in results:
                roots[result.vid] = result.vid
        else:
            # The k-th line result continues the k-th line operand.
            for operand, result in zip(operands, results):
                roots[result.vid] = roots.get(operand.vid, operand.vid)
    return roots


def _arm_paths(program: Program) -> list[tuple[tuple[int, bool], ...]]:
    """For each op, the (IF index, in else-arm) pairs of the arms enclosing it."""
    paths: list[tuple[tuple[int, bool], ...]] = []
    stack: list[tuple[int, bool]] = []
    for idx, op in enumerate(program.ops):
        if op.kind == OpKind.ELSE and stack:
            stack[-1] = (stack[-1][0], True)
        if op.kind == OpKind.END_IF and stack:
            stack.pop()
        paths.append(tuple(stack))
        if op.kind == OpKind.IF:
            stack.append((idx, False))
    return paths


def _exclusive(a: tuple[tuple[int, bool], ...], b: tuple[tuple[int, bool], ...]) -> bool:
    """Whether two ops lie in different arms of the same if, so never both run."""
    arms = dict(a)
    return any(arms.get(index, in_else) != in_else for index, in_else in b)


def check_monotone_time(program: Program) -> list[TypeCheckError]:
    """Verify that time only moves forward on each line."""
    errors: list[TypeCheckError] = []
    roots = _line_roots(program)
    line_clocks: dict[int, float] = {}
    # Each arm of an if starts from the clocks at the IF; they rejoin at the
    # later of the two arms.
    saved: list[tuple[dict[int, float], dict[int, float] | None]] = []

    for idx, op in enumerate(program.ops):
        if op.kind == OpKind.IF:
            saved.append((dict(line_clocks), None))
        elif op.kind == OpKind.ELSE and saved:
            entry, _ = saved[-1]
            saved[-1] = (entry, line_clocks)
            line_clocks = dict(entry)
        elif op.kind == OpKind.END_IF and saved:
            _, then_clocks = saved.pop()
            for key, value in (then_clocks or {}).items():
                if value > line_clocks.get(key, 0.0):
                    line_clocks[key] = value
        lid = line_id_of(op)
        if lid is None:
            continue

        root = roots.get(lid, lid)
        start = float(op.attrs.get("start_vtu", line_clocks.get(root, 0.0)))
        current = line_clocks.get(root, 0.0)

        if start < current - 1e-12:
            errors.append(
                BackwardTimeTravelError(
                    message=f"Time goes backward on line %{root}: "
                    f"expected >= {current:.4f}, got {start:.4f}",
                    op_index=idx,
                    line_vid=root,
                    expected_min=current,
                    actual=start,
                )
            )

        dur = duration_of(op)
        end = start + dur
        if end > current:
            line_clocks[root] = end

        for r in op.results:
            if r.vtype in (ValueType.DRIVE_LINE, ValueType.READOUT_LINE):
                line_clocks[r.vid] = line_clocks.get(root, 0.0)

    return errors


def check_drive_exclusivity(program: Program) -> list[TypeCheckError]:
    """Drive ops on the same line lineage must be totally ordered, non-overlapping.

    Only meaningful after scheduling — if no ops carry ``start_vtu`` attrs
    the check is skipped (unscheduled programs trivially alias at t=0).
    """
    errors: list[TypeCheckError] = []

    has_schedule = any("start_vtu" in op.attrs for op in program.ops if op.kind == OpKind.DRIVE)
    if not has_schedule:
        return errors

    roots = _line_roots(program)
    paths = _arm_paths(program)
    line_intervals: dict[int, list[tuple[float, float, int]]] = {}

    for idx, op in enumerate(program.ops):
        if op.kind != OpKind.DRIVE:
            continue
        lid = line_id_of(op)
        if lid is None:
            continue

        start = float(op.attrs.get("start_vtu", 0.0))
        dur = duration_of(op)
        end = start + dur

        root = roots.get(lid, lid)
        if root not in line_intervals:
            line_intervals[root] = []

        for prev_start, prev_end, prev_idx in line_intervals[root]:
            if _exclusive(paths[prev_idx], paths[idx]):
                continue
            if start < prev_end - 1e-12 and end > prev_start + 1e-12:
                errors.append(
                    UnintentionalOverlapError(
                        message=(
                            f"Drive ops overlap on line %{root}: "
                            f"op[{prev_idx}] [{prev_start:.2f},{prev_end:.2f}) vs "
                            f"op[{idx}] [{start:.2f},{end:.2f})"
                        ),
                        op_index=idx,
                        line_vid=root,
                        interval_a=(prev_start, prev_end),
                        interval_b=(start, end),
                    )
                )

        line_intervals[root].append((start, end, idx))

    return errors


def check_cr_miscalibration(program: Program) -> list[TypeCheckError]:
    """Heuristic check for cross-resonance tone tag mismatches.

    If a drive op is tagged as cross-resonance (attrs['cr_target'] is set),
    verify the tone tag matches the target qubit's frequency label.
    """
    errors: list[TypeCheckError] = []

    for idx, op in enumerate(program.ops):
        if op.kind != OpKind.DRIVE:
            continue
        cr_target = op.attrs.get("cr_target")
        if cr_target is None:
            continue

        tone_tag = op.attrs.get("tone_tag", "")
        expected_tag = f"q{cr_target}"

        if tone_tag and expected_tag not in tone_tag:
            errors.append(
                CrossResonanceMiscalibrationError(
                    message=(
                        f"Cross-resonance drive at op[{idx}] targets qubit {cr_target} "
                        f"but tone tag is '{tone_tag}' (expected to contain '{expected_tag}')"
                    ),
                    op_index=idx,
                    target_qubit=cr_target,
                    tone_tag=tone_tag,
                )
            )

    return errors


def check_if_structure(program: Program) -> list[TypeCheckError]:
    """Verify IF/ELSE/END_IF regions are well formed.

    The condition is a single measurement result, each IF has exactly one
    ELSE, and both arms yield values matching the END_IF results.
    """
    errors: list[TypeCheckError] = []
    stack: list[tuple[int, int | None]] = []

    def error(idx: int, message: str) -> None:
        errors.append(TypeCheckError(message=message, op_index=idx))

    for idx, op in enumerate(program.ops):
        if op.kind == OpKind.IF:
            if len(op.operands) != 1 or op.operands[0].vtype != ValueType.MEASUREMENT:
                error(idx, f"IF at op[{idx}] must take exactly one measurement result")
            stack.append((idx, None))
        elif op.kind == OpKind.ELSE:
            if not stack or stack[-1][1] is not None:
                error(idx, f"ELSE at op[{idx}] without an open IF")
                continue
            stack[-1] = (stack[-1][0], idx)
        elif op.kind == OpKind.END_IF:
            if not stack:
                error(idx, f"END_IF at op[{idx}] without matching IF")
                continue
            start, else_idx = stack.pop()
            if else_idx is None:
                error(idx, f"IF at op[{start}] has no ELSE")
                continue
            expected = [r.vtype for r in op.results]
            for arm, values in (
                ("then", program.ops[else_idx].operands),
                ("else", op.operands),
            ):
                if [v.vtype for v in values] != expected:
                    error(
                        idx,
                        f"IF at op[{start}]: {arm} arm yields "
                        f"{[str(v.vtype) for v in values]}, expected {[str(t) for t in expected]}",
                    )
    for start, _ in stack:
        error(start, f"IF at op[{start}] without matching END_IF")
    return errors


def check_loop_structure(program: Program) -> list[TypeCheckError]:
    """Verify FOR_LOOP/END_FOR are balanced and carry required attrs."""
    errors: list[TypeCheckError] = []
    stack: list[int] = []

    for idx, op in enumerate(program.ops):
        if op.kind == OpKind.FOR_LOOP:
            stack.append(idx)
            if "ub" not in op.attrs and "count" not in op.attrs:
                errors.append(
                    TypeCheckError(
                        message=f"FOR_LOOP at op[{idx}] missing 'ub' or 'count' attr",
                        op_index=idx,
                    )
                )
        elif op.kind == OpKind.END_FOR:
            if not stack:
                errors.append(
                    TypeCheckError(
                        message=f"END_FOR at op[{idx}] without matching FOR_LOOP",
                        op_index=idx,
                    )
                )
            else:
                stack.pop()

    for start_idx in stack:
        errors.append(
            TypeCheckError(
                message=f"FOR_LOOP at op[{start_idx}] without matching END_FOR",
                op_index=start_idx,
            )
        )

    return errors


def check_waveform_validity(program: Program) -> list[TypeCheckError]:
    """Check waveform construction attrs for basic validity."""
    errors: list[TypeCheckError] = []

    for idx, op in enumerate(program.ops):
        if op.kind != OpKind.MAKE_WAVEFORM:
            continue

        dur = op.attrs.get("duration_vtu")
        if dur is not None and float(dur) <= 0:
            errors.append(
                TypeCheckError(
                    message=f"Waveform at op[{idx}] has non-positive duration: {dur}",
                    op_index=idx,
                )
            )

        amp = op.attrs.get("amplitude")
        if amp is not None:
            amp_vals = amp if isinstance(amp, (list, tuple)) else [amp]
            for av in amp_vals:
                try:
                    a = abs(complex(av)) if isinstance(av, complex) else float(av)
                    if not (-1e6 < a < 1e6):
                        errors.append(
                            TypeCheckError(
                                message=f"Waveform at op[{idx}] has extreme amplitude: {a}",
                                op_index=idx,
                                severity="warning",
                            )
                        )
                except (TypeError, ValueError):
                    errors.append(
                        TypeCheckError(
                            message=(
                                f"Waveform at op[{idx}] has non-numeric amplitude component: {av!r}"
                            ),
                            op_index=idx,
                        )
                    )

        sigma = op.attrs.get("sigma")
        if sigma is not None and float(sigma) <= 0:
            errors.append(
                TypeCheckError(
                    message=f"Waveform at op[{idx}] has non-positive sigma: {sigma}",
                    op_index=idx,
                )
            )

    return errors


# ---------------------------------------------------------------------------
# Aggregate verifier
# ---------------------------------------------------------------------------


def verify(program: Program, *, strict: bool = False) -> list[TypeCheckError]:
    """Run all verification checks and return collected errors/warnings.

    Parameters
    ----------
    strict : bool
        If True, raise ``RuntimeError`` on the first error-severity issue.
    """
    errors: list[TypeCheckError] = []
    errors.extend(check_linearity(program))
    errors.extend(check_loop_structure(program))
    errors.extend(check_if_structure(program))
    errors.extend(check_waveform_validity(program))
    errors.extend(check_monotone_time(program))
    errors.extend(check_drive_exclusivity(program))
    errors.extend(check_cr_miscalibration(program))

    if strict:
        for e in errors:
            if e.severity == "error":
                raise RuntimeError(str(e))

    return errors
