# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-conditioned branches: the ``scf.if`` an ``if`` on a value
known only when the program runs becomes, with both arms traced from the
same state. The tracer imports this module when a kernel first branches
on such a value; a compile without one never reads it.
"""

from __future__ import annotations

import numbers
import operator
from typing import TYPE_CHECKING, Any

from . import _bytecode_normalize as bc
from ._emitter import (
    _CELL_ORDER,
    _CONTAINERS,
    _FIRST,
    _MISSING,
    _MUTABLE,
    _REGION_DONE,
    _SCALARS,
    _Choice,
    _container_unchanged,
    _describe,
    _Held,
    _NotMeasurement,
    _NotRollableError,
    _restore_channels,
    _slot_changes,
)
from .ir_builder import (
    LINEAR_TYPES,
    AnswerValue,
    CompilationError,
    IRValue,
    Parameter,
    ParameterExpression,
)

if TYPE_CHECKING:
    from ._emitter import PulseIREmitter, StackVal, _Chan

_PACKED_BRANCH_MESSAGE = "measurement-conditioned branches need an IR builder that supports regions"
_BRANCH_TYPE_MESSAGE = (
    "only measurement results can control a branch at runtime; got an IR value of type {}"
)
_IS_NOT = operator.is_not


def _choice_kind(value: Any) -> str | None:
    """What a measurement-conditioned if can choose *value* as: "number" or
    "waveform"; None if it cannot."""
    cls = value.__class__
    if cls is _Choice:
        return "number" if value.vtype is None else "waveform"
    if cls is IRValue:
        return "waveform" if value.vtype == "waveform" else None
    if cls is AnswerValue or cls is Parameter or cls is ParameterExpression:
        return "number"
    return "number" if isinstance(value, numbers.Real) else None


def _made_in_if(value: Any, first: int, scope: tuple[int, ...]) -> bool:
    """Whether the number or waveform *value* holds a value made inside a
    measurement-conditioned if, which does not exist after it. *first* is
    the first id made inside the if, *scope* the regions open around it."""
    cls = value.__class__
    if cls is _Choice:
        return (
            value.cond.vid >= first
            or _made_in_if(value.then, first, scope)
            or _made_in_if(value.other, first, scope)
        )
    if cls is IRValue:
        return bool(value.vid >= first)
    if cls is AnswerValue:
        return bool(scope[: len(value.scope)] != value.scope)
    if cls is ParameterExpression:
        return _made_in_if(value.lhs, first, scope) or _made_in_if(value.rhs, first, scope)
    return False


def runtime_branch(
    emitter: PulseIREmitter, prog: bc.DecodedCode, index: int, cond: Any, jump_if_true: bool
) -> int:
    """Emit the ``if`` whose conditional jump is ``prog.instrs[index]``.

    Both arms are traced from the same state into an ``scf.if`` region.
    Pulse resources rebound by either arm become results of the region;
    any other value must end up the same in both arms. Returns the
    instruction index where tracing continues.
    """
    negated = isinstance(cond, _NotMeasurement)
    value = cond.value if negated else cond
    if value.vtype not in ("measurement", "i1"):
        raise CompilationError(_BRANCH_TYPE_MESSAGE.format(value.vtype), code="runtime_value")
    if not emitter._can_roll:
        raise CompilationError(_PACKED_BRANCH_MESSAGE)

    target = prog.instrs[index][1]
    then_start, else_start = (target, index + 1) if jump_if_true else (index + 1, target)
    if negated:
        then_start, else_start = else_start, then_start
    join = bc.join_point(prog, index)
    view = bc.region_view(prog, join)
    returns = join >= len(prog.instrs)

    builder = emitter.b
    stack = emitter.stack
    saved_stack = list(stack)
    saved_locals = dict(emitter.locals)
    # A container is copied when an arm first changes it
    # (_note_mutation), with its length: a growing list of outcomes is
    # not copied at every branch. A list holding lines or tones, whose
    # slots the trace rebinds, is copied now, and its slots tracked again
    # after the first arm.
    containers: list[Any] = [v for v in saved_locals.values() if isinstance(v, _MUTABLE)]
    lazy: list[tuple[Any, int]] = []
    if containers:
        found = containers
        containers = []
        rebound = set(map(id, map(_FIRST, emitter._slots.values())))
        # id -> (list, n): the list's first n entries hold no line or
        # tone, looked at once each while they stay in it, unless a call
        # the trace does not follow ran. A line or tone a traced change
        # puts in a list has its slot tracked (rebound).
        plain = getattr(emitter._lines, "plain", None)
        if plain is None:
            plain = emitter._lines.plain = {}
        # What a generator yields is added to without a note.
        yielded: Any = saved_locals.get(".yielded")
        for v in found:
            if v is yielded:
                containers.append((v, list(v)))
                continue
            if v.__class__ is dict or v.__class__ is not list and not isinstance(v, list):
                lazy.append((v, len(v)))
                continue
            if id(v) not in rebound:
                seen = plain.get(id(v))
                i = 0 if seen is None or seen[0] is not v else seen[1]
                n = len(v)
                if i > n:
                    i = 0
                while i < n:
                    x = v[i]
                    if x.__class__ is IRValue and x.vtype in LINEAR_TYPES:
                        break
                    i += 1
                else:
                    plain[id(v)] = (v, n)
                    lazy.append((v, n))
                    continue
                plain.pop(id(v), None)
            containers.append((v, list(v)))

    lines = emitter._lines
    held = lines.held
    held_count = len(held)
    # Each channel the arms change, with its value entering the if: an if
    # costs what its arms do, not the number of channels. One acquired in
    # an arm has a seq of at least held_count. A tracing error leaves
    # this journal set: _roll, which goes on tracing, puts back its own.
    outer = lines.journal
    journal: dict[_Chan, IRValue] = {}
    lines.journal = journal

    first = builder._next_id  # the first id made in the if
    builder.emit("scf.if", (value,), ())
    then_locals, then_slots, then_copies = trace_arm(
        emitter, view, index, join, then_start, (containers, lazy)
    )
    then_stack = list(stack)
    then_held = {cell: cell.value for cell in journal}
    stack[:] = saved_stack
    emitter.locals.clear()
    emitter.locals.update(saved_locals)
    for obj, copy in containers:
        emitter._restore_container(obj, copy)
    if then_copies:
        for obj, copy in then_copies:
            emitter._restore_container(obj, copy)
    _restore_channels(held, held_count, ())
    for cell in journal:
        if cell.held.seq < held_count:
            before = journal[cell]
            cell.value = before
            before.chan = cell
    else_index = builder.mark()
    builder.emit("scf.else", (), ())
    else_locals, else_slots, else_copies = trace_arm(
        emitter, view, index, join, else_start, (containers, lazy)
    )
    # Names, and stack slots (ints), the arms leave different numbers,
    # waveforms or conditions on.
    chosen: list[tuple[str | int, Any, Any]] = []
    # Python joins the arms at one stack depth.
    if stack and not returns and any(map(_IS_NOT, stack, then_stack)):
        from ._conditions import stack_choices

        stack_choices(
            emitter, value, then_stack, saved_stack, jump_if_true, prog, index, first, chosen
        )

    merged: dict[str, StackVal] = {}
    results: list[tuple[str, IRValue, IRValue]] = []
    for name in sorted(then_locals.keys() | else_locals.keys()):
        a = then_locals.get(name, _MISSING)
        b = else_locals.get(name, _MISSING)
        if a is b:
            merged[name] = a
        elif (
            isinstance(a, IRValue)
            and isinstance(b, IRValue)
            and a.vtype == b.vtype
            and a.vtype != "qref"
        ):
            results.append((name, a, b))
        elif a is _MISSING or b is _MISSING:
            # Bound in one arm only: unbound after the if.
            try:
                emitter._one_arm[name] = (prog, index)
            except AttributeError:
                emitter._one_arm = {name: (prog, index)}
        elif type(a) is type(b) and isinstance(a, _SCALARS) and a == b:
            merged[name] = a
        else:
            from ._conditions import bits_chosen

            if not bits_chosen(value, a, b):
                check_choosable(emitter, name, a, b, prog, index, first, value)
            chosen.append((name, a, b))
    slot_results: list[tuple[Any, int, IRValue, IRValue]] = []
    # Channels either arm advanced become results, unless a name
    # carries the same pair of values.
    aliases: list[tuple[_Chan, str]] = []
    by_pair: dict[tuple[int, int], str] | None = None
    changed = [cell for cell in journal if cell.held.seq < held_count]
    changed.sort(key=_CELL_ORDER)
    for cell in changed:
        before = journal[cell]
        a = then_held[cell] if cell in then_held else before
        b = cell.value
        if a is before and b is before:
            continue
        if by_pair is None:
            by_pair = {(id(x), id(y)): name for name, x, y in results}
        alias = by_pair.get((id(a), id(b)))
        if alias is None:
            slot_results.append((cell.held, cell.index, a, b))
        else:
            aliases.append((cell, alias))
    # List slots holding lines or tones that either arm rebound. One that
    # holds the values a name or channel carries takes its result, so
    # that each arm yields a value once.
    listed: list[tuple[Any, int, IRValue, IRValue, IRValue | None]] = []
    if then_slots or else_slots:
        originals = {id(obj): copy for obj, copy in containers}
        if lazy:
            for obj, copy in then_copies + else_copies:
                originals[id(obj)] = copy
        keys: dict[tuple[int, int], list[Any]] = {}
        for obj, i, _ in then_slots + else_slots:
            keys[(id(obj), i)] = obj
        then_new = {(id(obj), i): new for obj, i, new in then_slots}
        else_new = {(id(obj), i): new for obj, i, new in else_slots}
        carrying = {(id(a), id(b)) for _, a, b in results}
        carrying.update((id(a), id(b)) for _, _, a, b in slot_results)
        for key, obj in keys.items():
            old = originals[key[0]][key[1]]
            a = then_new.get(key, old)
            b = else_new.get(key, old)
            if a is b:
                continue
            if (id(a), id(b)) in carrying:
                emitter._slots.pop(id(a), None)
                emitter._slots.pop(id(b), None)
                listed.append((obj, key[1], a, b, None))
            else:
                slot_results.append((obj, key[1], a, b))
    if slot_results:
        pairs = {name: (a, b) for name, a, b in results}
        order = emitter._carry_order(
            [name for name, _, _ in results],
            {n: ab[1] for n, ab in pairs.items()},
            slot_results,
        )
        labels = []
        then_list = []
        else_list = []
        for entry in order:
            if type(entry) is str:
                a, b = pairs[entry]
                labels.append(entry)
            else:
                _, _, a, b = slot_results[entry[0]]
                labels.append(entry[1])
            then_list.append(a)
            else_list.append(b)
        then_vals = tuple(then_list)
        else_vals = tuple(else_list)
    else:
        order = [name for name, _, _ in results]
        labels = order
        then_vals = tuple(a for _, a, _ in results)
        else_vals = tuple(b for _, _, b in results)
    builder.insert(else_index, "scf.yield", then_vals)
    builder.emit("scf.yield", else_vals, ())
    outs = builder.emit(
        "scf.if_end",
        (),
        tuple(v.vtype for v in then_vals),
        {"result_names": labels},
    )
    carried: dict[tuple[int, int], IRValue] = {}
    for entry, out in zip(order, outs, strict=False):
        if type(entry) is str:
            merged[entry] = out
            continue
        obj, i, a, b = slot_results[entry[0]]
        emitter._slots.pop(id(a), None)
        emitter._slots.pop(id(b), None)
        if isinstance(obj, _Held):
            carried[(id(a), id(b))] = out
            obj[i] = out
        else:
            listed.append((obj, i, a, b, out))
    if listed:
        for name, a, b in results:
            carried.setdefault((id(a), id(b)), merged[name])
    # A list slot that holds its channel's values takes the channel's result.
    for obj, i, a, b, out in listed:
        out = carried.get((id(a), id(b)), out)
        assert out is not None
        obj[i] = out
        emitter._slots[id(out)] = (obj, i, out)
    if chosen:
        from ._conditions import chosen_value

        line = prog.location(index)[1]
        scope = getattr(builder, "scope", ())
        for chose, a, b in chosen:
            if isinstance(chose, int):
                stack[chose] = chosen_value(emitter, value, a, b, "", line, scope)
            else:
                merged[chose] = chosen_value(emitter, value, a, b, chose, line, scope)
    for cell, alias in aliases:
        out = merged[alias]
        cell.value = out
        out.chan = cell
    # A channel first acquired in one arm is not held after the if.
    if len(held) != held_count:
        _restore_channels(held, held_count, ())
    lines.journal = outer
    if outer is not None:
        # The channels held before this if, for the if around it.
        for cell in journal:
            if cell.held.seq < held_count and cell not in outer:
                outer[cell] = journal[cell]
    emitter.locals.clear()
    emitter.locals.update(merged)
    if returns:
        stack[:] = saved_stack
        # Both arms returned: continue at the trailing ``return None``.
        return len(prog.instrs) - 2
    return join


def check_choosable(
    emitter: PulseIREmitter,
    name: str,
    a: Any,
    b: Any,
    prog: bc.DecodedCode,
    index: int,
    first: int,
    cond: IRValue,
) -> None:
    """Reject the values *a* and *b* the arms of the if on *cond* at
    ``prog.instrs[index]`` leave on *name* (on the stack, for ``""``:
    the value of a conditional expression, ``and`` or ``or``), unless
    the if can choose between them: two numbers or two waveforms that
    exist after it."""
    line = prog.location(index)[1]
    measurement = cond.vtype == "measurement"
    on = "a measurement result" if measurement else "a runtime condition"
    if name:
        subject = f"{name!r}"
        where = f"the {'measurement' if measurement else 'runtime'}-conditioned if at line {line}"
        inside, outside = f"use {name!r} inside that arm", "use it inside each arm"
    else:
        subject = "the value of the conditional expression, `and` or `or`"
        where = f"that expression at line {line}"
        inside = outside = "use an if statement with the op in each arm"
    from ._conditions import is_condition, truth_choice

    if truth_choice(cond, a, b):
        # A condition on measurement results, chosen as one.
        return
    if measurement and (is_condition(a) or is_condition(b)):
        raise CompilationError(
            f"{subject} would depend on both a measurement result and a condition on "
            f"answers at line {line}, which one test cannot combine; branch on them in "
            "nested ifs, or with `and` and `or` in the if condition",
            code="runtime_value",
        )
    kind = _choice_kind(a)
    if kind is not None and kind == _choice_kind(b):
        scope = getattr(emitter.b, "scope", ())[:-1]  # around the if: this is its else arm
        if not (_made_in_if(a, first, scope) or _made_in_if(b, first, scope)):
            return
        raise CompilationError(
            f"{subject} would depend on {on} through a value made inside an arm of "
            f"{where}, which does not exist after it; {inside}, or make the value "
            "before it",
            code="runtime_value",
        )
    raise CompilationError(
        f"{subject} would depend on {on}: the arms of {where} leave {_describe(a)} and "
        f"{_describe(b)} on it, and an if chooses only between two numbers or two "
        "waveforms, two lines or tones of one kind, or two conditions on answers; "
        f"compute it before the if, or {outside}",
        code="runtime_value",
    )


def trace_arm(
    emitter: PulseIREmitter,
    view: bc.DecodedCode,
    branch: int,
    join: int,
    start: int,
    snapshot: tuple[list[tuple[Any, Any]], list[tuple[Any, int]]],
) -> tuple[dict[str, StackVal], list[tuple[list[Any], int, IRValue]], list[tuple[Any, Any]]]:
    """Trace one arm until it reaches *join*; return its locals, rebound
    slots and the contents before it of the containers it changed. Of the
    *snapshot* of containers the arm starts with, the first are copied and
    the others (each with its length) are copied when first changed."""
    containers, lazy = snapshot
    returns = join >= len(view.instrs)
    # An arm may only move forward from the branch, except that a join at
    # an enclosing loop header (both arms continue) ends the arm.
    if join < branch:
        header = join
        # There a comprehension's if adds to what it builds, on the stack.
        if not _CONTAINERS.isdisjoint(map(type, emitter.stack)):
            from ._conditions import add_built

            add_built(emitter.stack, containers)
    else:
        header = -1
    outer = emitter._mutated
    mutated: dict[int, tuple[Any, Any]] = {}
    emitter._mutated = mutated
    if lazy:
        lines = emitter._lines
        around = getattr(lines, "watched", ())
        lines.watched = ((mutated, tuple(obj for obj, _ in lazy)), *around)
    try:
        outcome = emitter._loop(view, start, branch, len(view.instrs), header)
    except _NotRollableError:
        from ._conditions import rejoin_error

        raise rejoin_error(view, branch) from None
    finally:
        emitter._mutated = outer
        if lazy:
            lines.watched = around
        if outer is not None:
            for key, entry in mutated.items():
                outer.setdefault(key, entry)
    if header >= 0 and outcome is None:
        outcome = _REGION_DONE
    if returns:
        if outcome is not None:
            raise CompilationError(
                "a measurement-conditioned branch cannot return a value",
                code="runtime_value",
            )
    elif outcome is not _REGION_DONE:
        from ._conditions import rejoin_error

        raise rejoin_error(view, branch)
    copies: list[tuple[Any, Any]] = []
    slots: list[tuple[list[Any], int, IRValue]] | None
    if lazy:
        slots = None
        for obj, length in lazy:
            noted = mutated.get(id(obj))
            if noted is not None and noted[1] is not None:
                copies.append(noted)
            elif len(obj) != length:
                break
        else:
            slots = _slot_changes(containers + copies)
    elif containers:
        slots = _slot_changes(containers)
    else:
        slots = []
    if slots is not None and mutated:
        # Any other container the arm changed was made in it; one a call
        # the trace does not follow may have changed must be as it was.
        known = {id(obj) for obj, _ in containers}
        known.update(id(obj) for obj, _ in lazy)
        for key, (obj, copy) in mutated.items():
            if (
                copy is not None
                and key not in known
                and not (obj == copy if obj.__class__ is set else _container_unchanged(obj, copy))
            ):
                slots = None
                break
    if slots is None:
        from ._conditions import filtered

        if header >= 0 and filtered(emitter, containers):
            raise CompilationError(
                "a comprehension or generator expression cannot filter on a condition "
                "known only when the program runs: which items it has would be decided "
                "as the program runs; loop with a for statement and branch with an if "
                "statement in its body instead",
                code="runtime_value",
            )
        raise CompilationError(
            "a measurement-conditioned branch may only change list entries that "
            "hold lines or tones",
            code="runtime_value",
        )
    return dict(emitter.locals), slots, copies
