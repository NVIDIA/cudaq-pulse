# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measurement-conditioned ifs: PulseIREmitter._runtime_branch and
_trace_arm, imported at a trace's first such if.

Importing this module puts these functions on PulseIREmitter in place
of the stubs that import it, so later ifs call them directly. A trace
without such an if, as a first compile often is, never loads them.
"""

from __future__ import annotations

import operator
from typing import Any

from . import _bytecode_normalize as bc
from ._bytecode_normalize import JUMP
from ._emitter import (
    _BRANCH_TYPE_MESSAGE,
    _CONTAINERS,
    _IS_NOT,
    _MISSING,
    _MUTABLE,
    _PACKED_BRANCH_MESSAGE,
    _REGION_DONE,
    _SCALARS,
    PulseIREmitter,
    StackVal,
    _Chan,
    _container_unchanged,
    _Held,
    _Lines,
    _NotMeasurement,
    _NotRollableError,
    _restore_channels,
    _slot_changes,
)
from .ir_builder import LINEAR_TYPES, CompilationError, IRValue

_journal_lines: type[_Lines] | None = None


def _load_journal_lines() -> type[_Lines]:
    """The channels of a trace from its first measurement-conditioned if on,
    imported then."""
    global _journal_lines
    from ._branch_journal import JournalLines

    _journal_lines = JournalLines
    return JournalLines


def _runtime_branch(
    self: PulseIREmitter, prog: bc.DecodedCode, index: int, cond: Any, jump_if_true: bool
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
    if not self._can_roll:
        raise CompilationError(_PACKED_BRANCH_MESSAGE)

    target = prog.instrs[index][1]
    then_start, else_start = (target, index + 1) if jump_if_true else (index + 1, target)
    if negated:
        then_start, else_start = else_start, then_start
    join = bc.join_point(prog, index)
    returns = join >= len(prog.instrs)

    builder = self.b
    stack = self.stack
    saved_stack = list(stack)
    saved_locals = dict(self.locals)
    # A container is copied when an arm first changes it
    # (_note_mutation), with its length: a growing list of outcomes is
    # not copied at every branch. A list holding lines or tones, whose
    # slots the trace rebinds, is copied now, and its slots tracked again
    # after the first arm.
    containers: list[Any] = [v for v in saved_locals.values() if isinstance(v, _MUTABLE)]
    # A _branch_journal.Logged at an if holding lists of lines.
    lazy: Any = []
    if containers:
        found = containers
        containers = []
        # id -> (list, n): the list's first n entries hold no line or
        # tone, looked at once each while they stay in it, unless a call
        # the trace does not follow ran.
        journal_lines: Any = self._lines
        if getattr(journal_lines, "between", None) is not None:
            journal_lines.take_plain()
        plain = getattr(self._lines, "plain", None)
        if plain is None:
            plain = self._lines.plain = {}
        # What a generator yields is added to without a note.
        yielded: Any = saved_locals.get(".yielded")
        for v in found:
            if v is yielded:
                containers.append((v, list(v)))
                continue
            if v.__class__ is dict or v.__class__ is not list and not isinstance(v, list):
                lazy.append((v, len(v)))
                continue
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
            # A list holding lines or tones is not copied: each arm puts
            # back what it wrote into the list's slots from its slot log
            # (copy None), and copies the list, as it was when the arm
            # began, only if code changes it (_materialize). A subclass
            # is copied as before: its own indexing runs as often as it did.
            containers.append((v, None if v.__class__ is list else list(v)))
        if any(copy is None for _, copy in containers):
            # Lists of lines put back from the arms' slot logs: the
            # containers copied when first changed become a record of the
            # if's slot logs (_branch_journal.Logged).
            journal_class: Any = _journal_lines or _load_journal_lines()
            lazy = journal_class.logged(self, containers, lazy)

    # JournalLines from the first such if on (_branch_journal).
    lines: Any = self._lines
    held = lines.held
    held_count = len(held)
    # Each channel the arms change, with its value entering the if: an if
    # costs what its arms do, not the number of channels. One acquired in
    # an arm has a seq of at least held_count. The trace's first such if
    # makes its channels ones that note what an op rebinds while a
    # journal is open (_branch_journal.JournalLines), and numbers them,
    # as each one acquired from then on: before it nothing is noted.
    if lines.__class__ is _Lines:
        for seq, h in enumerate(held.values()):
            h.seq = seq
        lines.__class__ = _journal_lines or _load_journal_lines()
        lines.journal = None
    # The arms stop at join: written into the function's instructions
    # until the finally below.
    if returns:
        view = prog
    else:
        try:
            owned = self._owned
        except AttributeError:
            owned = self._owned = {}
        view = bc.region(prog, join, owned)
    outer = lines.journal
    journal: dict[Any, IRValue] = {}
    lines.journal = journal
    try:
        first = builder._next_id  # the first id made in the if
        builder.emit("scf.if", (value,), ())
        then_locals, then_slots, then_copies = self._trace_arm(
            view, index, join, then_start, (containers, lazy)
        )
        then_stack = list(stack)
        stack[:] = saved_stack
        self.locals.clear()
        self.locals.update(saved_locals)
        if len(held) != held_count:
            # The channels an arm acquired (_restore_channels, inline).
            for key in list(held)[held_count:]:
                del held[key]
        # Each channel's value after the then arm, and before the if.
        then_held: dict[Any, IRValue] = {}
        for cell in journal:
            then_held[cell] = cell.value
            if cell.held.seq < held_count:
                before = journal[cell]
                cell.value = before
                before.chan = cell
        else_index = builder.mark()
        builder.emit("scf.else", (), ())
        else_slots: list[tuple[list[Any], int, IRValue]]
        else_copies: list[tuple[Any, Any]]
        # The containers the then arm changed, put back (after the
        # channels: neither reads the other).
        if containers:
            undo: dict[int, list[tuple[int, IRValue]]] | None = None
            # The lists put back, in order: their records are made
            # canonical as each is put back, from the records written
            # since the last if unless the lists differ from its own.
            canon = tuple(id(obj) for obj, _ in containers if isinstance(obj, list))
            # Or every one if code may have moved the items of one since
            # (_branch_journal.Logged.fresh).
            started = lazy.__class__ is not list
            if started:
                then_log = lazy.log
                then_made = lazy.made
            full = started and (lazy.fresh or lines.dirty or lines.canon != canon)
            for obj, copy in containers:
                if copy is None:
                    copy = then_made.get(id(obj))
                    if copy is None:
                        # Only the slots the then arm wrote, put back.
                        if undo is None:
                            # As the then arm's end grouped it, if it did.
                            undo = lazy.by_list or lines.log_by_list(then_log)
                        slots = self._slots
                        for i, old in undo.get(id(obj), ()):
                            slots.pop(id(obj[i]), None)
                            obj[i] = old
                            slots[id(old)] = (obj, i, old)
                        lines.canonical_list(self, obj, full)
                        continue
                # lines.restore_container, inline for lists, dicts and sets.
                if isinstance(obj, list):
                    obj[:] = copy
                    n = len(copy)
                    if (
                        n > 8
                        and obj.__class__ is list
                        and set(map(type, copy)) == {IRValue}
                        and LINEAR_TYPES.issuperset(map(operator.attrgetter("vtype"), copy))
                    ):
                        entries = zip([obj] * n, range(n), copy, strict=False)
                        self._slots.update(zip(map(id, copy), entries, strict=False))
                    else:
                        self._track_slots(obj, 0)
                elif obj.__class__ is dict or obj.__class__ is set:
                    obj.clear()
                    obj.update(copy)
                else:
                    lines.restore_container(self, obj, copy)
            if started:
                lines.canon = canon
                lines.dirty = False
                logged: Any = self._slots
                logged.written.clear()
                # What the then arm wrote into list slots counts, for an
                # if around this one, as written by its arm, before the
                # else arm runs (a call it does not follow copies the
                # lists of that arm as they were when it began).
                around_log = lines.slotlog
                if around_log is not None:
                    for key, noted in then_log.items():
                        around_log.setdefault(key, noted)
            if then_copies:
                for obj, copy in then_copies:
                    lines.restore_container(self, obj, copy)
            if else_start == join or view.instrs[else_start] == (JUMP, join):
                # An else arm that runs nothing but a jump to where the
                # arms join (an if without else) changes nothing: each
                # container holds the copy it was put back to just before.
                else_locals, else_slots, else_copies = dict(self.locals), [], []
                else_log: Any = {}
                else_made: dict[int, list[Any]] = {}
            else:
                else_locals, else_slots, else_copies = self._trace_arm(
                    view, index, join, else_start, (containers, lazy)
                )
                if started:
                    else_log = lazy.log
                    else_made = lazy.made
            if started:
                # And what the else arm wrote.
                around_log = lines.slotlog
                if around_log is not None:
                    for key, noted in else_log.items():
                        around_log.setdefault(key, noted)
        else:
            if then_copies:
                for obj, copy in then_copies:
                    lines.restore_container(self, obj, copy)
            else_locals, else_slots, else_copies = self._trace_arm(
                view, index, join, else_start, (containers, lazy)
            )
        # Names, and stack slots (ints), the arms leave different numbers,
        # waveforms or conditions on.
        chosen: list[tuple[str | int, Any, Any]] = []
        # Python joins the arms at one stack depth.
        if stack and not returns and any(map(_IS_NOT, stack, then_stack)):
            from ._conditions import stack_choices

            stack_choices(
                self, value, then_stack, saved_stack, jump_if_true, prog, index, first, chosen
            )

        merged: dict[str, StackVal] = {}
        results: list[tuple[str, IRValue, IRValue]] = []
        # The name that carries each channel's pair of values, if one does,
        # and each other name that holds the same pair, with that name.
        named: dict[_Chan, str] = {}
        twins: list[tuple[str, str]] = []
        for name in sorted(then_locals.keys() | else_locals.keys()):
            a = then_locals.get(name, _MISSING)
            b = else_locals.get(name, _MISSING)
            if a is b:
                merged[name] = a
            elif isinstance(a, IRValue) and isinstance(b, IRValue):
                chan = a.chan
                if chan is b.chan and chan is not None:
                    if (
                        chan.value is b
                        and (
                            then_held.get(chan) is a
                            or (chan not in then_held and journal.get(chan) is a)
                        )
                        and chan not in named
                    ):
                        # The values its line had at the arms' ends: the
                        # name carries that channel's pair.
                        named[chan] = name
                        results.append((name, a, b))
                    else:
                        # An old value of the line (an op reached it through
                        # another name), or a second name of the line.
                        lines.join_name(
                            name, a, b, then_held, journal, named, twins, merged, results
                        )
                elif a.vtype == b.vtype and a.vtype != "qref":
                    # Two lines, or values of no channel.
                    results.append((name, a, b))
                else:
                    from ._conditions import bits_chosen

                    if not bits_chosen(value, a, b):
                        self._check_choosable(name, a, b, prog, index, first, value)
                    chosen.append((name, a, b))
            elif a is _MISSING or b is _MISSING:
                # Bound in one arm only: unbound after the if.
                try:
                    self._one_arm[name] = (prog, index)
                except AttributeError:
                    self._one_arm = {name: (prog, index)}
            elif type(a) is type(b) and isinstance(a, _SCALARS) and a == b:
                merged[name] = a
            else:
                from ._conditions import bits_chosen

                if not bits_chosen(value, a, b):
                    self._check_choosable(name, a, b, prog, index, first, value)
                chosen.append((name, a, b))
        slot_results: list[tuple[Any, int, IRValue, IRValue]] = []
        # Channels either arm advanced become results, unless a name
        # carries the same pair of values.
        aliases: list[tuple[_Chan, str]] = []
        if journal:
            changed = [cell for cell in journal if cell.held.seq < held_count]
            changed.sort(key=lines.CELL_ORDER)
            for cell in changed:
                before = journal[cell]
                a = then_held.get(cell, before)
                b = cell.value
                if a is before and b is before:
                    continue
                alias = named.get(cell)
                if alias is None:
                    slot_results.append((cell.held, cell.index, a, b))
                else:
                    aliases.append((cell, alias))
        # List slots holding lines or tones that either arm rebound. One that
        # holds the values a name or channel carries takes its result, so
        # that each arm yields a value once.
        listed: list[tuple[Any, int, IRValue, IRValue, IRValue | None]] = []
        if then_slots or else_slots:
            originals = {id(obj): copy for obj, copy in containers if copy is not None}
            if lazy.__class__ is not list:
                for made in (else_made, then_made):
                    originals.update(made)
            if lazy:
                for obj, copy in then_copies + else_copies:
                    originals[id(obj)] = copy
            keys: dict[tuple[int, int], list[Any]] = {}
            for obj, i, _ in then_slots + else_slots:
                keys[(id(obj), i)] = obj
            then_new = {(id(obj), i): new for obj, i, new in then_slots}
            else_new = {(id(obj), i): new for obj, i, new in else_slots}
            records = self._slots
            carrying: Any = ()
            if results or slot_results:
                carrying = {(id(a), id(b)) for _, a, b in results}
                carrying.update((id(a), id(b)) for _, _, a, b in slot_results)
            for key, obj in keys.items():
                copied: Any = originals.get(key[0])
                if copied is not None:
                    old = copied[key[1]]
                else:
                    # A list put back from the arms' slot logs: what the
                    # slot held entering the if.
                    entry: Any = then_log.get(key) or else_log.get(key)
                    old = obj[key[1]] if entry is None else entry[2]
                a = then_new.get(key, old)
                b = else_new.get(key, old)
                if a is b:
                    continue
                if (id(a), id(b)) in carrying:
                    records.pop(id(a), None)
                    records.pop(id(b), None)
                    listed.append((obj, key[1], a, b, None))
                elif a.chan is None or a.chan is not b.chan:
                    slot_results.append((obj, key[1], a, b))
                else:
                    # Two values of one line or tone that neither its
                    # channel nor a name carries: one is an old value (from
                    # a name an op did not rebind), which stands for the
                    # value its channel had at that arm's end, as a name's.
                    lines.list_now(listed, obj, key, a, b, then_held, journal)
        if slot_results:
            pairs = {name: (a, b) for name, a, b in results}
            order = self._carry_order(
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
            if results:
                _, then_vals, else_vals = zip(*results, strict=True)
            else:
                then_vals = else_vals = ()
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
            self._slots.pop(id(a), None)
            self._slots.pop(id(b), None)
            if isinstance(obj, _Held):
                carried[(id(a), id(b))] = out
                obj[i] = out
            else:
                listed.append((obj, i, a, b, out))
        if twins:
            # Names that hold one line on both paths take its one result.
            for name, carrier in twins:
                merged[name] = merged[carrier]
        if listed:
            for name, a, b in results:
                carried.setdefault((id(a), id(b)), merged[name])
            # The log of the arm around this if, if any.
            around_log = getattr(lines, "slotlog", None)
        # A list slot that holds its channel's values takes the channel's result.
        for obj, i, a, b, out in listed:
            out = carried.get((id(a), id(b)), out)
            assert out is not None
            if around_log is not None and obj.__class__ is list:
                at = i + len(obj) if i < 0 else i
                around_log.setdefault((id(obj), at), (obj, at, obj[i]))
            obj[i] = out
            self._slots[id(out)] = (obj, i, out)
        if chosen:
            from ._conditions import chosen_value

            line = prog.location(index)[1]
            scope = getattr(builder, "scope", ())
            for chose, a, b in chosen:
                if isinstance(chose, int):
                    stack[chose] = chosen_value(self, value, a, b, "", line, scope)
                else:
                    merged[chose] = chosen_value(self, value, a, b, chose, line, scope)
        for cell, alias in aliases:
            out = merged[alias]
            cell.value = out
            out.chan = cell
        # A channel first acquired in one arm is not held after the if.
        if len(held) != held_count:
            _restore_channels(held, held_count, ())
    finally:
        lines.journal = outer
        if view is not prog:
            bc.end_region(view)
    if outer is not None:
        # The channels held before this if, for the if around it.
        for cell in journal:
            if cell.held.seq < held_count and cell not in outer:
                outer[cell] = journal[cell]
    self.locals.clear()
    self.locals.update(merged)
    if returns:
        stack[:] = saved_stack
        # Both arms returned: continue at the trailing ``return None``.
        return len(prog.instrs) - 2
    return join


def _trace_arm(
    self: PulseIREmitter,
    view: bc.DecodedCode,
    branch: int,
    join: int,
    start: int,
    snapshot: tuple[list[tuple[Any, Any]], list[tuple[Any, int]]],
) -> tuple[dict[str, StackVal], list[tuple[list[Any], int, IRValue]], list[tuple[Any, Any]]]:
    """Trace one arm until it reaches *join*; return its locals, rebound
    slots and the contents before it of the containers it changed. Of the
    *snapshot* of containers the arm starts with, those with a copy are
    copied, those without (lists of lines and tones) are put back from
    the arm's slot log (the containers copied when first changed are then
    a ``_branch_journal.Logged``, which keeps that log), and the others
    (each with its length) are copied when first changed."""
    lazy: Any  # a _branch_journal.Logged at an if holding lists of lines
    containers, lazy = snapshot
    end = len(view.instrs)
    returns = join >= end
    # An arm may only move forward from the branch, except that a join at
    # an enclosing loop header (both arms continue) ends the arm.
    if join < branch:
        header = join
        # There a comprehension's if adds to what it builds, on the stack.
        if not _CONTAINERS.isdisjoint(map(type, self.stack)):
            from ._conditions import add_built

            add_built(self.stack, containers)
    else:
        header = -1
    outer = self._mutated
    mutated: dict[int, tuple[Any, Any]] = {}
    self._mutated = mutated
    if lazy:
        lines: Any = self._lines
        around = getattr(lines, "watched", ())
        if lazy.__class__ is list:
            lines.watched = ((mutated, tuple(obj for obj, _ in lazy)), *around)
        else:
            # Lists of lines put back from the arm's slot log
            # (_branch_journal.Logged).
            lazy.begin(lines, mutated, around)
    try:
        outcome = self._loop(view, start, branch, end, header)
    except _NotRollableError:
        from ._conditions import rejoin_error

        raise rejoin_error(view, branch) from None
    finally:
        self._mutated = outer
        if lazy:
            lines.watched = around
            if lazy.__class__ is not list:
                lines.slotlog = lazy.around_log
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
            if lazy.__class__ is list:
                slots = _slot_changes(containers + copies)
            else:
                slots = lazy.end(mutated, containers, copies)
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

        if header >= 0 and filtered(self, containers):
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
    return dict(self.locals), slots, copies


# In place of the stubs that imported this module, where they still are.
for _name, _function in (("_runtime_branch", _runtime_branch), ("_trace_arm", _trace_arm)):
    if getattr(PulseIREmitter.__dict__.get(_name), "loads_measured_if", False):
        setattr(PulseIREmitter, _name, _function)
del _name, _function
