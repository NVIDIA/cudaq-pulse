# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The channels of a trace from its first measurement-conditioned if on,
imported by that if; a trace without one runs none of this.

While an if is traced, each op that rebinds a channel notes it in the if's
journal (``_Lines.journal``) with its value entering the if, and the if
restores and compares only those channels; outside an if no journal is
open. Each channel has its place in ``_Lines.held`` (``_Held.seq``), which
orders an if's results as the channels were acquired.
"""

from __future__ import annotations

import itertools
import operator
from typing import Any

from .._native._cudaq_pulse_native import identity_keys
from ._emitter import PulseIREmitter, StackVal, _Chan, _Held, _Lines, _slot_holds
from .ir_builder import LINEAR_TYPES, IRValue


class JournalLines(_Lines):
    """``_Lines`` that note what an op rebinds in an open journal, and give
    each channel acquired its place."""

    __slots__ = ()

    # Where a cell is among the channels: its channel's place, then its own.
    CELL_ORDER = operator.attrgetter("held.seq", "index")

    def add(self, key: tuple[Any, ...], values: tuple[IRValue, ...]) -> None:
        # Only a channel not held is added, and ``held`` only ever loses its
        # newest entries: its length is the channel's place.
        held = _Held(values, key)
        held.seq = len(self.held)
        self.held[key] = held

    @staticmethod
    def now(
        a: IRValue, b: IRValue, then_held: dict[_Chan, IRValue], journal: dict[_Chan, IRValue]
    ) -> tuple[IRValue, IRValue]:
        """What an if's then arm left on a line or tone *a*, and its else arm
        on *b*, once the else arm is traced: the value each one's channel had
        at that arm's end (for *a*, what the then arm left on a channel it
        changed, in *then_held*, else what the channel held entering the if),
        or the value itself if it is no channel's."""
        chan = a.chan
        if chan is not None:
            if chan in then_held:
                a = then_held[chan]
            elif chan in journal:
                a = journal[chan]
            else:
                a = chan.value
        chan = b.chan
        return a, b if chan is None else chan.value

    @classmethod
    def join_name(
        cls,
        name: str,
        a: IRValue,
        b: IRValue,
        then_held: dict[_Chan, IRValue],
        journal: dict[_Chan, IRValue],
        named: dict[_Chan, str],
        twins: list[tuple[str, str]],
        merged: dict[str, Any],
        results: list[tuple[str, IRValue, IRValue]],
    ) -> None:
        """Join the values *a* and *b* of one line or tone that an if's arms
        left on *name*, one an old value or the pair another name carries:
        an old value stands for its channel's value at that arm's end
        (``now``), as an op on it would take; *name* then holds one value
        (*merged*), takes the result of the name that carries the same pair
        (*twins*), or carries it (*named*, *results*)."""
        chan = a.chan
        a, b = cls.now(a, b, then_held, journal)
        if a is b:
            merged[name] = a
        elif chan in named:
            twins.append((name, named[chan]))
        else:
            named[chan] = name
            results.append((name, a, b))

    @classmethod
    def list_now(
        cls,
        listed: list[Any],
        obj: Any,
        key: tuple[int, int],
        a: IRValue,
        b: IRValue,
        then_held: dict[_Chan, IRValue],
        journal: dict[_Chan, IRValue],
    ) -> None:
        """Note in *listed* the list place *obj[key[1]]* an if's arms left on
        values *a* and *b* of one line or tone, one an old value, as its
        channel's values at the arms' ends (``now``), unless they are one."""
        a, b = cls.now(a, b, then_held, journal)
        if a is not b:
            listed.append((obj, key[1], a, b, None))

    def rebind_linear(
        self,
        emitter: PulseIREmitter,
        fname: str,
        call_args: list[StackVal],
        results: tuple[IRValue, ...],
    ) -> None:
        """``_Lines.rebind_linear``, noting in an open journal each channel
        it moves."""
        slots = emitter._slots
        names: dict[int, str | None] | None = None
        if fname == "sync" and len(call_args) == 2 and call_args[0] is not call_args[1]:
            # Two distinct lines: their names are found in one pass.
            first, second = call_args
            if first.chan is not None and second.chan is not None and len(results) == 2:
                # Both are claimable, so each claims its own result.
                self.rebind_pair(emitter, first, results[0], second, results[1])
                return
            found = emitter._local_names(first, second)
            names = {id(first): found[0], id(second): found[1]}
        # The operands each result may replace, by type, the first in operand
        # order last: a sync over n lines rebinds in O(n), not O(n^2).
        unclaimed: dict[str, list[tuple[str | None, IRValue, Any, Any]]] = {}
        for a in reversed(call_args):
            if a.__class__ is IRValue and a.vtype in LINEAR_TYPES:
                slot = slots.get(id(a))
                if slot is not None and not _slot_holds(slot, a):
                    slot = None
                name = emitter._find_local_name(a) if names is None else names[id(a)]
                if name is not None or slot is not None or a.chan is not None:
                    if a.vtype in unclaimed:
                        unclaimed[a.vtype].append((name, a, slot, a.chan))
                    else:
                        unclaimed[a.vtype] = [(name, a, slot, a.chan)]

        for res in results:
            if res.vtype not in unclaimed or not unclaimed[res.vtype]:
                continue
            name, op_val, slot, chan = unclaimed[res.vtype].pop()
            if chan is not None and chan.value is op_val:
                journal = self.journal
                if journal is not None and chan not in journal:
                    journal[chan] = op_val
                chan.value = res
                res.chan = chan
                if chan.shared:
                    for other, value in emitter.locals.items():
                        if value is op_val:
                            emitter.locals[other] = res
            if name is not None:
                emitter.locals[name] = res
            if slot is not None:
                slot[0][slot[1]] = res
                slots.pop(id(op_val), None)
                slots[id(res)] = (slot[0], slot[1], res)

    def rebind_pair(
        self, emitter: PulseIREmitter, old: IRValue, new: IRValue, old2: IRValue, new2: IRValue
    ) -> None:
        """``_Lines.rebind_pair``, noting in an open journal the channels it
        moves."""
        chan = old.chan
        chan2 = old2.chan
        journal = self.journal
        if journal is not None:
            # Noted for _rebind_one too, which only this calls.
            if chan is not None and chan not in journal:
                journal[chan] = chan.value
            if chan2 is not None and chan2 not in journal:
                journal[chan2] = chan2.value
        if (chan is not None and chan.shared) or (chan2 is not None and chan2.shared):
            emitter._rebind_one(old, new)
            emitter._rebind_one(old2, new2)
            return
        name, name2 = identity_keys(emitter.locals, old, old2)
        if chan is not None:
            chan.value = new
            new.chan = chan
        if name is not None:
            emitter.locals[name] = new
        emitter._rebind_slot(old, new)
        if chan2 is not None:
            chan2.value = new2
            new2.chan = chan2
        if name2 is not None:
            emitter.locals[name2] = new2
        emitter._rebind_slot(old2, new2)

    def restore_container(self, emitter: PulseIREmitter, obj: Any, copy: Any) -> None:
        """*emitter*'s ``_restore_container``, as an if restores a container
        an arm changed: a list of more than 8 lines and tones has its slots
        tracked in one step."""
        if isinstance(obj, _Held):
            for i, value in enumerate(copy):
                obj[i] = value
        elif isinstance(obj, list):
            obj[:] = copy
            if (
                len(copy) > 8
                and obj.__class__ is list
                and set(map(type, copy)) == {IRValue}
                and LINEAR_TYPES.issuperset(map(operator.attrgetter("vtype"), copy))
            ):
                # Lines or tones only, as a measurement-conditioned if restores
                # a list of them: every slot at once. A subclass is read by
                # index (its own len() and indexing run).
                entries = zip(itertools.repeat(obj), itertools.count(), copy, strict=False)
                emitter._slots.update(zip(map(id, copy), entries, strict=False))
            else:
                emitter._track_slots(obj, 0)
        elif isinstance(obj, (dict, set)):
            obj.clear()
            obj.update(copy)
        else:
            obj.clear()
            obj.extend(copy)
