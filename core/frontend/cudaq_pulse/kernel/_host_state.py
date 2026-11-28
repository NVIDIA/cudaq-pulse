# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python state a rolled loop's body may change that the tracer cannot put
back: an iterator made before the loop, an object a method changes, a
random generator, a NumPy array changed in place.

A rolled body is traced once, and every iteration would then read the state
that one trace left. So a call or an assignment that may change such state
stops the trace before it runs, and the loop is unrolled from the state
before it. The tracer imports this module when a branch arm or a rolled
body first calls Python or changes an object in place.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from ._emitter import _CONTAINERS, _MUTABLE, PulseIREmitter, _NotRollableError

# A host call into these packages changes no Python state but an iterator
# passed to it, unless it is to one of the functions in _WRITES.
_COMPUTES = frozenset({"builtins", "math", "cmath", "numpy", "cudaq_pulse"})
_WRITES = frozenset(
    {
        "setattr",
        "delattr",
        "exec",
        "eval",
        "input",
        "copyto",
        "put",
        "place",
        "putmask",
        "put_along_axis",
        "fill_diagonal",
    }
)
_IMMUTABLE = frozenset({int, float, complex, bool, str, bytes, tuple, frozenset, range})
# The methods through which an augmented assignment, an item assignment or a
# del of an item changes an object in place.
_IN_PLACE = (
    "__iadd__",
    "__isub__",
    "__imul__",
    "__imatmul__",
    "__itruediv__",
    "__ifloordiv__",
    "__imod__",
    "__ipow__",
    "__ilshift__",
    "__irshift__",
    "__iand__",
    "__ixor__",
    "__ior__",
    "__setitem__",
    "__delitem__",
)
# The methods of a NumPy array or scalar that change it; the others compute.
_NUMPY_WRITES = frozenset(
    {
        "fill",
        "sort",
        "put",
        "resize",
        "partition",
        "setfield",
        "setflags",
        "itemset",
        "byteswap",
        "__setstate__",
        *_IN_PLACE,
    }
)
# The methods of the containers ``_note_mutation`` copies that consume an
# iterable they are given.
_CONSUMES = frozenset(
    {
        "extend",
        "extendleft",
        "update",
        "difference_update",
        "intersection_update",
        "symmetric_difference_update",
        "union",
        "intersection",
        "difference",
        "symmetric_difference",
        "isdisjoint",
        "issubset",
        "issuperset",
    }
)
# Lazy iterators a call makes from its arguments, without advancing them.
_ITERATORS = frozenset({zip, map, filter, enumerate, reversed})


class ChangesStateError(_NotRollableError):
    """The traced iteration of a rolled loop was about to change Python state
    that the iterations after it would read, and that cannot be put back."""


def in_rolled_body() -> bool:
    """Whether a rolled loop's body is being traced: ``PulseIREmitter._roll``
    runs further up this thread's stack, around the trace, inlined call or
    branch arm that asks. Asked only where a call or an assignment would
    change state, so that tracing pays nothing for it elsewhere."""
    roll = PulseIREmitter._roll.__code__
    frame: types.FrameType | None = sys._getframe(1)
    while frame is not None:
        if frame.f_code is roll:
            return True
        frame = frame.f_back
    return False


def may_change_state(fn: Any, owner: Any, fname: str, kwargs: dict[str, Any] | None) -> bool:
    """Whether a host call of *fn*, not a method of a container
    ``_note_mutation`` copies, may change Python state other than an
    iterator passed to it: the object it is a method of, unless that is an
    immutable value or a NumPy array or scalar it does not write to; a
    random generator it draws from; anything a user function reaches."""
    cls = owner.__class__
    if cls in _IMMUTABLE:
        return False
    if cls.__module__ == "numpy":
        return fname in _NUMPY_WRITES or (kwargs is not None and "out" in kwargs)
    module = getattr(fn, "__module__", None) or ""
    top = module.partition(".")[0]
    if top not in _COMPUTES or module.startswith("numpy.random"):
        return True
    return fname in _WRITES or (kwargs is not None and "out" in kwargs)


def changes_in_place(obj: Any) -> bool:
    """Whether an item assignment or an augmented assignment may change
    *obj*, which is not one of the containers ``_MUTABLE`` names, in place."""
    cls = obj.__class__
    return cls not in _IMMUTABLE and any(hasattr(cls, name) for name in _IN_PLACE)


def advances_an_iterator(
    mutated: dict[int, tuple[Any, Any]], args: list[Any], kwargs: dict[str, Any] | None
) -> bool:
    """Whether a call is passed an iterator that the body or arm being
    traced, whose record of changed containers is *mutated*, did not make."""
    for value in [*args, *kwargs.values()] if kwargs else args:
        if hasattr(value.__class__, "__next__"):
            noted = mutated.get(id(value))
            if noted is None or noted[1] is not None:
                return True
    return False


def checked_call(
    emitter: PulseIREmitter,
    fn: Any,
    owner: Any,
    fname: str,
    args: list[Any],
    kwargs: dict[str, Any] | None,
) -> Any:
    """Before *emitter* calls *fn* as Python in a branch arm or a rolled
    body: raise ``ChangesStateError`` in a rolled body if the call may change
    state a later iteration reads. Returns what to call: *fn*, or for one
    that makes an iterator, *fn* that marks the iterator free to advance.
    Between measured ifs (a record of another class) nothing is checked."""
    mutated = emitter._mutated
    if mutated.__class__ is not dict:
        return fn
    assert mutated is not None
    if owner.__class__ in _CONTAINERS or isinstance(owner, _MUTABLE):
        # A method of a container changes that container only, which the
        # tracer copies, or also an iterator it consumes (lst.extend(it)).
        if fname in _CONSUMES and advances_an_iterator(mutated, args, kwargs) and in_rolled_body():
            raise ChangesStateError
        return fn
    if (
        may_change_state(fn, owner, fname, kwargs) or advances_an_iterator(mutated, args, kwargs)
    ) and in_rolled_body():
        raise ChangesStateError
    if fn is iter or (fn.__class__ is type and fn in _ITERATORS):

        def made_here(*args: Any, **kwargs: Any) -> Any:
            # An iterator made here, as by zip(...), is free to advance.
            made = fn(*args, **kwargs)
            emitter._note_fresh(made)
            return made

        return made_here
    return fn


def check_in_place(obj: Any) -> None:
    """Stop a rolled body's trace before it changes *obj*, which is not one
    of the containers ``_note_mutation`` copies, in place: the change could
    not be put back."""
    if changes_in_place(obj) and in_rolled_body():
        raise ChangesStateError
