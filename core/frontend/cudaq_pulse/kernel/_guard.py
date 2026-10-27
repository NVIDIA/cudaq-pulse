# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Guards that keep a cached kernel compilation faithful to Python semantics.

A kernel reads module globals, closure cells and attribute chains of either
(``cfg.amp``, ``math.pi``) while it is traced, and the compiled IR bakes the
values in. The guard records those reads, found statically in the decoded
bytecode of the kernel and of every Python helper it can reach, and generates
one Python function that re-reads them and compares with the values seen at
trace time. A cache entry is used only while its guard holds, so rebinding a
global or mutating a list a kernel reads is seen on the next call.

A mutable container is compared by its contents, copied when the guard is
built. A dict the kernel only subscripts (``TABLE[key]`` at every read, and
never iterated or tested with ``in``) is instead compared at the items the
trace read, so the cost of a guard does not grow with the dict's size.
"""

from __future__ import annotations

import builtins
import copy
import types
from itertools import compress
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from . import _bytecode_normalize as bc

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping

    _Read = tuple[str, str, tuple[str, ...], bool]

_IMMUTABLE = (int, float, complex, str, bytes, bool, type(None), range)
_SNAPSHOT = (list, dict, set, bytearray)
_MAX_HELPER_DEPTH = 8


def _always_false() -> bool:
    return False


def _immutable(value: Any) -> bool:
    if isinstance(value, _IMMUTABLE):
        return True
    if isinstance(value, (tuple, frozenset)):
        return all(_immutable(v) for v in value)
    return False


def _snapshot(value: Any) -> Any | None:
    """A copy to compare a mutable container's contents against, or None."""
    if isinstance(value, _SNAPSHOT):
        try:
            return copy.deepcopy(value)
        except Exception:
            return None
    if type(value).__module__ == "numpy" and type(value).__name__ == "ndarray":
        return value.copy()
    return None


def _item_snapshot(value: Any) -> Any | None:
    """A copy of a dict item to compare its contents against, or None when
    a copy does not compare equal to it (an object without ``__eq__``)."""
    snap = _snapshot(value)
    if snap is not None:
        return snap
    try:
        snap = copy.deepcopy(value)
    except Exception:
        return None
    return snap if _same_contents(value, snap) else None


def _same_contents(value: Any, snapshot: Any) -> bool:
    if type(value) is not type(snapshot):
        return False
    if type(value).__name__ == "ndarray":
        return bool(
            value.shape == snapshot.shape
            and value.dtype == snapshot.dtype
            and (value == snapshot).all()
        )
    try:
        return bool(value == snapshot)
    except Exception:
        return False


# A code's canonical opcodes are read as bytes: 1 in _IS_READ at a global or
# closure read; _END follows the last instruction.
_IS_READ = bytearray(256)
_IS_READ[bc.LOAD_GLOBAL] = _IS_READ[bc.LOAD_DEREF] = 1
_END = b"\xff\xff"
# Instructions that push one value and read nothing from the stack.
_PUSH_ONE = frozenset({bc.LOAD_CONST, bc.LOAD_FAST, bc.LOAD_DEREF, bc.LOAD_GLOBAL})
_OP = itemgetter(0)
_ARG = itemgetter(1)

# The reads of each code object; equal code objects read the same names.
_READS: dict[types.CodeType, list[_Read]] = {}
_READS_LIMIT = 4096


def _reads(code: types.CodeType) -> list[_Read]:
    """``(kind, name, attribute chain, indexed)`` for each global or closure
    read; *indexed* when the value read is subscripted at once with a key
    pushed by one instruction, so nothing else ever holds it."""
    if code in _READS:
        return _READS[code]
    try:
        prog = bc.decode(code)
    except Exception:
        return []
    instrs = prog.instrs
    ops = bytes(map(_OP, instrs)) + _END
    cellvars = code.co_cellvars
    found: list[_Read] = []
    for start in compress(range(len(instrs)), ops.translate(_IS_READ)):
        op, arg = instrs[start]
        if op == bc.LOAD_GLOBAL:
            kind = "g"
        elif arg not in cellvars:
            kind = "c"
        else:
            continue
        end = start + 1
        while ops[end] == bc.LOAD_ATTR:
            end += 1
        chain = () if end == start + 1 else tuple(map(_ARG, instrs[start + 1 : end]))
        found.append((kind, arg, chain, ops[end] in _PUSH_ONE and ops[end + 1] == bc.SUBSCR))
    if len(_READS) >= _READS_LIMIT:
        _READS.clear()
    _READS[code] = found
    return found


class Guard:
    """Re-reads a kernel's globals and closures; true while none changed."""

    __slots__ = ("check", "reads", "impure")

    def __init__(self, check: Callable[[], bool], reads: int, impure: bool) -> None:
        self.check = check
        self.reads = reads
        self.impure = impure


def build_guard(
    fn: types.FunctionType,
    *,
    impure: bool = False,
    dict_items: Mapping[tuple[int, Any], tuple[dict[Any, Any], Any, Any]] | None = None,
    dict_whole: Collection[int] = (),
) -> Guard:
    """Generate the guard for *fn* from the values its reads have right now.

    *impure* marks a trace that evaluated a host call whose result may change
    between calls (``random.random()``); its guard never holds. *dict_items*
    are the dict items the trace read, ``(id(dict), key) -> (dict, key,
    value)``, and *dict_whole* the ids of the dicts it iterated or tested
    with ``in``; without *dict_items* every dict is compared whole.
    """
    if impure:
        return Guard(_always_false, 0, True)

    env: dict[str, Any] = {"_B": vars(builtins), "_same": _same_contents}
    # [var, expr, value, item] for each value to compare, rendered once every
    # read is known; [None, line, None, False] for a guard line as it is.
    lines: list[list[Any]] = []
    slots: dict[tuple[Any, ...], str] = {}
    scopes: dict[int, str] = {}
    visited: set[int] = set()
    # Values some read uses other than by an immediate subscript.
    escaped: set[int] = set()

    def scope_name(prefix: str, obj: Any) -> str:
        key = id(obj)
        name = scopes.get(key)
        if name is None:
            name = f"_{prefix}{len(scopes)}"
            scopes[key] = name
            env[name] = obj
        return name

    def bind(value: Any, expr: str, key: tuple[Any, ...], indexed: bool = False) -> str:
        if not indexed:
            escaped.add(id(value))
        var = slots.get(key)
        if var is not None:
            return var
        var = f"v{len(slots)}"
        slots[key] = var
        lines.append([var, expr, value, False])
        return var

    def walk(func: types.FunctionType, depth: int) -> None:
        if id(func.__code__) in visited or depth > _MAX_HELPER_DEPTH:
            return
        visited.add(id(func.__code__))
        fn_globals = func.__globals__
        freevars = func.__code__.co_freevars
        cells = func.__closure__ or ()
        for kind, name, chain, indexed in _reads(func.__code__):
            if kind == "g":
                g = scope_name("G", fn_globals)
                if name in fn_globals:
                    value = fn_globals[name]
                    expr = f"{g}[{name!r}]"
                elif name in vars(builtins):
                    value = vars(builtins)[name]
                    expr = f"({g}[{name!r}] if {name!r} in {g} else _B[{name!r}])"
                else:
                    # Unresolved at trace time: a later binding must retrace.
                    lines.append([None, f"if {name!r} in {g}: return False", None, False])
                    continue
                root_key: tuple[Any, ...] = ("g", id(fn_globals), name)
            else:
                if name not in freevars:
                    continue
                cell = cells[freevars.index(name)]
                try:
                    value = cell.cell_contents
                except ValueError:
                    continue
                expr = f"{scope_name('C', cell)}.cell_contents"
                root_key = ("c", id(cell))
            var = bind(value, expr, root_key, indexed and not chain)
            follow(value, depth)
            key = root_key
            for n, attr in enumerate(chain, start=1):
                try:
                    value = getattr(value, attr)
                except Exception:
                    break
                key = (*key, attr)
                var = bind(value, f"{var}.{attr}", key, indexed and n == len(chain))
                follow(value, depth)

    def follow(value: Any, depth: int) -> None:
        if isinstance(value, types.FunctionType):
            module = getattr(value, "__module__", "") or ""
            if not module.startswith("cudaq_pulse."):
                walk(value, depth + 1)
        inner = getattr(value, "__pulse_kernel_fn__", None)
        if isinstance(inner, types.FunctionType):
            walk(inner, depth + 1)

    walk(fn, 0)
    items: dict[int, list[tuple[Any, Any]]] = {}
    if dict_items:
        for container, index, value in dict_items.values():
            items.setdefault(id(container), []).append((index, value))
    rendered: list[str] = []
    # The items of a dict compared at the items read are appended to *lines*
    # as it is rendered, and so rendered after it by this same loop.
    for line in lines:
        var, expr, value, item = line
        if var is None:
            rendered.append(expr)
            continue
        n = len(env)
        expected = f"_o{n}"
        env[expected] = value
        if _immutable(value):
            rendered += (
                f"{var} = {expr}",
                f"if {var} is not {expected} and ({var}.__class__ is not "
                f"{expected}.__class__ or {var} != {expected}): return False",
            )
        elif isinstance(value, (types.MethodType, types.BuiltinMethodType)):
            # A bound method is made anew on every attribute read; two are
            # equal when they bind the same function to the same object.
            rendered += (f"{var} = {expr}", f"if {var} != {expected}: return False")
        else:
            rendered += (f"{var} = {expr}", f"if {var} is not {expected}: return False")
            if (
                value.__class__ is dict
                and dict_items is not None
                and id(value) not in escaped
                and id(value) not in dict_whole
            ):
                # Only subscripted: the items read are all the trace used.
                for k, (index, read) in enumerate(items.get(id(value), ())):
                    # What the trace did with an item is not known: compare it whole.
                    escaped.add(id(read))
                    env[f"_k{n}_{k}"] = index
                    lines.append([f"{var}_{k}", f"{var}[_k{n}_{k}]", read, True])
                continue
            snap = _item_snapshot(value) if item else _snapshot(value)
            if snap is not None:
                env[f"_s{n}"] = snap
                rendered.append(f"if not _same({var}, _s{n}): return False")
    body = "".join(f"            {line}\n" for line in rendered) or "            pass\n"
    src = (
        "def _make():\n"
        "    def _guard():\n"
        "        try:\n"
        f"{body}"
        "        except Exception:\n"
        "            return False\n"
        "        return True\n"
        "    return _guard\n"
    )
    namespace = dict(env)
    exec(compile(src, f"<pulse guard {fn.__qualname__}>", "exec"), namespace)
    return Guard(namespace["_make"](), len(slots), False)
