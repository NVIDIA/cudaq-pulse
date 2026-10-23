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
"""

from __future__ import annotations

import builtins
import copy
import types
from collections.abc import Callable
from typing import Any

from . import _bytecode_normalize as bc

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


def _reads(code: types.CodeType) -> list[tuple[str, str, tuple[str, ...]]]:
    """``(kind, name, attribute chain)`` for each global or closure read."""
    try:
        prog = bc.decode(code)
    except Exception:
        return []
    instrs = prog.instrs
    cellvars = set(code.co_cellvars)
    found: list[tuple[str, str, tuple[str, ...]]] = []
    for i, (op, arg) in enumerate(instrs):
        if op == bc.LOAD_GLOBAL:
            kind = "g"
        elif op == bc.LOAD_DEREF and arg not in cellvars:
            kind = "c"
        else:
            continue
        chain: list[str] = []
        j = i + 1
        while j < len(instrs) and instrs[j][0] == bc.LOAD_ATTR:
            chain.append(instrs[j][1])
            j += 1
        found.append((kind, arg, tuple(chain)))
    return found


class Guard:
    """Re-reads a kernel's globals and closures; true while none changed."""

    __slots__ = ("check", "reads", "impure")

    def __init__(self, check: Callable[[], bool], reads: int, impure: bool) -> None:
        self.check = check
        self.reads = reads
        self.impure = impure


def build_guard(fn: types.FunctionType, *, impure: bool = False) -> Guard:
    """Generate the guard for *fn* from the values its reads have right now.

    *impure* marks a trace that evaluated a host call whose result may change
    between calls (``random.random()``); its guard never holds.
    """
    if impure:
        return Guard(_always_false, 0, True)

    env: dict[str, Any] = {"_B": vars(builtins), "_same": _same_contents}
    lines: list[str] = []
    slots: dict[tuple[Any, ...], str] = {}
    scopes: dict[int, str] = {}
    visited: set[int] = set()

    def scope_name(prefix: str, obj: Any) -> str:
        key = id(obj)
        name = scopes.get(key)
        if name is None:
            name = f"_{prefix}{len(scopes)}"
            scopes[key] = name
            env[name] = obj
        return name

    def bind(value: Any, expr: str, key: tuple[Any, ...]) -> str:
        var = slots.get(key)
        if var is not None:
            return var
        var = f"v{len(slots)}"
        slots[key] = var
        lines.append(f"{var} = {expr}")
        expected = f"_o{len(slots)}"
        if _immutable(value):
            env[expected] = value
            lines.append(
                f"if {var} is not {expected} and ({var}.__class__ is not "
                f"{expected}.__class__ or {var} != {expected}): return False"
            )
        else:
            env[expected] = value
            lines.append(f"if {var} is not {expected}: return False")
            snap = _snapshot(value)
            if snap is not None:
                env[f"_s{len(slots)}"] = snap
                lines.append(f"if not _same({var}, _s{len(slots)}): return False")
        return var

    def walk(func: types.FunctionType, depth: int) -> None:
        if id(func.__code__) in visited or depth > _MAX_HELPER_DEPTH:
            return
        visited.add(id(func.__code__))
        fn_globals = func.__globals__
        freevars = func.__code__.co_freevars
        cells = func.__closure__ or ()
        for kind, name, chain in _reads(func.__code__):
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
                    lines.append(f"if {name!r} in {g}: return False")
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
            var = bind(value, expr, root_key)
            follow(value, depth)
            key = root_key
            for attr in chain:
                try:
                    value = getattr(value, attr)
                except Exception:
                    break
                key = (*key, attr)
                var = bind(value, f"{var}.{attr}", key)
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
    body = "".join(f"            {line}\n" for line in lines) or "            pass\n"
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
