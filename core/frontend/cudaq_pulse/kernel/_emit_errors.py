# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The tracer's errors that take more than a line to word, loaded only
when one is raised.
"""

from __future__ import annotations

import difflib
import types
from typing import Any

from ._emitter import (
    _LINE_TYPES,
    _OPERAND_TYPES,
    _QUDIT,
    _TYPE_WORDS,
    _WAVEFORM,
    _describe,
    _find_choice,
    _parameter_names,
)
from .ir_builder import OP_TABLE, CompilationError, IRValue, SymbolicValueError


def did_you_mean(name: str, candidates: Any, call: bool = False) -> str:
    """``"; did you mean 'x'?"`` for the closest of *candidates*, or ``""``."""
    names = [c for c in dict.fromkeys(candidates) if isinstance(c, str) and not c.startswith("__")]
    close = difflib.get_close_matches(name, names, n=1)
    if not close:
        return ""
    return f"; did you mean {close[0]}()?" if call else f"; did you mean {close[0]!r}?"


def attribute_error(obj: Any, attr: str) -> str:
    """The error for ``obj.attr`` when *obj* has no attribute *attr*."""
    if isinstance(obj, types.ModuleType):
        module = obj.__name__
        if module.split(".")[0] == "cudaq_pulse" and attr in OP_TABLE:
            return (
                f"module {module!r} has no attribute {attr!r}: the kernel language is in "
                f"cudaq_pulse.ops; write {attr}(...) or cudaq_pulse.ops.{attr}(...)"
            )
        return f"cannot resolve attribute {attr!r} on module {module!r}" + did_you_mean(
            attr, dir(obj)
        )
    try:
        names = dir(obj)
    except Exception:  # a broken __dir__ only loses the suggestion
        names = []
    return f"cannot resolve attribute {attr!r} on {type(obj).__name__}" + did_you_mean(attr, names)


def _qudit_name(local_vars: dict[str, Any], qudit: IRValue) -> str | None:
    """The name *local_vars* give *qudit*: ``"qs[1]"`` for an element of a
    list of qudits, else the first name bound to it."""
    for name, value in local_vars.items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is qudit:
                    return f"{name}[{i}]"
    for name, value in local_vars.items():
        if value is qudit and not name.startswith("."):
            return name
    return None


def qudit_attribute_error(
    local_vars: dict[str, Any], qudit: IRValue, attr: str
) -> CompilationError:
    """The error for reading attribute *attr* of *qudit* while tracing."""
    name = _qudit_name(local_vars, qudit)
    if name is None or "[" not in name:
        where = "a qudit" if name is None else f"qudit {name!r}"
        handle = "q" if name is None else name
        example = f"value: float, called with TABLE[{handle}.{attr}]"
    else:
        where = f"qudit {name!r}"
        register = name.split("[")[0]
        example = f"values: list[float], called with [TABLE[q.{attr}] for q in {register}]"
    return CompilationError(
        f"cannot read .{attr} of {where} while the kernel is traced: in a kernel a qudit "
        "is a value of the program, not a handle. Look the value up on the host and pass "
        f"it in as a parameter, e.g. {example}"
    )


def operand_error(fname: str, call_args: list[Any]) -> CompilationError:
    """The error for the first operand of *fname* that is not of its type."""
    index, want = 0, _OPERAND_TYPES[fname][0]
    for index, (v, want) in enumerate(zip(call_args, _OPERAND_TYPES[fname])):
        if v.__class__ is not IRValue or v.vtype not in want:
            break
    value = call_args[index]
    wanted = " or ".join(_TYPE_WORDS[t] for t in want)
    message = f"{fname}(): argument {index + 1} must be a {wanted}, got {_describe(value)}"
    vtype = value.vtype if value.__class__ is IRValue else None
    play = fname in ("drive", "readout")
    if play and index == 0 and vtype == "tone":
        message += (
            f"; the line comes first and the tone by keyword: {fname}(line, waveform) "
            f"or {fname}(line, waveform, tone=tone)"
        )
    elif play and index == 2 and vtype in _LINE_TYPES:
        message += (
            f"; pass a tone, such as line.tone, not a line: {fname}(line, waveform, tone=tone)"
        )
    elif play and vtype in _LINE_TYPES:
        message += "; drive() takes a drive line and readout() a readout line"
    elif fname == "wf_scale":
        message += "; call it as wf_scale(waveform, scalar)"
    elif want is _WAVEFORM and vtype is None:
        message += "; pass a waveform such as square(duration, amplitude)"
    elif want is _QUDIT and vtype is None:
        from .._device import Qudit

        if isinstance(value, Qudit):
            message = (
                f"{fname}(): argument {index + 1} is a qudit handle that reached the kernel "
                "inside a list; pass qudits as separate arguments, a Register "
                "(dev.qudit_handles[0:4]) or a tuple, not a list"
            )
        else:
            message += "; pass one of the kernel's qudit arguments"
    return CompilationError(message)


def play_arity_error(fname: str, count: int) -> CompilationError:
    """The error for *fname* (drive or readout) given *count* positional arguments."""
    message = (
        f"{fname}() takes a line and a waveform, and a tone only by keyword: "
        f"{fname}(line, waveform) plays on the line's own tone, line.tone, and "
        f"{fname}(line, waveform, tone=tone) on another; got {count} positional arguments"
    )
    return CompilationError(message)


def range_error(args: list[Any]) -> CompilationError:
    choice = _find_choice(args)
    if choice is not None:
        return CompilationError(
            f"a range() bound is {choice.origin}, which is known only when the program "
            "runs; loop inside each arm of the if instead"
        )
    names = sorted({n for a in args for n in _parameter_names(a)})
    if names:
        return SymbolicValueError(
            "range() bound depends on kernel parameter "
            + ", ".join(repr(n) for n in names)
            + "; a float parameter stays symbolic, so annotate it as int to "
            "specialize the kernel on its value",
            tuple(names),
            "a range() bound",
        )
    kinds = ", ".join(type(a).__name__ for a in args)
    return CompilationError(f"range() bounds must be integers known at compile time; got ({kinds})")
