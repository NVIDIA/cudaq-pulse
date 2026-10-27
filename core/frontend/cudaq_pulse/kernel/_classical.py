# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tracing calls to classical functions inside a kernel.

``fn.submit(*args)`` in a kernel records a ``classical.submit`` and gives a
``Future``; its ``.result()`` records a ``classical.wait`` and gives the
answer as ``AnswerValue``s, which later pulse ops take like float kernel
parameters and which branches test. The tracer calls these instead of the
methods themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..classical import (
    _F64,
    _KIND_FLOATS,
    _LIST_NAME,
    _UNSIZED,
    ClassicalFunction,
    _enter_slow,
)
from ..classical._values import ValueType
from .ir_builder import (
    AnswerValue,
    CompilationError,
    IRValue,
    UndecidableComparison,
    contains_answer,
    is_symbolic,
)

if TYPE_CHECKING:
    from .._units import Time


class Future:
    """A request to a classical function, sent from inside a kernel."""

    __slots__ = ("function", "index", "vid", "flow", "answers", "timeout_ns")

    def __init__(
        self,
        function: ClassicalFunction[..., Any],
        index: int,
        vid: int,
        flow: tuple[tuple[int, bool], ...],
    ) -> None:
        self.function = function
        self.index = index
        self.vid = vid
        self.flow = flow
        self.answers: Any = None
        self.timeout_ns: float | None = None

    def result(self, timeout: int | Time | None = None) -> Any:
        """Take the answer; only meaningful while a kernel is traced."""
        raise CompilationError("a future's result() is taken inside the kernel that submitted it")

    def __repr__(self) -> str:
        return f"<future of {self.function.name}>"


def _builder_supports(builder: Any) -> None:
    if not hasattr(builder, "classical_submit"):
        raise CompilationError("classical calls need the packed IR builder")


def _measurements(value: Any, spec: ValueType, name: str, position: int) -> list[int]:
    items = list(value) if isinstance(value, (list, tuple)) else [value]
    if len(items) != spec.count:
        raise CompilationError(
            f"{name}: argument {position} is {spec.descriptor}, which takes "
            f"{spec.count} measurement result{'s' if spec.count != 1 else ''}, got {len(items)}"
        )
    vids = []
    for item in items:
        if not (isinstance(item, IRValue) and item.vtype == "measurement"):
            raise CompilationError(
                f"{name}: argument {position} ({spec.descriptor}) takes measurement "
                f"results from readout(), got {item!r}"
            )
        vids.append(item.vid)
    return vids


def _numbers(builder: Any, value: Any, spec: ValueType, name: str, position: int) -> list[int]:
    if spec.is_scalar:
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise CompilationError(
            f"{name}: argument {position} is {spec.descriptor}; pass a list of {spec.count} numbers"
        )
    if len(items) != spec.count:
        raise CompilationError(
            f"{name}: argument {position} is {spec.descriptor}, which takes "
            f"{spec.count} numbers, got {len(items)}"
        )
    dtype = "f64" if spec.source == "float" else "i64"
    vids = []
    builder._answer_integers = True
    try:
        for item in items:
            if isinstance(item, bool) or not (isinstance(item, (int, float)) or is_symbolic(item)):
                raise CompilationError(
                    f"{name}: argument {position} ({spec.descriptor}) takes numbers, got {item!r}"
                )
            if dtype == "i64" and isinstance(item, float) and not item.is_integer():
                raise CompilationError(
                    f"{name}: argument {position} ({spec.descriptor}) takes integers, got {item!r}"
                )
            vids.append(
                builder._get_numeric_vid(
                    int(item) if dtype == "i64" and isinstance(item, float) else item, dtype
                )
            )
    finally:
        builder._answer_integers = False
    return vids


def submit(
    builder: Any,
    function: ClassicalFunction[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any] | None,
) -> Future:
    """Record ``function.submit(*args)``."""
    _builder_supports(builder)
    if kwargs:
        raise CompilationError(f"{function.name}.submit() takes positional arguments only")
    try:
        arity = len(function.args)
    except TypeError as error:  # a mistake in its declaration
        raise CompilationError(str(error)) from None
    if len(args) != arity:
        # A template has args == (): its calls give the list lengths.
        plan = function._plan
        if plan is None or len(args) != len(plan):
            arity = arity if plan is None else len(plan)
            raise CompilationError(
                f"{function.name}.submit() takes {arity} arguments, got {len(args)}"
            )
        return _submit_template(builder, function, args, plan)
    operands: list[int] = []
    for position, (spec, value) in enumerate(zip(function.args, args)):
        if spec is _F64 and (value.__class__ is float or value.__class__ is int):
            # What _numbers gives a plain number, without its checks.
            operands.append(builder._get_numeric_vid(value, "f64"))
        elif spec.source == "measurement":
            operands.extend(_measurements(value, spec, function.name, position))
        else:
            operands.extend(_numbers(builder, value, spec, function.name, position))
    index = builder.classical_function(function)
    vid = builder.classical_submit(index, operands)
    return Future(function, index, vid, builder.flow)


def _submit_template(
    builder: Any,
    template: ClassicalFunction[..., Any],
    args: Sequence[Any],
    plan: tuple[Any, ...],
) -> Future:
    """Record a call of a template, *args* one per entry of its *plan*: one
    pass over the arguments makes the operands and the version's key (per
    unsized list its length, negative for a ``list[int]`` of numbers), then
    one lookup finds the version."""
    name = template.name
    operands: list[int] = []
    key: Any = 0 if template._n == 1 else []
    for position, (p, value) in enumerate(zip(plan, args)):
        source = p.source
        if source is not _UNSIZED:
            if p is _F64 and (value.__class__ is float or value.__class__ is int):
                # What _numbers gives a plain number, without its checks.
                operands.append(builder._get_numeric_vid(value, "f64"))
            elif source == "measurement":
                operands.extend(_measurements(value, p, name, position))
            else:
                operands.extend(_numbers(builder, value, p, name, position))
        else:
            kind = p.kind
            if value.__class__ is not list and value.__class__ is not tuple:
                _not_a_list(template, position, value)
            if not value:
                raise CompilationError(
                    f"{name}.submit(): {_parameter(template, position)!r} is an empty list"
                )
            if kind >= _KIND_FLOATS and (
                kind == _KIND_FLOATS or not _measured(value, template, position)
            ):
                operands.extend(_number_list(builder, value, name, position, kind == _KIND_FLOATS))
                n = len(value) if kind == _KIND_FLOATS else -len(value)
            else:
                for item in value:
                    if not (isinstance(item, IRValue) and item.vtype == "measurement"):
                        raise CompilationError(
                            f"{name}.submit(): {_parameter(template, position)!r}"
                            f" ({_LIST_NAME[kind]}) takes measurement results from readout(),"
                            f" got {item!r}"
                        )
                    operands.append(item.vid)
                n = len(value)
            if key.__class__ is int:
                key = n
            else:
                key.append(n)
    if key.__class__ is list:
        key = tuple(key)
    version = template._fast.get(key)
    if version is None:
        version = _enter_slow(builder, template, key)
    index = builder.classical_function(version)
    vid = builder.classical_submit(index, operands)
    return Future(version, index, vid, builder.flow)


def _parameter(template: ClassicalFunction[..., Any], position: int) -> str:
    shape = template._shape
    assert shape is not None
    return shape.names[position]


def _measured(value: Sequence[Any], template: ClassicalFunction[..., Any], position: int) -> bool:
    """Whether a ``list[int]`` holds measurements (levels) or numbers."""
    measured = [isinstance(v, IRValue) and v.vtype == "measurement" for v in value]
    if any(measured) and not all(measured):
        raise CompilationError(
            f"{template.name}.submit(): {_parameter(template, position)!r} is list[int]: all"
            " measurements (sent as levels) or all numbers, not both"
        )
    return measured[0]


def _number_list(
    builder: Any, items: Sequence[Any], name: str, position: int, floats: bool
) -> list[int]:
    dtype = "f64" if floats else "i64"
    vids = []
    builder._answer_integers = True
    try:
        for item in items:
            if isinstance(item, bool) or not (isinstance(item, (int, float)) or is_symbolic(item)):
                raise CompilationError(f"{name}: argument {position} takes numbers, got {item!r}")
            if dtype == "i64" and isinstance(item, float):
                if not item.is_integer():
                    raise CompilationError(
                        f"{name}: argument {position} takes integers, got {item!r}"
                    )
                item = int(item)
            vids.append(builder._get_numeric_vid(item, dtype))
    finally:
        builder._answer_integers = False
    return vids


def _not_a_list(template: ClassicalFunction[..., Any], position: int, value: Any) -> None:
    parameter = _parameter(template, position)
    shape = template._shape
    assert shape is not None
    kind = _LIST_NAME[shape.plan[position]]
    if isinstance(value, IRValue) and value.vtype == "measurement":
        raise CompilationError(
            f"{template.name}.submit(): {parameter!r} is {kind}, a list; got one measurement,"
            " so pass [m]"
        )
    raise CompilationError(
        f"{template.name}.submit(): {parameter!r} is {kind}; pass a list, got {value!r}"
    )


def _group(function: ClassicalFunction[..., Any], answers: list[AnswerValue]) -> Any:
    """Answers as the kernel sees them: per result a value or a list."""
    grouped: list[Any] = []
    cursor = 0
    for spec in function.results:
        part = answers[cursor : cursor + spec.answer_count]
        cursor += spec.answer_count
        grouped.append(part[0] if spec.is_scalar else list(part))
    return grouped[0] if len(grouped) == 1 else tuple(grouped)


def take(builder: Any, future: Future, args: Sequence[Any], kwargs: dict[str, Any] | None) -> Any:
    """Record ``future.result(timeout=...)``."""
    timeout = None
    if args:
        if len(args) > 1:
            raise CompilationError("result() takes at most one argument, timeout")
        timeout = args[0]
    for key, value in (kwargs or {}).items():
        if key != "timeout":
            raise CompilationError(f"result() got an unexpected keyword {key!r}")
        timeout = value
    function = future.function
    timeout_ns = None
    if timeout is not None:
        from ._durations import timeout_ns as _timeout_ns

        timeout_ns = _timeout_ns(builder, timeout)
        if function.latency_mode == "blocking":
            raise CompilationError(f"{function.name} blocks until it answers; it has no timeout")
    if future.answers is not None:
        if timeout_ns != future.timeout_ns:
            raise CompilationError("an answer is taken once; call result() with the same timeout")
        return future.answers
    if not function.results:
        raise CompilationError(f"{function.name} returns nothing; there is no answer to take")
    builder.check_wait(future.flow, function.name)
    timeout = timeout_ns is not None or function.latency_mode == "timeout"
    dtypes = (["i1"] if timeout else []) + [
        spec.answer for spec in function.results for _ in range(spec.answer_count)
    ]
    answers = builder.classical_wait(future.index, future.vid, dtypes, timeout_ns)
    for k, answer in enumerate(answers):
        answer.label = f"{function.name}[{k}]"
    if timeout:
        ok, rest = answers[0], answers[1:]
        value = _group(function, rest)
        result: Any = (ok, *value) if isinstance(value, tuple) else (ok, value)
    else:
        result = _group(function, answers)
    future.answers = result
    future.timeout_ns = timeout_ns
    return result


def is_answer_condition(value: Any) -> bool:
    """Whether *value* is a branch condition computed from an answer."""
    return isinstance(value, (AnswerValue, UndecidableComparison)) and contains_answer(value)


def condition(builder: Any, value: Any) -> IRValue:
    """The ``i1`` a branch on *value*, computed from an answer, tests."""
    if isinstance(value, AnswerValue):
        builder.check_scope(value.scope, "an answer of a classical call")
        if value.dtype == "i1":
            return IRValue(value.vid, "i1")
        return IRValue(builder.classical_compare("!=", value, 0), "i1")
    if isinstance(value, UndecidableComparison):
        if isinstance(value.lhs, IRValue) or isinstance(value.rhs, IRValue):
            raise CompilationError("an answer cannot be compared with a measurement result")
        return IRValue(builder.classical_compare(value.op, value.lhs, value.rhs), "i1")
    raise CompilationError(f"cannot branch on {value!r}")
