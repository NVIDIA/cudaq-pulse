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

import numbers
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..classical import (
    _F64,
    _KIND_BITS,
    _KIND_FLOATS,
    _KIND_IQ,
    _LIST_NAME,
    ClassicalFunction,
    ValueType,
    _enter_slow,
    _spelled,
)
from .ir_builder import (
    AnswerValue,
    CompilationError,
    IRValue,
    ParameterExpression,
    SymbolicValueError,
    UndecidableComparison,
    _UndecidableComparisons,
    contains_answer,
    is_symbolic,
)

if TYPE_CHECKING:
    from .._units import Time
    from .packed_ir_builder import PackedIRBuilder

# The values an f64 argument takes as they are: _numbers would pass them
# straight to _get_numeric_vid.
_DIRECT_F64 = frozenset({float, int, AnswerValue})


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

    def result(self, timeout: float | Time | None = None) -> Any:
        """Take the answer; only meaningful while a kernel is traced."""
        raise CompilationError("a future's result() is taken inside the kernel that submitted it")

    def __repr__(self) -> str:
        return f"<future of {self.function.name}>"


def _builder_supports(builder: Any) -> None:
    if not hasattr(builder, "classical_submit"):
        raise CompilationError("classical calls need the packed IR builder")


def _wrong_count(
    function: ClassicalFunction[..., Any], spec: ValueType, position: int, got: int
) -> CompilationError:
    """A call passing a list of another length than its parameter's."""
    parameter = f"argument {position}"
    code = getattr(function.__wrapped__, "__code__", None)
    if code is not None and position < code.co_argcount:
        parameter = repr(code.co_varnames[position])
    spelled, unit = _spelled(spec)
    return CompilationError(
        f"{function.name}.submit(): {parameter} takes {spec.count} {unit}"
        f"{'s' if spec.count != 1 else ''} ({spelled}), got {got}"
    )


# What a list of each kind holds, for an error.
_HOLDS = {
    _KIND_BITS: "a list of outcomes",
    _KIND_IQ: "a list of IQ points",
    _KIND_FLOATS: "a list of numbers",
}


def _is_readout(value: Any) -> bool:
    return isinstance(value, IRValue) and value.vtype == "measurement"


def _local_name(value: Any) -> str | None:
    """The name the traced kernel binds *value* to, if it has one: a local
    of the emitter running its code (``_emitter``'s ``locals``)."""
    frame = sys._getframe(1)
    while frame is not None:
        found = getattr(frame.f_locals.get("self"), "locals", None)
        if found.__class__ is dict:
            for name, bound in found.items():
                if bound is value and name.isidentifier():
                    return str(name)
            return None
        frame = frame.f_back  # type: ignore[assignment]
    return None


def _one_readout(
    function: ClassicalFunction[..., Any], position: int, kind: str, holds: str, value: Any
) -> CompilationError:
    """A call passing one readout where its parameter takes a list."""
    code = getattr(function.__wrapped__, "__code__", None)
    names = list(code.co_varnames[: code.co_argcount]) if code is not None else []
    parameter = names[position] if position < len(names) else f"argument {position}"
    shown = _local_name(value)
    listed = f"[{shown or 'm'}]"
    arguments = [listed if k == position else n for k, n in enumerate(names)] or [listed]
    called = function.base_name
    return CompilationError(
        f"{called}.submit(): {parameter} is {kind} ({holds}) but {shown or 'the argument'} is"
        f" one readout. Write {called}.submit({', '.join(arguments)})."
    )


def _measurements(
    value: Any, spec: ValueType, function: ClassicalFunction[..., Any], position: int
) -> list[int]:
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        if not spec.is_scalar and _is_readout(value):
            holds = (
                "a list of IQ points" if spec.descriptor.startswith("iq") else "a list of outcomes"
            )
            raise _one_readout(function, position, _spelled(spec)[0], holds, value)
        items = [value]
    if len(items) != spec.count:
        raise _wrong_count(function, spec, position, len(items))
    vids = []
    for item in items:
        if not (isinstance(item, IRValue) and item.vtype == "measurement"):
            raise CompilationError(
                f"{function.name}: argument {position} ({spec.descriptor}) takes measurement "
                f"results from readout(), got {item!r}"
            )
        vids.append(item.vid)
    return vids


def _numbers(
    builder: Any, value: Any, spec: ValueType, function: ClassicalFunction[..., Any], position: int
) -> list[int]:
    name = function.name
    if spec.is_scalar:
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    elif _is_readout(value):
        raise _one_readout(function, position, _spelled(spec)[0], "a list of numbers", value)
    else:
        raise CompilationError(
            f"{name}: argument {position} is {spec.descriptor}; pass a list of {spec.count} numbers"
        )
    if len(items) != spec.count:
        raise _wrong_count(function, spec, position, len(items))
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
    for position, (spec, value) in enumerate(zip(function.args, args, strict=False)):
        if spec.source == "measurement":
            operands.extend(_measurements(value, spec, function, position))
        elif spec is _F64 and value.__class__ in _DIRECT_F64:
            # What _numbers gives a plain number or an answer, without its checks.
            operands.append(builder._get_numeric_vid(value, "f64"))
        else:
            operands.extend(_numbers(builder, value, spec, function, position))
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
    for position, (kind, value) in enumerate(zip(plan, args, strict=False)):
        if kind.__class__ is not int:  # a wire type
            if kind.source == "measurement":
                operands.extend(_measurements(value, kind, template, position))
            elif kind is _F64 and value.__class__ in _DIRECT_F64:
                # What _numbers gives a plain number or an answer, without its checks.
                operands.append(builder._get_numeric_vid(value, "f64"))
            else:
                operands.extend(_numbers(builder, value, kind, template, position))
        else:
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
    plan = shape.plan[position]
    kind = _LIST_NAME[plan]
    if _is_readout(value):
        holds = _HOLDS.get(plan, "a list of outcomes or numbers")
        raise _one_readout(template, position, kind, holds, value)
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


class _NotMeasurement(_UndecidableComparisons):
    """``not value`` for a measurement result, or for the ``i1`` IR value
    a condition on answers is tested as, pending use as a condition."""

    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value

    def __bool__(self) -> bool:
        raise CompilationError(
            f"cannot decide {self!r} at trace time: it is known only when the program runs; "
            "use it directly as an if condition"
        )

    def __eq__(self, other: object) -> UndecidableComparison:  # type: ignore[override]
        return UndecidableComparison(_negated(self), "==", other)

    def __ne__(self, other: object) -> UndecidableComparison:  # type: ignore[override]
        return UndecidableComparison(_negated(self), "!=", other)

    def __repr__(self) -> str:
        return f"(not {self.value!r})"


def _negated(value: _NotMeasurement) -> Any:
    """``not`` of a condition on answers as the comparison it is; a
    measurement's stays itself."""
    inner = value.value
    if _answer_bit(inner):
        # The IR value carries no regions to check its uses against, so () checks none.
        return UndecidableComparison(AnswerValue(inner.vid, "i1", ()), "==", False)
    return value


def _answer_bit(value: Any) -> bool:
    """Whether *value* is the ``i1`` a condition on answers is tested as."""
    return value.__class__ is IRValue and value.vtype == "i1"


def _constant_bool(value: Any) -> Any:
    """*value* if it is a bool (Python's or NumPy's) or the int 0 or 1, which
    ``&``, ``|`` and ``^`` treat as one; else None."""
    cls = value.__class__
    if cls is bool or (cls is int and (value == 0 or value == 1)):
        return value
    if cls.__name__ == "bool" and cls.__module__ == "numpy":
        return bool(value)
    return None


def measured(value: Any) -> bool:
    """Whether the bool *value* is computed from a measurement result."""
    cls = value.__class__
    if cls is IRValue:
        return bool(value.vtype == "measurement")
    if cls is _NotMeasurement:
        return measured(value.value)
    if cls is UndecidableComparison:
        return measured(value.lhs) or measured(value.rhs)
    return False


def is_condition(value: Any) -> bool:
    """Whether *value* is a bool known only when the program runs that is
    computed from answers or float arguments: a comparison, an answer bit,
    or ``not`` of one."""
    cls = value.__class__
    if cls is UndecidableComparison:
        return not measured(value)
    if cls is AnswerValue:
        return bool(value.dtype == "i1")
    return cls is _NotMeasurement and _answer_bit(value.value)


_LOGIC_WORDS = {"&": "and", "|": "or"}


def bit_logic(symbol: str, lhs: Any, rhs: Any) -> Any:
    """``lhs & rhs``, ``lhs | rhs`` or ``lhs ^ rhs``, as Python computes them
    on bools, where one is known only when the program runs: the value
    when the other operand decides it, else a condition traced as ``and``,
    ``or`` or ``!=``. Measurement results and numbers raise."""
    word = _LOGIC_WORDS.get(symbol)
    for a, b in ((lhs, rhs), (rhs, lhs)):
        constant = _constant_bool(a)
        if constant is not None and is_condition(b):
            if word is None:
                return UndecidableComparison(b, "!=", constant)
            # True | b and False & b are the constant; True & b and False | b are b.
            return constant if constant == (word == "or") else b
    if not (is_condition(lhs) and is_condition(rhs)):
        if measured(lhs) or measured(rhs):
            fix = (
                "branch on a measurement result with `if result:` or `if not result:`, "
                "combining results with `and`, `or` and `not` in the if condition"
            )
        else:
            fix = (
                f"`{symbol}` combines only conditions (comparisons and answer bits) and "
                "True or False; bitwise operations on numbers known only when the program "
                "runs are not supported"
            )
        raise CompilationError(
            f"cannot trace {lhs!r} {symbol} {rhs!r}: it is known only when the program runs; " + fix
        )
    if lhs.__class__ is _NotMeasurement:
        lhs = _negated(lhs)
    if rhs.__class__ is _NotMeasurement:
        rhs = _negated(rhs)
    return UndecidableComparison(lhs, word or "!=", rhs)


def bit_select(cond: AnswerValue, then: Any, other: Any) -> Any:
    """*then* where the answer bit *cond* is set, else *other*: bools, at
    least one known only when the program runs."""
    if then is True:
        return bit_logic("|", cond, other)
    if other is False:
        return bit_logic("&", cond, then)
    unset = UndecidableComparison(cond, "==", False)
    if then is False:
        return bit_logic("&", unset, other)
    if other is True:
        return bit_logic("|", unset, then)
    return bit_logic("|", bit_logic("&", cond, then), bit_logic("&", unset, other))


def runtime_value(value: Any) -> bool:
    """Whether *value* is known only when the program runs, so that a set or
    dict cannot look it up by its hash."""
    cls = value.__class__
    if cls is IRValue:
        return bool(value.vtype == "measurement")
    return cls is UndecidableComparison or cls is _NotMeasurement or is_symbolic(value)


# Past this many items, `answer in range(...)` of a float answer is too
# many comparisons; one of an integer answer is two whatever its length.
_MAX_MEMBERS = 64


def contains(left: Any, right: Any, error: CompilationError) -> Any:
    """``left in right`` for a tuple, list or range *right*, where comparing
    *left* with its items raised *error* because the outcome is known only
    when the program runs: an ``or`` of the comparisons, as ``in`` tests
    each item with ``is`` and ``==``."""
    cls = right.__class__
    if (
        isinstance(error, SymbolicValueError)
        or not (cls is tuple or cls is list or cls is range)
        or measured(left)
    ):
        raise error
    if cls is range and left.__class__ is AnswerValue and left.dtype == "i64":
        start, stop, step = right.start, right.stop, right.step
        if not right:
            return False
        if step > 0:
            found: Any = UndecidableComparison(left >= start, "and", left < stop)
        else:
            found = UndecidableComparison(left <= start, "and", left > stop)
        if step != 1 and step != -1:
            offset = ParameterExpression("sub", left, start, dtype="i64")
            remainder = ParameterExpression("mod", offset, abs(step), dtype="i64")
            on_step = UndecidableComparison(remainder, "==", 0)
            found = UndecidableComparison(found, "and", on_step)
        return found
    if len(right) > _MAX_MEMBERS:
        raise CompilationError(
            f"cannot trace {left!r} in a {cls.__name__} of {len(right)}: it is known only "
            f"when the program runs, and `in` would compare it with each of more than "
            f"{_MAX_MEMBERS} items; compare it with <, <= and == instead"
        ) from error
    found = False
    for item in right:
        if item is left:
            return True
        if (
            left.__class__ is AnswerValue
            and not isinstance(item, numbers.Real)
            and _constant_bool(item) is None
        ):
            continue  # an answer is a number, equal to no other kind of value
        term = left == item
        if term.__class__ is bool:
            if term:
                return True
            continue
        found = term if found is False else UndecidableComparison(found, "or", term)
    return found


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
        if value.lhs.__class__ is AnswerValue and value.rhs.__class__ in _NUMBERS:
            try:
                return IRValue(builder.classical_compare(value.op, value.lhs, value.rhs), "i1")
            except CompilationError:
                # classical_compare rejects an answer bit before it records anything.
                if value.lhs.dtype != "i1":
                    raise
        lhs, rhs, op = value.lhs, value.rhs, value.op
        if measured(value):
            raise CompilationError(_MEASURED_MESSAGE.format(value))
        if op == "and" or op == "or":
            a = condition(builder, lhs).vid
            return IRValue(builder.compare_bits(op, a, condition(builder, rhs).vid), "i1")
        if is_condition(lhs) or is_condition(rhs):
            return IRValue(_compare_bits(builder, op, lhs, rhs), "i1")
        if lhs is None or rhs is None:
            # An answer is a number, never None.
            return IRValue(builder.bit_constant(op == "!="), "i1")
        if isinstance(lhs, IRValue) or isinstance(rhs, IRValue):
            raise CompilationError(f"cannot compare {value!r}: a pulse value is not a number")
        return IRValue(builder.classical_compare(op, lhs, rhs), "i1")
    if value.__class__ is _NotMeasurement and _answer_bit(value.value):
        bit = value.value.vid
        return IRValue(builder.compare_bits("==", bit, builder.bit_constant(False)), "i1")
    raise CompilationError(f"cannot branch on {value!r}")


_NUMBERS = frozenset({int, float, bool})

_MEASURED_MESSAGE = (
    "cannot trace {0!r}: a measurement result is known only when the program runs, and "
    "a comparison of it is not traced; branch on it directly with `if result:` or "
    "`if not result:`, and combine results and answers with `and`, `or` and `not` in "
    "the if condition"
)


def _compare_bits(builder: PackedIRBuilder, op: str, lhs: Any, rhs: Any) -> int:
    """The ``i1`` of *lhs* ``op`` *rhs*, one of them a bool known only when
    the program runs and the other such a bool or a number, as Python
    compares bools."""
    if op not in ("==", "!="):
        raise CompilationError(
            f"cannot order the conditions in ({lhs!r} {op} {rhs!r}), which are known only "
            "when the program runs; compare them with == or !=, or combine them with "
            "and, or, not"
        )
    for constant, bit in ((lhs, rhs), (rhs, lhs)):
        if is_condition(constant):
            continue
        if constant is None:
            return builder.bit_constant(op == "!=")
        if not isinstance(constant, numbers.Real):
            given, constant = constant, _constant_bool(constant)
            if constant is None:
                raise CompilationError(
                    f"cannot compare the condition {bit!r} with {given!r}; a condition "
                    "known only when the program runs compares with another such condition, "
                    "True or False"
                )
        if constant != 0 and constant != 1:
            return builder.bit_constant(op == "!=")
        vid = condition(builder, bit).vid
        if bool(constant) == (op == "=="):
            return vid
        return builder.compare_bits("==", vid, builder.bit_constant(False))
    return builder.compare_bits(op, condition(builder, lhs).vid, condition(builder, rhs).vid)
