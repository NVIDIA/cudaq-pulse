# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The body of a classical function a controller computes, as a program of
run-time expressions: an export's ``functions`` table.

The body is read as the cost model reads it (``_controller_cost``): the
same walk of its Python source, through the helpers it calls, with the same
names resolved, the same constants folded and the same refusals, so that
the operations ``controller_usage`` counts are the terms of the program.
It becomes a straight-line program: a ``let`` for each value the body
names, in order, and its results, each an expression of the export's
vocabulary (``controller.evaluate_expression``) over the function's
arguments (``{"argument": k}``, with ``"index"`` for an item of a list)
and the names bound before it (``{"local": name}``). An ``if`` computes
both arms and selects (the vocabulary has no side effects), a loop of a
trip count known when the program compiles is unrolled, and a helper's
body is inlined. Python's ``%``, ``**`` and the math functions are the
terms ``mod``, ``pow``, ``exp``, ``log``, ``sqrt``, ``sin``, ``cos``,
``atan2``, ``abs``, ``floor`` and ``ceil``.

A body is lowered once per function and kept; an export of a program
whose control system has no processor loads none of this.
"""

from __future__ import annotations

import ast
import math
import threading
import types
import weakref
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ._controller_cost import _MATH, _NotAnalyzableError, _source_tree, _Walker, analyze_callable
from ._errors import UnsupportedError
from ._units import as_nanoseconds

# The arithmetic number a controller computes in.
FLOAT64 = "float64"

# Python's operators on numbers, as the terms they are.
_OPERATORS: Mapping[type, str] = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floor_div",
    ast.Mod: "mod",
    ast.Pow: "pow",
}
_COMPARED: Mapping[type, str] = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
}
# Python applied to constants, as the cost model folds them.
_FOLDS: Mapping[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a**b,
}
# The math functions of one float, as the terms they are.
_UNARY_MATH = {
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "sin": "sin",
    "cos": "cos",
    "fabs": "abs",
    "floor": "floor",
    "ceil": "ceil",
}
# The terms that are arithmetic or an elementary function: the operations
# the cost model counts by class, one each (``pow`` as its classes).
ARITHMETIC_TERMS = frozenset(
    {"add", "sub", "mul", "div", "neg", "abs", "floor_div", "mod", "pow"}
    | {"sqrt", "exp", "log", "sin", "cos", "atan2"}
)
# The value types of the wire's scalar descriptors.
_SCALAR_TYPES = {"f64": "float", "f32": "float", "i64": "int", "i32": "int", "u8": "int"}
_ITEM_TYPES = {"bits": "int", "levels": "int", "f64": "float", "f32": "float", "i32": "int"}
_INT_LIMIT = 1 << 63
_EXACT_INT = 1 << 53


class _RefusedError(Exception):
    """A body, or a part of it, outside what a controller program holds."""


@dataclass(frozen=True)
class Lowered:
    """A function's program (its ``functions`` entry), and the operations
    its arithmetic terms are counted as, by the cost model's classes
    (``classes``) and in all (``counted``, one per such term)."""

    entry: Mapping[str, Any]
    classes: Mapping[str, int]
    counted: int


# ── Values ──────────────────────────────────────────────────────────────────


class _Scalar:
    """One number of the body: its expression and its type, ``"int"``,
    ``"float"`` or ``"bool"``."""

    __slots__ = ("expression", "kind")

    def __init__(self, expression: dict[str, Any], kind: str) -> None:
        self.expression, self.kind = expression, kind


class _Object:
    """Something the body names that is not a number: a function or a module."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value


class _Unbound:
    """A name one path through an ``if`` assigns and the other does not."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


# A tuple or list of the body (a list argument too) is a Python tuple of values.
Value = Any


def _const(value: bool | int | float) -> _Scalar:
    kind = "bool" if isinstance(value, bool) else "int" if isinstance(value, int) else "float"
    return _Scalar({"const": value}, kind)


def _is_leaf(expression: Mapping[str, Any]) -> bool:
    return "op" not in expression


def _constant_of(value: Any) -> bool | int | float | None:
    if value.__class__ is _Scalar and "const" in value.expression:
        found: bool | int | float = value.expression["const"]
        return found
    return None


# ── The program ─────────────────────────────────────────────────────────────


class _Program:
    """The lets of a body as it is lowered, and what its terms count."""

    def __init__(self) -> None:
        self.lets: list[dict[str, Any]] = []
        self.taken: set[str] = set()
        self.temporaries = 0
        self.classes: Counter[str] = Counter()
        self.counted = 0

    def term(
        self,
        op: str,
        kind: str,
        of: Sequence[_Scalar],
        result: str,
        classes: Sequence[str] = (),
    ) -> _Scalar:
        """The term *op* over *of*, computed in *kind*, of type *result*;
        *classes* are the operations the cost model counts it as."""
        self.classes.update(classes)
        if op in ARITHMETIC_TERMS:
            self.counted += 1
        expression = {"op": op, "type": kind, "of": [v.expression for v in of]}
        return _Scalar(expression, result)

    def named(self, name: str, value: _Scalar) -> _Scalar:
        """*value* bound to *name* (made unique) when it is a term, so that
        what reads it reads the let."""
        if _is_leaf(value.expression):
            return value
        unique, n = name, 0
        while unique in self.taken:
            n += 1
            unique = f"{name}.{n}"
        self.taken.add(unique)
        self.lets.append({"name": unique, "value": value.expression})
        return _Scalar({"local": unique}, value.kind)

    def shared(self, value: _Scalar) -> _Scalar:
        """*value*, bound to a temporary when it is a term the lowering reads
        more than once."""
        if _is_leaf(value.expression):
            return value
        self.temporaries += 1
        return self.named(f"#{self.temporaries}", value)


# ── Lowering ────────────────────────────────────────────────────────────────


def _returns(nodes: Sequence[ast.AST]) -> bool:
    """Whether a return is among *nodes* or inside them."""
    for node in nodes:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Return):
                return True
    return False


class _Loop:
    """An unrolled loop's pass: bind the loop's target, then run its body."""

    __slots__ = ("node", "value")

    def __init__(self, node: ast.For, value: Value) -> None:
        self.node, self.value = node, value


class _Lowerer(_Walker):
    """Lowers one body, the function's or an inlined helper's, into a
    ``_Program``: the cost model's walker, reading the same source with the
    same names and constants, producing terms instead of counts."""

    def __init__(
        self,
        fn: Callable[..., Any],
        sizes: Mapping[str, int],
        stack: tuple[Any, ...],
        program: _Program,
        env: dict[str, Value],
        prefix: str,
    ) -> None:
        super().__init__(fn, sizes, stack)
        self.program = program
        self.env = env
        self.prefix = prefix

    def refuse(self, node: ast.AST, what: str) -> _RefusedError:
        return _RefusedError(f"{self.where(node)}: {what}")

    # ── Statements ───────────────────────────────────────────────────────

    def run(self, work: list[Any]) -> Value:
        """The value the statements *work* return, in order; an ``if`` with
        a return in an arm runs the statements after it in each arm."""
        i = 0
        while i < len(work):
            node = work[i]
            if node.__class__ is _Loop:
                self.assign(node.node.target, node.value)
                self.bind(node.node.target, None)
                i += 1
                continue
            if isinstance(node, ast.Return):
                if node.value is None:
                    raise self.refuse(node, "a return without a value")
                return self.value(node.value)
            if isinstance(node, ast.If) and _returns([node]):
                test = self.program.shared(self.test(node.test))
                rest = work[i + 1 :]
                before = dict(self.env)
                then = self.run([*node.body, *rest])
                self.env = dict(before)
                other = self.run([*node.orelse, *rest])
                return self.select(test, then, other, node)
            if isinstance(node, ast.For) and _returns([node]):
                work = [*work[:i], *self.passes(node), *work[i + 1 :]]
                continue
            self.execute(node)
            i += 1
        raise _RefusedError(f"{self.fn.__name__}: it can finish without returning a value")

    def execute(self, node: ast.AST) -> None:
        """Lower one statement without a return in it."""
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant):  # a docstring
                return
            value = self.value(node.value)
            if value.__class__ is _Scalar:
                self.program.shared(value)
            return
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:
                return
            value = self.value(node.value)
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                self.assign(target, value)
                self.bind(target, node.value)
            return
        if isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise self.refuse(node, "a store into a sequence or attribute")
            current = self.read(node.target, node.target.id)
            value = self.arithmetic(node.op, current, self.value(node.value), node, node.value)
            self.assign(node.target, value)
            self.bind(node.target, None)
            return
        if isinstance(node, ast.If):
            test = self.program.shared(self.test(node.test))
            before = dict(self.env)
            for statement in node.body:
                self.execute(statement)
            then = self.env
            self.env = dict(before)
            for statement in node.orelse:
                self.execute(statement)
            other = self.env
            self.env = self.merge(test, then, other, node)
            return
        if isinstance(node, ast.For):
            for step in self.passes(node):
                if step.__class__ is _Loop:
                    self.assign(step.node.target, step.value)
                    self.bind(step.node.target, None)
                else:
                    self.execute(step)
            return
        if isinstance(node, ast.Pass):
            return
        raise self.refuse(node, f"a {type(node).__name__} statement")

    def passes(self, node: ast.For) -> list[Any]:
        """The statements of a loop, unrolled: each pass binds its target
        and runs the body."""
        if node.orelse:
            raise self.refuse(node, "a for-else")
        items = self.iterated(node.iter, node)
        work: list[Any] = []
        for item in items:
            work.append(_Loop(node, item))
            work.extend(node.body)
        return work

    def iterated(self, node: ast.AST, where: ast.AST) -> Sequence[Value]:
        """The values a loop takes, one per pass."""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            name = node.func.id
            if name not in self.env and name in ("range", "enumerate"):
                if name == "range":
                    bounds = [self.constant(a) for a in node.args]
                    if 1 <= len(bounds) <= 3 and all(
                        isinstance(b, int) and not isinstance(b, bool) for b in bounds
                    ):
                        return [_const(k) for k in range(*bounds)]
                    raise self.refuse(where, "a range whose bounds are not known when it compiles")
                if len(node.args) == 1:
                    sequence = self.sequence(self.value(node.args[0]), node.args[0])
                    return [(_const(k), v) for k, v in enumerate(sequence)]
        return self.sequence(self.value(node), where)

    def sequence(self, value: Value, node: ast.AST) -> tuple[Value, ...]:
        if value.__class__ is not tuple:
            raise self.refuse(node, f"`{ast.unparse(node)}` is not a sequence of known length")
        found: tuple[Value, ...] = value
        return found

    def assign(self, target: ast.AST, value: Value) -> None:
        if isinstance(target, ast.Name):
            if value.__class__ is _Scalar:
                value = self.program.named(self.prefix + target.id, value)
            elif value.__class__ is tuple:
                value = tuple(
                    self.program.named(f"{self.prefix}{target.id}[{k}]", v)
                    if v.__class__ is _Scalar
                    else v
                    for k, v in enumerate(value)
                )
            self.env[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            items = self.sequence(value, target)
            if any(isinstance(e, ast.Starred) for e in target.elts):
                raise self.refuse(target, "a starred assignment")
            if len(items) != len(target.elts):
                raise self.refuse(target, f"{len(target.elts)} names unpack {len(items)} values")
            for element, item in zip(target.elts, items, strict=True):
                self.assign(element, item)
            return
        raise self.refuse(target, "a store into a sequence or attribute")

    def merge(
        self, test: _Scalar, then: dict[str, Value], other: dict[str, Value], node: ast.AST
    ) -> dict[str, Value]:
        """The names after an ``if`` whose arms leave *then* and *other*."""
        out: dict[str, Value] = {}
        for name in {**then, **other}:
            a, b = then.get(name), other.get(name)
            if a is None or b is None or a.__class__ is _Unbound or b.__class__ is _Unbound:
                out[name] = _Unbound(name)
            elif a is b:
                out[name] = a
            else:
                merged = self.select(test, a, b, node)
                out[name] = self.named_value(name, merged)
        return out

    def named_value(self, name: str, value: Value) -> Value:
        if value.__class__ is _Scalar:
            return self.program.named(self.prefix + name, value)
        if value.__class__ is tuple:
            return tuple(self.named_value(f"{name}[{k}]", v) for k, v in enumerate(value))
        return value

    def select(self, test: _Scalar, a: Value, b: Value, node: ast.AST) -> Value:
        """*a* where *test* holds, else *b*: both computed."""
        if a is b:
            return a
        if a.__class__ is tuple and b.__class__ is tuple and len(a) == len(b):
            return tuple(self.select(test, x, y, node) for x, y in zip(a, b, strict=True))
        if a.__class__ is not _Scalar or b.__class__ is not _Scalar:
            raise self.refuse(node, "the paths through it give values of different shapes")
        if a.expression == b.expression:
            return a
        a, b = self.unified(a, b)
        return self.program.term("select", a.kind, (test, a, b), a.kind)

    # ── Expressions ──────────────────────────────────────────────────────

    def read(self, node: ast.AST, name: str) -> Value:
        value = self.env[name]
        if value.__class__ is _Unbound:
            raise self.refuse(node, f"`{name}` is assigned on one path through an if only")
        return value

    def value(self, node: ast.AST) -> Value:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (bool, int, float)):
                return self.number(node.value, node)
            raise self.refuse(node, f"the constant {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id in self.env:
                return self.read(node, node.id)
            return self.outside(node)
        if isinstance(node, ast.Attribute):
            return self.outside(node)
        if isinstance(node, ast.BinOp):
            if self.numeric(node) is not None and (folded := self.folded(node)) is not None:
                return self.number(folded, node)
            left, right = self.value(node.left), self.value(node.right)
            return self.arithmetic(node.op, left, right, node, node.right)
        if isinstance(node, ast.UnaryOp):
            if self.numeric(node) is not None and (folded := self.folded(node)) is not None:
                return self.number(folded, node)
            if isinstance(node.op, ast.Not):
                return self.test(node)
            operand = self.value(node.operand)
            if isinstance(node.op, ast.USub):
                value = self.integer(self.scalar(operand, node.operand))
                return self.program.term("neg", value.kind, (value,), value.kind, ("add",))
            if isinstance(node.op, ast.UAdd):
                return self.scalar(operand, node.operand)
            raise self.refuse(node, f"the operator {type(node.op).__name__}")
        if isinstance(node, ast.BoolOp):
            return self.boolean(node)
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.IfExp):
            test = self.test(node.test)
            return self.select(test, self.value(node.body), self.value(node.orelse), node)
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, (ast.Tuple, ast.List)):
            if any(isinstance(e, ast.Starred) for e in node.elts):
                raise self.refuse(node, "a starred item")
            return tuple(self.value(e) for e in node.elts)
        if isinstance(node, ast.Call):
            return self.invoke(node)
        if isinstance(node, ast.NamedExpr):
            value = self.value(node.value)
            self.assign(node.target, value)
            self.bind(node.target, node.value)
            return self.env[node.target.id]
        raise self.refuse(node, f"a {type(node).__name__} expression")

    def number(self, value: Any, node: ast.AST) -> _Scalar:
        if isinstance(value, int) and not isinstance(value, bool):
            if not -_INT_LIMIT <= value < _INT_LIMIT:
                raise self.refuse(node, f"the integer {value}, which no 64-bit integer holds")
        elif not isinstance(value, (bool, float)):
            raise self.refuse(node, f"the value {value!r}")
        return _const(value)

    def folded(self, node: ast.AST) -> Any:
        """The value of an expression the cost model folds (``numeric``),
        as Python computes it; None when a name in it holds no constant on
        this path (the cost model's constants hold on any)."""
        if isinstance(node, ast.BinOp) and type(node.op) in _FOLDS:
            left, right = self.folded(node.left), self.folded(node.right)
            if left is None or right is None:
                return None
            return _FOLDS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.folded(node.operand)
            if value is None:
                return None
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.Name) and node.id in self.env:
            return _constant_of(self.env[node.id])
        return self.constant(node)

    def outside(self, node: ast.AST) -> Value:
        """A name or attribute from outside the body: a number, a tuple of
        numbers, or a function or module."""
        try:
            value = self.resolve(node)
        except (_NotAnalyzableError, AttributeError):
            raise self.refuse(node, f"`{ast.unparse(node)}`, which can't be followed") from None
        return self.from_object(value, node)

    def from_object(self, value: Any, node: ast.AST) -> Value:
        if isinstance(value, (bool, int, float)):
            return self.number(value, node)
        if isinstance(value, (tuple, list)):
            return tuple(self.from_object(v, node) for v in value)
        return _Object(value)

    def scalar(self, value: Value, node: ast.AST) -> _Scalar:
        if value.__class__ is not _Scalar:
            raise self.refuse(node, f"`{ast.unparse(node)}` is not a number")
        found: _Scalar = value
        return found

    def integer(self, value: _Scalar) -> _Scalar:
        """*value*, a bool as Python counts it in arithmetic: 1 or 0."""
        if value.kind != "bool":
            return value
        known = _constant_of(value)
        if known is not None:
            return _const(int(known))
        return self.program.term("select", "int", (value, _const(1), _const(0)), "int")

    def floating(self, value: _Scalar) -> _Scalar:
        """*value* as a float, as Python converts an integer operand."""
        value = self.integer(value)
        if value.kind == "float":
            return value
        known = _constant_of(value)
        if known is not None and abs(known) <= _EXACT_INT:
            return _const(float(known))
        return self.program.term("to_float", "int", (value,), "float")

    def unified(self, a: _Scalar, b: _Scalar) -> tuple[_Scalar, _Scalar]:
        """*a* and *b* in one type: bools as integers beside numbers, and
        integers as floats beside floats."""
        if a.kind == b.kind:
            return a, b
        a, b = self.integer(a), self.integer(b)
        if a.kind == b.kind:
            return a, b
        return self.floating(a), self.floating(b)

    def test(self, node: ast.AST) -> _Scalar:
        """Whether the test *node* holds: an ``and`` or ``or`` of tests
        combines their truths, whatever values Python's would give."""
        if isinstance(node, ast.BoolOp):
            truths = [self.test(v) for v in node.values]
            self.program.classes["compare"] += len(truths) - 1
            out = truths[-1]
            for truth in reversed(truths[:-1]):
                pair = (out, _const(False)) if isinstance(node.op, ast.And) else (_const(True), out)
                out = self.program.term("select", "bool", (truth, *pair), "bool")
            return out
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            truth = self.test(node.operand)
            return self.program.term("ne", "bool", (truth, _const(True)), "bool", ("compare",))
        return self.truth(self.value(node), node)

    def truth(self, value: Value, node: ast.AST) -> _Scalar:
        """Whether *value* is true, as a bool: a number is true when nonzero."""
        number = self.scalar(value, node)
        if number.kind == "bool":
            return number
        zero = _const(0.0 if number.kind == "float" else 0)
        return self.program.term("ne", number.kind, (number, zero), "bool")

    def arithmetic(
        self, op: ast.operator, a: Value, b: Value, node: ast.AST, right: ast.AST
    ) -> _Scalar:
        """``a <op> b`` as Python computes it on numbers; *right* is the
        right operand's source, which the cost model classes a division by."""
        name = _OPERATORS.get(type(op))
        if name is None:
            raise self.refuse(node, f"the operator {type(op).__name__}")
        a = self.integer(self.scalar(a, node))
        b = self.integer(self.scalar(b, right))
        if name == "pow":
            return self.power(a, b, node, right)
        kind = self.binary_class(op, right)
        if name == "div":
            a, b = self.floating(a), self.floating(b)
        elif name == "floor_div" and "float" in (a.kind, b.kind):
            raise self.refuse(node, "a floor division of floats")
        else:
            a, b = self.unified(a, b)
        return self.program.term(name, a.kind, (a, b), a.kind, (kind,))

    def binary_class(self, op: ast.operator, right: ast.AST) -> str:
        kind = _Walker.binary(self, op, right)
        assert kind is not None
        return kind

    def power(self, a: _Scalar, b: _Scalar, node: ast.AST, right: ast.AST) -> _Scalar:
        exponent = self.constant(right)
        small = isinstance(exponent, int) and not isinstance(exponent, bool) and 0 <= exponent <= 16
        if a.kind == "int" and b.kind == "int":
            if not small:
                raise self.refuse(node, "an integer power whose exponent is not a constant 0 to 16")
            return self.repeated(a, int(exponent))
        classes = (
            ["mul"] * max(exponent.bit_length() + bin(exponent).count("1") - 2, 0)
            if small
            else ["log", "mul", "exp"]
        )
        a, b = self.floating(a), self.floating(b)
        return self.program.term("pow", "float", (a, b), "float", classes)

    def repeated(self, base: _Scalar, exponent: int) -> _Scalar:
        """An integer to a constant power, by squaring and multiplying, as
        the cost model counts it."""
        if exponent == 0:
            return _const(1)
        result: _Scalar | None = None
        square = base
        while True:
            if exponent & 1:
                result = (
                    square
                    if result is None
                    else self.program.term("mul", "int", (result, square), "int", ("mul",))
                )
            exponent >>= 1
            if not exponent:
                break
            square = self.program.shared(square)
            square = self.program.term("mul", "int", (square, square), "int", ("mul",))
            if result is not None:
                result = self.program.shared(result)
        assert result is not None
        return result

    def compare(self, node: ast.Compare) -> _Scalar:
        left = self.value(node.left)
        found: _Scalar | None = None
        operands = [node.left, *node.comparators]
        for k, (op, comparator) in enumerate(zip(node.ops, node.comparators, strict=True)):
            name = _COMPARED.get(type(op))
            if name is None:
                raise self.refuse(node, f"the comparison {type(op).__name__}")
            right = self.value(comparator)
            if k + 1 < len(node.ops):
                right = self.program.shared(self.scalar(right, comparator))
            a, b = self.scalar(left, operands[k]), self.scalar(right, comparator)
            if a.kind == "bool" and b.kind == "bool" and name not in ("eq", "ne"):
                a, b = self.integer(a), self.integer(b)
            a, b = self.unified(a, b)
            pair = self.program.term(name, a.kind, (a, b), "bool", ("compare",))
            found = (
                pair
                if found is None
                else self.program.term("select", "bool", (found, pair, _const(False)), "bool")
            )
            left = right
        assert found is not None
        return found

    def boolean(self, node: ast.BoolOp) -> _Scalar:
        """``a and b`` (``or``): Python's value, the first operand that is
        false (true), else the last."""
        values = [self.scalar(self.value(v), v) for v in node.values]
        conjunction = isinstance(node.op, ast.And)
        out = values[-1]
        for value, source in zip(reversed(values[:-1]), reversed(node.values[:-1]), strict=True):
            self.program.classes["compare"] += 1
            if value.kind == "bool" and out.kind == "bool":
                pair = (out, _const(False)) if conjunction else (_const(True), out)
                out = self.program.term("select", "bool", (value, *pair), "bool")
                continue
            value, out = self.unified(value, out)
            value = self.program.shared(value)
            truth = self.truth(value, source)
            pair = (out, value) if conjunction else (value, out)
            out = self.program.term("select", value.kind, (truth, *pair), value.kind)
        return out

    def subscript(self, node: ast.Subscript) -> Value:
        base = self.sequence(self.value(node.value), node.value)
        if isinstance(node.slice, ast.Slice):
            bounds = [
                None if part is None else self.index(self.value(part), part)
                for part in (node.slice.lower, node.slice.upper, node.slice.step)
            ]
            return base[slice(*bounds)]
        at = self.index(self.value(node.slice), node.slice)
        if not -len(base) <= at < len(base):
            raise self.refuse(node, f"the index {at} of a sequence of {len(base)}")
        return base[at]

    def index(self, value: Value, node: ast.AST) -> int:
        """An index the body computes from constants, as a number."""
        value = self.scalar(value, node)
        known = _constant_of(value)
        if known is None and value.kind == "int" and _constants_only(value.expression):
            from .controller import evaluate_expression

            known = evaluate_expression(value.expression)
        if not isinstance(known, int):
            raise self.refuse(node, "an index known only when the function runs")
        return int(known)

    # ── Calls ────────────────────────────────────────────────────────────

    def invoke(self, node: ast.Call) -> Value:
        if any(isinstance(a, ast.Starred) for a in node.args) or any(
            k.arg is None for k in node.keywords
        ):
            raise self.refuse(node, "a call with *arguments or **keywords")
        if isinstance(node.func, ast.Name) and node.func.id in self.env:
            raise self.refuse(node, f"a call of `{node.func.id}`, which can't be followed")
        target = self.outside(node.func)
        if target.__class__ is not _Object:
            raise self.refuse(node, f"a call of `{ast.unparse(node.func)}`, which is a number")
        fn = target.value
        args = [self.value(a) for a in node.args]
        keywords = {k.arg: self.value(k.value) for k in node.keywords if k.arg is not None}
        name = getattr(fn, "__name__", "")
        if fn is as_nanoseconds and len(args) == 1 and not keywords:
            return args[0]
        if getattr(fn, "__module__", None) == "math" or (
            isinstance(fn, types.BuiltinFunctionType) and getattr(math, name, None) is fn
        ):
            return self.math(name, args, keywords, node)
        if fn in (min, max) and not keywords:
            return self.extreme(fn is max, args, node)
        if fn is len and len(args) == 1:
            return _const(len(self.sequence(args[0], node.args[0])))
        if fn is abs and len(args) == 1:
            return self.absolute(self.integer(self.scalar(args[0], node.args[0])), ("add",))
        if fn is float and len(args) == 1:
            value = self.scalar(args[0], node.args[0])
            self.program.classes["convert"] += 1
            return self.floating(value)
        if fn is int and len(args) == 1:
            value = self.integer(self.scalar(args[0], node.args[0]))
            if value.kind == "int":
                self.program.classes["convert"] += 1
                return value
            return self.program.term("to_int", "float", (value,), "int", ("convert",))
        if fn is bool and len(args) == 1:
            self.program.classes["compare"] += 1
            return self.truth(args[0], node.args[0])
        wrapped = getattr(fn, "implementation", None) or getattr(fn, "__wrapped__", None)
        if wrapped is not None and not isinstance(fn, types.FunctionType):
            fn = wrapped
        if not isinstance(fn, types.FunctionType):
            raise self.refuse(
                node, f"a call of `{ast.unparse(node.func)}`, which can't be followed"
            )
        return self.inline(fn, args, keywords, node)

    def math(
        self, name: str, args: list[Value], keywords: Mapping[str, Value], node: ast.Call
    ) -> _Scalar:
        if keywords:
            raise self.refuse(node, f"math.{name} with keywords")
        values = [self.scalar(a, n) for a, n in zip(args, node.args, strict=True)]
        classes = _MATH.get(name, ())
        if name in _UNARY_MATH and len(values) == 1:
            x = self.floating(values[0])
            term = _UNARY_MATH[name]
            if term in ("floor", "ceil"):
                rounded = self.program.term(term, "float", (x,), "float", classes)
                return self.program.term("to_int", "float", (rounded,), "int")
            return self.program.term(term, "float", (x,), "float", classes)
        if name == "atan2" and len(values) == 2:
            y, x = self.floating(values[0]), self.floating(values[1])
            return self.program.term("atan2", "float", (y, x), "float", classes)
        if name in ("isnan", "isinf", "isfinite") and len(values) == 1:
            self.program.classes["compare"] += 1
            x = self.program.shared(self.floating(values[0]))
            if name == "isnan":
                return self.program.term("ne", "float", (x, x), "bool")
            inf, low = _const(math.inf), _const(-math.inf)
            if name == "isinf":
                above = self.program.term("eq", "float", (x, inf), "bool")
                below = self.program.term("eq", "float", (x, low), "bool")
                return self.program.term("select", "bool", (above, _const(True), below), "bool")
            under = self.program.term("lt", "float", (x, inf), "bool")
            over = self.program.term("gt", "float", (x, low), "bool")
            return self.program.term("select", "bool", (under, over, _const(False)), "bool")
        raise self.refuse(node, f"math.{name}, which a controller program does not have")

    def absolute(self, value: _Scalar, classes: Sequence[str]) -> _Scalar:
        if value.kind == "float":
            return self.program.term("abs", "float", (value,), "float", classes)
        value = self.program.shared(value)
        negative = self.program.term("lt", "int", (value, _const(0)), "bool")
        flipped = self.program.term("neg", "int", (value,), "int", classes)
        return self.program.term("select", "int", (negative, flipped, value), "int")

    def extreme(self, largest: bool, args: list[Value], node: ast.Call) -> _Scalar:
        """``max(...)`` (``min``): Python keeps the first of equal values, and
        a later value replaces it only when greater (less)."""
        if len(args) == 1:
            items = list(self.sequence(args[0], node.args[0]))
            sources: list[ast.AST] = [node.args[0]] * len(items)
        else:
            items, sources = args, list(node.args)
        if not items:
            raise self.refuse(node, f"`{ast.unparse(node)}` of nothing")
        values = [self.scalar(v, s) for v, s in zip(items, sources, strict=True)]
        out = self.integer(values[0])
        for value in values[1:]:
            a, b = self.unified(self.integer(value), out)
            out = self.program.term(
                "max" if largest else "min", a.kind, (a, b), a.kind, ("compare",)
            )
        return out

    def inline(
        self,
        fn: types.FunctionType,
        args: list[Value],
        keywords: Mapping[str, Value],
        node: ast.Call,
    ) -> Value:
        """A helper's body, run on *args*: its parameters are the values
        passed, each computed once."""
        if fn in self.stack or fn is self.fn:
            raise self.refuse(node, f"recursion into {fn.__name__}")
        try:
            tree = _source_tree(fn)
        except (OSError, TypeError, SyntaxError, _NotAnalyzableError) as error:
            raise self.refuse(node, f"{fn.__name__}'s source is unavailable ({error})") from None
        arguments = tree.args
        if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
            raise self.refuse(
                node, f"{fn.__name__} takes *args, **kwargs or keyword-only arguments"
            )
        names = [a.arg for a in (*arguments.posonlyargs, *arguments.args)]
        if len(args) > len(names):
            raise self.refuse(node, f"{len(args)} arguments for {fn.__name__}")
        defaults = fn.__defaults__ or ()
        prefix = f"{self.prefix}{fn.__name__}."
        env: dict[str, Value] = {}
        sizes: dict[str, int] = {}
        program = self.program
        for k, name in enumerate(names):
            if k < len(args):
                value = args[k]
            elif name in keywords:
                value = keywords[name]
            elif k >= len(names) - len(defaults):
                value = self.from_object(defaults[k - (len(names) - len(defaults))], node)
            else:
                raise self.refuse(node, f"{fn.__name__} is not given `{name}`")
            if value.__class__ is _Scalar:
                value = program.named(prefix + name, value)
            elif value.__class__ is tuple:
                sizes[name] = len(value)
            env[name] = value
        unknown = set(keywords) - set(names)
        if unknown:
            raise self.refuse(node, f"{fn.__name__} has no parameter {sorted(unknown)[0]}")
        helper = _Lowerer(fn, sizes, (*self.stack, self.fn), program, env, prefix)
        helper.names |= set(names)
        return helper.run(list(tree.body))


def _constants_only(expression: Mapping[str, Any]) -> bool:
    if "const" in expression:
        return True
    if "op" not in expression:
        return False
    return all(_constants_only(x) for x in expression["of"])


# ── Entry point ─────────────────────────────────────────────────────────────


def _parameter_kind(spec: Any) -> tuple[str, int | None] | None:
    """The value type of an argument's numbers and its length (None for a
    scalar); None for an argument a controller program does not take."""
    descriptor = spec.descriptor
    if spec.is_scalar:
        if descriptor == "bits:1":
            return "int", None
        kind = _SCALAR_TYPES.get(descriptor)
        return None if kind is None else (kind, None)
    head, colon, _ = descriptor.partition(":")
    if not colon:
        head = descriptor[: descriptor.index("[")]
    kind = _ITEM_TYPES.get(head)
    return None if kind is None else (kind, int(spec.count))


_cache: weakref.WeakKeyDictionary[Any, Lowered | str] = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def lower(function: Any) -> Lowered:
    """*function*'s program; raises ``UnsupportedError`` naming the line and
    the construct for a body outside what a controller program holds."""
    with _lock:
        kept = _cache.get(function)
    if kept is None:
        try:
            kept = _lower(function)
        except _RefusedError as error:
            kept = str(error)
        with _lock:
            _cache[function] = kept
    if isinstance(kept, str):
        raise UnsupportedError(
            f"{function.name} can't be lowered to a controller program: {kept}. On a link no "
            "processor serves, it runs as before"
        )
    return kept


def _lower(function: Any) -> Lowered:
    from ._controller_cost import _value_lengths

    fn = function.implementation
    if fn is None or function.uniform is not None:
        raise _RefusedError("it has no body to read")
    sizes = _value_lengths(function)
    cost = analyze_callable(fn, sizes)
    if cost.violations:
        more = len(cost.violations) - 4
        raise _RefusedError(
            "; ".join(cost.violations[:4]) + (f" (and {more} more)" if more > 0 else "")
        )
    try:
        tree = _source_tree(fn)
    except (OSError, TypeError, SyntaxError, _NotAnalyzableError) as error:
        raise _RefusedError(f"its source is unavailable ({error})") from None
    arguments = tree.args
    names = [a.arg for a in (*arguments.posonlyargs, *arguments.args)]
    env: dict[str, Value] = {}
    descriptors = [spec.descriptor for spec in function.args]
    for k, (name, spec) in enumerate(zip(names, function.args, strict=False)):
        found = _parameter_kind(spec)
        if found is None:
            raise _RefusedError(
                f"{fn.__name__}: the argument {name} ({spec.descriptor}), which a controller "
                "program does not take"
            )
        kind, count = found
        if count is None:
            env[name] = _Scalar({"argument": k}, kind)
        else:
            env[name] = tuple(_Scalar({"argument": k, "index": i}, kind) for i in range(count))
    program = _Program()
    lowerer = _Lowerer(fn, sizes, (), program, env, "")
    lowerer.names |= set(names)
    returned = lowerer.run(list(tree.body))
    results = list(function.results)
    values = list(returned) if len(results) > 1 else [returned]
    if returned.__class__ is tuple and len(results) == 1 and len(returned) != 1:
        raise _RefusedError(f"{fn.__name__}: it returns {len(returned)} values for one result")
    if len(values) != len(results):
        raise _RefusedError(f"{fn.__name__}: it returns {len(values)} values for {len(results)}")
    out = []
    for k, (value, spec) in enumerate(zip(values, results, strict=True)):
        if value.__class__ is not _Scalar:
            raise _RefusedError(f"{fn.__name__}: result {k} is not a number")
        out.append(_result(lowerer, value, spec.descriptor, k).expression)
    entry = {
        "arguments": descriptors,
        "results": [spec.descriptor for spec in results],
        "number_format": FLOAT64,
        "body": {"let": program.lets, "results": out},
    }
    return Lowered(entry, dict(program.classes), program.counted)


def _result(lowerer: _Lowerer, value: _Scalar, descriptor: str, k: int) -> _Scalar:
    """*value* as result *k* of type *descriptor* takes it."""
    if descriptor in ("f64", "f32"):
        return lowerer.floating(value)
    if descriptor in ("i64", "i32", "u8"):
        if value.kind == "float":
            raise _RefusedError(
                f"{lowerer.fn.__name__}: result {k} is a float, and it is declared {descriptor}"
            )
        return lowerer.integer(value)
    if descriptor == "bits:1":
        if value.kind == "bool":
            return value
        return lowerer.program.term(
            "ne", value.kind, (value, _const(0.0 if value.kind == "float" else 0)), "bool"
        )
    raise _RefusedError(
        f"{lowerer.fn.__name__}: result {k} ({descriptor}), which a controller program does "
        "not have"
    )


def functions_table(lowered: Any, controller: int) -> dict[str, Any]:
    """The ``functions`` of the export of *lowered* for *controller*: each
    function a processor of it computes, by its compiled name. A function
    on a processor with ``fixed_cost_only`` must lower; on another, one
    whose body does not is left out (its processor runs it as it can)."""
    machine = lowered.options.machine
    system = machine._system
    placement = system._placement
    table: dict[str, Any] = {}
    for fn in lowered.classical:
        where = placement.get(machine.link_id(fn.link))
        if where is None or controller not in where or fn.implementation is None:
            continue
        processor = system.controllers[controller].processors[where[controller]]
        if processor.fixed_cost_only:
            table[fn.name] = lower(fn).entry
            continue
        try:
            table[fn.name] = lower(fn).entry
        except UnsupportedError:
            continue
    return table
