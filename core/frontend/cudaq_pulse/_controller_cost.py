# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""What a classical function costs a controller, from its body.

The body of a function a controller runs is read as Python source and walked
once: every arithmetic operation, comparison, branch, load and elementary
function is counted by class, through the helper functions it calls, with
loops multiplied by their trip counts. An expression of constants costs
nothing (it folds when the program compiles) and dividing by a constant is
a multiply. The walk also checks the body is in
*controller form*: a fixed number of scalars of state, loops of a trip count
known when the program compiles (``range`` of a constant, or a fixed-size
argument), no ``while``, recursion, allocation, exceptions, closures over
mutable state or calls it cannot follow. A helper with a
``controller_cost`` attribute (a function of its arguments' lengths giving
operations by class) is an intrinsic, counted by it instead of its body. See
``cudaq_pulse.controller``.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import math
import textwrap
import threading
import types
import weakref
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

# The classes operations are counted in.
OP_CLASSES = (
    "add",
    "mul",
    "div",
    "compare",
    "branch",
    "load",
    "convert",
    "popcount",
    "sqrt",
    "exp",
    "log",
    "trig",
)

_MATH = {
    "sqrt": ("sqrt",),
    "exp": ("exp",),
    "log": ("log",),
    "log2": ("log", "mul"),
    "log10": ("log", "mul"),
    "atan2": ("trig",),
    "atan": ("trig",),
    "cos": ("trig",),
    "sin": ("trig",),
    "tan": ("trig",),
    "floor": ("convert",),
    "ceil": ("convert",),
    "fabs": ("add",),
    "copysign": ("add",),
    "isnan": ("compare",),
    "isfinite": ("compare",),
    "isinf": ("compare",),
    "hypot": ("mul", "mul", "add", "sqrt"),
}
_BUILTINS = {"abs": ("add",), "float": ("convert",), "int": ("convert",), "bool": ("compare",)}
_BINARY = {
    ast.Add: "add",
    ast.Sub: "add",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "div",
    ast.Mod: "div",
    ast.BitAnd: "add",
    ast.BitOr: "add",
    ast.BitXor: "add",
    ast.LShift: "add",
    ast.RShift: "add",
}
# Operators an expression of constants folds through when the program compiles.
_FOLDS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: float(a) ** float(b),
}
# Each scalar a function holds is a 64-bit register.
_REGISTER_BYTES = 8


@dataclass(frozen=True)
class FunctionCost:
    """The static cost of one classical function's body.

    ``ops`` counts the operations of the longest path through the body by
    class (what one call executes at most); ``size`` counts every operation
    of the body, both arms of each branch (what the controller stores).
    ``locals`` is how many scalars the body and its helpers hold at once at
    most, beyond the arguments and results. ``violations`` lists what keeps
    the body out of controller form; empty when it is in it. ``helpers``
    names the functions it calls whose bodies were counted too.

    For a ``Processor``: ``iterations`` counts the loop iterations on the
    longest path (each also counted in ``ops`` as an ``add`` and a
    ``branch``, the loop's counter and jump); ``unrolled_size`` is ``size``
    with every loop unrolled; ``uses`` gives, by operation class (and
    ``"loop"``), where the body or a helper first uses it.
    """

    name: str
    ops: Mapping[str, int]
    size: int
    locals: int
    loops: int
    helpers: tuple[str, ...]
    violations: tuple[str, ...]
    iterations: int = 0
    unrolled_size: int = 0
    uses: Mapping[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Operations on the longest path."""
        return sum(self.ops.values())

    @property
    def in_controller_form(self) -> bool:
        return not self.violations

    def cycles(self, op_cycles: Mapping[str, int]) -> int:
        """Clock cycles of the longest path, at *op_cycles* per operation
        of each class (classes not given cost one cycle)."""
        return sum(n * int(op_cycles.get(kind, 1)) for kind, n in self.ops.items())

    def processor_cycles(self, cycles_per_op: Mapping[str, int], loops: bool) -> int:
        """Clock cycles of the longest path on a processor: ``cycles_per_op``
        by class (absent: 1), and a loop's counter and jump as one ``"loop"``
        per iteration if the processor *loops*, else nothing (unrolled)."""
        overhead = self.iterations
        total = 0
        for kind, n in self.ops.items():
            if kind == "add" or kind == "branch":
                n -= overhead
            total += n * int(cycles_per_op.get(kind, 1))
        if loops:
            total += overhead * int(cycles_per_op.get("loop", 1))
        return total


@dataclass
class _Cost:
    # The body's own operations, by class, on the longest path.
    ops: Counter[str] = field(default_factory=Counter)
    size: int = 0
    # Operations counted beside the body's own: loop counters and jumps,
    # inlined calls and the operations of the helpers they inline.
    placed: Counter[str] = field(default_factory=Counter)
    # Every class the body's own operations use, on any path.
    kinds: set[str] = field(default_factory=set)
    unrolled: int = 0
    iterations: int = 0

    @property
    def weight(self) -> int:
        return sum(self.ops.values()) + sum(self.placed.values())

    def add(self, kind: str, n: int = 1) -> None:
        self.ops[kind] += n
        self.size += n
        self.unrolled += n
        if n:
            self.kinds.add(kind)

    def extend(self, other: _Cost, times: int = 1) -> None:
        for kind, n in other.ops.items():
            self.ops[kind] += n * times
        for kind, n in other.placed.items():
            self.placed[kind] += n * times
        self.size += other.size
        self.unrolled += other.unrolled * times
        self.iterations += other.iterations * times
        self.kinds |= other.kinds


class _NotAnalyzableError(Exception):
    pass


def _source_tree(fn: Callable[..., Any]) -> ast.FunctionDef:
    lines, start = inspect.getsourcelines(fn)
    module = ast.parse(textwrap.dedent("".join(lines)))
    ast.increment_lineno(module, max(start - 1, 0))
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if isinstance(node, ast.AsyncFunctionDef):
                raise _NotAnalyzableError("an async function")
            return node
    raise _NotAnalyzableError("no function definition in its source")


class _Walker:
    """Counts one function body, inlining the helpers it calls."""

    def __init__(self, fn: Callable[..., Any], sizes: Mapping[str, int], stack: tuple[Any, ...]):
        self.fn = fn
        self.sizes = dict(sizes)  # name -> length, for fixed-size sequences
        self.stack = stack
        self.violations: list[str] = []
        self.helpers: list[str] = []
        self.names: set[str] = set()
        self.loops = 0
        self.uses: dict[str, str] = {}
        self.extra_locals = 0
        self.constants: dict[str, Any] = {}
        self.globals = getattr(fn, "__globals__", {})
        closure = getattr(fn, "__closure__", None) or ()
        free = getattr(getattr(fn, "__code__", None), "co_freevars", ())
        self.closure = {n: c.cell_contents for n, c in zip(free, closure) if _has_contents(c)}

    def where(self, node: ast.AST) -> str:
        return f"{self.fn.__name__} line {getattr(node, 'lineno', '?')}"

    def flag(self, node: ast.AST, what: str) -> None:
        self.violations.append(f"{self.where(node)}: {what}")

    # ── Name resolution ──────────────────────────────────────────────────

    def resolve(self, node: ast.AST) -> Any:
        """The object a name or attribute chain refers to, outside the body."""
        if isinstance(node, ast.Name):
            if node.id in self.names:
                raise _NotAnalyzableError(node.id)
            if node.id in self.closure:
                return self.closure[node.id]
            if node.id in self.globals:
                return self.globals[node.id]
            if hasattr(builtins, node.id):
                return getattr(builtins, node.id)
            raise _NotAnalyzableError(node.id)
        if isinstance(node, ast.Attribute):
            return getattr(self.resolve(node.value), node.attr)
        raise _NotAnalyzableError(type(node).__name__)

    def constant(self, node: ast.AST) -> Any:
        """The value of *node* when it is known without running the body."""
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name) and node.id in self.constants:
            return self.constants[node.id]
        if isinstance(node, (ast.Name, ast.Attribute)):
            try:
                value = self.resolve(node)
            except (_NotAnalyzableError, AttributeError):
                return None
            return value if isinstance(value, (int, float, bool)) else None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "len" and len(node.args) == 1:
                return self.length(node.args[0])
        if isinstance(node, ast.BinOp):
            left, right = self.constant(node.left), self.constant(node.right)
            if isinstance(left, int) and isinstance(right, int):
                op = type(node.op)
                if op is ast.Add:
                    return left + right
                if op is ast.Sub:
                    return left - right
                if op is ast.Mult:
                    return left * right
                if op is ast.FloorDiv and right:
                    return left // right
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = self.constant(node.operand)
            return -value if isinstance(value, (int, float)) else None
        return None

    def numeric(self, node: ast.AST) -> int | float | None:
        """The number *node* folds to when the program compiles, if any."""
        if isinstance(node, ast.BinOp) and type(node.op) in _FOLDS:
            left, right = self.numeric(node.left), self.numeric(node.right)
            if left is None or right is None:
                return None
            try:
                value = _FOLDS[type(node.op)](left, right)
            except (ArithmeticError, ValueError):
                return None
            return value if isinstance(value, (int, float)) and math.isfinite(value) else None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            value = self.numeric(node.operand)
            if value is None:
                return None
            return -value if isinstance(node.op, ast.USub) else value
        value = self.constant(node)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        return None

    def binary(self, op: ast.operator, right: ast.AST) -> str | None:
        """The class of ``left <op> right``: dividing by a constant is a
        multiply by its reciprocal."""
        if isinstance(op, ast.Div) and self.numeric(right) not in (None, 0):
            return "mul"
        return _BINARY.get(type(op))

    def length(self, node: ast.AST) -> int | None:
        if isinstance(node, ast.Name) and node.id in self.sizes:
            return self.sizes[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            return len(node.elts)
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Slice):
            whole = self.length(node.value)
            lo = self.constant(node.slice.lower) if node.slice.lower is not None else 0
            hi = self.constant(node.slice.upper) if node.slice.upper is not None else whole
            if whole is not None and isinstance(lo, int) and isinstance(hi, int):
                return max(0, min(hi, whole) - max(lo, 0))
        return None

    # ── Statements ───────────────────────────────────────────────────────

    def body(self, statements: list[ast.stmt]) -> _Cost:
        cost = _Cost()
        for statement in statements:
            done = self.statement(statement)
            for kind in done.kinds - self.uses.keys():
                self.uses[kind] = self.where(statement)
            cost.extend(done)
        return cost

    def statement(self, node: ast.stmt) -> _Cost:
        cost = _Cost()
        if isinstance(node, ast.Expr):
            if isinstance(node.value, ast.Constant):  # a docstring
                return cost
            return self.expression(node.value)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if value is not None:
                cost.extend(self.expression(value))
            for target in targets:
                self.bind(target, value)
            return cost
        if isinstance(node, ast.AugAssign):
            cost.extend(self.expression(node.value))
            kind = self.binary(node.op, node.value)
            if kind is None:
                self.flag(node, f"the operator {type(node.op).__name__}")
            else:
                cost.add(kind)
            if isinstance(node.target, ast.Subscript):
                self.flag(node, "a store into a sequence (allocation)")
            self.bind(node.target, None)
            return cost
        if isinstance(node, ast.Return):
            if node.value is not None:
                cost.extend(self.results(node.value))
            return cost
        if isinstance(node, ast.If):
            test = self.expression(node.test)
            test.add("branch")
            then, other = self.body(node.body), self.body(node.orelse)
            return _branch(test, then, other)
        if isinstance(node, ast.For):
            return self.loop(node)
        if isinstance(node, ast.While):
            self.flag(node, "a while loop (its trip count is not fixed)")
            return cost
        if isinstance(node, ast.Pass):
            return cost
        if isinstance(node, (ast.Raise, ast.Try, ast.Assert)):
            self.flag(node, "an exception (report failure through a result)")
            return cost
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            self.flag(node, "state kept between calls (pass it in and out)")
            return cost
        if isinstance(node, (ast.Break, ast.Continue)):
            self.flag(node, "an early exit from a loop (use a flag)")
            return cost
        self.flag(node, f"a {type(node).__name__} statement")
        return cost

    def bind(self, target: ast.AST, value: ast.AST | None) -> None:
        if isinstance(target, ast.Name):
            self.names.add(target.id)
            known = self.constant(value) if value is not None else None
            if isinstance(known, int) and not isinstance(known, bool):
                self.constants[target.id] = known
            else:
                self.constants.pop(target.id, None)
            if value is not None:
                length = self.length(value)
                if length is not None and isinstance(value, (ast.Name, ast.Subscript)):
                    self.sizes[target.id] = length
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = value.elts if isinstance(value, (ast.Tuple, ast.List)) else None
            for k, element in enumerate(target.elts):
                self.bind(element, values[k] if values and k < len(values) else None)
        else:
            self.flag(target, "a store into a sequence or attribute (allocation)")

    def loop(self, node: ast.For) -> _Cost:
        cost = _Cost()
        trips: int | None = None
        iterated = node.iter
        if (
            isinstance(iterated, ast.Call)
            and isinstance(iterated.func, ast.Name)
            and iterated.func.id == "range"
            and not iterated.keywords
        ):
            bounds = [self.constant(a) for a in iterated.args]
            if all(isinstance(b, int) for b in bounds) and 1 <= len(bounds) <= 3:
                trips = len(range(*bounds))
        elif (
            isinstance(iterated, ast.Call)
            and isinstance(iterated.func, ast.Name)
            and iterated.func.id == "enumerate"
            and len(iterated.args) == 1
        ):
            trips = self.length(iterated.args[0])
            if trips is not None:
                cost.add("load", trips)
        else:
            trips = self.length(iterated)
            if trips is not None:
                cost.add("load", trips)
        if trips is None:
            self.flag(node, "a loop whose trip count is not known when the program compiles")
            trips = 1
        if node.orelse:
            self.flag(node, "a for-else")
        self.loops += 1
        self.bind(node.target, None)
        body = self.body(node.body)
        # The counter and the jump back.
        body.placed["add"] += 1
        body.placed["branch"] += 1
        body.size += 2
        body.iterations += 1
        self.uses.setdefault("loop", self.where(node))
        # A controller stores a loop's body once.
        cost.extend(body, trips)
        return cost

    # ── Expressions ──────────────────────────────────────────────────────

    def results(self, node: ast.AST) -> _Cost:
        """A returned value: a list or tuple there is the fixed set of
        results, not an allocation."""
        if isinstance(node, (ast.Tuple, ast.List)):
            cost = _Cost()
            for element in node.elts:
                cost.extend(self.results(element))
            return cost
        return self.expression(node)

    def expression(self, node: ast.AST) -> _Cost:
        cost = _Cost()
        if isinstance(node, (ast.Constant, ast.Name)):
            return cost
        if isinstance(node, ast.Attribute):
            if self.constant(node) is None:
                try:
                    self.resolve(node)
                except (_NotAnalyzableError, AttributeError):
                    self.flag(node, f"the attribute {node.attr}")
            return cost
        if isinstance(node, ast.BinOp):
            if self.numeric(node) is not None:
                return cost
            cost.extend(self.expression(node.left))
            cost.extend(self.expression(node.right))
            if isinstance(node.op, ast.Pow):
                exponent = self.constant(node.right)
                if isinstance(exponent, int) and 0 <= exponent <= 16:
                    cost.add("mul", max(exponent.bit_length() + bin(exponent).count("1") - 2, 0))
                else:
                    cost.add("log")
                    cost.add("mul")
                    cost.add("exp")
                return cost
            kind = self.binary(node.op, node.right)
            if kind is None:
                self.flag(node, f"the operator {type(node.op).__name__}")
            else:
                cost.add(kind)
            return cost
        if isinstance(node, ast.UnaryOp):
            cost.extend(self.expression(node.operand))
            cost.add("compare" if isinstance(node.op, ast.Not) else "add")
            return cost
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                cost.extend(self.expression(value))
            cost.add("compare", len(node.values) - 1)
            return cost
        if isinstance(node, ast.Compare):
            cost.extend(self.expression(node.left))
            for comparator in node.comparators:
                cost.extend(self.expression(comparator))
            cost.add("compare", len(node.ops))
            for op in node.ops:
                if isinstance(op, (ast.In, ast.NotIn, ast.Is, ast.IsNot)):
                    self.flag(node, f"the comparison {type(op).__name__}")
            return cost
        if isinstance(node, ast.IfExp):
            test = self.expression(node.test)
            test.add("branch")
            return _branch(test, self.expression(node.body), self.expression(node.orelse))
        if isinstance(node, ast.Subscript):
            cost.extend(self.expression(node.value))
            if isinstance(node.slice, ast.Slice):
                # A constant range of a fixed-size sequence is a view of it.
                if self.length(node) is None:
                    self.flag(node, "a slice whose bounds are not known when the program compiles")
            else:
                cost.extend(self.expression(node.slice))
                cost.add("load")
            return cost
        if isinstance(node, ast.Tuple):
            for element in node.elts:
                cost.extend(self.expression(element))
            return cost
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, (ast.List, ast.Dict, ast.Set)):
            self.flag(node, f"a {type(node).__name__.lower()} (allocation)")
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            self.flag(node, "a comprehension (allocation; use a loop)")
        elif isinstance(node, ast.Lambda):
            self.flag(node, "a lambda")
        elif isinstance(node, (ast.JoinedStr, ast.FormattedValue)):
            self.flag(node, "a string")
        elif isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
            self.flag(node, "a generator")
        elif isinstance(node, ast.NamedExpr):
            cost.extend(self.expression(node.value))
            self.bind(node.target, node.value)
        else:
            self.flag(node, f"a {type(node).__name__} expression")
        return cost

    def call(self, node: ast.Call) -> _Cost:
        cost = _Cost()
        if node.keywords and any(k.arg is None for k in node.keywords):
            self.flag(node, "a call with **keywords")
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                self.flag(node, "a call with *arguments")
                continue
            cost.extend(self.expression(arg))
        for keyword in node.keywords:
            cost.extend(self.expression(keyword.value))
        try:
            target = self.resolve(node.func)
        except (_NotAnalyzableError, AttributeError):
            self.flag(node, f"a call of {ast.unparse(node.func)}, which cannot be followed")
            return cost
        if target is math or isinstance(target, types.ModuleType):
            self.flag(node, f"a call of {ast.unparse(node.func)}")
            return cost
        name = getattr(target, "__name__", "")
        intrinsic = getattr(target, "controller_cost", None)
        if callable(intrinsic):
            lengths = [self.length(arg) for arg in node.args]
            if any(n is None for n in lengths):
                self.flag(node, f"{name} of a sequence of unknown size")
                return cost
            for kind, n in intrinsic(*lengths).items():
                cost.add(kind, n)
            return cost
        if getattr(target, "__module__", None) == "math" or (
            isinstance(target, types.BuiltinFunctionType) and getattr(math, name, None) is target
        ):
            kinds = _MATH.get(name)
            if kinds is None:
                self.flag(node, f"math.{name}, which a controller does not provide")
            else:
                for kind in kinds:
                    cost.add(kind)
            return cost
        if target in (min, max):
            n = len(node.args)
            if n == 1:
                length = self.length(node.args[0])
                if length is None:
                    self.flag(node, f"{name} of a sequence of unknown size")
                    return cost
                cost.add("load", length)
                n = length
            cost.add("compare", max(n - 1, 0))
            return cost
        if target is len:
            if not node.args or self.length(node.args[0]) is None:
                self.flag(node, "len of a sequence of unknown size")
            return cost
        if name in _BUILTINS and getattr(builtins, name, None) is target:
            for kind in _BUILTINS[name]:
                cost.add(kind)
            return cost
        wrapped = getattr(target, "implementation", None) or getattr(target, "__wrapped__", None)
        if wrapped is not None and not isinstance(target, types.FunctionType):
            target = wrapped
        if not isinstance(target, types.FunctionType):
            self.flag(node, f"a call of {ast.unparse(node.func)}, which cannot be followed")
            return cost
        if target in self.stack:
            self.flag(node, f"recursion into {target.__name__}")
            return cost
        sizes = {}
        try:
            parameters = list(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            parameters = []
        for parameter, arg in zip(parameters, node.args):
            length = self.length(arg)
            if length is not None:
                sizes[parameter] = length
        for keyword in node.keywords:
            if keyword.arg is not None:
                length = self.length(keyword.value)
                if length is not None:
                    sizes[keyword.arg] = length
        helper = analyze_callable(target, sizes, (*self.stack, self.fn))
        self.helpers.append(f"{target.__module__}.{target.__qualname__}")
        self.helpers.extend(helper.helpers)
        self.violations.extend(helper.violations)
        self.loops += helper.loops
        self.extra_locals = max(self.extra_locals, helper.locals + len(parameters))
        for kind, where in helper.uses.items():
            self.uses.setdefault(kind, where)
        cost.placed["branch"] += 1  # the call, inlined
        cost.size += 1
        cost.unrolled += 1
        for kind, n in helper.ops.items():
            cost.placed[kind] += n
        cost.size += helper.size
        cost.unrolled += helper.unrolled_size
        cost.iterations += helper.iterations
        return cost


def _has_contents(cell: Any) -> bool:
    try:
        cell.cell_contents  # noqa: B018
    except ValueError:
        return False
    return True


def _branch(test: _Cost, then: _Cost, other: _Cost) -> _Cost:
    """The longest path through a branch, and both arms' size."""
    out = _Cost()
    out.extend(test)
    longer = then if then.weight >= other.weight else other
    for kind, n in longer.ops.items():
        out.ops[kind] += n
    for kind, n in longer.placed.items():
        out.placed[kind] += n
    out.iterations += longer.iterations
    out.size += then.size + other.size
    out.unrolled += then.unrolled + other.unrolled
    out.kinds |= then.kinds | other.kinds
    return out


_cache: weakref.WeakKeyDictionary[Any, dict[Any, FunctionCost]] = weakref.WeakKeyDictionary()
_lock = threading.Lock()


def analyze_callable(
    fn: Callable[..., Any], sizes: Mapping[str, int], stack: tuple[Any, ...] = ()
) -> FunctionCost:
    """The cost of *fn*'s body, given the lengths of its sequence arguments
    by name (*sizes*)."""
    key = tuple(sorted(sizes.items()))
    with _lock:
        kept = _cache.get(fn, {}).get(key)
    if kept is not None:
        return kept
    name = getattr(fn, "__qualname__", getattr(fn, "__name__", "function"))
    try:
        tree = _source_tree(fn)
    except (OSError, TypeError, SyntaxError, _NotAnalyzableError) as error:
        return FunctionCost(
            name, {}, 0, 0, 0, (), (f"{name}: its source is unavailable ({error})",)
        )
    walker = _Walker(fn, sizes, stack)
    arguments = tree.args
    if arguments.vararg or arguments.kwarg:
        walker.flag(tree, "*args or **kwargs")
    parameters = {a.arg for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)}
    walker.names |= parameters
    cost = walker.body(tree.body)
    local_count = len(walker.names - parameters) + walker.extra_locals
    ops = Counter(cost.ops)
    for kind, n in cost.placed.items():
        ops[kind] += n
    result = FunctionCost(
        name,
        dict(sorted(ops.items())),
        cost.size,
        local_count,
        walker.loops,
        tuple(dict.fromkeys(walker.helpers)),
        tuple(dict.fromkeys(walker.violations)),
        cost.iterations,
        cost.unrolled,
        walker.uses,
    )
    with _lock:
        try:
            _cache.setdefault(fn, {})[key] = result
        except TypeError:
            pass
    return result


def register_bytes(count: int) -> int:
    return count * _REGISTER_BYTES
