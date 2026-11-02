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
operations by class) is an intrinsic, counted by it instead of its body.
Converting a time argument to ns (``as_nanoseconds``) costs nothing. See
``cudaq_pulse.controller``.

The reports built from these costs (``resources``, ``analyze``, ``usage``)
are here too, which ``cudaq_pulse.controller`` names: exports never build
one, so this module loads when one is first asked for. Each report class
says it is ``controller``'s.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import math
import threading
import types
import weakref
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ._errors import InvalidTypeError
from ._units import as_nanoseconds
from .controller import ControllerLimits

if TYPE_CHECKING:
    from ._system import Controller, Processor
    from .compiler import Lowered

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
    """What a function body costs, as its operations are counted."""

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
    import textwrap

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
        self.closure = {
            n: c.cell_contents for n, c in zip(free, closure, strict=False) if _has_contents(c)
        }

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
        if target is as_nanoseconds:
            # A time converts to its ns where Python calls the function; on a
            # controller the argument is a number already.
            return cost
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
        for parameter, arg in zip(parameters, node.args, strict=False):
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


# ── What the classical functions cost ───────────────────────────────────────

# A request and an answer carry a 24-byte header (classical.wire).
_HEADER_BYTES = 24


@dataclass(frozen=True)
class FunctionResources:
    """What one classical function costs, from ``resources`` or ``analyze``.

    ``cost`` is its body's static cost (operations by class on the longest
    path, size, locals, whether it is in controller form). ``state_bytes``
    adds its arguments, results and locals; ``request_bytes`` and
    ``answer_bytes`` are its messages with their headers. ``compute_ns`` is
    the longest path's time at the limits' ``op_cycles`` and clock (``None``
    without them), against ``max_latency_ns``. ``calls_per_shot`` counts its
    executions in one shot of the program (0 from ``analyze``).
    ``on_controller`` says whether it runs on the controller (its link is
    one of the limits' ``compute_links``). ``function`` is the name it is
    declared with and ``compiled_name`` the version's, such as
    ``"adaptive_t1__1"`` (the same name for one of one schema).
    """

    __module__ = "cudaq_pulse.controller"

    function: str
    compiled_name: str
    link: int | str
    on_controller: bool
    cost: FunctionCost
    state_bytes: int
    request_bytes: int
    answer_bytes: int
    max_latency_ns: float
    compute_cycles: int | None
    compute_ns: float | None
    calls_per_shot: int = 0

    @property
    def margin_ns(self) -> float | None:
        """How much of ``max_latency_ns`` the compute time leaves."""
        return None if self.compute_ns is None else self.max_latency_ns - self.compute_ns


@dataclass(frozen=True)
class LinkUsage:
    """One link's use in one shot: its calls, the most requests it holds
    at once against its slots, its largest message against its slot size,
    and the bytes of all its requests and answers. A per-controller link has
    a row for each controller's instance (``controller``)."""

    __module__ = "cudaq_pulse.controller"

    link: int | str
    calls_per_shot: int
    peak_in_flight: int
    max_in_flight: int
    largest_message_bytes: int
    slot_bytes: int
    bytes_per_shot: int
    controller: int | None = None


@dataclass(frozen=True)
class ResourceReport:
    """What a program's classical functions cost the controller, per
    function and per shot, and the limits it exceeds (``violations``).
    ``str(report)`` is a table."""

    __module__ = "cudaq_pulse.controller"

    functions: tuple[FunctionResources, ...]
    links: tuple[LinkUsage, ...]
    calls_per_shot: int
    buffer_bytes: int
    limits: ControllerLimits | None
    violations: tuple[str, ...]

    def __str__(self) -> str:
        return format_report(self)


def _payload_bytes(types: Sequence[Any]) -> int:
    return sum(int(t.wire_bytes) for t in types)


def _value_lengths(function: Any) -> dict[str, int]:
    """The lengths of a function's sequence arguments, by parameter name."""
    import inspect

    implementation = function.implementation
    try:
        names = list(inspect.signature(implementation).parameters)
    except (TypeError, ValueError):
        return {}
    lengths = {}
    for name, spec in zip(names, function.args, strict=False):
        if not spec.is_scalar:
            lengths[name] = int(spec.count)
    return lengths


def _link_key(function: Any) -> int | str:
    link: int | str = function.link
    return link


def _on_controller(function: Any, limits: ControllerLimits | None, machine: Any) -> bool:
    links = limits.compute_links if limits is not None else ("controller",)
    if function.link in links:
        return True
    if machine is not None:
        try:
            link_id = machine.link_id(function.link)
        except (KeyError, ValueError):
            return False
        return any(
            (isinstance(link, int) and link == link_id)
            or (
                isinstance(link, str)
                and link in machine.link_names
                and machine.link_id(link) == link_id
            )
            for link in links
        )
    return False


def _function_resources(
    function: Any,
    limits: ControllerLimits | None,
    clock_ghz: float | None,
    machine: Any = None,
    calls: int = 0,
) -> FunctionResources:
    if function.implementation is None:
        cost = FunctionCost(
            function.name, {}, 0, 0, 0, (), (f"{function.name}: external, with no body to read",)
        )
    elif function.uniform is not None:
        cost = FunctionCost(
            function.name,
            {"load": 1},
            1,
            0,
            0,
            (),
            (),
            unrolled_size=1,
            uses={"load": function.name},
        )
    else:
        cost = analyze_callable(function.implementation, _value_lengths(function))
    args_bytes = _payload_bytes(function.args)
    results_bytes = _payload_bytes(function.results)
    state = args_bytes + results_bytes + register_bytes(cost.locals)
    cycles = compute_ns = None
    if limits is not None and limits.op_cycles:
        cycles = cost.cycles(limits.op_cycles)
        rate = limits.compute_clock_ghz or limits.clock_ghz or clock_ghz
        if rate:
            compute_ns = cycles / rate
    return FunctionResources(
        function=function.base_name,
        compiled_name=function.name,
        link=_link_key(function),
        on_controller=_on_controller(function, limits, machine),
        cost=cost,
        state_bytes=state,
        request_bytes=_HEADER_BYTES + args_bytes,
        answer_bytes=_HEADER_BYTES + results_bytes,
        max_latency_ns=function.max_latency_ns,
        compute_cycles=cycles,
        compute_ns=compute_ns,
        calls_per_shot=calls,
    )


def analyze(function: Any, limits: ControllerLimits | None = None) -> FunctionResources:
    """What one classical function costs, on its own (see ``resources``).
    A function whose list lengths come from its calls has a cost per
    version: analyze one, ``analyze(flip.specialize(outcomes=3))``."""
    from .classical import ClassicalFunction, _resolve, _unsized_names

    if isinstance(function, ClassicalFunction):
        _resolve(function)
        if function._shape is not None:
            names = _unsized_names(function)
            example = ", ".join(f"{n}=3" for n in names)
            raise InvalidTypeError(
                f"{function.name}'s list lengths come from its calls ({', '.join(names)}), so it"
                f" has no single cost; analyze one version, e.g."
                f" analyze({function.name}.specialize({example}))"
            )
    return _function_resources(function, limits, None)


def _function_violations(row: FunctionResources, limits: ControllerLimits) -> list[str]:
    found = []
    name = row.compiled_name
    if limits.controller_form and row.cost.violations:
        found.append(
            f"{name} is not in controller form: "
            + "; ".join(row.cost.violations[:4])
            + (
                f" (and {len(row.cost.violations) - 4} more)"
                if len(row.cost.violations) > 4
                else ""
            )
        )
    if limits.max_ops is not None and row.cost.size > limits.max_ops:
        found.append(
            f"{name} holds {row.cost.size} operations; the controller holds {limits.max_ops} "
            "per function"
        )
    if limits.state_bytes is not None and row.state_bytes > limits.state_bytes:
        found.append(
            f"{name} needs {row.state_bytes} bytes of state (arguments, results and "
            f"{row.cost.locals} locals); the controller holds {limits.state_bytes} per function"
        )
    if row.compute_ns is not None and row.compute_ns > row.max_latency_ns:
        found.append(
            f"{name} computes for about {row.compute_ns:.0f} ns ({row.compute_cycles} cycles on "
            f"its longest path), beyond its max_latency_ns={row.max_latency_ns:g}"
        )
    return found


def resources(lowered: Lowered, limits: ControllerLimits | None = None) -> ResourceReport:
    """What the classical functions of *lowered* cost the controller.

    *limits*, else the machine model's, give the cycle costs and the limits
    checked; every function is reported, and the limits apply to those on
    the controller's ``compute_links``. Calls per shot, slots in flight and
    buffer bytes come from the scheduled program (``classical_report``).
    """
    limits = limits if limits is not None else lowered.options.machine.controller
    machine = lowered.options.machine
    calls = lowered.classical_report() if lowered.classical and lowered.scheduled else []
    per_function: dict[str, int] = {}
    for call in calls:
        per_function[call.compiled_name] = per_function.get(call.compiled_name, 0) + 1
    rows = tuple(
        _function_resources(
            fn, limits, lowered.target.clock_ghz, machine, per_function.get(fn.name, 0)
        )
        for fn in lowered.classical
    )
    by_name = {row.compiled_name: row for row in rows}
    usage: dict[tuple[int, int], dict[str, Any]] = {}
    for call in calls:
        row = by_name.get(call.compiled_name)
        instance = -1 if call.controller is None else call.controller
        entry = usage.setdefault(
            (call.link, instance),
            {"calls": 0, "bytes": 0, "largest": 0, "spans": [], "name": call.link_name},
        )
        entry["calls"] += 1
        if row is not None:
            entry["bytes"] += row.request_bytes + row.answer_bytes
            entry["largest"] = max(entry["largest"], row.request_bytes, row.answer_bytes)
        end = call.taken_cycles if call.taken_cycles is not None else call.due_cycles
        entry["spans"].append((call.submit_cycles, max(end, call.submit_cycles + 1)))
    links = []
    for (link_id, instance), entry in sorted(usage.items()):
        edges = sorted([(a, 1) for a, _ in entry["spans"]] + [(b, -1) for _, b in entry["spans"]])
        peak = held = 0
        for _, step in edges:
            held += step
            peak = max(peak, held)
        link = machine.link(link_id)
        links.append(
            LinkUsage(
                link=entry["name"] if entry["name"] is not None else link_id,
                calls_per_shot=entry["calls"],
                peak_in_flight=peak,
                max_in_flight=link.max_in_flight,
                largest_message_bytes=entry["largest"],
                slot_bytes=link.slot_bytes,
                bytes_per_shot=entry["bytes"],
                controller=None if instance < 0 else instance,
            )
        )
    found: list[str] = []
    controller_rows = [row for row in rows if row.on_controller]
    controller_calls = sum(row.calls_per_shot for row in controller_rows)
    controller_bytes = sum(
        row.calls_per_shot * (row.request_bytes + row.answer_bytes) for row in controller_rows
    )
    if limits is not None:
        for row in controller_rows:
            found.extend(_function_violations(row, limits))
        if limits.max_calls_per_shot is not None and controller_calls > limits.max_calls_per_shot:
            found.append(
                f"one shot makes {controller_calls} calls to the controller; it takes "
                f"{limits.max_calls_per_shot}"
            )
        if limits.buffer_bytes is not None and controller_bytes > limits.buffer_bytes:
            found.append(
                f"one shot's requests and answers to the controller take {controller_bytes} "
                f"bytes; its buffer holds {limits.buffer_bytes}"
            )
    for usage_row in links:
        where = (
            f"link {usage_row.link!r}"
            if usage_row.controller is None
            else f"link {usage_row.link!r} on controllers[{usage_row.controller}]"
        )
        if usage_row.peak_in_flight > usage_row.max_in_flight:
            found.append(
                f"{where} holds {usage_row.peak_in_flight} requests at once; it "
                f"has {usage_row.max_in_flight} slots"
            )
        if usage_row.largest_message_bytes > usage_row.slot_bytes:
            found.append(
                f"{where} carries a {usage_row.largest_message_bytes}-byte "
                f"message; its slots hold {usage_row.slot_bytes}"
            )
    return ResourceReport(
        functions=rows,
        links=tuple(links),
        calls_per_shot=len(calls),
        buffer_bytes=sum(entry["bytes"] for entry in usage.values()),
        limits=limits,
        violations=tuple(found),
    )


def format_report(report: ResourceReport) -> str:
    """The report as a table, one row per function and per link."""
    lines = ["Classical functions"]
    header = (
        f"  {'function':<24} {'link':<11} {'ops':>5} {'size':>5} {'state B':>7} "
        f"{'compute':>9} {'budget':>8} {'calls':>5}  form"
    )
    lines.append(header)
    for row in report.functions:
        compute = "-" if row.compute_ns is None else f"{row.compute_ns:.0f} ns"
        form = "ok" if row.cost.in_controller_form else "no"
        lines.append(
            f"  {row.compiled_name:<24} {str(row.link):<11} {row.cost.total:>5} {row.cost.size:>5} "
            f"{row.state_bytes:>7} {compute:>9} {row.max_latency_ns:>5.0f} ns "
            f"{row.calls_per_shot:>5}  {form}"
        )
        ops = ", ".join(f"{k} {n}" for k, n in row.cost.ops.items() if n)
        if ops:
            lines.append(f"      {ops}")
    if report.links:
        lines.append("Links, per shot")
        lines.append(f"  {'link':<11} {'calls':>5} {'in flight':>9} {'largest B':>9} {'bytes':>7}")
        for usage in report.links:
            name = str(usage.link)
            if usage.controller is not None:
                name += f"@{usage.controller}"
            lines.append(
                f"  {name:<11} {usage.calls_per_shot:>5} "
                f"{usage.peak_in_flight:>4}/{usage.max_in_flight:<4} "
                f"{usage.largest_message_bytes:>4}/{usage.slot_bytes:<4} {usage.bytes_per_shot:>7}"
            )
    lines.append(f"Per shot: {report.calls_per_shot} calls, {report.buffer_bytes} bytes")
    if report.violations:
        lines.append("Exceeds:")
        lines.extend(f"  {v}" for v in report.violations)
    return "\n".join(lines)


# ── What the processors of a control system run ─────────────────────────────


@dataclass(frozen=True)
class FunctionUsage:
    """One classical function on the processor that runs it, in
    ``Lowered.controller_usage``.

    ``ops`` is the operations it holds there: its body as written, or with
    every loop unrolled (``unrolled``) on a processor without ``"loop"``.
    ``state_bytes`` adds its arguments, results and locals;
    ``request_bytes`` and ``answer_bytes`` are its messages with their
    headers. ``compute_ns`` is its longest path at the processor's
    ``cycles_per_op`` and clock, against ``max_latency_ns``;
    ``calls_per_shot`` counts its calls in one shot.
    """

    __module__ = "cudaq_pulse.controller"

    function: str
    link: int | str
    ops: int
    unrolled: bool
    state_bytes: int
    request_bytes: int
    answer_bytes: int
    compute_cycles: int
    compute_ns: float
    max_latency_ns: float
    calls_per_shot: int
    cost: FunctionCost

    @property
    def slack_ns(self) -> float:
        """How much of ``max_latency_ns`` the compute time leaves."""
        return self.max_latency_ns - self.compute_ns


@dataclass(frozen=True)
class ProcessorUsage:
    """What one processor runs (``functions``) and its totals: the
    operations and state bytes of all its functions, its calls and their
    bytes per shot, and the most calls it computes at once; ``processor``
    has the limits."""

    __module__ = "cudaq_pulse.controller"

    controller: int
    index: int
    processor: Processor
    clock_ghz: float
    functions: tuple[FunctionUsage, ...]
    ops: int
    state_bytes: int
    calls_per_shot: int
    bytes_per_shot: int
    peak_concurrent_calls: int


@dataclass(frozen=True)
class ControllerUsage:
    """One controller of the system, by its index, and its processors."""

    __module__ = "cudaq_pulse.controller"

    controller: int
    model: str | None
    processors: tuple[ProcessorUsage, ...]


@dataclass(frozen=True)
class LimitExceeded:
    """A limit of a processor that a program exceeds, in
    ``UsageReport.exceeded``: the function, or the processor, that exceeds
    it (``where``), the limit (a ``Processor`` field, or the function's
    ``max_latency_ns``), what the program needs of it (``value``, in
    ``unit``) and what the limit allows (``most``)."""

    __module__ = "cudaq_pulse.controller"

    where: str
    limit: str
    value: float
    most: float
    unit: str

    @property
    def over(self) -> float:
        """By how much the program exceeds the limit."""
        return self.value - self.most


@dataclass(frozen=True)
class UsageReport:
    """What a program's classical functions use of the processors of its
    control system (``Lowered.controller_usage``): one row per controller
    with processors, the functions that run off them (``off_board``, by
    name, whose links alone are checked) and the limits exceeded, as text
    (``violations``) and for each numeric limit by how much (``exceeded``).
    ``str(report)`` is a table. A program over a limit does not compile:
    its ``ControllerLimitError`` carries the report as ``usage``."""

    __module__ = "cudaq_pulse.controller"

    controllers: tuple[ControllerUsage, ...]
    off_board: tuple[str, ...]
    violations: tuple[str, ...]
    exceeded: tuple[LimitExceeded, ...] = ()

    @property
    def within_limits(self) -> bool:
        """Whether the program fits every processor."""
        return not self.violations

    def __str__(self) -> str:
        return _format_usage(self)


def _peak(spans: Sequence[tuple[int, int]]) -> int:
    edges = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
    peak = held = 0
    for _, step in edges:
        held += step
        peak = max(peak, held)
    return peak


def usage(lowered: Lowered) -> UsageReport:
    """What *lowered*'s classical functions use of its control system's
    processors, and the limits they exceed (see ``Processor``)."""
    machine = lowered.options.machine
    system = machine._system
    placement = system._placement if system is not None else {}
    if system is None or not placement:
        return UsageReport((), tuple(fn.name for fn in lowered.classical), ())
    calls = lowered.classical_report() if lowered.classical and lowered.scheduled else []
    # By function and the controller whose instance of a per-controller
    # link runs the call (None: shared, or not known).
    spans: dict[tuple[str, int | None], list[tuple[int, int]]] = {}
    for call in calls:
        end = max(call.due_cycles, call.submit_cycles + 1)
        spans.setdefault((call.compiled_name, call.controller), []).append(
            (call.submit_cycles, end)
        )
    # Calls over a per-controller link whose instance is not known count on each.
    placed: dict[tuple[int, int], list[tuple[Any, list[tuple[int, int]]]]] = {}
    off_board = []
    for fn in lowered.classical:
        where = placement.get(machine.link_id(fn.link))
        if where is None or fn.implementation is None:
            off_board.append(fn.name)
            continue
        own = machine.link(fn.link).per_controller
        unknown = spans.get((fn.name, None), [])
        for index, number in where.items():
            mine = spans.get((fn.name, index), []) + unknown if own else unknown
            placed.setdefault((index, number), []).append((fn, mine))
    found = _Found()
    rows: list[ControllerUsage] = []
    for index, controller in enumerate(system.controllers):
        processors = []
        for number, processor in enumerate(controller.processors):
            label = f"processors[{number}] of {_system_label(index, controller)}"
            clock = processor.clock_ghz or controller.clock_ghz or lowered.target.clock_ghz
            assigned = placed.get((index, number), [])
            functions = [
                _function_usage(fn, processor, clock, len(made), label, found)
                for fn, made in assigned
            ]
            local = {fn.name: made for fn, made in assigned}
            processors.append(
                _processor_usage(index, number, processor, clock, functions, local, label, found)
            )
        if processors:
            rows.append(ControllerUsage(index, controller.model, tuple(processors)))
    return UsageReport(tuple(rows), tuple(off_board), tuple(found), tuple(found.exceeded))


class _Found(list[str]):
    """The limits a program exceeds, as text, and the numeric ones also as
    ``LimitExceeded`` (``exceeded``)."""

    def __init__(self) -> None:
        super().__init__()
        self.exceeded: list[LimitExceeded] = []

    def over(self, text: str, where: str, limit: str, value: float, most: float, unit: str) -> None:
        """*text*, with by how much *value* exceeds *most*."""
        found = LimitExceeded(where, limit, value, most, unit)
        self.exceeded.append(found)
        amount = f"{found.over:.0f}" if isinstance(value, float) else str(found.over)
        self.append(f"{text}: {amount} {unit} over")


def _system_label(index: int, controller: Controller) -> str:
    name = f"controllers[{index}]"
    return f"{name} ({controller.model})" if controller.model else name


def _function_usage(
    fn: Any, processor: Processor, clock: float, calls: int, label: str, found: _Found
) -> FunctionUsage:
    base = _function_resources(fn, None, clock)
    cost = base.cost
    unrolled = not processor.loops and cost.loops > 0
    ops = cost.unrolled_size if unrolled else cost.size
    cycles = cost.processor_cycles(processor.cycles_per_op, processor.loops)
    row = FunctionUsage(
        function=fn.name,
        link=base.link,
        ops=ops,
        unrolled=unrolled,
        state_bytes=base.state_bytes,
        request_bytes=base.request_bytes,
        answer_bytes=base.answer_bytes,
        compute_cycles=cycles,
        compute_ns=cycles / clock,
        max_latency_ns=fn.max_latency_ns,
        calls_per_shot=calls,
        cost=cost,
    )
    name = fn.name
    for op in processor.unsupported:
        if op != "loop" and op in cost.uses:
            found.append(f"{name} uses {op} ({cost.uses[op]}), which {label} does not support")
    if processor.fixed_cost_only and cost.violations:
        more = len(cost.violations) - 4
        found.append(
            f"{name} is not fixed-cost, which {label} runs only: "
            + "; ".join(cost.violations[:4])
            + (f" (and {more} more)" if more > 0 else "")
        )
    most = processor.max_ops_per_function
    if most is not None and ops > most:
        how = f" with its loops unrolled ({label} does not support 'loop')" if unrolled else ""
        found.over(
            f"{name} holds {ops} operations{how}; {label} holds {most} per function "
            "(max_ops_per_function)",
            name,
            "max_ops_per_function",
            ops,
            most,
            "operations",
        )
    most = processor.max_state_bytes_per_function
    if most is not None and row.state_bytes > most:
        found.over(
            f"{name} needs {row.state_bytes} bytes of state (arguments, results and "
            f"{cost.locals} locals); {label} holds {most} per function "
            "(max_state_bytes_per_function)",
            name,
            "max_state_bytes_per_function",
            row.state_bytes,
            most,
            "bytes",
        )
    if row.compute_ns > row.max_latency_ns:
        found.over(
            f"{name} computes for about {row.compute_ns:.0f} ns on {label} ({cycles} cycles on "
            f"its longest path), beyond its max_latency_ns={row.max_latency_ns:g}",
            name,
            "max_latency_ns",
            row.compute_ns,
            float(row.max_latency_ns),
            "ns",
        )
    return row


def _processor_usage(
    controller: int,
    index: int,
    processor: Processor,
    clock: float,
    functions: list[FunctionUsage],
    spans: Mapping[str, list[tuple[int, int]]],
    label: str,
    found: _Found,
) -> ProcessorUsage:
    ops = sum(row.ops for row in functions)
    state = sum(row.state_bytes for row in functions)
    calls = sum(row.calls_per_shot for row in functions)
    bytes_per_shot = sum(
        row.calls_per_shot * (row.request_bytes + row.answer_bytes) for row in functions
    )
    peak = _peak([span for row in functions for span in spans.get(row.function, ())])
    for value, most, what, field_name, unit in (
        (ops, processor.max_ops_total, "operations", "max_ops_total", "operations"),
        (
            state,
            processor.max_state_bytes_total,
            "bytes of state",
            "max_state_bytes_total",
            "bytes",
        ),
    ):
        if most is not None and value > most:
            found.over(
                f"{label}: its classical functions hold {value} {what}; it holds {most} in all "
                f"({field_name})",
                label,
                field_name,
                value,
                most,
                unit,
            )
    most = processor.max_calls_per_shot
    if most is not None and calls > most:
        found.over(
            f"{label}: one shot makes {calls} calls to it; it takes {most} (max_calls_per_shot)",
            label,
            "max_calls_per_shot",
            calls,
            most,
            "calls",
        )
    most = processor.max_bytes_per_shot
    if most is not None and bytes_per_shot > most:
        found.over(
            f"{label}: one shot's requests and answers to it take {bytes_per_shot} bytes; it "
            f"takes {most} (max_bytes_per_shot)",
            label,
            "max_bytes_per_shot",
            bytes_per_shot,
            most,
            "bytes",
        )
    most = processor.max_concurrent_calls
    if peak > most:
        found.over(
            f"{label}: {peak} calls compute on it at once; it computes {most} "
            "(max_concurrent_calls)",
            label,
            "max_concurrent_calls",
            peak,
            most,
            "calls",
        )
    return ProcessorUsage(
        controller=controller,
        index=index,
        processor=processor,
        clock_ghz=clock,
        functions=tuple(functions),
        ops=ops,
        state_bytes=state,
        calls_per_shot=calls,
        bytes_per_shot=bytes_per_shot,
        peak_concurrent_calls=peak,
    )


def _of(value: int, most: int | None, unit: str = "") -> str:
    return f"{value}{unit}" if most is None else f"{value} / {most}{unit}"


def _format_usage(report: UsageReport, exceeds: bool = True) -> str:
    lines = ["Controller usage: " + ("within limits" if report.within_limits else "over limits")]
    for row in report.controllers:
        name = f"controller {row.controller}" + (f" ({row.model})" if row.model else "")
        for usage_row in row.processors:
            p = usage_row.processor
            links = ", ".join(str(link) for link in p.links)
            lines.append(
                f"{name}, processor {usage_row.index} at {usage_row.clock_ghz:g} GHz "
                f"(links: {links})"
            )
            lines.append(
                f"  in all: ops {_of(usage_row.ops, p.max_ops_total)}, "
                f"state {_of(usage_row.state_bytes, p.max_state_bytes_total, ' B')}, "
                f"calls per shot {_of(usage_row.calls_per_shot, p.max_calls_per_shot)}, "
                f"bytes per shot {_of(usage_row.bytes_per_shot, p.max_bytes_per_shot)}, "
                f"at once {usage_row.peak_concurrent_calls} / {p.max_concurrent_calls}"
            )
            if not usage_row.functions:
                continue
            lines.append(
                f"  {'function':<24} {'ops':>13} {'state B':>13} {'compute / budget':>20} "
                f"{'calls':>5}"
            )
            for f in usage_row.functions:
                ops = _of(f.ops, p.max_ops_per_function) + (" unrolled" if f.unrolled else "")
                state = _of(f.state_bytes, p.max_state_bytes_per_function)
                compute = f"{f.compute_ns:.0f} / {f.max_latency_ns:.0f} ns"
                lines.append(
                    f"  {f.function:<24} {ops:>13} {state:>13} {compute:>20} {f.calls_per_shot:>5}"
                )
    if report.off_board:
        lines.append("Off the processors: " + ", ".join(report.off_board))
    if exceeds and report.violations:
        lines.append("Exceeds:")
        lines.extend(f"  {v}" for v in report.violations)
    return "\n".join(lines)
