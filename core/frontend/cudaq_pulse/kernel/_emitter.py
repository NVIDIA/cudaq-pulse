# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""PulseIREmitter -- executes decoded kernel bytecode and emits pulse IR.

Kernels are traced by running their canonical instruction stream (see
``_bytecode_normalize``) on an emulated stack. Python values are computed
concretely, so compile-time conditions, short-circuit operators and loop
bounds resolve exactly as Python would; pulse operations (``drive``,
``gaussian``, ...) are emitted as IR operations instead of being called.

A ``for ... in range(n)`` loop whose induction variable is never read is
emitted once as a structured ``scf.for`` when the builder supports regions.
Every other loop, and every loop whose body turns out to carry Python state
from one iteration to the next, is unrolled.
"""

from __future__ import annotations

import builtins as _builtins_mod
import inspect
import types
from typing import Any

from . import _bytecode_normalize as bc
from ._bytecode_normalize import (
    BINARY,
    CALL,
    COMPARE,
    FOR_ITER,
    GET_ITER,
    ITER_SLOTS,
    JUMP,
    JUMP_IF_FALSE,
    JUMP_IF_TRUE,
    LOAD_ATTR,
    LOAD_CONST,
    LOAD_DEREF,
    LOAD_FAST,
    LOAD_GLOBAL,
    POP_TOP,
    STORE_FAST,
    SUBSCR,
    UNPACK_SEQUENCE,
    decode,
)
from .ir_builder import (
    LINEAR_TYPES,
    OP_TABLE,
    CompilationError,
    IRValue,
    PythonIRBuilder,
    is_symbolic,
)

_ALLOC_OPS = {"qudit_ref": "pulse.qudit_alloc", "qvec_ref": "pulse.qvec_alloc"}
_PULSE_OP_NAMES = frozenset(OP_TABLE.keys()) | frozenset(_ALLOC_OPS.keys())

_RANGE_BOUNDS_MESSAGE = "range() bounds must be compile-time integers in pulse kernels"
_BRANCH_MESSAGE = (
    "runtime-dependent branches are not supported in the pulse research "
    "preview; measurement results cannot control Python if statements"
)

# The emulated stack mixes IR values with arbitrary Python constants
# (tuples, dicts, callables), so entries are dynamically typed.
StackVal = Any

_BUILTINS = vars(_builtins_mod)
_SCALARS = (int, float, complex, str, bytes, bool, type(None))

# Bad-argument failures of a Python call evaluated at trace time. Anything
# else (CompilationError, MemoryError, KeyboardInterrupt, ...) propagates.
_EVALUATION_ERRORS = (TypeError, ValueError, ArithmeticError, LookupError)


class _EmptyCell:
    """A closure variable that was never assigned in the enclosing scope."""

    __slots__ = ()


_EMPTY_CELL = _EmptyCell()


def _extract_closures(fn: Any) -> dict[str, Any]:
    """Return the values captured by a Python function, read now.

    Called on every trace, so rebinding a captured variable between calls
    takes effect; the decoded bytecode is cached separately.
    """
    closures: dict[str, Any] = {}
    cells = getattr(fn, "__closure__", None)
    if cells and fn.__code__.co_freevars:
        for name, cell in zip(fn.__code__.co_freevars, cells):
            try:
                closures[name] = cell.cell_contents
            except ValueError:
                closures[name] = _EMPTY_CELL
    return closures


_PULSE_FUNCTIONS: dict[Any, str] = {}


def _load_pulse_functions() -> None:
    """Map each DSL function object, by identity, to its op name."""
    from .. import ops
    from . import decorator

    for name in _PULSE_OP_NAMES:
        op = getattr(ops, name, None) or getattr(decorator, name, None)
        if op is not None:
            _PULSE_FUNCTIONS[op] = name


def _uses_pulse_ops(fn: types.FunctionType, seen: set[int] | None = None) -> bool:
    """Whether *fn* directly or transitively calls the pulse DSL."""
    seen = seen or set()
    if id(fn) in seen:
        return False
    seen.add(id(fn))

    for name in fn.__code__.co_names:
        if name in _PULSE_OP_NAMES:
            return True
        value = fn.__globals__.get(name)
        if isinstance(value, types.FunctionType) and _uses_pulse_ops(value, seen):
            return True
    return False


def _truth(value: StackVal) -> bool:
    if isinstance(value, IRValue):
        raise CompilationError(_BRANCH_MESSAGE)
    return bool(value)


class _BadIter:
    """Iterator over anything but a compile-time ``range()``."""

    __slots__ = ()


_BAD_ITER = _BadIter()
_MISSING = object()


class _Loop:
    """Iterator state of a ``for ... in range()`` loop on the stack."""

    __slots__ = ("rng", "it", "rolled")

    def __init__(self, rng: range):
        self.rng = rng
        self.it = iter(rng)
        self.rolled: bool | None = None  # None until the header decides


class _NotRollableError(Exception):
    """The single traced iteration of a rolled loop left the loop body."""


class PulseIREmitter:
    """Traces decoded bytecode and emits pulse IR via a builder."""

    __slots__ = ("b", "stack", "locals", "_fn_globals", "_fn_closures", "_builtins", "_can_roll")

    def __init__(
        self,
        builder: Any,
        fn_globals: dict[str, Any] | None = None,
        fn_closures: dict[str, Any] | None = None,
    ):
        self.b = builder
        self.stack: list[StackVal] = []
        self.locals: dict[str, StackVal] = {}
        self._fn_globals = {} if fn_globals is None else fn_globals
        self._fn_closures = {} if fn_closures is None else fn_closures
        self._builtins = _BUILTINS
        self._can_roll = getattr(builder, "supports_regions", False)
        if not _PULSE_FUNCTIONS:
            _load_pulse_functions()

    # ── Public entry point ───────────────────────────────────────────

    def run(self, code: types.CodeType) -> StackVal:
        """Trace *code* with the current locals; return its return value.

        An exception raised while tracing gets a ``pulse_source`` attribute,
        ``(DecodedCode, instruction_index)``, naming the innermost kernel or
        helper instruction that failed; ``DecodedCode.location(index)``
        resolves it to a source span.
        """
        prog = decode(code)
        return self._loop(prog, 0, -1, len(prog.instrs), -1)

    # ── Interpreter loop ─────────────────────────────────────────────

    def _loop(
        self,
        prog: bc.DecodedCode,
        pc: int,
        lo: int,
        hi: int,
        header: int,
        # Tracing hot loop: opcodes bound as defaults are plain local reads.
        _load_fast: int = LOAD_FAST,
        _load_const: int = LOAD_CONST,
        _load_deref: int = LOAD_DEREF,
        _call: int = CALL,
        _store_fast: int = STORE_FAST,
        _load_global: int = LOAD_GLOBAL,
        _load_attr: int = LOAD_ATTR,
        _subscr: int = SUBSCR,
        _binary: int = BINARY,
        _compare: int = COMPARE,
        _jif: int = JUMP_IF_FALSE,
        _jit: int = JUMP_IF_TRUE,
        _jump: int = JUMP,
        _pop_top: int = POP_TOP,
        _for_iter: int = FOR_ITER,
        _get_iter: int = GET_ITER,
        _unpack: int = UNPACK_SEQUENCE,
        _iter_slots: int = ITER_SLOTS,
    ) -> Any:
        """Run from *pc*. Jumps must stay inside ``(lo, hi)``.

        With ``header >= 0`` this traces one iteration of a rolled loop:
        it returns when control jumps back to the loop header and raises
        ``_NotRollableError`` if control leaves the body any other way.
        An exception raised here gets ``pulse_source = (prog, index)`` unless
        a nested trace already attached one.
        """
        instrs = prog.instrs
        stack = self.stack
        local_vars = self.locals
        closures = self._fn_closures
        fn_globals = self._fn_globals
        push = stack.append
        pop = stack.pop

        try:
            while True:
                op, arg = instrs[pc]
                pc += 1

                if op == _load_fast:
                    try:
                        push(local_vars[arg])
                    except KeyError:
                        raise CompilationError(f"undefined local: {arg}") from None

                elif op == _load_global:
                    value = fn_globals.get(arg, _MISSING)
                    push(self._resolve_global(arg) if value is _MISSING else value)

                elif op == _call:
                    if arg:
                        call_args = stack[-arg:]
                        del stack[-arg:]
                    else:
                        call_args = []
                    self._dispatch_call(pop(), call_args, None)

                elif op == _pop_top:
                    pop()

                elif op == _load_deref:
                    value = local_vars.get(arg, _MISSING)
                    if value is _MISSING:
                        value = closures.get(arg, _MISSING)
                        if value is _MISSING:
                            value = self._resolve_global(arg)
                        elif value is _EMPTY_CELL:
                            raise CompilationError(
                                f"closure variable {arg!r} is referenced before assignment"
                            )
                    push(value)

                elif op == _load_const:
                    push(arg)

                elif op == _store_fast:
                    local_vars[arg] = pop()

                elif op == _load_attr:
                    obj = pop()
                    if isinstance(obj, tuple) and len(obj) == 2 and obj[0] == "__unresolved__":
                        push(("__unresolved__", arg))
                    else:
                        try:
                            push(getattr(obj, arg))
                        except (AttributeError, TypeError) as e:
                            raise CompilationError(
                                f"cannot resolve attribute {arg!r} on {type(obj).__name__}"
                            ) from e

                elif op == _subscr:
                    index = pop()
                    obj = pop()
                    try:
                        push(obj[index])
                    except (TypeError, IndexError, KeyError) as e:
                        raise CompilationError(
                            f"subscript failed: {type(obj).__name__}[{index!r}]: {e}"
                        ) from e

                elif op == _binary or op == _compare:
                    right = pop()
                    left = pop()
                    push(arg(left, right))

                elif op == _jif or op == _jit:
                    if _truth(pop()) == (op == _jit):
                        pc = self._jump(arg, lo, hi, header)
                        if pc < 0:
                            return None

                elif op == _jump:
                    pc = self._jump(arg, lo, hi, header)
                    if pc < 0:
                        return None

                elif op == _for_iter:
                    loop = stack[-_iter_slots]
                    if type(loop) is not _Loop:
                        raise CompilationError(_RANGE_BOUNDS_MESSAGE)
                    exit_idx, var_is_read = arg
                    if loop.rolled is None:
                        loop.rolled = (
                            self._can_roll
                            and not var_is_read
                            and self._roll(loop, prog, pc - 1, exit_idx)
                        )
                    value = loop if loop.rolled else next(loop.it, loop)
                    if value is loop:
                        del stack[-_iter_slots:]
                        pc = self._jump(exit_idx, lo, hi, header)
                        if pc < 0:
                            return None
                    else:
                        push(value)

                elif op == bc.RETURN:
                    if header >= 0:
                        raise _NotRollableError
                    return pop()

                elif op == _get_iter:
                    value = pop()
                    push(_Loop(value) if isinstance(value, range) else _BAD_ITER)
                    if _iter_slots > 1:
                        stack.extend([None] * (_iter_slots - 1))

                elif op == _unpack:
                    val = pop()
                    if not isinstance(val, (tuple, list)):
                        raise CompilationError(f"cannot unpack {type(val)}")
                    if len(val) != arg:
                        raise CompilationError(f"unpack mismatch: expected {arg}, got {len(val)}")
                    stack.extend(reversed(val))

                elif op == bc.BUILD_TUPLE or op == bc.BUILD_LIST:
                    if arg:
                        items = stack[-arg:]
                        del stack[-arg:]
                    else:
                        items = []
                    push(tuple(items) if op == bc.BUILD_TUPLE else items)

                elif op == bc.LIST_APPEND or op == bc.LIST_EXTEND:
                    val = pop()
                    lst = stack[-arg]
                    if not isinstance(lst, list):
                        raise CompilationError(f"list update on non-list: {type(lst)}")
                    if op == bc.LIST_APPEND:
                        lst.append(val)
                    else:
                        lst.extend(val)

                elif op == bc.UNARY:
                    push(arg(pop()))

                elif op == bc.NOT:
                    push(not _truth(pop()))

                elif op == bc.IS:
                    right = pop()
                    left = pop()
                    push((left is right) != arg)

                elif op == bc.CONTAINS:
                    right = pop()
                    left = pop()
                    push((left in right) != arg)

                elif op == bc.CALL_KW:
                    names = pop()
                    values = stack[-arg:]
                    del stack[-arg:]
                    n_pos = arg - len(names)
                    kwargs = dict(zip(names, values[n_pos:]))
                    self._dispatch_call(pop(), values[:n_pos], kwargs)

                elif op == bc.COPY:
                    push(stack[-arg])

                elif op == bc.SWAP:
                    stack[-1], stack[-arg] = stack[-arg], stack[-1]

                elif op == bc.JUMP_IF_NONE or op == bc.JUMP_IF_NOT_NONE:
                    if (pop() is None) == (op == bc.JUMP_IF_NONE):
                        pc = self._jump(arg, lo, hi, header)
                        if pc < 0:
                            return None

                elif op == bc.STORE_DEREF:
                    local_vars[arg] = pop()

                elif op == bc.DELETE_FAST:
                    local_vars.pop(arg, None)

                elif op == bc.BUILD_SLICE:
                    items = stack[-arg:]
                    del stack[-arg:]
                    push(slice(*items))

                elif op == bc.STORE_SUBSCR:
                    index = pop()
                    obj = pop()
                    value = pop()
                    obj[index] = value

                elif op == bc.BUILD_CONST_KEY_MAP:
                    keys = pop()
                    vals = stack[-arg:] if arg else []
                    del stack[len(stack) - arg :]
                    push(dict(zip(keys, vals)))

                elif op == bc.BUILD_MAP:
                    items = stack[len(stack) - 2 * arg :]
                    del stack[len(stack) - 2 * arg :]
                    push(dict(zip(items[::2], items[1::2])))

                elif op == bc.BUILD_STRING:
                    parts = stack[-arg:] if arg else []
                    del stack[len(stack) - arg :]
                    push("".join(parts))

                elif op == bc.CONVERT:
                    push((str, repr, ascii)[arg - 1](pop()))

                elif op == bc.FORMAT:
                    spec = pop() if arg else ""
                    push(format(pop(), spec))

                elif op == bc.IMPORT_NAME:
                    fromlist = pop()
                    level = pop()
                    push(__import__(arg, self._fn_globals, None, fromlist, level))

                elif op == bc.IMPORT_FROM:
                    push(getattr(stack[-1], arg))

                elif op == bc.RAISE:
                    exc = pop() if arg else None
                    raise CompilationError(f"kernel raised {exc!r}")

                elif op == bc.UNSUPPORTED:
                    raise CompilationError(arg)

                else:  # pragma: no cover - decode only produces the ops above
                    raise CompilationError(f"unknown canonical instruction {op}")
        except _NotRollableError:
            raise
        except Exception as e:
            if getattr(e, "pulse_source", None) is None:
                try:
                    e.pulse_source = (prog, pc - 1)  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            raise

    @staticmethod
    def _jump(target: int, lo: int, hi: int, header: int) -> int:
        """Return the jump target, or -1 when a rolled iteration completes."""
        if target == header:
            return -1
        if not lo < target < hi:
            raise _NotRollableError
        return target

    # ── Structured loops ─────────────────────────────────────────────

    def _roll(self, loop: _Loop, prog: bc.DecodedCode, for_idx: int, exit_idx: int) -> bool:
        """Emit *loop* as one ``scf.for``; return False to unroll it instead.

        The body is traced once. Rolling is kept only if that iteration
        returned to the loop header and left every Python-level value
        (non-IR locals and the lists and dicts they hold) unchanged, so all
        iterations would trace identically. Otherwise the emitted ops are
        rolled back.
        """
        rng = loop.rng
        stack = self.stack
        saved_stack = list(stack)
        saved_locals = dict(self.locals)
        containers = [
            (v, list(v) if isinstance(v, list) else dict(v))
            for v in saved_locals.values()
            if isinstance(v, (list, dict))
        ]
        checkpoint = self.b.checkpoint()

        snap = self._ir_snapshot()
        count = len(rng)
        var = prog.instrs[for_idx + 1][1]
        self.b.emit("scf.for", (), (), {"lb": 0, "ub": count, "step": 1, "var": var})

        # The iterator stays below the loop value, where ``break`` pops it.
        stack.append(rng[0] if count else 0)
        try:
            self._loop(prog, for_idx + 1, for_idx, exit_idx, for_idx)
            ok = (
                len(stack) == len(saved_stack)
                and self._python_state_unchanged(saved_locals)
                and all(_container_unchanged(obj, copy) for obj, copy in containers)
            )
        except _NotRollableError:
            ok = False

        if not ok:
            self.b.rollback(checkpoint)
            stack[:] = saved_stack
            self.locals.clear()
            self.locals.update(saved_locals)
            for obj, copy in containers:
                _restore_container(obj, copy)
            return False

        delta = self._ir_delta(snap)
        if delta:
            names = sorted(delta)
            vals = tuple(delta[n] for n in names)
            self.b.emit("scf.yield", vals, ())
            results = self.b.emit(
                "scf.for_end", (), tuple(v.vtype for v in vals), {"iter_args": names}
            )
            for n, r in zip(names, results):
                self.locals[n] = r
        else:
            self.b.emit("scf.for_end", (), ())

        if count:
            self.locals[var] = rng[-1]
        elif var in saved_locals:
            self.locals[var] = saved_locals[var]
        else:
            self.locals.pop(var, None)
        return True

    def _python_state_unchanged(self, before: dict[str, StackVal]) -> bool:
        after = self.locals
        for name, old in before.items():
            if isinstance(old, IRValue):
                if name in after and not isinstance(after[name], IRValue):
                    return False
                continue
            if name not in after:
                return False
            new = after[name]
            if new is not old and not (
                type(new) is type(old) and isinstance(old, _SCALARS) and new == old
            ):
                return False
        return True

    # ── Global / closure resolution ──────────────────────────────────

    def _resolve_global(self, name: str) -> Any:
        """Resolve a global name to its Python value."""
        if name in self._fn_globals:
            return self._fn_globals[name]
        if name in self._fn_closures:
            value = self._fn_closures[name]
            if value is _EMPTY_CELL:
                raise CompilationError(f"closure variable {name!r} is referenced before assignment")
            return value
        if name in self._builtins:
            return self._builtins[name]
        return ("__unresolved__", name)

    # ── Call dispatch ────────────────────────────────────────────────

    def _dispatch_call(
        self,
        callable_val: Any,
        call_args: list[StackVal],
        kwargs: dict[str, StackVal] | None,
    ) -> None:
        """Route a call to pulse IR emission or Python evaluation."""

        # Unresolved sentinel (fallback from _resolve_global)
        if isinstance(callable_val, tuple) and len(callable_val) == 2:
            tag, name = callable_val
            if tag == "__unresolved__":
                if kwargs:
                    from .. import ops

                    op_fn = getattr(ops, name, None) if name in _PULSE_OP_NAMES else None
                    if op_fn is None:
                        raise CompilationError(f"unknown pulse op: {name}")
                    call_args = _bind_positional(op_fn, call_args, kwargs)
                if name in OP_TABLE or name in _ALLOC_OPS:
                    self._push_pulse_results(name, call_args)
                elif name == "range":
                    self._push_range(call_args)
                else:
                    raise CompilationError(f"unknown pulse op: {name}")
                return

        # Resolved callable: a DSL op only if it is the DSL function itself,
        # so a user function that happens to be named ``gaussian`` stays Python.
        try:
            # Functions hash and compare by identity.
            op_name = _PULSE_FUNCTIONS.get(callable_val)
        except TypeError:  # unhashable callable
            op_name = None
        if op_name is not None:
            if kwargs:
                call_args = _bind_positional(callable_val, call_args, kwargs)
            # Qudit/qvec allocation (check before OP_TABLE so qvec_ref gets expanded)
            if op_name in _ALLOC_OPS:
                self._push_alloc_results(op_name, call_args)
            else:
                self._push_pulse_results(op_name, call_args)
            return

        fname = getattr(callable_val, "__name__", "")
        if callable_val is range:
            if kwargs:
                raise CompilationError("range() takes no keyword arguments")
            self._push_range(call_args)
            return

        # Inline user helpers which consume IR values or call the pulse DSL.
        # Executing these as ordinary Python would invoke pulse operations
        # outside the compiler context and lose their IR effects.
        if isinstance(callable_val, types.FunctionType) and (
            any(isinstance(arg, IRValue) for arg in call_args)
            or (kwargs and any(isinstance(v, IRValue) for v in kwargs.values()))
            or _uses_pulse_ops(callable_val)
        ):
            self._inline_helper(callable_val, call_args, kwargs)
            return

        # Regular Python callable (int, float, len, abs, math.sin, etc.)
        # int()/float() on a symbolic value create an explicit IR cast.
        if callable(callable_val):
            if not kwargs and len(call_args) == 1 and is_symbolic(call_args[0]):
                if callable_val is int:
                    self.stack.append(call_args[0].cast("i64"))
                    return
                if callable_val is float:
                    self.stack.append(call_args[0].cast("f64"))
                    return
            if any(is_symbolic(a) for a in call_args) or (
                kwargs and any(is_symbolic(v) for v in kwargs.values())
            ):
                raise CompilationError(
                    f"Cannot call {fname}() with symbolic parameter "
                    "expressions. Only int(), float(), arithmetic, and pulse "
                    "numeric arguments are supported."
                )
            try:
                result = callable_val(*call_args, **(kwargs or {}))
            except _EVALUATION_ERRORS as e:
                raise CompilationError(f"failed to evaluate {fname}({call_args!r}): {e}") from e
            self.stack.append(result)
            return

        raise CompilationError(f"cannot call: {callable_val!r}")

    def _push_range(self, call_args: list[StackVal]) -> None:
        if 1 <= len(call_args) <= 3 and all(isinstance(a, int) for a in call_args):
            try:
                self.stack.append(range(*call_args))
            except ValueError as exc:
                raise CompilationError(str(exc)) from exc
        else:
            self.stack.append(_BAD_ITER)

    def _inline_helper(
        self,
        fn: types.FunctionType,
        call_args: list[StackVal],
        kwargs: dict[str, StackVal] | None,
    ) -> None:
        """Emit a pulse-aware Python helper into the current IR builder."""
        code = fn.__code__
        params = list(code.co_varnames[: code.co_argcount])
        if kwargs or fn.__defaults__:
            call_args = _bind_positional(fn, call_args, kwargs or {})
        if len(call_args) != len(params):
            raise CompilationError(
                f"{fn.__name__}: expected {len(params)} args, got {len(call_args)}"
            )

        helper = PulseIREmitter(
            self.b,
            fn_globals=fn.__globals__,
            fn_closures=_extract_closures(fn),
        )
        helper.locals.update(zip(params, call_args))
        result = helper.run(code)

        # Propagate rebindings of directly passed linear values back to the
        # caller, then leave the helper's return value on the caller stack.
        for name, original in zip(params, call_args):
            updated = helper.locals.get(name)
            if not isinstance(original, IRValue) or updated is original:
                continue
            for caller_name, caller_value in self.locals.items():
                if caller_value is original:
                    self.locals[caller_name] = updated

        self.stack.append(result)

    def _push_pulse_results(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> None:
        results = self._emit_pulse_call(fname, call_args)
        if fname == "readout":
            self.stack.append(results[-1])
        elif fname in {"drive", "wait", "sync"}:
            self.stack.append(None)
        elif len(results) == 1:
            self.stack.append(results[0])
        elif len(results) > 1:
            self.stack.append(results)
        else:
            self.stack.append(None)

    def _push_alloc_results(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> None:
        """Handle qudit_ref() and qvec_ref(n) allocation inside kernels."""
        if fname == "qvec_ref":
            if len(call_args) == 1 and isinstance(call_args[0], int):
                n = call_args[0]
                ir_vals = []
                for i in range(n):
                    (v,) = self.b.emit("pulse.qudit_alloc", (), ("qref",), {"index": i})
                    ir_vals.append(v)
                self.stack.append(ir_vals)
                return
        # qudit_ref() or fallback qvec_ref
        results = self._emit_pulse_call(fname, call_args)
        if len(results) == 1:
            self.stack.append(results[0])
        elif len(results) > 1:
            self.stack.append(results)
        else:
            self.stack.append(None)

    # ── Pulse IR emission ────────────────────────────────────────────

    def _emit_pulse_call(
        self,
        fname: str,
        call_args: list[StackVal],
    ) -> tuple[IRValue, ...]:
        entry = OP_TABLE.get(fname)
        if entry is None:
            raise CompilationError(f"unknown pulse op: {fname}")

        n_val, attr_names, rtypes = entry

        if n_val == -1:
            operands = tuple(v for v in call_args if isinstance(v, IRValue))
            attrs: dict[str, Any] = {}
            out_types = tuple(v.vtype for v in operands) if rtypes is None else rtypes
        else:
            operands = tuple(call_args[:n_val])
            attr_vals = call_args[n_val:]
            if len(attr_vals) != len(attr_names):
                raise CompilationError(
                    f"{fname}: expected {n_val + len(attr_names)} args, got {len(call_args)}"
                )
            attrs = dict(zip(attr_names, attr_vals))
            out_types = (operands[0].vtype,) if (rtypes is None and operands) else (rtypes or ())

        op_name = _ALLOC_OPS.get(fname, f"pulse.{fname}")
        results: tuple[IRValue, ...] = self.b.emit(op_name, operands, out_types, attrs)

        self._rebind_linear(fname, call_args, results)

        return results

    def _rebind_linear(
        self,
        fname: str,
        call_args: list[StackVal],
        results: tuple[IRValue, ...],
    ) -> None:
        """Rebind local names for linear-typed results (drive_line, tone, etc)."""
        arg_info: list[tuple[str | None, IRValue | None]] = []
        for a in call_args:
            if isinstance(a, IRValue):
                name = self._find_local_name(a)
                arg_info.append((name, a))
            else:
                arg_info.append((None, None))

        claimed: set[int] = set()
        for res in results:
            if res.vtype not in LINEAR_TYPES:
                continue
            for i, (name, op_val) in enumerate(arg_info):
                if i in claimed or op_val is None or name is None:
                    continue
                if op_val.vtype == res.vtype:
                    self.locals[name] = res
                    claimed.add(i)
                    break

    def _find_local_name(self, val: IRValue) -> str | None:
        """Find the local variable name bound to a given IRValue."""
        for name, v in self.locals.items():
            if v is val:
                return name
        return None

    # ── Snapshot/delta for structured control flow ───────────────────

    def _ir_snapshot(self) -> dict[str, IRValue]:
        return {k: v for k, v in self.locals.items() if isinstance(v, IRValue)}

    def _ir_delta(self, snap: dict[str, IRValue]) -> dict[str, IRValue]:
        return {
            k: v
            for k, v in self.locals.items()
            if isinstance(v, IRValue) and (k not in snap or snap[k] is not v)
        }


def _container_unchanged(obj: list[Any] | dict[Any, Any], copy: Any) -> bool:
    if len(obj) != len(copy):
        return False
    if isinstance(obj, list):
        return all(a is b for a, b in zip(obj, copy))
    return all(k in copy and obj[k] is copy[k] for k in obj)


def _restore_container(obj: list[Any] | dict[Any, Any], copy: Any) -> None:
    if isinstance(obj, list):
        obj[:] = copy
    else:
        obj.clear()
        obj.update(copy)


_SIGNATURES: dict[int, tuple[Any, inspect.Signature]] = {}


def _bind_positional(fn: Any, args: list[StackVal], kwargs: dict[str, StackVal]) -> list[StackVal]:
    """Bind a keyword call to *fn*'s parameters as a positional list."""
    hit = _SIGNATURES.get(id(fn))
    if hit is None or hit[0] is not fn:
        try:
            hit = (fn, inspect.signature(fn))
        except (TypeError, ValueError) as e:
            raise CompilationError(f"cannot bind keyword arguments for {fn!r}") from e
        _SIGNATURES[id(fn)] = hit
    try:
        bound = hit[1].bind(*args, **kwargs)
    except TypeError as e:
        raise CompilationError(f"{getattr(fn, '__name__', fn)}: {e}") from e
    bound.apply_defaults()
    if bound.kwargs:
        raise CompilationError(f"{getattr(fn, '__name__', fn)}: keyword-only parameters")
    return list(bound.args)


__all__ = ["PulseIREmitter", "PythonIRBuilder"]
