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
_PACKED_BRANCH_MESSAGE = (
    "measurement-conditioned branches are retained in kernel IR (call the kernel "
    "or lower it with the Program passes), but compile() writes a packed program "
    "for the MLIR pipeline, which cannot represent them yet"
)
_REJOIN_MESSAGE = (
    "both branches of a measurement-conditioned if must continue after it; "
    "break, continue or return in only one branch is not supported"
)
_BRANCH_TYPE_MESSAGE = (
    "only measurement results can control a branch at runtime; got an IR value of type {}"
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
    if isinstance(value, (IRValue, _NotMeasurement)):
        raise CompilationError(
            "a measurement result is only known at runtime; use it directly as "
            "an if condition (`if result:` or `if not result:`)"
        )
    return bool(value)


class _NotMeasurement:
    """``not result`` for a measurement result, pending use as a condition."""

    __slots__ = ("value",)

    def __init__(self, value: IRValue):
        self.value = value


class _RegionDone:
    """Returned by ``_loop`` when an if-arm reaches its join point."""

    __slots__ = ()


_REGION_DONE = _RegionDone()


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

    __slots__ = (
        "b",
        "stack",
        "locals",
        "_fn_globals",
        "_fn_closures",
        "_builtins",
        "_can_roll",
        "_slots",
    )

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
        # id(line or tone) -> (list, index, value) for values read out of a
        # list, so the op that consumes one can rebind the list slot.
        self._slots: dict[int, tuple[list[Any], int, IRValue]] = {}
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
        # Tracing hot loop: opcodes bound as defaults are plain local reads, and
        # are small ints, which CPython caches, so ``is`` compares them exactly.
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
        _return: int = bc.RETURN,
        _build_tuple: int = bc.BUILD_TUPLE,
        _build_list: int = bc.BUILD_LIST,
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

                if op is _load_fast:
                    try:
                        push(local_vars[arg])
                    except KeyError:
                        raise CompilationError(f"undefined local: {arg}") from None

                elif op is _load_global:
                    value = fn_globals.get(arg, _MISSING)
                    push(self._resolve_global(arg) if value is _MISSING else value)

                elif op is _call:
                    if arg:
                        call_args = stack[-arg:]
                        del stack[-arg:]
                    else:
                        call_args = []
                    self._dispatch_call(pop(), call_args, None)

                elif op is _pop_top:
                    pop()

                elif op is _load_deref:
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

                elif op is _load_const:
                    push(arg)

                elif op is _store_fast:
                    local_vars[arg] = pop()

                elif op is _return:
                    if header >= 0:
                        raise _NotRollableError
                    return pop()

                elif op is _load_attr:
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

                elif op is _subscr:
                    index = pop()
                    obj = pop()
                    try:
                        value = obj[index]
                    except (TypeError, IndexError, KeyError) as e:
                        raise CompilationError(
                            f"subscript failed: {type(obj).__name__}[{index!r}]: {e}"
                        ) from e
                    push(value)
                    if (
                        value.__class__ is IRValue
                        and obj.__class__ is list
                        and value.vtype in LINEAR_TYPES
                    ):
                        self._slots[id(value)] = (obj, index, value)

                elif op is _binary or op is _compare:
                    right = pop()
                    left = pop()
                    push(arg(left, right))

                elif op is _jif or op is _jit:
                    cond = pop()
                    if cond.__class__ is not bool:
                        if isinstance(cond, (IRValue, _NotMeasurement)):
                            pc = self._runtime_branch(prog, pc - 1, cond, op is _jit)
                            pc = self._jump(pc, lo, hi, header)
                            if pc < 0:
                                return None
                            continue
                        cond = _truth(cond)
                    if cond == (op is _jit):
                        pc = self._jump(arg, lo, hi, header)
                        if pc < 0:
                            return None

                elif op is _jump:
                    pc = self._jump(arg, lo, hi, header)
                    if pc < 0:
                        return None

                elif op is _for_iter:
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

                elif op is _get_iter:
                    value = pop()
                    push(_Loop(value) if isinstance(value, range) else _BAD_ITER)
                    if _iter_slots > 1:
                        stack.extend([None] * (_iter_slots - 1))

                elif op is _unpack:
                    val = pop()
                    if not isinstance(val, (tuple, list)):
                        raise CompilationError(f"cannot unpack {type(val)}")
                    if len(val) != arg:
                        raise CompilationError(f"unpack mismatch: expected {arg}, got {len(val)}")
                    stack.extend(reversed(val))

                elif op is _build_tuple or op is _build_list:
                    if arg:
                        items = stack[-arg:]
                        del stack[-arg:]
                    else:
                        items = []
                    if op is _build_tuple:
                        push(tuple(items))
                    else:
                        push(items)
                        self._track_slots(items, 0)

                elif op is bc.LIST_APPEND or op is bc.LIST_EXTEND:
                    val = pop()
                    lst = stack[-arg]
                    if not isinstance(lst, list):
                        raise CompilationError(f"list update on non-list: {type(lst)}")
                    if op is bc.LIST_APPEND:
                        lst.append(val)
                    else:
                        lst.extend(val)

                elif op is bc.UNARY:
                    push(arg(pop()))

                elif op is bc.NOT:
                    value = pop()
                    if isinstance(value, IRValue):
                        push(_NotMeasurement(value))
                    elif isinstance(value, _NotMeasurement):
                        push(value.value)
                    else:
                        push(not _truth(value))

                elif op is bc.IS:
                    right = pop()
                    left = pop()
                    push((left is right) != arg)

                elif op is bc.CONTAINS:
                    right = pop()
                    left = pop()
                    push((left in right) != arg)

                elif op is bc.CALL_KW:
                    names = pop()
                    values = stack[-arg:]
                    del stack[-arg:]
                    n_pos = arg - len(names)
                    kwargs = dict(zip(names, values[n_pos:]))
                    self._dispatch_call(pop(), values[:n_pos], kwargs)

                elif op is bc.COPY:
                    push(stack[-arg])

                elif op is bc.SWAP:
                    stack[-1], stack[-arg] = stack[-arg], stack[-1]

                elif op is bc.JUMP_IF_NONE or op is bc.JUMP_IF_NOT_NONE:
                    if (pop() is None) == (op is bc.JUMP_IF_NONE):
                        pc = self._jump(arg, lo, hi, header)
                        if pc < 0:
                            return None

                elif op is bc.STORE_DEREF:
                    local_vars[arg] = pop()

                elif op is bc.DELETE_FAST:
                    local_vars.pop(arg, None)

                elif op is bc.BUILD_SLICE:
                    items = stack[-arg:]
                    del stack[-arg:]
                    push(slice(*items))

                elif op is bc.STORE_SUBSCR:
                    index = pop()
                    obj = pop()
                    value = pop()
                    obj[index] = value

                elif op is bc.BUILD_CONST_KEY_MAP:
                    keys = pop()
                    vals = stack[-arg:] if arg else []
                    del stack[len(stack) - arg :]
                    push(dict(zip(keys, vals)))

                elif op is bc.BUILD_MAP:
                    items = stack[len(stack) - 2 * arg :]
                    del stack[len(stack) - 2 * arg :]
                    push(dict(zip(items[::2], items[1::2])))

                elif op is bc.BUILD_STRING:
                    parts = stack[-arg:] if arg else []
                    del stack[len(stack) - arg :]
                    push("".join(parts))

                elif op is bc.CONVERT:
                    push((str, repr, ascii)[arg - 1](pop()))

                elif op is bc.FORMAT:
                    spec = pop() if arg else ""
                    push(format(pop(), spec))

                elif op is bc.IMPORT_NAME:
                    fromlist = pop()
                    level = pop()
                    push(__import__(arg, self._fn_globals, None, fromlist, level))

                elif op is bc.IMPORT_FROM:
                    push(getattr(stack[-1], arg))

                elif op is bc.RAISE:
                    exc = pop() if arg else None
                    raise CompilationError(f"kernel raised {exc!r}")

                elif op is bc.UNSUPPORTED:
                    raise CompilationError(arg)

                elif op is bc.REGION_END:
                    return _REGION_DONE

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

    # ── Measurement-conditioned branches ─────────────────────────────

    def _runtime_branch(
        self, prog: bc.DecodedCode, index: int, cond: Any, jump_if_true: bool
    ) -> int:
        """Emit the ``if`` whose conditional jump is ``prog.instrs[index]``.

        Both arms are traced from the same state into an ``scf.if`` region.
        Pulse resources rebound by either arm become results of the region;
        any other value must end up the same in both arms. Returns the
        instruction index where tracing continues.
        """
        negated = isinstance(cond, _NotMeasurement)
        value = cond.value if negated else cond
        if value.vtype != "measurement":
            raise CompilationError(_BRANCH_TYPE_MESSAGE.format(value.vtype))
        if not self._can_roll:
            raise CompilationError(_PACKED_BRANCH_MESSAGE)

        target = prog.instrs[index][1]
        then_start, else_start = (target, index + 1) if jump_if_true else (index + 1, target)
        if negated:
            then_start, else_start = else_start, then_start
        join = bc.join_point(prog, index)
        view = bc.region_view(prog, join)
        returns = join >= len(prog.instrs)

        builder = self.b
        stack = self.stack
        saved_stack = list(stack)
        saved_locals = dict(self.locals)
        containers = [
            (v, list(v) if isinstance(v, list) else dict(v))
            for v in saved_locals.values()
            if isinstance(v, (list, dict))
        ]

        builder.emit("scf.if", (value,), ())
        then_locals, then_slots = self._trace_arm(
            view, index, join, then_start, saved_stack, containers
        )
        stack[:] = saved_stack
        self.locals.clear()
        self.locals.update(saved_locals)
        for obj, copy in containers:
            _restore_container(obj, copy)
        else_index = len(builder.ops)
        builder.emit("scf.else", (), ())
        else_locals, else_slots = self._trace_arm(
            view, index, join, else_start, saved_stack, containers
        )

        merged: dict[str, StackVal] = {}
        results: list[tuple[str, IRValue, IRValue]] = []
        for name in sorted(then_locals.keys() | else_locals.keys()):
            a = then_locals.get(name, _MISSING)
            b = else_locals.get(name, _MISSING)
            if a is b:
                merged[name] = a
            elif isinstance(a, IRValue) and isinstance(b, IRValue) and a.vtype == b.vtype:
                results.append((name, a, b))
            elif a is _MISSING or b is _MISSING:
                continue  # bound in one arm only: unbound after the if
            elif type(a) is type(b) and isinstance(a, _SCALARS) and a == b:
                merged[name] = a
            else:
                raise CompilationError(
                    f"{name!r} would depend on a measurement result; only pulse "
                    "resources (lines, tones, waveforms) may differ between the "
                    "arms of a measurement-conditioned if"
                )
        # List slots holding lines or tones that either arm rebound.
        slot_results: list[tuple[list[Any], int, IRValue, IRValue]] = []
        originals = {id(obj): copy for obj, copy in containers}
        keys: dict[tuple[int, int], list[Any]] = {}
        for obj, i, _ in then_slots + else_slots:
            keys[(id(obj), i)] = obj
        then_new = {(id(obj), i): new for obj, i, new in then_slots}
        else_new = {(id(obj), i): new for obj, i, new in else_slots}
        for key, obj in keys.items():
            old = originals[key[0]][key[1]]
            a = then_new.get(key, old)
            b = else_new.get(key, old)
            if a is not b:
                slot_results.append((obj, key[1], a, b))

        labels = [name for name, _, _ in results]
        labels += [self._slot_label(obj, i) for obj, i, _, _ in slot_results]
        then_vals = tuple(a for _, a, _ in results) + tuple(a for _, _, a, _ in slot_results)
        else_vals = tuple(b for _, _, b in results) + tuple(b for _, _, _, b in slot_results)
        builder.insert(else_index, "scf.yield", then_vals)
        builder.emit("scf.yield", else_vals, ())
        outs = builder.emit(
            "scf.if_end",
            (),
            tuple(v.vtype for v in then_vals),
            {"result_names": labels},
        )
        for (name, _, _), out in zip(results, outs):
            merged[name] = out
        for (obj, i, _, _), out in zip(slot_results, outs[len(results) :]):
            obj[i] = out
            self._slots[id(out)] = (obj, i, out)
        self.locals.clear()
        self.locals.update(merged)
        stack[:] = saved_stack
        # Both arms returned: continue at the trailing ``return None``.
        return len(prog.instrs) - 2 if returns else join

    def _trace_arm(
        self,
        view: bc.DecodedCode,
        branch: int,
        join: int,
        start: int,
        saved_stack: list[StackVal],
        containers: list[tuple[Any, Any]],
    ) -> tuple[dict[str, StackVal], list[tuple[list[Any], int, IRValue]]]:
        """Trace one arm until it reaches *join*; return its locals and rebound slots."""
        returns = join >= len(view.instrs)
        # An arm may only move forward from the branch, except that a join at
        # an enclosing loop header (both arms continue) ends the arm.
        header = join if join < branch else -1
        try:
            outcome = self._loop(view, start, branch, len(view.instrs), header)
        except _NotRollableError:
            raise CompilationError(_REJOIN_MESSAGE) from None
        if header >= 0 and outcome is None:
            outcome = _REGION_DONE
        if returns:
            if outcome is not None:
                raise CompilationError("a measurement-conditioned branch cannot return a value")
        elif outcome is not _REGION_DONE or len(self.stack) != len(saved_stack):
            raise CompilationError(_REJOIN_MESSAGE)
        slots = _slot_changes(containers)
        if slots is None:
            raise CompilationError(
                "a measurement-conditioned branch may only change list entries that "
                "hold lines or tones"
            )
        return dict(self.locals), slots

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
            slot_changes = _slot_changes(containers)
            ok = (
                len(stack) == len(saved_stack)
                and self._python_state_unchanged(saved_locals)
                and slot_changes is not None
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
        assert slot_changes is not None
        if delta or slot_changes:
            names = sorted(delta)
            vals = tuple(delta[n] for n in names) + tuple(new for _, _, new in slot_changes)
            labels = names + [self._slot_label(obj, i) for obj, i, _ in slot_changes]
            self.b.emit("scf.yield", vals, ())
            results = self.b.emit(
                "scf.for_end", (), tuple(v.vtype for v in vals), {"iter_args": labels}
            )
            for n, r in zip(names, results):
                self.locals[n] = r
            for (obj, i, _), r in zip(slot_changes, results[len(names) :]):
                obj[i] = r
                self._slots[id(r)] = (obj, i, r)
        else:
            self.b.emit("scf.for_end", (), ())

        if count:
            self.locals[var] = rng[-1]
        elif var in saved_locals:
            self.locals[var] = saved_locals[var]
        else:
            self.locals.pop(var, None)
        return True

    def _slot_label(self, obj: list[Any], index: int) -> str:
        name = next((n for n, v in self.locals.items() if v is obj), "")
        return f"{name}[{index}]"

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
            owner = getattr(callable_val, "__self__", None)
            if owner.__class__ is list and fname in ("append", "extend", "insert"):
                self._track_slots(owner, len(owner) - 1 if fname == "append" else 0)
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
        helper._slots = self._slots
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
        """Rebind the locals and list slots holding consumed linear values.

        Each linear result (drive_line, tone, ...) replaces the first
        unclaimed operand of the same type, wherever the kernel holds it.
        """
        slots = self._slots
        if fname == "drive" and len(results) == 2:
            # drive(line, waveform, tone) -> (line, tone): the common case.
            self._rebind_one(call_args[0], results[0])
            self._rebind_one(call_args[2], results[1])
            return
        arg_info: list[tuple[str | None, IRValue | None, Any]] = []
        for a in call_args:
            if isinstance(a, IRValue):
                slot = slots.get(id(a))
                if slot is not None and (slot[2] is not a or slot[0][slot[1]] is not a):
                    slot = None
                arg_info.append((self._find_local_name(a), a, slot))
            else:
                arg_info.append((None, None, None))

        claimed: set[int] = set()
        for res in results:
            if res.vtype not in LINEAR_TYPES:
                continue
            for i, (name, op_val, slot) in enumerate(arg_info):
                if i in claimed or op_val is None or (name is None and slot is None):
                    continue
                if op_val.vtype == res.vtype:
                    if name is not None:
                        self.locals[name] = res
                    if slot is not None:
                        slot[0][slot[1]] = res
                        slots[id(res)] = (slot[0], slot[1], res)
                    claimed.add(i)
                    break

    def _track_slots(self, items: list[Any], start: int) -> None:
        """Remember the list slots from *start* on that hold lines or tones."""
        slots = self._slots
        for i in range(start, len(items)):
            v = items[i]
            if v.__class__ is IRValue and v.vtype in LINEAR_TYPES:
                slots[id(v)] = (items, i, v)

    def _rebind_one(self, old: Any, new: IRValue) -> None:
        if old.__class__ is not IRValue or old.vtype != new.vtype:
            return
        name = self._find_local_name(old)
        if name is not None:
            self.locals[name] = new
        slot = self._slots.get(id(old))
        if slot is not None and slot[2] is old and slot[0][slot[1]] is old:
            slot[0][slot[1]] = new
            self._slots[id(new)] = (slot[0], slot[1], new)

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


def _slot_changes(
    containers: list[tuple[Any, Any]],
) -> list[tuple[list[Any], int, IRValue]] | None:
    """List slots whose linear value was replaced, or None for any other change."""
    changes: list[tuple[list[Any], int, IRValue]] = []
    for obj, copy in containers:
        if len(obj) != len(copy):
            return None
        if not isinstance(obj, list):
            if not _container_unchanged(obj, copy):
                return None
            continue
        for i, (new, old) in enumerate(zip(obj, copy)):
            if new is old:
                continue
            if not (
                isinstance(new, IRValue)
                and isinstance(old, IRValue)
                and new.vtype == old.vtype
                and new.vtype in LINEAR_TYPES
            ):
                return None
            changes.append((obj, i, new))
    return changes


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
