# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Kernel objects: calling, compiling and caching pulse kernels.

Calling a kernel binds its arguments as Python binds a call, then splits
them by annotation. A ``float`` parameter is a runtime value: it stays a
function argument of the compiled module, so a new value never recompiles.
Every other parameter (qudits, ints, anything unannotated) is static: it can
shape the program, so the kernel is specialized on its value.

A specialization is found by its static values and the target, and is used
while its guard holds: the globals, closure cells and attribute chains the
kernel read when it was traced are re-read and compared on every call
(``_guard.py``). A miss traces and builds the module under the dispatcher's
lock; a hit takes no lock.
"""

from __future__ import annotations

import atexit
import copy
import inspect
import itertools
import os
import threading
import types
import warnings
import weakref
from collections.abc import Callable
from dataclasses import replace
from typing import Any, NamedTuple

from .._device import Device, DeviceMismatchError, NoDeviceError, Qudit, Register
from ..compiler import CompileOptions, Lowered, build
from ._guard import Guard, build_guard
from ._schedule_blocks import root_schedule, validate_kernel_schedule
from ._unroll import resolve_unroll_threshold
from .ir_builder import CompilationError, IRValue, Parameter, SymbolicValueError, is_symbolic

CACHE_SIZE_ENV = "CUDAQ_PULSE_CACHE_SIZE"
DEFAULT_CACHE_SIZE = 32

_ticks = itertools.count()


# A kernel's guard holds its module globals, which hold the kernel: release
# the compiled modules before the interpreter tears the extension down.
_dispatchers: weakref.WeakSet[_Dispatcher] = weakref.WeakSet()


@atexit.register
def _release_compiled_modules() -> None:
    for dispatcher in list(_dispatchers):
        dispatcher._entries.clear()
        dispatcher.general = None


def _cache_size(value: int | None) -> int:
    if value is None:
        raw = os.environ.get(CACHE_SIZE_ENV, "").strip()
        if not raw:
            return DEFAULT_CACHE_SIZE
        if not raw.isdigit() or int(raw) < 1:
            raise ValueError(f"{CACHE_SIZE_ENV} must be a positive integer, got {raw!r}")
        return int(raw)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"cache_size must be a positive integer, got {value!r}")
    return value


def _is_runtime(annotation: Any) -> bool:
    return annotation is float or annotation == "float"


def _make_binder(fn: types.FunctionType) -> Callable[..., tuple[Any, ...]]:
    """A function with *fn*'s signature that returns its bound arguments.

    Python itself binds the call, so defaults, keyword-only parameters and
    every ``TypeError`` behave exactly as for *fn*.
    """
    code = fn.__code__
    names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
    npos = code.co_posonlyargcount
    params = list(names[: code.co_argcount])
    if npos:
        params.insert(npos, "/")
    if code.co_kwonlyargcount:
        params.append("*")
        params.extend(names[code.co_argcount :])
    body = f"({', '.join(names)},)" if names else "()"
    src = f"def _bind({', '.join(params)}):\n    return {body}\n"
    namespace: dict[str, Any] = {}
    exec(compile(src, code.co_filename, "exec"), namespace)
    binder: Any = namespace["_bind"]
    binder.__code__ = binder.__code__.replace(co_name=code.co_name, co_qualname=code.co_qualname)
    binder.__name__ = fn.__name__
    binder.__qualname__ = fn.__qualname__
    binder.__defaults__ = fn.__defaults__
    binder.__kwdefaults__ = fn.__kwdefaults__
    return binder  # type: ignore[no-any-return]


def _static_key(value: Any) -> Any:
    t = type(value)
    if t is Qudit:
        return (Qudit, value.carrier, value.index)
    if t is Register:
        return (Register, tuple((q.carrier, q.index) for q in value))
    if t in (int, bool, float, str, complex, type(None)):
        return (t, value)
    if t in (tuple, list):
        return (t, tuple(_static_key(v) for v in value))
    if t is dict:
        return (t, tuple((k, _static_key(v)) for k, v in value.items()))
    try:
        hash(value)
    except TypeError:
        return (id, id(value))
    return (t, value)


# ── Return values ────────────────────────────────────────────────────────


class _Measured(NamedTuple):
    readout: int


class _Sequence(NamedTuple):
    kind: type
    items: tuple[Any, ...]


_NO_RETURN = object()


def _return_spec(value: Any, readouts: dict[int, int], name: str) -> Any:
    """What a call returns, from the value the trace returned."""
    if isinstance(value, IRValue):
        if value.vtype == "measurement":
            ordinal = readouts.get(value.vid)
            if ordinal is None:
                raise CompilationError(
                    f"{name}() returns a measurement made inside a loop or branch; "
                    "only measurements made at the top level of the kernel can be returned"
                )
            return _Measured(ordinal)
        raise CompilationError(
            f"{name}() returns a pulse {value.vtype} value; a kernel returns "
            "measurements or values known at compile time"
        )
    if is_symbolic(value):
        raise CompilationError(
            f"{name}() returns a value computed from a float parameter, which is "
            "only known when the kernel runs"
        )
    if type(value) in (tuple, list):
        items = tuple(_return_spec(v, readouts, name) for v in value)
        if any(isinstance(i, (_Measured, _Sequence)) for i in items):
            return _Sequence(type(value), items)
    return value


def _has_measurements(spec: Any) -> bool:
    if isinstance(spec, _Measured):
        return True
    return isinstance(spec, _Sequence) and any(_has_measurements(i) for i in spec.items)


def _materialize(spec: Any, outcome: Callable[[int], int]) -> Any:
    if isinstance(spec, _Measured):
        return outcome(spec.readout)
    if isinstance(spec, _Sequence):
        return spec.kind(_materialize(i, outcome) for i in spec.items)
    # Each call returns its own copy of a container built by the kernel.
    return spec if isinstance(spec, (int, float, complex, str, type(None))) else copy.deepcopy(spec)


def _execute(lowered: Lowered, returns: Any) -> Any:
    from ..runtime.simulate import _simulate_lowered

    measured = _has_measurements(returns)
    result = _simulate_lowered(lowered, shots=1 if measured else None)
    if not measured:
        return _materialize(returns, int)
    assert result.records is not None
    row = result.records[0]
    column = {m.readout: i for i, m in enumerate(result.measurements) if not m.iteration}

    def outcome(readout: int) -> int:
        i = column.get(readout)
        return -1 if i is None else int(row[i])

    return _materialize(returns, outcome)


def _devices(value: Any, found: Device | None, name: str, seen: set[int]) -> Device | None:
    """The device of the handles in *value*, checked against *found*.

    *seen* collects the carriers passed so far: one qudit passed twice is
    an error, as it would be one degree of freedom bound to two names.
    """
    t = type(value)
    if t is Qudit:
        if value.index in seen:
            raise TypeError(f"{name}() got qudit {value.logical} more than once")
        seen.add(value.index)
        device: Device = value.device
        if found is None or found is device:
            return device
        if found != device:
            raise DeviceMismatchError(
                f"{name}() got qudits of two devices, {found.name!r} and {device.name!r}"
            )
        return found
    if t is Register or t is tuple or t is list:
        for item in value:
            found = _devices(item, found, name, seen)
    return found


# ── Specializations ──────────────────────────────────────────────────────


class CacheInfo(NamedTuple):
    size: int
    compiled: int
    evicted: int
    capacity: int | None


class _Entry:
    """One compiled specialization and the guard that keeps it valid."""

    __slots__ = ("lowered", "guard", "returns", "tick", "pins", "runtime")

    def __init__(
        self,
        lowered: Lowered,
        guard: Guard,
        returns: Any,
        pins: tuple[Any, ...],
        runtime: tuple[int, ...],
    ):
        self.lowered = lowered
        self.guard = guard
        self.returns = returns
        self.tick = next(_ticks)
        # The parameters that are function arguments of the module.
        self.runtime = runtime
        # Static arguments keyed by identity are kept alive with the entry.
        self.pins = pins


class _Dispatcher:
    """Binds a call, finds or builds its specialization, and runs it."""

    def __init__(self, kernel: Kernel[Any], options: CompileOptions, capacity: int | None) -> None:
        self.kernel = kernel
        self.options = options
        self._capacity = capacity
        self._entries: dict[Any, _Entry] = {}
        self._lock = threading.Lock()
        self.misses = 0
        self.evictions = 0
        self.pin: Device | None = None
        # The specialization a pinned all-float kernel is built with.
        self.general: Lowered | None = None
        _dispatchers.add(self)

    def _device(self, values: tuple[Any, ...]) -> Device:
        name = self.kernel.__name__
        found = None
        seen: set[int] = set()
        for i in self.kernel._split[0]:
            found = _devices(values[i], found, name, seen)
        pin = self.pin
        if pin is not None:
            if found is not None and found != pin:
                raise DeviceMismatchError(
                    f"{name}() is compiled for device {pin.name!r} and got qudits of {found.name!r}"
                )
            return pin
        if found is not None:
            return found
        raise NoDeviceError(
            f"{name}() was called without qudit handles, so it has no device: pass "
            "handles of a device (dev.qudit_handles[i]), or compile it for one with "
            "cudaq_pulse.compile(k, device=dev)"
        )

    def lookup(self, values: tuple[Any, ...]) -> _Entry:
        kernel = self.kernel
        target = self._device(values)
        static = kernel._split[0]
        options = self.options
        unroll = options.unroll_threshold
        if unroll is None:
            unroll = resolve_unroll_threshold()
        key = (target.fingerprint, unroll, static, *[_static_key(values[i]) for i in static])
        entry = self._entries.get(key)
        if entry is not None and entry.guard.check():
            entry.tick = next(_ticks)
            return entry
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.guard.check():
                return entry
            if options.unroll_threshold is None:
                options = replace(options, unroll_threshold=unroll)
            entry = kernel._specialize(values, target, options)
            self.misses += 1
            # Tracing may have made float parameters static.
            static = kernel._split[0]
            key = (target.fingerprint, unroll, static, *[_static_key(values[i]) for i in static])
            self._insert(key, entry)
        return entry

    def _insert(self, key: Any, entry: _Entry) -> None:
        entries = self._entries
        entries[key] = entry
        if self._capacity is not None and len(entries) > self._capacity:
            oldest = min(entries, key=lambda k: entries[k].tick)
            del entries[oldest]
            self.evictions += 1

    def call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        kernel = self.kernel
        values = kernel._bind(*args, **kwargs)
        entry = self.lookup(values)
        lowered = entry.lowered
        if entry.runtime:
            lowered = lowered.bind(kernel._runtime_values(values, entry.runtime))
        return _execute(lowered, entry.returns)

    def lower(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Lowered:
        kernel = self.kernel
        values = kernel._bind(*args, **kwargs)
        entry = self.lookup(values)
        lowered = entry.lowered
        if entry.runtime:
            lowered = lowered.bind(kernel._runtime_values(values, entry.runtime))
        return lowered

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class Kernel[**P]:
    """A pulse kernel. ``k(*args)`` compiles on first use and runs it.

    ``k.lower(*args)`` returns the compiled module without running it.
    ``cudaq_pulse.compile(k)`` compiles ahead of the call. ``P`` is the
    decorated function's signature, so calls are checked against it.
    """

    __name__: str
    __qualname__: str

    def __init__(
        self,
        fn: Callable[P, Any],
        *,
        cache_size: int | None = None,
        schedule: str | None = None,
        unroll_threshold: int | None = None,
    ) -> None:
        if not isinstance(fn, types.FunctionType):
            raise TypeError(f"@kernel expects a function, got {type(fn).__name__}")
        code = fn.__code__
        if code.co_flags & (inspect.CO_VARARGS | inspect.CO_VARKEYWORDS):
            raise TypeError(f"pulse kernel {fn.__name__}() cannot take *args or **kwargs")
        self.__wrapped__ = fn
        self.__pulse_kernel_fn__ = fn
        # Called from another kernel, the kernel is scheduled on its own with
        # this policy; None schedules it with its caller.
        self.schedule = validate_kernel_schedule(schedule)
        # None takes the environment's or the default threshold per call.
        self.unroll_threshold = (
            None if unroll_threshold is None else resolve_unroll_threshold(unroll_threshold)
        )
        for attr in ("__module__", "__name__", "__qualname__", "__doc__"):
            setattr(self, attr, getattr(fn, attr))
        signature = inspect.signature(fn)
        self.__signature__ = signature
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        self._names = names
        runtime = tuple(
            i for i, n in enumerate(names) if _is_runtime(signature.parameters[n].annotation)
        )
        # (static, runtime) parameter indices, replaced as one value when a
        # float parameter turns out to shape the program.
        self._split = (tuple(i for i in range(len(names)) if i not in runtime), runtime)
        self._promote_lock = threading.Lock()
        self._bind = _make_binder(fn)
        self._dispatch = _Dispatcher(
            self,
            CompileOptions(
                schedule=root_schedule(self.schedule, None),
                unroll_threshold=self.unroll_threshold,
            ),
            _cache_size(cache_size),
        )

    @property
    def runtime_parameters(self) -> tuple[str, ...]:
        """Parameters annotated ``float``: function arguments of the module."""
        return tuple(self._names[i] for i in self._split[1])

    @property
    def static_parameters(self) -> tuple[str, ...]:
        """Parameters the kernel is specialized on.

        Includes float parameters whose value the program needs while it
        compiles: one that sets a waveform duration, goes through ``int()``
        or decides Python control flow. A wait length is not one of them.
        """
        return tuple(self._names[i] for i in self._split[0])

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        return self._dispatch.call(args, kwargs)

    def lower(self, *args: P.args, **kwargs: P.kwargs) -> Lowered:
        """Compile for these arguments without running; see ``Lowered``."""
        return self._dispatch.lower(args, kwargs)

    def cache_info(self) -> CacheInfo:
        """Specializations held, compiled and evicted by ``k(...)`` calls."""
        d = self._dispatch
        return CacheInfo(len(d._entries), d.misses, d.evictions, d._capacity)

    def cache_clear(self) -> None:
        self._dispatch.clear()

    def __repr__(self) -> str:
        return f"<pulse kernel {self.__qualname__}{self.__signature__}>"

    def _runtime_values(self, values: tuple[Any, ...], runtime: tuple[int, ...]) -> list[float]:
        out = []
        for i in runtime:
            v = values[i]
            if type(v) is not float:
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    try:
                        v = v.__float__()
                    except AttributeError:
                        raise TypeError(
                            f"{self.__name__}(): argument {self._names[i]!r} must be a "
                            f"real number, not {type(v).__name__}"
                        ) from None
                v = float(v)
            out.append(v)
        return out

    def _specialize(self, values: tuple[Any, ...], target: Any, options: CompileOptions) -> _Entry:
        fn = self.__wrapped__
        names = self._names
        while True:
            static, runtime = self._split
            trace_args = list(values)
            for n, i in enumerate(runtime):
                trace_args[i] = Parameter(names[i], n, "f64")
            try:
                lowered, builder = build(
                    fn, trace_args, tuple(names[i] for i in runtime), target, options
                )
                break
            except SymbolicValueError as e:
                promote = {i for i in runtime if names[i] in e.names}
                if not promote:
                    raise
                operation = e.operation or "an operation that needs its value"
            _warn_static(self.__name__, [names[i] for i in sorted(promote)], operation)
            self._promote(promote)
        returns = _return_spec(builder.returned, builder.readouts, self.__name__)
        guard = build_guard(fn, impure=builder.impure)
        pins = tuple(values[i] for i in static)
        return _Entry(lowered, guard, returns, pins, runtime)

    def _promote(self, indices: set[int]) -> None:
        """Make float parameters static: the program needs their values."""
        with self._promote_lock:
            static, runtime = self._split
            self._split = (
                tuple(sorted({*static, *indices})),
                tuple(i for i in runtime if i not in indices),
            )


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _warn_static(kernel: str, params: list[str], operation: str) -> None:
    """Warn that tracing made the float parameters *params* static. Each is
    made static once per kernel, so this warns once per parameter."""
    quoted = ", ".join(repr(p) for p in params)
    one = len(params) == 1
    warnings.warn(
        f"pulse kernel {kernel}(): float argument{'' if one else 's'} {quoted} "
        f"{'is' if one else 'are'} used in {operation}, which needs "
        f"{'its value' if one else 'their values'} while tracing, so the kernel is "
        f"compiled again for every new value of {quoted}. To keep "
        f"{'it a runtime parameter' if one else 'them runtime parameters'}, compute "
        "that value on the host and pass it in, or annotate the parameter as int to "
        "make the specialization explicit.",
        RuntimeWarning,
        skip_file_prefixes=(_PACKAGE_DIR,),
    )


class CompiledKernel[**P]:
    """A kernel compiled ahead of its call by ``cudaq_pulse.compile``.

    ``ck(*args)`` runs the kernel on the device of the qudit handles passed
    and returns what the kernel returns. It holds one specialization per
    device and static argument values, built on first use and never
    evicted. A kernel pinned to a device and without static parameters is
    built when it is compiled.
    """

    def __init__(self, kernel: Kernel[P], options: CompileOptions, device: Device | None) -> None:
        if device is not None and not isinstance(device, Device):
            raise TypeError(f"device must be a cudaq_pulse device, got {type(device).__name__}")
        self.kernel = kernel
        self.device = device
        self._dispatch = _Dispatcher(kernel, options, None)
        self._dispatch.pin = device
        if device is not None and not kernel._split[0]:
            # Every parameter is a float: the values are not needed to build
            # it, unless one of them shapes the program.
            try:
                self._dispatch.general = self._dispatch.lookup((None,) * len(kernel._names)).lowered
            except Exception:
                if not kernel._split[0]:
                    raise

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> Any:
        return self._dispatch.call(args, kwargs)

    def lower(self, *args: P.args, **kwargs: P.kwargs) -> Lowered:
        """The module for these arguments, specialized and scheduled."""
        return self._dispatch.lower(args, kwargs)

    def _view(self) -> Lowered:
        general = self._dispatch.general
        if general is None:
            if self.kernel._split[0]:
                why = f"is specialized on {', '.join(self.kernel.static_parameters)}"
            else:
                why = "has no device until it is called; pin one with compile(k, device=dev)"
            raise TypeError(
                f"{self.kernel.__name__}() {why}; inspect one specialization with .lower(...)"
            )
        return general

    def mlir(self, stage: str = "pulse") -> str:
        """MLIR text of the compiled module; see ``Lowered.mlir``."""
        return self._view().mlir(stage)

    @property
    def schedule(self) -> list[Any]:
        return self._view().schedule

    def timeline(self, **kwargs: Any) -> Any:
        return self._view().timeline(**kwargs)

    @property
    def metrics(self) -> Any:
        return self._view().metrics

    @property
    def parameters(self) -> tuple[str, ...]:
        """The float parameters, function arguments of the module."""
        return self.kernel.runtime_parameters

    def cache_info(self) -> CacheInfo:
        d = self._dispatch
        return CacheInfo(len(d._entries), d.misses, d.evictions, None)

    def __repr__(self) -> str:
        return f"<compiled pulse kernel {self.kernel.__qualname__}{self.kernel.__signature__}>"
