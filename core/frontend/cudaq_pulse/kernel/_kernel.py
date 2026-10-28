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
So is each element of a ``list[float]`` parameter, whose length is part of
the specialization. A parameter annotated ``Qudit`` or ``Register`` is a
runtime qudit: the module takes it as a slot, and a call binds the slot to
its handle's carrier, so calls with other handles, or on a recalibrated
device, share the module. Every other parameter (unannotated qudits, ints,
anything unannotated) is static: it can shape the program, so the kernel is
specialized on its value.

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
import typing
import warnings
import weakref
from collections.abc import Callable
from dataclasses import replace
from typing import Any, NamedTuple

from .._device import Device, DeviceMismatchError, NoDeviceError, Qudit, Register, _transition_hz
from .._errors import InvalidTypeError
from ..compiler import _PLACEMENTS, CompileOptions, Lowered, _Role, build, with_line_delays
from ._guard import Guard, build_guard
from ._schedule_blocks import root_schedule, validate_kernel_schedule
from ._unroll import resolve_unroll_threshold
from .ir_builder import (
    CompilationError,
    IRValue,
    Parameter,
    Slot,
    SlotRegister,
    SymbolicValueError,
    is_symbolic,
)

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
        dispatcher.kernel._traced.clear()


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


# What a runtime parameter is.
_FLOAT, _VECTOR, _QUDIT, _REGISTER = range(4)
_KINDS = {float: _FLOAT, Qudit: _QUDIT, Register: _REGISTER}


def _runtime_kind(annotation: Any, scope: dict[str, Any]) -> int | None:
    """The kind of runtime parameter *annotation* declares, if any: ``float``,
    ``list[float]``, ``Qudit``, or ``Register`` (also ``list[Qudit]`` and
    ``tuple[Qudit, ...]``). A string annotation is evaluated in *scope*."""
    if isinstance(annotation, str):
        if annotation == "float":
            return _FLOAT
        try:
            annotation = eval(annotation, scope)  # noqa: S307
        except Exception:
            return None
    try:
        kind = _KINDS.get(annotation)
    except TypeError:
        return None
    if kind is not None:
        return kind
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is list and args == (float,):
        return _VECTOR
    if (origin is list and args == (Qudit,)) or (origin is tuple and args == (Qudit, ...)):
        return _REGISTER
    return None


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
        # The dispatcher's key holds the device's fingerprint.
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


class _QuditTwiceError(Exception):
    """A qudit ``_devices`` met twice; the caller names the parameters."""

    def __init__(self, qudit: Qudit) -> None:
        self.qudit = qudit


def _devices(value: Any, found: Device | None, name: str, seen: set[int]) -> Device | None:
    """The device of the handles in *value*, checked against *found*.

    *seen* collects the carriers passed so far: one qudit passed twice is
    an error, as it would be one degree of freedom bound to two names, and
    raises ``_QuditTwiceError``.
    """
    t = type(value)
    if t is Qudit:
        if value.index in seen:
            raise _QuditTwiceError(value)
        seen.add(value.index)
        device: Device = value.device
        if found is None or found is device:
            return device
        if found != device:
            from .._errors import _ranges

            both = (
                f"both of target {device.name!r}"
                if found.name == device.name
                else f"of targets {found.name!r} and {device.name!r}"
            )
            raise DeviceMismatchError(
                f"{name}() got qudits of two different devices, {both}: one with layout "
                f"{_ranges(found.layout)}, the other with layout {_ranges(device.layout)}; "
                "pass handles of one device"
            )
        return found
    if t is Register or t is tuple or t is list:
        for item in value:
            found = _devices(item, found, name, seen)
    return found


def _count(value: Any, index: int) -> int:
    """How many handles in *value* are of the qudit at *index*."""
    t = type(value)
    if t is Qudit:
        return int(value.index == index)
    if t is Register or t is tuple or t is list:
        return sum(_count(item, index) for item in value)
    return 0


def _passed_twice(name: str, qudit: Qudit, arguments: list[tuple[str, Any]]) -> InvalidTypeError:
    """The error for *qudit* passed twice among *arguments*, the (parameter,
    value) pairs in the order ``_devices`` met them."""
    holders = [
        parameter for parameter, value in arguments for _ in range(_count(value, qudit.index))
    ]
    first, second = holders[0], holders[1]
    if first == second:
        return InvalidTypeError(
            f"{name}() got qudit {qudit.logical} twice in {first!r}; each qudit of an "
            "argument needs to be distinct"
        )
    return InvalidTypeError(
        f"{name}() got qudit {qudit.logical} for both {first!r} and {second!r}; each "
        "argument needs a distinct qudit"
    )


# ── Specializations ──────────────────────────────────────────────────────


class CacheInfo(NamedTuple):
    size: int
    compiled: int
    evicted: int
    capacity: int | None


# Traced programs a kernel keeps for scheduling again with other line delays.
_TRACED_KEPT = 8


class _Entry:
    """One compiled specialization and the guard that keeps it valid."""

    __slots__ = (
        "lowered",
        "guard",
        "returns",
        "tick",
        "pins",
        "runtime",
        "n_qudits",
        "relocated",
    )
    # Set only on the entries kept to be scheduled again.
    n_qudits: int

    def __init__(
        self,
        lowered: Lowered,
        guard: Guard,
        returns: Any,
        pins: tuple[Any, ...],
        runtime: tuple[int, ...],
        relocated: bool = False,
    ):
        self.lowered = lowered
        self.guard = guard
        self.returns = returns
        self.tick = next(_ticks)
        # The parameters that are function arguments of the module.
        self.runtime = runtime
        # Whether some are list[float], Qudit or Register parameters.
        self.relocated = relocated
        # Static arguments keyed by identity are kept alive with the entry.
        self.pins = pins


class _Dispatcher:
    """Binds a call, finds or builds its specialization, and runs it."""

    def __init__(self, kernel: Kernel[...], options: CompileOptions, capacity: int | None) -> None:
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
        try:
            for i in self.kernel._split[0]:
                found = _devices(values[i], found, name, seen)
        except _QuditTwiceError as twice:
            names = self.kernel._names
            arguments = [(names[i], values[i]) for i in self.kernel._split[0]]
            raise _passed_twice(name, twice.qudit, arguments) from None
        pin = self.pin
        if pin is not None:
            if found is not None and found != pin:
                raise _pin_mismatch(name, pin, found)
            return pin
        if found is not None:
            return found
        raise _no_device(name)

    def lookup(self, values: tuple[Any, ...]) -> _Entry:
        kernel = self.kernel
        target = self._device(values)
        static = kernel._split[0]
        key = (target.fingerprint, static, *[_static_key(values[i]) for i in static])
        entry = self._entries.get(key)
        if entry is not None and entry.guard.check():
            entry.tick = next(_ticks)
            return entry
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.guard.check():
                return entry
            entry = kernel._specialize(values, target, self.options)
            self.misses += 1
            # Tracing may have made float parameters static.
            static = kernel._split[0]
            key = (target.fingerprint, static, *[_static_key(values[i]) for i in static])
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
            lowered = lowered._with_values(kernel._runtime_values(values, entry.runtime))
        return _execute(lowered, entry.returns)

    def lower(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Lowered:
        kernel = self.kernel
        values = kernel._bind(*args, **kwargs)
        entry = self.lookup(values)
        lowered = entry.lowered
        if entry.runtime:
            lowered = lowered._with_values(kernel._runtime_values(values, entry.runtime))
        return lowered

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class _Placing(_Dispatcher):
    """The dispatcher of a kernel with list, qudit or register parameters:
    the module depends on lists only through their lengths and, with runtime
    qudits, on the device only through its structure; each call binds them."""

    def _device(self, values: tuple[Any, ...]) -> Device:
        kernel = self.kernel
        slotted = kernel._slotted
        if len(slotted) == 1 and not kernel._split[0]:
            value = values[slotted[0]]
            if type(value) is Qudit:
                device: Device = value.device
                pin = self.pin
                if pin is None or pin is device:
                    return device
                if device != pin:
                    raise _pin_mismatch(kernel.__name__, pin, device)
                return pin
        name = kernel.__name__
        found: Device | None = None
        seen: set[int] = set()
        for i in kernel._split[0]:
            found = _devices(values[i], found, name, seen)
        wrong = None
        for i in kernel._slotted:
            value = values[i]
            if type(value) is not Qudit and not (
                type(value) is Register or type(value) is list or type(value) is tuple
            ):
                wrong = i
                continue
            found = _devices(value, found, name, seen)
        pin = self.pin
        if wrong is not None:
            if found is None and pin is None:
                raise _no_device(name)
            raise InvalidTypeError(
                f"{name}(): argument {kernel._names[wrong]!r} must be a qudit handle "
                f"or a register of them, not {type(values[wrong]).__name__}"
            )
        if pin is not None:
            if found is not None and found != pin:
                raise _pin_mismatch(name, pin, found)
            return pin
        if found is not None:
            return found
        raise _no_device(name)

    def _key(self, target: Device, values: tuple[Any, ...]) -> Any:
        kernel = self.kernel
        static = kernel._split[0]
        fingerprint = target._structure if kernel._slotted else target.fingerprint
        lengths = [len(values[i]) for i in kernel._shaped]
        return (fingerprint, static, *[_static_key(values[i]) for i in static], *lengths)

    def lookup(self, values: tuple[Any, ...]) -> _Entry:
        kernel = self.kernel
        target = self._device(values)
        key = self._key(target, values)
        entry = self._entries.get(key)
        if entry is not None and entry.guard.check():
            entry.tick = next(_ticks)
            return entry
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.guard.check():
                return entry
            entry = kernel._specialize(values, target, self.options)
            self.misses += 1
            # Tracing may have made float parameters static.
            key = self._key(target, values)
            self._insert(key, entry)
        return entry

    def call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        values = self.kernel._bind(*args, **kwargs)
        entry = self.lookup(values)
        return _execute(self._bound(entry, values), entry.returns)

    def lower(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Lowered:
        values = self.kernel._bind(*args, **kwargs)
        return self._bound(self.lookup(values), values)

    def _bound(self, entry: _Entry, values: tuple[Any, ...]) -> Lowered:
        kernel = self.kernel
        if entry.relocated:
            return kernel._relocate(entry.lowered, values, entry.runtime)
        if entry.runtime:
            return entry.lowered.bind(kernel._runtime_values(values, entry.runtime))
        return entry.lowered


class _Deciding(_Dispatcher):
    """The dispatcher of a kernel with unannotated parameters until its first
    call: each one passed a qudit handle or a register becomes a runtime
    parameter, and the dispatcher becomes the one for the kernel it makes."""

    def _decide(self, values: tuple[Any, ...]) -> None:
        kernel = self.kernel
        kernel._adopt(values)
        self.__class__ = kernel._dispatcher_class()  # type: ignore[assignment]

    def lookup(self, values: tuple[Any, ...]) -> _Entry:
        self._decide(values)
        return self.lookup(values)

    def call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        self._decide(self.kernel._bind(*args, **kwargs))
        return self.call(args, kwargs)

    def lower(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Lowered:
        self._decide(self.kernel._bind(*args, **kwargs))
        return self.lower(args, kwargs)


def _pin_mismatch(name: str, pin: Device, found: Device) -> DeviceMismatchError:
    return DeviceMismatchError(
        f"{name}() is compiled for device {pin.name!r} and got qudits of {found.name!r}"
    )


def _no_device(name: str) -> NoDeviceError:
    return NoDeviceError(
        f"{name}() was called without qudit handles, so it has no device: pass "
        "handles of a device (dev.qudit_handles[i]), or compile it for one with "
        "cudaq_pulse.compile(k, device=dev)"
    )


class Kernel[**P]:
    """A pulse kernel. ``k(*args)`` compiles on first use and runs it.

    ``k.lower(*args)`` returns the compiled module without running it.
    ``cudaq_pulse.compile(k)`` compiles ahead of the call. ``P`` is the
    decorated function's signature, so calls are checked against it.

    A kernel pickles as a function does. One that its module and qualified
    name find, outside ``__main__``, pickles as that name. Any other (in
    ``__main__``, at the prompt, inside a function) pickles as its function
    and options, the function by reference with pickle and by value with
    cloudpickle (Ray, Dask): pickle cannot send one defined inside a
    function. A process keeps the kernels it unpickles this way, so sending
    one again does not compile it again.
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
        # None leaves the threshold to cudaq_pulse.compile(), else the default.
        self.unroll_threshold = (
            None if unroll_threshold is None else resolve_unroll_threshold(unroll_threshold)
        )
        for attr in ("__module__", "__name__", "__qualname__", "__doc__"):
            setattr(self, attr, getattr(fn, attr))
        signature = inspect.signature(fn)
        self.__signature__ = signature
        names = code.co_varnames[: code.co_argcount + code.co_kwonlyargcount]
        self._names = names
        kinds = {}
        unannotated = []
        for i, n in enumerate(names):
            annotation = signature.parameters[n].annotation
            kind = _runtime_kind(annotation, fn.__globals__)
            if kind is not None:
                kinds[i] = kind
            elif annotation is inspect.Parameter.empty:
                unannotated.append(i)
        # Unannotated parameters, until the first call decides which of them
        # are runtime qudits and registers.
        self._undecided = tuple(unannotated)
        self._classify(kinds, tuple(i for i in range(len(names)) if i not in kinds))
        self._promote_lock = threading.Lock()
        # The latest scheduled specializations with line delays of the
        # programs it traced, the last _TRACED_KEPT of them, which other line
        # delays schedule again without tracing.
        self._traced: dict[Any, _Entry] = {}
        self._bind = _make_binder(fn)
        self._dispatch = self._dispatcher_class()(
            self,
            CompileOptions(
                schedule=root_schedule(self.schedule, None),
                unroll_threshold=resolve_unroll_threshold(self.unroll_threshold),
            ),
            _cache_size(cache_size),
        )

    @property
    def runtime_parameters(self) -> tuple[str, ...]:
        """Parameters annotated ``float``, ``list[float]``, ``Qudit`` or
        ``Register``, and unannotated ones the first call passed a qudit
        handle or a register: function arguments of the module."""
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
        self._traced.clear()

    def __reduce__(self) -> str | tuple[Any, ...]:
        from .._pickling import kernel_reduce

        return kernel_reduce(self)

    @property
    def __pulse_pickled_fn__(self) -> types.FunctionType:
        """The function a kernel that its name does not find pickles with:
        named after this attribute, so that pickle finds it by reference and
        cloudpickle sends it by value, as each does a function."""
        from .._pickling import function_copy

        return function_copy(self)

    def __repr__(self) -> str:
        return f"<pulse kernel {self.__qualname__}{self.__signature__}>"

    def _runtime_values(self, values: tuple[Any, ...], runtime: tuple[int, ...]) -> list[float]:
        out = []
        for i in runtime:
            v = values[i]
            if type(v) is not float:
                from .._units import Time

                if v.__class__ is Time:
                    name = self._names[i]
                    raise InvalidTypeError(
                        f"{self.__name__}(): argument {name!r} is a time ({v}), and a float "
                        f"parameter is a plain number; pass {name} in nanoseconds and write "
                        f"{name} * pulse.ns in the kernel, or drop its float annotation to "
                        "compile the kernel for it"
                    )
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

    def _relocate(
        self, lowered: Lowered, values: tuple[Any, ...], runtime: tuple[int, ...]
    ) -> Lowered:
        """The call's view of a module with list or qudit parameters: its
        float values flattened, and the carrier of each slot."""
        kinds = self._kinds
        numbers: list[float] = []
        carriers: list[int] = []
        device = None
        for i in runtime:
            kind = kinds[i]
            v = values[i]
            if kind == _FLOAT:
                if type(v) is float:
                    numbers.append(v)
                else:
                    numbers.extend(self._runtime_values(values, (i,)))
            elif kind == _VECTOR:
                numbers.extend(self._vector_values(v, i))
            elif kind == _QUDIT:
                carriers.append(v.index)
                device = v.device
            else:
                for q in v:
                    carriers.append(q.index)
                    device = q.device
        role = lowered._role
        placement = tuple(carriers)
        views = None
        if not numbers and role is not None and device is not None:
            views = role.views
            view = views.get((placement, device))
            if view is not None:
                return view
        if role is not None and role.frequencies:
            assert device is not None
            for slot, lower, upper in role.frequencies:
                numbers.append(_transition_hz(device, carriers[slot], lower, upper))
        # A module with list or qudit parameters is a _Placed or _CheckedPlaced.
        place: Callable[..., Lowered] = lowered.bind
        view = place(numbers, placement, device)
        if views is not None and device is not None and len(views) < _PLACEMENTS:
            views[(placement, device)] = view
        return view

    def _vector_values(self, value: Any, i: int) -> list[float]:
        try:
            return [x if type(x) is float else _real(x) for x in value]
        except TypeError:
            raise InvalidTypeError(
                f"{self.__name__}(): argument {self._names[i]!r} must be a list of real "
                f"numbers, not {type(value).__name__}"
            ) from None

    def _specialize(self, values: tuple[Any, ...], target: Any, options: CompileOptions) -> _Entry:
        if options.machine.line_delays_ns or self._traced:
            found = self._retimed(values, target, options)
            if found is not None:
                return found
        fn = self.__wrapped__
        names = self._names
        kinds = self._kinds
        while True:
            static, runtime = self._split
            trace_args = list(values)
            for i in self._declared_float:
                if i in runtime:
                    continue
                value = values[i]
                if value.__class__ is float:
                    from ._durations import Cycles

                    trace_args[i] = Cycles(value)
                elif kinds.get(i) == _VECTOR:
                    from ._durations import Cycles

                    # Each element is a float parameter's value, so a duration too.
                    trace_args[i] = [Cycles(v) if v.__class__ is float else v for v in value]
            parameters: list[str] = []
            symbols: list[tuple[str, str, str, int | None]] = []
            slotted: list[tuple[str, str, str, int | None]] = []
            for i in runtime:
                kind = kinds[i]
                if kind == _FLOAT:
                    trace_args[i] = Parameter(names[i], len(parameters), "f64")
                    parameters.append(names[i])
                    symbols.append((names[i], "float", names[i], None))
                elif kind == _VECTOR:
                    elements = []
                    for k in range(len(values[i])):
                        element = f"{names[i]}[{k}]"
                        elements.append(Parameter(element, len(parameters), "f64"))
                        parameters.append(element)
                        symbols.append((element, "vector", names[i], k))
                    trace_args[i] = elements
                elif kind == _QUDIT:
                    trace_args[i] = Slot(len(slotted))
                    slotted.append((f"s{len(slotted)}", "slot", names[i], None))
                else:
                    first = len(slotted)
                    trace_args[i] = SlotRegister(Slot(first + k) for k in range(len(values[i])))
                    for k in range(len(values[i])):
                        slotted.append((f"s{first + k}", "slot", names[i], k))
            relocated = any(kinds[i] != _FLOAT for i in runtime)
            role = _Role((*symbols, *slotted)) if relocated else None
            try:
                lowered, builder = build(
                    fn, trace_args, tuple(parameters), target, options, len(slotted), role
                )
                break
            except SymbolicValueError as e:
                promote = {
                    i
                    for i in runtime
                    if kinds[i] in (_FLOAT, _VECTOR)
                    and any(n == names[i] or n.startswith(f"{names[i]}[") for n in e.names)
                }
                if not promote:
                    raise
                operation = e.operation or "an operation that needs its value"
                where = _traced_line(e)
            promoted = self._promote(promote)
            if promoted:
                _warn_static(self.__name__, [names[i] for i in promoted], operation, where)
        returns = _return_spec(builder.returned, builder.readouts, self.__name__)
        guard = build_guard(fn, impure=builder.impure)
        pins = tuple(values[i] for i in static)
        entry = _Entry(lowered, guard, returns, pins, runtime, relocated)
        if options.machine.line_delays_ns and lowered.scheduled:
            # Only a kept entry is scheduled again, for its qudits.
            entry.n_qudits = builder.n_qubits
            self._keep_traced(self._traced_key(values, target, options), entry)
        return entry

    def _keep_traced(self, key: Any, entry: _Entry) -> None:
        traced = self._traced
        traced.pop(key, None)
        traced[key] = entry
        if len(traced) > _TRACED_KEPT:
            del traced[next(iter(traced))]

    def _traced_key(self, values: tuple[Any, ...], target: Any, options: CompileOptions) -> Any:
        """What a specialization is traced for: everything but line delays,
        which only move its schedule."""
        if options.machine.line_delays_ns:
            options = replace(options, machine=replace(options.machine, line_delays_ns={}))
        static = self._split[0]
        return (target.fingerprint, options, static, *[_static_key(values[i]) for i in static])

    def _retimed(
        self, values: tuple[Any, ...], target: Any, options: CompileOptions
    ) -> _Entry | None:
        """The specialization for *values* scheduled again with the line
        delays of *options*, from one traced with other delays, if any."""
        prior = self._traced.get(self._traced_key(values, target, options))
        if (
            prior is None
            or prior.lowered.options.machine.line_delays_ns == options.machine.line_delays_ns
            or not prior.guard.check()
        ):
            return None
        lowered = with_line_delays(prior.lowered, options, prior.n_qudits)
        if lowered is None:
            return None
        entry = _Entry(
            lowered, prior.guard, prior.returns, prior.pins, prior.runtime, prior.relocated
        )
        entry.n_qudits = prior.n_qudits
        self._keep_traced(self._traced_key(values, target, options), entry)
        return entry

    def _classify(self, kinds: dict[int, int], static: tuple[int, ...]) -> None:
        # The kind of each runtime parameter, by index.
        self._kinds = kinds
        runtime = tuple(i for i in sorted(kinds) if i not in static)
        # Runtime qudit and register parameters, and those whose length
        # (lists, registers) is part of the specialization.
        self._slotted = tuple(i for i in runtime if kinds[i] in (_QUDIT, _REGISTER))
        self._shaped = tuple(i for i in runtime if kinds[i] in (_VECTOR, _REGISTER))
        self._relocatable = bool(self._slotted or self._shaped)
        # (static, runtime) parameter indices, replaced as one value when a
        # float parameter turns out to shape the program.
        self._split = (static, runtime)
        # The float and list[float] parameters, static or not: a static one's
        # values trace as cycles, so that a duration may take them.
        self._declared_float = tuple(i for i in sorted(kinds) if kinds[i] in (_FLOAT, _VECTOR))

    def _dispatcher_class(self) -> type[_Dispatcher]:
        if self._undecided:
            return _Deciding
        return _Placing if self._relocatable else _Dispatcher

    def _adopt(self, values: tuple[Any, ...]) -> None:
        """Make the unannotated parameters passed a qudit handle or a register
        runtime ones; the first call decides, once per kernel."""
        with self._promote_lock:
            undecided = self._undecided
            if not undecided:
                return
            found = {}
            for i in undecided:
                t = type(values[i])
                if t is Qudit:
                    found[i] = _QUDIT
                elif t is Register:
                    found[i] = _REGISTER
            if found:
                static = tuple(i for i in self._split[0] if i not in found)
                self._classify({**self._kinds, **found}, static)
            self._undecided = ()

    def _promote(self, indices: set[int]) -> list[int]:
        """Make float parameters static: the program needs their values.
        Returns those of *indices* that were not static yet, so that threads
        tracing at once report each parameter once."""
        with self._promote_lock:
            static, runtime = self._split
            promoted = [i for i in runtime if i in indices]
            self._split = (
                tuple(sorted({*static, *indices})),
                tuple(i for i in runtime if i not in indices),
            )
        return promoted


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _traced_line(e: BaseException) -> str:
    """`` at line N`` for the kernel line where tracing raised *e*, with the
    helper's line when the value was needed inside a helper."""
    source = getattr(e, "pulse_source", None)
    if source is None:
        return ""
    frames = [source, *getattr(e, "pulse_callers", ())]
    prog, index = frames[-1]
    where = f" at line {prog.location(index)[1]}"
    if len(frames) > 1:
        prog, index = frames[0]
        where += f" (in {prog.code.co_name}(), line {prog.location(index)[1]})"
    return where


def _real(value: Any) -> float:
    """*value* as a float, if it is a real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            return float(value.__float__())
        except AttributeError:
            raise InvalidTypeError(f"{type(value).__name__} is not a real number") from None
    return float(value)


def _warn_static(kernel: str, params: list[str], operation: str, where: str = "") -> None:
    """Warn that tracing made the float parameters *params* static. Each is
    made static once per kernel, so this warns once per parameter."""
    from .._errors import _SpecializationWarning

    quoted = ", ".join(repr(p) for p in params)
    one = len(params) == 1
    it = "it" if one else "them"
    warnings.warn(
        f"pulse kernel {kernel}(): float argument{'' if one else 's'} {quoted} "
        f"{'is' if one else 'are'} used in {operation}{where}, which needs "
        f"{'its value' if one else 'their values'} while tracing, so the kernel is "
        f"compiled again for every new value of {quoted}. To keep "
        f"{'it a runtime parameter' if one else 'them runtime parameters'}, compute "
        f"that value on the host and pass it in; otherwise annotate {it} as int, or "
        f"give {it} a fixed value, to make the specialization explicit.",
        _SpecializationWarning,
        skip_file_prefixes=(_PACKAGE_DIR,),
    )


class CompiledKernel[**P]:
    """A kernel compiled ahead of its call by ``cudaq_pulse.compile``.

    ``ck(*args)`` runs the kernel on the device of the qudit handles passed
    and returns what the kernel returns. It holds one specialization per
    device and static argument values, built on first use and never
    evicted. A kernel pinned to a device and without static parameters is
    built when it is compiled.

    It pickles as its kernel (by reference), options and device, without
    its specializations, which compile again on first use where it is
    unpickled; pickle ``ck.lower(*args)`` to send a program that runs
    without compiling.
    """

    def __init__(self, kernel: Kernel[P], options: CompileOptions, device: Device | None) -> None:
        if device is not None and not isinstance(device, Device):
            raise TypeError(f"device must be a cudaq_pulse device, got {type(device).__name__}")
        self.kernel = kernel
        self.device = device
        self._dispatch = kernel._dispatcher_class()(kernel, options, None)
        self._dispatch.pin = device
        if (
            device is not None
            and not kernel._split[0]
            and not kernel._shaped
            and not kernel._slotted
        ):
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
        kernel = self.kernel
        kinds = kernel._kinds
        return tuple(kernel._names[i] for i in kernel._split[1] if kinds[i] in (_FLOAT, _VECTOR))

    def cache_info(self) -> CacheInfo:
        d = self._dispatch
        return CacheInfo(len(d._entries), d.misses, d.evictions, None)

    def __reduce__(self) -> tuple[Any, ...]:
        from .._pickling import rebuilt_compiled

        d = self._dispatch
        return (rebuilt_compiled, (self.kernel, d.options, self.device, d.general))

    def __repr__(self) -> str:
        return f"<compiled pulse kernel {self.kernel.__qualname__}{self.kernel.__signature__}>"
