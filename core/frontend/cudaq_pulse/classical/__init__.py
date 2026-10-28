# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Classical co-processing at QPU runtime.

``classical`` is a classical real-time dialect for co-processing at QPU
runtime: the GPU, host or FPGA services that a pulse program calls while the
QPU runs, to decode syndromes, update calibrations or receive bulk readout
data, and whose answers later pulses use.

``pulse.classical`` is both this module and the decorator that declares such
a function::

    import cudaq_pulse as pulse
    from cudaq_pulse.classical import Bits

    @pulse.classical(args=(Bits(4),), results=(Bits(1),), max_latency_ns=3000)
    def decode(syndromes):
        return [syndromes[0] ^ syndromes[3]]

    @pulse.kernel
    def round(...):
        ...
        answer = decode.submit([m0, m1, m2, m3])   # send the request
        fix = answer.result()                      # take the answer
        if fix[0]:
            drive(...)

Inside a kernel, ``.submit(...)`` sends a request and returns a future at
once; pulses that do not need the answer keep playing. ``.result()`` is where
the answer is taken: it is due the maximum latency after the request is sent,
and the pulses whose amplitude, phase, frequency or branch condition come from
it start no earlier. An answer sets a duration only through a variable
wait, ``wait(line, answer, bounds=(lo, hi))``, which begins a new epoch of
the timeline. Outside a kernel, calling the
function runs its body, its reference implementation, directly.

The body is what simulation runs. On hardware the service registered under
the function's id (the 32-bit FNV-1a hash of its name, as CUDA-Q Realtime
uses) answers instead.
"""

from __future__ import annotations

import math
import sys
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Required, TypedDict, Unpack, overload

from .._errors import InvalidTypeError, InvalidValueError
from .._units import Time, as_nanoseconds

if TYPE_CHECKING:
    from .._system import Controller

__all__ = [
    "Bits",
    "Byte",
    "Bytes",
    "ClassicalDecorator",
    "ClassicalFunction",
    "Float",
    "Floats",
    "IQ",
    "Integer",
    "Integers",
    "Levels",
    "Link",
    "Pending",
    "external",
    "function",
    "function_id",
    "random_index",
]

_LATENCY_MODES = ("bounded", "timeout", "blocking")
_SUBMIT_POLICIES = ("eager", "lazy")
# The handler schema of the CUDA-Q Realtime dispatcher.
_MAX_ARGS = 8
_MAX_RESULTS = 4


def function_id(name: str) -> int:
    """The id of a function named *name* on the wire: 32-bit FNV-1a."""
    value = 2166136261
    for byte in name.encode("utf-8"):
        value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
    return value


# ── Argument and result types ────────────────────────────────────────────────


class ValueType:
    """The type of one argument or result of a classical function.

    ``descriptor`` is its spelling in the IR, ``count`` how many values it
    holds, ``source`` what a kernel passes for it (``"measurement"``,
    ``"float"`` or ``"integer"``), and ``answer`` the type of the values a
    kernel gets back for it (``"i1"``, ``"i64"`` or ``"f64"``).
    """

    __slots__ = ("count", "descriptor", "source", "answer", "answer_count", "wire_bytes")

    def __init__(
        self,
        descriptor: str,
        count: int,
        source: str,
        answer: str,
        answer_count: int,
        wire_bytes: int,
    ) -> None:
        self.descriptor = descriptor
        self.count = count
        self.source = source
        self.answer = answer
        self.answer_count = answer_count
        self.wire_bytes = wire_bytes

    @property
    def is_scalar(self) -> bool:
        return ":" not in self.descriptor and "[" not in self.descriptor

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ValueType) and other.descriptor == self.descriptor

    def __hash__(self) -> int:
        return hash(self.descriptor)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.descriptor!r})"


def _count(n: int, what: str) -> int:
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise InvalidValueError(f"{what} needs a positive whole number of values, got {n!r}")
    return n


def _bits_width(bits: int, allowed: tuple[int, ...], what: str) -> int:
    if bits not in allowed:
        raise InvalidValueError(f"{what} has {' or '.join(map(str, allowed))} bits, got {bits!r}")
    return bits


class Bits(ValueType):
    """``n`` measurement results, sent as ``outcome != 0``, bit-packed.

    As a result, ``n`` booleans: the answer's bits, which a kernel can
    branch on.
    """

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "Bits")
        super().__init__(f"bits:{n}", n, "measurement", "i1", n, (n + 7) // 8)


class Levels(ValueType):
    """``n`` measured levels, one byte each, for leakage-aware decoding.

    As a result, ``n`` integers.
    """

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "Levels")
        super().__init__(f"levels:{n}", n, "measurement", "i64", n, n)


class IQ(ValueType):
    """The IQ points of ``n`` readouts, as ``2n`` 32-bit floats (I, then Q).

    As a result, ``2n`` floats.
    """

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "IQ")
        super().__init__(f"iq:{n}", n, "measurement", "f64", 2 * n, 8 * n)


class Float(ValueType):
    """One floating-point number, of 64 bits (the default) or 32."""

    __slots__ = ()

    def __init__(self, bits: int = 64) -> None:
        bits = _bits_width(bits, (64, 32), "Float")
        super().__init__(f"f{bits}", 1, "float", "f64", 1, bits // 8)


class Integer(ValueType):
    """One signed integer, of 64 bits (the default) or 32."""

    __slots__ = ()

    def __init__(self, bits: int = 64) -> None:
        bits = _bits_width(bits, (64, 32), "Integer")
        super().__init__(f"i{bits}", 1, "integer", "i64", 1, bits // 8)


class Byte(ValueType):
    """One unsigned byte."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__("u8", 1, "integer", "i64", 1, 1)


class Floats(ValueType):
    """``n`` floating-point numbers, of 64 bits (the default) or 32 each."""

    __slots__ = ()

    def __init__(self, n: int, bits: int = 64) -> None:
        n = _count(n, "Floats")
        bits = _bits_width(bits, (64, 32), "Floats")
        super().__init__(f"f{bits}[{n}]", n, "float", "f64", n, n * bits // 8)


class Integers(ValueType):
    """``n`` signed 32-bit integers."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "Integers")
        super().__init__(f"i32[{n}]", n, "integer", "i64", n, 4 * n)


class Bytes(ValueType):
    """``n`` unsigned bytes."""

    __slots__ = ()

    def __init__(self, n: int) -> None:
        n = _count(n, "Bytes")
        super().__init__(f"u8[{n}]", n, "integer", "i64", n, n)


def _value_type(value: Any, where: str) -> ValueType:
    if isinstance(value, ValueType):
        return value
    if isinstance(value, type) and issubclass(value, (Float, Integer, Byte)):
        return value()
    raise InvalidTypeError(
        f"{where} must be a type such as Bits(4), Float() or Floats(8), got {value!r}"
    )


# What ``args`` and ``results`` entries may be: ``Float`` stands for ``Float()``.
_TypeSpec = ValueType | type[Float] | type[Integer] | type[Byte]


class _Options(TypedDict, total=False):
    """The keyword options of ``pulse.classical`` and ``external``."""

    args: Sequence[_TypeSpec]
    results: Sequence[_TypeSpec]
    max_latency_ns: Required[float | Time]
    min_latency_ns: float | Time
    nominal_latency_ns: float | Time | None
    latency_mode: str
    timeout_ns: float | Time | None
    submit: str
    deadline_ns: float | Time | None
    link: int | str
    id: int | None


class _NamedOptions(_Options, total=False):
    name: str | None


# ── Links ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Link:
    """What one link to a classical device costs, for scheduling.

    ``fixed_ns`` is the fixed cost of sending a request and ``bytes_per_ns``
    its bandwidth (0 for unlimited): a request holds the link for
    ``fixed_ns + bytes / bytes_per_ns``, at least one clock cycle.
    ``feedback_ns`` is added to every latency: the time from an answer
    arriving to a pulse being able to use it. ``fixed_ns`` and
    ``feedback_ns`` are times (``100 * pulse.ns``) or numbers of ns. ``max_in_flight`` is how many
    requests the link holds at once (its slots), and ``slot_bytes`` the
    largest request or answer, its 24-byte header included.

    ``controllers`` are the controllers of a ``ControlSystem`` the link
    reaches (``None``: all of them): a call over it can send only their
    measurements, and only they can use its answers.
    """

    fixed_ns: float | Time = 0.0
    bytes_per_ns: float = 0.0
    feedback_ns: float | Time = 0.0
    max_in_flight: int = 64
    slot_bytes: int = 1024
    controllers: Sequence[Controller] | None = None

    def __post_init__(self) -> None:
        # A program that uses a link it does not list makes a Link() as it
        # compiles: no call here for the common case of no times.
        if self.fixed_ns.__class__ is Time:
            object.__setattr__(self, "fixed_ns", float(self.fixed_ns))
        if self.feedback_ns.__class__ is Time:
            object.__setattr__(self, "feedback_ns", float(self.feedback_ns))
        if self.controllers is not None:
            from .._system import Controller

            reached = tuple(self.controllers)
            for controller in reached:
                if not isinstance(controller, Controller):
                    raise InvalidTypeError(
                        f"Link.controllers are pulse.Controller objects, got {controller!r}"
                    )
            object.__setattr__(self, "controllers", reached)
        for name in ("fixed_ns", "bytes_per_ns", "feedback_ns"):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise InvalidValueError(f"Link.{name} must be a finite number >= 0, got {value!r}")
        if not isinstance(self.max_in_flight, int) or self.max_in_flight < 1:
            raise InvalidValueError(
                f"Link.max_in_flight must be at least 1, got {self.max_in_flight!r}"
            )
        if not isinstance(self.slot_bytes, int) or self.slot_bytes <= 24:
            raise InvalidValueError(
                f"Link.slot_bytes must exceed the 24-byte header, got {self.slot_bytes!r}"
            )

    def cost(self) -> dict[str, float | int]:
        """What the link costs, as JSON-serializable data."""
        return {
            "fixed_ns": self.fixed_ns,
            "bytes_per_ns": self.bytes_per_ns,
            "feedback_ns": self.feedback_ns,
            "max_in_flight": self.max_in_flight,
            "slot_bytes": self.slot_bytes,
        }

    def attribute(self) -> str:
        """The link as an entry of the module attribute ``classical.links``."""
        return (
            f"{{fixed_ns = {float(self.fixed_ns)!r} : f64, "
            f"bytes_per_ns = {float(self.bytes_per_ns)!r} : f64, "
            f"feedback_ns = {float(self.feedback_ns)!r} : f64, "
            f"max_in_flight = {self.max_in_flight} : i64, "
            f"slot_bytes = {self.slot_bytes} : i64}}"
        )


# ── Classical functions ──────────────────────────────────────────────────────


class ClassicalFunction:
    """A classical function a pulse kernel can call while the QPU runs.

    Made by ``@pulse.classical(...)`` or ``pulse.classical.external(...)``.
    In a kernel, ``.submit(*args)`` sends a request and returns a future
    whose ``.result()`` takes the answer. Outside a kernel, calling it runs
    its reference implementation.
    """

    def __init__(
        self,
        implementation: Callable[..., Any] | None,
        *,
        name: str | None = None,
        args: Sequence[_TypeSpec] = (),
        results: Sequence[_TypeSpec] = (),
        max_latency_ns: float | Time,
        min_latency_ns: float | Time = 0.0,
        nominal_latency_ns: float | Time | None = None,
        latency_mode: str = "bounded",
        timeout_ns: float | Time | None = None,
        submit: str = "eager",
        deadline_ns: float | Time | None = None,
        link: int | str = 0,
        id: int | None = None,
    ) -> None:
        max_latency_ns = as_nanoseconds(max_latency_ns)
        min_latency_ns = as_nanoseconds(min_latency_ns)
        nominal_latency_ns = as_nanoseconds(nominal_latency_ns)
        timeout_ns = as_nanoseconds(timeout_ns)
        deadline_ns = as_nanoseconds(deadline_ns)
        if name is None:
            if implementation is None:
                raise InvalidTypeError("an external classical function needs a name")
            name = implementation.__name__
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            raise InvalidValueError(
                f"a classical function name must be an identifier, got {name!r}"
            )
        self.name = name
        self.implementation = implementation
        self.args: tuple[ValueType, ...] = tuple(_value_type(a, "args entry") for a in args)
        self.results: tuple[ValueType, ...] = tuple(
            _value_type(r, "results entry") for r in results
        )
        if len(self.args) > _MAX_ARGS:
            raise InvalidValueError(f"{name}: at most {_MAX_ARGS} arguments, got {len(self.args)}")
        if len(self.results) > _MAX_RESULTS:
            raise InvalidValueError(
                f"{name}: at most {_MAX_RESULTS} results, got {len(self.results)}"
            )
        nominal = max_latency_ns if nominal_latency_ns is None else nominal_latency_ns
        for label, value in (
            ("max_latency_ns", max_latency_ns),
            ("min_latency_ns", min_latency_ns),
            ("nominal_latency_ns", nominal),
        ):
            if not (isinstance(value, (int, float)) and math.isfinite(value) and value >= 0):
                raise InvalidValueError(
                    f"{name}: {label} must be a finite number >= 0, got {value!r}"
                )
        if not min_latency_ns <= nominal <= max_latency_ns:
            raise InvalidValueError(
                f"{name}: latencies must satisfy min <= nominal <= max, got "
                f"{min_latency_ns}, {nominal}, {max_latency_ns}"
            )
        self.min_latency_ns = float(min_latency_ns)
        self.nominal_latency_ns = float(nominal)
        self.max_latency_ns = float(max_latency_ns)
        if latency_mode not in _LATENCY_MODES:
            raise InvalidValueError(
                f"{name}: latency_mode must be one of {', '.join(_LATENCY_MODES)}, "
                f"got {latency_mode!r}"
            )
        if (latency_mode == "timeout") != (timeout_ns is not None):
            raise InvalidValueError(
                f"{name}: timeout_ns goes with latency_mode='timeout' (or pass "
                "timeout to result() at the call)"
            )
        if timeout_ns is not None and not timeout_ns > 0:
            raise InvalidValueError(f"{name}: timeout_ns must be positive, got {timeout_ns!r}")
        self.latency_mode = latency_mode
        self.timeout_ns = None if timeout_ns is None else float(timeout_ns)
        if submit not in _SUBMIT_POLICIES:
            raise InvalidValueError(f"{name}: submit must be 'eager' or 'lazy', got {submit!r}")
        self.submit_policy = submit
        if deadline_ns is not None and not deadline_ns > 0:
            raise InvalidValueError(f"{name}: deadline_ns must be positive, got {deadline_ns!r}")
        self.deadline_ns = None if deadline_ns is None else float(deadline_ns)
        if isinstance(link, str):
            if not link.isidentifier():
                raise InvalidValueError(f"{name}: a link name must be an identifier, got {link!r}")
        elif isinstance(link, bool) or not isinstance(link, int) or link < 0:
            raise InvalidValueError(
                f"{name}: link must be a link name or a whole number >= 0, got {link!r}"
            )
        self.link = link
        if id is not None and not (isinstance(id, int) and 0 <= id <= 0xFFFFFFFF):
            raise InvalidValueError(f"{name}: id must be a 32-bit unsigned integer, got {id!r}")
        self.id = function_id(name) if id is None else id
        self.custom_id = id is not None
        # For an on-device random source: how many equally likely values.
        self.uniform: int | None = None
        if implementation is not None:
            self.__doc__ = implementation.__doc__
            self.__wrapped__ = implementation

    @property
    def __name__(self) -> str:  # noqa: PLW3201
        return self.name

    def __call__(self, *args: Any) -> Any:
        """Run the reference implementation on *args* (outside a kernel)."""
        if self.implementation is None:
            raise InvalidTypeError(
                f"{self.name} is external and has no reference implementation; "
                "simulate it with handlers={...}, or declare it with a body"
            )
        if len(args) != len(self.args):
            raise InvalidTypeError(
                f"{self.name}() takes {len(self.args)} arguments, got {len(args)}"
            )
        return self.implementation(*args)

    def submit(self, *args: Any) -> Pending:
        """Call the function; outside a kernel this runs it at once."""
        return Answer(self(*args), self)

    # A declaration, like a function: a copy is the function itself.
    def __copy__(self) -> ClassicalFunction:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ClassicalFunction:
        return self

    def __reduce_ex__(self, protocol: Any) -> Any:
        from .._pickling import classical_reduce

        return classical_reduce(self, protocol)

    def __repr__(self) -> str:
        args = ", ".join(a.descriptor for a in self.args)
        results = ", ".join(r.descriptor for r in self.results)
        return f"<classical function {self.name}({args}) -> ({results}) on link {self.link!r}>"


class Pending(Protocol):
    """What ``fn.submit(...)`` returns: a future while a kernel is traced, an
    answer already taken when called outside a kernel.

    ``.result()`` gives the declared results; with ``timeout`` (an int number
    of program cycles, or a time with units) it gives ``(in_time, *results)``.
    """

    def result(self, timeout: int | Time | None = None) -> Any: ...


class Answer:
    """The answer of a classical function called outside a kernel."""

    __slots__ = ("_value", "_function")

    def __init__(self, value: Any, function: ClassicalFunction) -> None:
        self._value = value
        self._function = function

    def done(self) -> bool:
        return True

    def result(self, timeout: int | Time | None = None) -> Any:
        if timeout is not None:
            if isinstance(self._value, tuple):
                return (True, *self._value)
            return (True, self._value)
        return self._value


@overload
def function(
    implementation: Callable[..., Any], /, **options: Unpack[_NamedOptions]
) -> ClassicalFunction: ...
@overload
def function(
    implementation: None = None, /, **options: Unpack[_NamedOptions]
) -> Callable[[Callable[..., Any]], ClassicalFunction]: ...
def function(
    implementation: Callable[..., Any] | None = None, /, **options: Any
) -> ClassicalFunction | Callable[[Callable[..., Any]], ClassicalFunction]:
    """Declare a classical function; ``pulse.classical`` is this decorator.

    Keyword options: ``args`` and ``results`` (sequences of types such as
    ``Bits(4)``, ``Float()``, ``Floats(8)``), ``max_latency_ns`` (required),
    ``min_latency_ns``, ``nominal_latency_ns``, ``latency_mode``
    (``"bounded"``, ``"timeout"`` with ``timeout_ns``, or ``"blocking"``),
    ``submit`` (``"eager"`` or ``"lazy"``), ``deadline_ns``, ``link`` (the
    link its requests travel on: a name from ``MachineModel.links``, such as
    ``"controller"``, or a link id; default 0; ``device`` is not used for
    it, because a cudaq-pulse device is the QPU), ``name`` and ``id``. A
    latency, timeout or deadline is a time (``400 * pulse.ns``) or a number
    of ns.
    """
    if implementation is not None:
        return ClassicalFunction(implementation, **options)

    def decorate(fn: Callable[..., Any]) -> ClassicalFunction:
        return ClassicalFunction(fn, **options)

    return decorate


def external(name: str, /, **options: Unpack[_Options]) -> ClassicalFunction:
    """A classical function with no reference implementation, by *name*.

    It takes the keyword options of ``pulse.classical``. Simulating a program
    that calls it needs a handler for it.
    """
    return ClassicalFunction(None, name=name, **options)


def random_index(
    n: int,
    /,
    *,
    name: str = "random_index",
    max_latency_ns: float | Time = 0.0,
    link: int | str = 0,
    **options: Any,
) -> ClassicalFunction:
    """An on-device random source: a classical function of no arguments whose
    answer is an integer drawn uniformly from ``0 .. n - 1``.

    On hardware the controller's random number generator answers it (the
    manifest marks it ``"random"``). Simulation without shots follows every
    value, each with probability 1/n; with shots each shot draws one.
    Emulation draws from its own seeded generator. Use it to pick a
    randomized-benchmarking gate per shot, branching on the index.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise InvalidValueError(f"random_index needs at least 2 values, got {n!r}")
    import random

    function = ClassicalFunction(
        lambda: random.randrange(n),
        name=name,
        args=(),
        results=(Integer(),),
        max_latency_ns=max_latency_ns,
        link=link,
        **options,
    )
    function.uniform = n
    return function


_Function = ClassicalFunction


class _Declare(Protocol):
    @overload
    def __call__(
        self, implementation: Callable[..., Any], /, **options: Unpack[_NamedOptions]
    ) -> _Function: ...
    @overload
    def __call__(
        self, implementation: None = None, /, **options: Unpack[_NamedOptions]
    ) -> Callable[[Callable[..., Any]], _Function]: ...


class _External(Protocol):
    def __call__(self, name: str, /, **options: Unpack[_Options]) -> _Function: ...


class ClassicalDecorator(_Declare, Protocol):
    """What type checkers take ``pulse.classical`` to be: the decorator, with
    this module's names as attributes.

    A module can't be declared callable, so ``cudaq_pulse`` declares its
    ``classical`` attribute with this type for checkers only; at run time it
    is this module.
    """

    Bits: type[Bits]
    Levels: type[Levels]
    IQ: type[IQ]
    Float: type[Float]
    Integer: type[Integer]
    Byte: type[Byte]
    Floats: type[Floats]
    Integers: type[Integers]
    Bytes: type[Bytes]
    Link: type[Link]
    ClassicalFunction: type[_Function]
    Pending: type[Pending]
    function: _Declare
    external: _External
    function_id: Callable[[str], int]
    random_index: Callable[..., _Function]
    # Submodules, once imported.
    calibration: types.ModuleType
    emulation: types.ModuleType
    loop: types.ModuleType
    qec: types.ModuleType
    wire: types.ModuleType


class _ClassicalModule(types.ModuleType):
    """This module, callable as the ``pulse.classical`` decorator."""

    def __call__(self, implementation: Callable[..., Any] | None = None, /, **options: Any) -> Any:
        return function(implementation, **options)


sys.modules[__name__].__class__ = _ClassicalModule
