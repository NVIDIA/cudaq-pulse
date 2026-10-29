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
a function. Its type hints say what it takes and returns::

    import cudaq_pulse as pulse

    @pulse.classical(max_latency_ns=3000)
    def decode(syndromes: pulse.Bits) -> bool:
        return bool(syndromes[0] ^ syndromes[3])

    @pulse.kernel
    def round(...):
        ...
        answer = decode.submit([m0, m1, m2, m3])   # send the request
        if answer.result():                        # take the answer
            drive(...)

``pulse.Bits`` takes measurements read as bits, ``list[int]`` measured
levels or numbers, ``list[complex]`` IQ points and ``list[float]`` numbers;
``float``, ``int`` and ``complex`` take one value. A list's length comes from
the call: a function called with lists of two lengths compiles to two
versions, each named by its lengths (``decode__4``). Results are typed the
same way, and a list result's length is inferred from the body per version.
``pulse.Length[pulse.Bits, 8]`` states a fixed length, for a function with
one wire schema (``id=``) or a device that takes one size.

Inside a kernel, ``.submit(...)`` sends a request and returns a future at
once; pulses that do not need the answer keep playing. ``.result()`` is where
the answer is taken: it is due the maximum latency after the request is sent,
and the pulses whose amplitude, phase, frequency or branch condition come from
it start no earlier. An answer sets a duration only through a variable
wait, ``wait(line, answer, bounds=(lo, hi))``, which begins a new epoch of
the timeline. Outside a kernel, calling the function runs its body, its
reference implementation, directly.

The body is what simulation runs. On hardware the service registered under
the function's id (the 32-bit FNV-1a hash of its name, as CUDA-Q Realtime
uses) answers instead.
"""

from __future__ import annotations

import builtins
import collections.abc
import math
import sys
import types
import typing
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, Required, TypedDict, Unpack, overload

from .._errors import InvalidTypeError, InvalidValueError
from .._units import Time, as_nanoseconds

if TYPE_CHECKING:
    from ._link import Link as Link
from ._values import (
    _IQ,
    ValueType,
    _Bits,
    _Float,
    _Floats,
    _Integer,
    _Integers,
    _Levels,
    _OneBit,
    _OnePoint,
)

__all__ = [
    "Bits",
    "ClassicalDecorator",
    "ClassicalFunction",
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


# Type checkers see a list of ints, so a body returns any list of bits and
# ``pulse.Bits()``, an external's placeholder result, checks too (an alias made
# with a ``type`` statement can't be called).
if typing.TYPE_CHECKING:
    Bits: typing.TypeAlias = list[int]  # noqa: UP040
else:

    class Bits(list):
        """Measurements read as bits (``outcome != 0``), bit-packed on the wire.

        A kernel passes a list of measurements, ``fn.submit([m0, m1])``; the
        body reads ``bits[0]``, ``len(bits)`` or iterates. As a result, the
        bits are answered as booleans a kernel can branch on. The length
        comes from the call, or is stated, ``pulse.Length[pulse.Bits, 8]``.
        ``pulse.Bits()`` is an empty list, an external's placeholder result.
        """

        __slots__ = ()


# ── Reading type hints ───────────────────────────────────────────────────────

# List kinds: measurements read as bits, measurements read as IQ points,
# numbers as f64, and list[int] (measured levels or i32 numbers, by what
# the call passes).
_KIND_BITS, _KIND_IQ, _KIND_FLOATS, _KIND_INTS = 2, 3, 4, 5
_LIST_KIND: dict[Any, int] = {
    Bits: _KIND_BITS,
    list[complex]: _KIND_IQ,
    list[float]: _KIND_FLOATS,
    list[int]: _KIND_INTS,
    collections.abc.Sequence[complex]: _KIND_IQ,
    collections.abc.Sequence[float]: _KIND_FLOATS,
    collections.abc.Sequence[int]: _KIND_INTS,
    typing.Sequence[complex]: _KIND_IQ,  # noqa: UP006
    typing.Sequence[float]: _KIND_FLOATS,  # noqa: UP006
    typing.Sequence[int]: _KIND_INTS,  # noqa: UP006
}
_LIST_NAME = {
    _KIND_BITS: "pulse.Bits",
    _KIND_IQ: "list[complex]",
    _KIND_FLOATS: "list[float]",
    _KIND_INTS: "list[int]",
}
_F64, _I64 = _Float(), _Integer()
_SCALAR_ARG: dict[Any, ValueType] = {float: _F64, int: _I64, complex: _OnePoint()}
_SCALAR_RESULT: dict[Any, ValueType] = {float: _F64, int: _I64, bool: _OneBit()}
_VARIADIC = 0x04 | 0x08  # CO_VARARGS | CO_VARKEYWORDS


def _sized(kind: int, n: int, measured: bool = True) -> ValueType:
    if kind == _KIND_BITS:
        return _Bits(n)
    if kind == _KIND_IQ:
        return _IQ(n)
    if kind == _KIND_FLOATS:
        return _Floats(n)
    return _Levels(n) if measured else _Integers(n)


_NOT_A_NAME: Any = object()
_HINTS: dict[tuple[int, str], Any] = {}
_BUILTIN_NAMES = {"float": float, "int": int, "bool": bool, "complex": complex, "None": None}


def _hint(t: Any, fn: Any) -> Any:
    """An annotation's value. A ``from __future__`` string is looked up when
    it is a plain or dotted name or a one-level ``list[...]``/``tuple[...]``,
    else evaluated; each is read once per module and string."""
    if t.__class__ is not str:
        return t
    g = fn.__globals__
    key = (id(g), t)
    try:
        return _HINTS[key]
    except KeyError:
        pass
    value = _name(t, g)
    if value is _NOT_A_NAME:
        value = _subscript(t, g)
    if value is _NOT_A_NAME:
        try:
            value = eval(t, g)  # noqa: S307 - the function's own annotation
        except Exception as error:
            raise InvalidTypeError(
                f"{fn.__qualname__}: can't read the type hint {t!r} ({error}); a hint is read"
                " in the function's module, so the names it uses must be defined there"
            ) from None
    _HINTS[key] = value
    return value


def _name(t: str, g: dict[str, Any]) -> Any:
    if t in _BUILTIN_NAMES and t not in g:
        return _BUILTIN_NAMES[t]
    if not t.replace(".", "").replace("_", "a").isalnum() or t[:1].isdigit():
        return _NOT_A_NAME
    head, *rest = t.split(".")
    if head in g:
        value = g[head]
    elif hasattr(builtins, head):
        value = getattr(builtins, head)
    else:
        return _NOT_A_NAME  # eval raises the NameError
    for part in rest:
        value = getattr(value, part, _NOT_A_NAME)
        if value is _NOT_A_NAME:
            return _NOT_A_NAME
    return value


def _subscript(t: str, g: dict[str, Any]) -> Any:
    """``tuple[float, float]``, ``list[int]`` or ``pulse.Length[pulse.Bits,
    8]``, one level deep, without eval."""
    i = t.find("[")
    if i <= 0 or t[-1] != "]" or "[" in t[i + 1 :]:
        return _NOT_A_NAME
    head = _name(t[:i].strip(), g)
    items = [_item(s.strip(), g) for s in t[i + 1 : -1].split(",")]
    if head is _NOT_A_NAME or any(x is _NOT_A_NAME for x in items):
        return _NOT_A_NAME
    try:
        return head[tuple(items)] if len(items) > 1 else head[items[0]]
    except TypeError:
        return _NOT_A_NAME


def _item(s: str, g: dict[str, Any]) -> Any:
    if s.isdigit():
        return int(s)
    if len(s) > 1 and s[0] == s[-1] and s[0] in "\"'" and s[1:-1].isidentifier():
        return s[1:-1]
    return _name(s, g)


def _result_items(t: Any) -> tuple[Any, ...]:
    if t is None or t is type(None):
        return ()
    if typing.get_origin(t) is tuple:
        return typing.get_args(t)
    if isinstance(t, type) and issubclass(t, tuple) and hasattr(t, "_fields"):
        hints = typing.get_type_hints(t)
        return tuple(hints[f] for f in t._fields)
    return (t,)


class _Rule:
    """A list result whose length each version infers from the body."""

    __slots__ = ("kind",)

    def __init__(self, kind: int) -> None:
        self.kind = kind


class _Shape:
    """A function's declaration read from its hints: parameter names, per
    parameter a wire type or an unsized list kind (an int), and per result
    a wire type or a ``_Rule``; how many parameters are unsized and whether
    any result is a rule."""

    __slots__ = ("names", "plan", "results", "unsized", "rules")

    def __init__(
        self,
        names: tuple[str, ...],
        plan: tuple[Any, ...],
        results: tuple[Any, ...],
        unsized: int,
        rules: bool,
    ):
        self.names, self.plan, self.results = names, plan, results
        self.unsized, self.rules = unsized, rules


def _read_hints(fn: Any) -> _Shape:
    """Read a function's annotations from ``__code__`` and
    ``__annotations__`` (no ``inspect.signature`` or ``get_type_hints``)."""
    where = fn.__qualname__
    code = fn.__code__
    if code.co_flags & _VARIADIC or code.co_kwonlyargcount:
        raise InvalidTypeError(f"{where}: only positional parameters are sent")
    names = code.co_varnames[: code.co_argcount]
    ann = fn.__annotations__
    plan: list[Any] = []
    unsized = 0
    for name in names:
        if name not in ann:
            raise InvalidTypeError(
                f"{where}: parameter {name!r} needs a type, such as float or pulse.Bits"
            )
        t = _hint(ann[name], fn)
        spec: Any = _SCALAR_ARG.get(t)
        if spec is None:
            kind = _LIST_KIND.get(t)
            if kind is None:
                spec = _stated_parameter(where, name, t)
            else:
                spec = kind
                unsized += 1
        plan.append(spec)
    if "return" not in ann:
        raise InvalidTypeError(f"{where}: annotate what it returns (-> None if it answers nothing)")
    returned = _hint(ann["return"], fn)
    if len(plan) <= _MAX_ARGS:
        try:
            scalar = _SCALAR_RESULT.get(returned)
        except TypeError:  # unhashable: not a type, the checks below say so
            scalar = None
        if scalar is not None:
            return _Shape(tuple(names), tuple(plan), (scalar,), unsized, False)
    items = _result_items(returned)
    results: list[Any] = []
    for t in items:
        spec = _SCALAR_RESULT.get(t)
        if spec is None:
            kind = _LIST_KIND.get(t)
            if kind is None:
                spec = _stated_result(where, t)
            elif kind == _KIND_IQ:
                _bad_result(where, t)
            else:
                spec = _Rule(kind)
        results.append(spec)
    if len(plan) > _MAX_ARGS:
        raise InvalidTypeError(f"{where}: at most {_MAX_ARGS} arguments, got {len(plan)}")
    if len(results) > _MAX_RESULTS:
        raise InvalidTypeError(f"{where}: at most {_MAX_RESULTS} results, got {len(results)}")
    rules = any(r.__class__ is _Rule for r in results)
    return _Shape(tuple(names), tuple(plan), tuple(results), unsized, rules)


def _stated_parameter(where: str, name: str, t: Any) -> ValueType:
    """A parameter of fixed length, ``pulse.Length[kind, n]``: *n*
    measurements read as bits for ``pulse.Bits``, else *n* numbers."""
    if typing.get_origin(t) is not typing.Annotated:
        _bad_parameter(where, name, t)
    base, meta = t.__origin__, t.__metadata__
    kind = _LIST_KIND.get(base)
    if kind is None:
        if base in _SCALAR_ARG or base is bool:
            raise InvalidTypeError(
                f"{where}: parameter {name!r} is {_spell(base)}; pulse.Length applies to"
                " pulse.Bits and lists"
            )
        _bad_parameter(where, name, base)
    return _sized(kind, _length(where, kind, meta, "parameter"), measured=False)


def _stated_result(where: str, t: Any) -> ValueType:
    """A list result of fixed length, ``pulse.Length[kind, n]``."""
    if typing.get_origin(t) is not typing.Annotated:
        _bad_result(where, t)
    base, meta = t.__origin__, t.__metadata__
    kind = _LIST_KIND.get(base)
    if kind is None or kind == _KIND_IQ:
        if base in _SCALAR_RESULT or base is complex:
            raise InvalidTypeError(
                f"{where}: its result is {_spell(base)}; pulse.Length applies to pulse.Bits"
                " and lists"
            )
        _bad_result(where, base)
    return _sized(kind, _length(where, kind, meta, "result"), measured=False)


def _length(where: str, kind: int, meta: tuple[Any, ...], what: str) -> int:
    """The one positive whole number a ``pulse.Length`` states."""
    spelled = _spell_length(kind, meta)
    if len(meta) != 1:
        raise InvalidTypeError(
            f"{where}: {spelled}: state one length, such as pulse.Length[{_LIST_NAME[kind]}, 8]"
        )
    n = meta[0]
    if n.__class__ is int and n > 0:
        return int(n)
    tail = ""
    if n.__class__ is str and what == "result":
        tail = f"; a list result without one takes its length from the body: -> {_LIST_NAME[kind]}"
    elif n.__class__ is str and what == "parameter":
        tail = f"; a list parameter without one takes its length from the call: {_LIST_NAME[kind]}"
    raise InvalidTypeError(
        f"{where}: {spelled}: a length is a positive whole number, got {n!r}{tail}"
    )


_SPELLED = {
    "bits": ("pulse.Bits", "", "bit"),
    "levels": ("list[int]", "", "measurement"),
    "iq": ("list[complex]", "", "IQ point"),
    "f64": ("list[float]", "", "number"),
    "i32": ("list[int]", "", "number"),
}


def _spelled(spec: ValueType) -> tuple[str, str]:
    """A list type's ``pulse.Length[...]`` spelling and what it counts
    (``"bit"``, ``"number"``...), for errors."""
    head = spec.descriptor.split(":")[0].split("[")[0]
    name, flag, unit = _SPELLED.get(head, (spec.descriptor, "", "value"))
    return f"pulse.Length[{name}, {spec.count}{flag}]", unit


def _wrong_length(function: ClassicalFunction[..., Any], position: int, got: int) -> str:
    """A handler's or body's answer of the wrong length, at run time."""
    spec = function.results[position]
    spelled, unit = _spelled(spec)
    which = f" as result {position}" if len(function.results) > 1 else ""
    return (
        f"{function.name} returned {got} {unit}{'s' if got != 1 else ''}{which}; its result is"
        f" {spelled} (id={function.id:#x})"
    )


def _spell_length(kind: int, meta: tuple[Any, ...]) -> str:
    items = ", ".join(f'"{m}"' if m.__class__ is str else repr(m) for m in meta)
    return f"pulse.Length[{_LIST_NAME[kind]}, {items}]"


def _bad_parameter(where: str, name: str, t: Any) -> typing.NoReturn:
    from ..ops import MeasurementResult

    if t is bool:
        raise InvalidTypeError(
            f"{where}: parameter {name!r} is bool; a measured bit is pulse.Bits"
            " (submit [m], read bits[0])"
        )
    if t is MeasurementResult:
        raise InvalidTypeError(
            f"{where}: parameter {name!r} is MeasurementResult; say how to read it: pulse.Bits,"
            " list[int] (levels) or complex (an IQ point)"
        )
    if typing.get_origin(t) in (list, collections.abc.Sequence):
        raise InvalidTypeError(
            f"{where}: parameter {name!r} is {_spell(t)}; lists hold float, int or complex,"
            " and measured bits are pulse.Bits"
        )
    raise InvalidTypeError(
        f"{where}: parameter {name!r} can't be {_spell(t)}; use float, int, complex,"
        " pulse.Bits, list[float], list[int] or list[complex]"
    )


def _bad_result(where: str, t: Any) -> typing.NoReturn:
    if t is complex or typing.get_args(t) == (complex,):
        raise InvalidTypeError(
            f"{where}: can't return {_spell(t)}: a kernel can't use a complex answer;"
            " return tuple[float, float] instead"
        )
    raise InvalidTypeError(
        f"{where}: can't return {_spell(t)}; use float, int, bool, pulse.Bits, list[float],"
        " list[int], a tuple of them, or None"
    )


def _spell(t: Any) -> str:
    if isinstance(t, type) and not typing.get_args(t):
        return t.__name__
    return repr(t).replace("typing.", "")


# ── Classical functions ──────────────────────────────────────────────────────

# Declarations whose hints are read at the next compile (``flush``).
_PENDING: list[ClassicalFunction[..., Any]] = []


class _LazyTypes:
    """Stands in for ``args`` or ``results`` until the hints are read.
    Touching it resolves the function, which replaces both attributes with
    tuples, so later reads are plain attribute loads."""

    __slots__ = ("owner", "which")

    def __init__(self, owner: ClassicalFunction[..., Any], which: str) -> None:
        self.owner = owner
        self.which = which

    def _real(self) -> tuple[ValueType, ...]:
        _resolve(self.owner)
        real: tuple[ValueType, ...] = getattr(self.owner, self.which)
        return real

    def __len__(self) -> int:
        return len(self._real())

    def __iter__(self) -> Iterator[ValueType]:
        return iter(self._real())

    def __getitem__(self, i: Any) -> Any:
        return self._real()[i]

    def __eq__(self, other: object) -> bool:
        return self._real() == other

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        return repr(self._real())


class _Unsized:
    """An unsized parameter in the plan a template's submit reads. Its
    ``source`` sets it apart from a wire type's."""

    __slots__ = ("source", "kind")

    def __init__(self, kind: int) -> None:
        self.source = _UNSIZED
        self.kind = kind


_UNSIZED = "unsized"
_NO_VERSIONS: Mapping[Any, Any] = types.MappingProxyType({})


class ClassicalFunction[**P, R]:
    """A classical function a pulse kernel can call while the QPU runs.

    Made by ``@pulse.classical(...)`` or ``@pulse.classical.external(...)``.
    In a kernel, ``.submit(*args)`` sends a request and returns a future
    whose ``.result()`` takes the answer. Outside a kernel, calling it runs
    its reference implementation.

    A function with a list parameter of no stated length is a template: each
    set of list lengths it is called with in a kernel is a version, named
    ``name__<lengths>`` (``decode__4``, ``levels__i3`` for three i32 numbers)
    with the id of that name. ``specialize(...)`` gives one version.
    """

    def __init__(
        self,
        implementation: Callable[P, R] | None,
        *,
        name: str | None = None,
        max_latency_ns: float | Time,
        min_latency_ns: float | Time = 0.0,
        nominal_latency_ns: float | Time | None = None,
        latency_mode: str = "bounded",
        timeout_ns: float | Time | None = None,
        submit: str = "eager",
        deadline_ns: float | Time | None = None,
        link: int | str = 0,
        id: int | None = None,
        _signature: Callable[..., Any] | None = None,
    ) -> None:
        max_latency_ns = as_nanoseconds(max_latency_ns)
        min_latency_ns = as_nanoseconds(min_latency_ns)
        nominal_latency_ns = as_nanoseconds(nominal_latency_ns)
        timeout_ns = as_nanoseconds(timeout_ns)
        deadline_ns = as_nanoseconds(deadline_ns)
        hinted = implementation if implementation is not None else _signature
        if name is None:
            if hinted is None:
                raise InvalidTypeError("an external classical function needs a name")
            name = hinted.__name__
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            raise InvalidValueError(
                f"a classical function name must be an identifier, got {name!r}"
            )
        self.name = name
        self.implementation = implementation
        self.args: tuple[ValueType, ...] = _LazyTypes(self, "args")  # type: ignore[assignment]
        self.results: tuple[ValueType, ...] = _LazyTypes(self, "results")  # type: ignore[assignment]
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
        self.__doc__ = hinted.__doc__ if hinted is not None else None
        self.__wrapped__ = hinted
        # Every attribute is set here, in one order, so all functions share
        # one attribute layout (attributes added later, or in other orders,
        # make every read slower).
        self._error: TypeError | None = None
        # A template's declaration, the plan its submit reads and how many
        # lengths its calls give; None, None, 0 for a function of one schema.
        self._shape: _Shape | None = None
        self._plan: tuple[Any, ...] | None = None
        self._n = 0
        self._made: dict[Any, ClassicalFunction[P, R]] | None = None
        # The versions a submit takes without a per-program check: _made, or
        # none with id= (its one version per program is checked each time).
        self._fast: Mapping[Any, ClassicalFunction[P, R]] = _NO_VERSIONS
        # A version's template and lengths.
        self._template: ClassicalFunction[P, R] | None = None
        self._key: Any = None
        self.base_name = name
        # A template's list-result lengths per version (classical._infer).
        self._length_rules: Any = None

    @classmethod
    def _declared(
        cls,
        implementation: Callable[..., Any] | None,
        args: Sequence[ValueType],
        results: Sequence[ValueType],
        **options: Any,
    ) -> ClassicalFunction[..., Any]:
        """A function of one schema given as wire types, for the functions
        cudaq-pulse declares itself."""
        function: ClassicalFunction[..., Any] = cls(implementation, **options)
        function.args = tuple(args)
        function.results = tuple(results)
        return function

    @property
    def __name__(self) -> str:  # noqa: PLW3201
        return self.name

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Run the reference implementation on *args* (outside a kernel)."""
        if self.implementation is None:
            how = ""
            if self.__wrapped__ is not None:
                how = " it is declared with @pulse.classical.external, so"
            raise InvalidTypeError(
                f"{self.base_name} runs on link {self.link!r};{how} it has no body to run here."
                " Give it a handler to simulate it:"
                f' pulse.simulate(..., handlers={{"{self.base_name}": fn}})'
            )
        expected = len(self.args)
        if self._plan is not None:
            expected = len(self._plan)
        if kwargs or len(args) != expected:
            raise InvalidTypeError(
                f"{self.name}() takes {expected} positional arguments, got {len(args)}"
            )
        return self.implementation(*args, **kwargs)

    def submit(self, *args: P.args, **kwargs: P.kwargs) -> Pending[R]:
        """Call the function; outside a kernel this runs it at once."""
        return Answer(self(*args, **kwargs))

    def specialize(self, **lengths: int) -> ClassicalFunction[P, R]:
        """The version of this function for list parameters of *lengths*
        (by parameter name), as a kernel call with lists of those lengths
        makes it. A function of one schema is its own version."""
        _resolve(self)
        if self._shape is None:
            if lengths:
                raise InvalidTypeError(
                    f"{self.name} has one schema; it has no lengths to specialize"
                )
            return self
        names = _unsized_names(self)
        unknown = sorted(set(lengths) - set(names))
        if unknown:
            raise InvalidTypeError(
                f"{self.name}.specialize(): {', '.join(unknown)} isn't a list parameter of"
                " unstated length"
            )
        missing = [n for n in names if n not in lengths]
        if missing:
            raise InvalidTypeError(
                f"{self.name}.specialize(): give lengths for {', '.join(missing)}"
            )
        key: list[int] = []
        for name in names:
            n = lengths[name]
            if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
                raise InvalidTypeError(
                    f"{self.name}.specialize(): {name} must be a positive length"
                )
            key.append(n)
        version = _version(self, key[0] if self._n == 1 else tuple(key))
        return typing.cast("ClassicalFunction[P, R]", version)

    # A declaration, like a function: a copy is the function itself.
    def __copy__(self) -> ClassicalFunction[P, R]:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ClassicalFunction[P, R]:
        return self

    def __reduce_ex__(self, protocol: Any) -> Any:
        from .._pickling import classical_reduce

        return classical_reduce(self, protocol)

    def __repr__(self) -> str:
        try:
            _resolve(self)
        except TypeError:
            return f"<classical function {self.name}, not declared correctly>"
        if self._shape is not None:
            plan = self._shape.plan
            names = self._shape.names
            args = ", ".join(
                f"{_LIST_NAME[p]} {n}" if p.__class__ is int else p.descriptor
                for n, p in zip(names, plan, strict=True)
            )
            results = ", ".join(
                f"{_LIST_NAME[r.kind]}" if r.__class__ is _Rule else r.descriptor
                for r in self._shape.results
            )
        else:
            args = ", ".join(a.descriptor for a in self.args)
            results = ", ".join(r.descriptor for r in self.results)
        return f"<classical function {self.name}({args}) -> ({results}) on link {self.link!r}>"


def _resolve(function: ClassicalFunction[..., Any]) -> None:
    """Read a declaration's hints, once. Never touch ``__dict__``: that
    changes the attribute layout every read goes through."""
    if function.args.__class__ is not _LazyTypes:  # type: ignore[comparison-overlap]
        return
    if function._error is not None:
        raise function._error
    hinted = function.__wrapped__
    try:
        shape = _read_hints(hinted)
        if not shape.unsized:
            if shape.rules:
                stated = {
                    n: p.count
                    for n, p in zip(shape.names, shape.plan, strict=True)
                    if not p.is_scalar
                }
                function.results = _result_specs(function, shape, stated, "")
            else:
                function.results = shape.results
            function.args = shape.plan
            return
    except TypeError as error:
        function._error = error
        raise
    function.results = () if shape.rules else shape.results
    # A template: args == () sends every kernel submit to the arity check,
    # where the template path starts (see kernel._classical.submit).
    function.args = ()
    function._shape = shape
    function._plan = tuple(_Unsized(p) if p.__class__ is int else p for p in shape.plan)
    function._n = shape.unsized
    function._made = {}
    if not function.custom_id:
        function._fast = function._made


def flush() -> None:
    """Read the hints of every declaration made since the last compile; an
    error stays on its function and is raised where the function is used."""
    while _PENDING:
        function = _PENDING.pop()
        try:
            _resolve(function)
        except TypeError:
            pass


def _result_specs(
    function: ClassicalFunction[..., Any],
    shape: _Shape,
    lengths: dict[str, int],
    text: str,
) -> tuple[ValueType, ...]:
    """A version's results: rules made sized with its input lengths."""
    rules = [(i, r) for i, r in enumerate(shape.results) if r.__class__ is _Rule]
    if not rules:
        return shape.results
    out = list(shape.results)
    inferred = [(i, _LIST_NAME[r.kind]) for i, r in rules]
    found: dict[int, int] = {}
    if inferred:
        known = function._length_rules
        if known is None:
            from ._infer import LengthRules

            bits = frozenset(n for n, p in zip(shape.names, shape.plan) if p is _KIND_BITS)
            # An external's stub is a declaration: its placeholder return,
            # such as `return outcomes`, gives the result's length.
            known = function._length_rules = LengthRules(
                function.__wrapped__, shape.names, inferred, len(out), bits
            )
        where = f"{function.name} at {text}" if text else function.name
        try:
            found = dict(zip((i for i, _ in inferred), known.lengths(where, lengths), strict=True))
        except TypeError as error:
            raise InvalidTypeError(f"{error} {_length_advice(function, shape, inferred)}") from None
    for i, r in rules:
        n = found[i]
        if n <= 0:
            where = f"{function.name} at {text}" if text else function.name
            if function.implementation is None:
                raise InvalidTypeError(
                    f"{where}: its {_LIST_NAME[r.kind]} result has no length: the placeholder"
                    f" says only its type. {_length_advice(function, shape, inferred)}"
                )
            raise InvalidTypeError(f"{where}: its {_LIST_NAME[r.kind]} result would be empty")
        out[i] = _sized(r.kind, n, measured=False)
    return tuple(out)


def _length_advice(
    function: ClassicalFunction[..., Any], shape: _Shape, inferred: list[tuple[int, str]]
) -> str:
    """How to give a list result a length the compiler can see."""
    kind = _LIST_NAME[shape.results[inferred[0][0]].kind]
    fixed = f"give it a fixed size: -> pulse.Length[{kind}, 8]"
    if function.implementation is None:
        plan = zip(shape.names, shape.plan, strict=True)
        inputs = [n for n, p in plan if p.__class__ is int or not p.is_scalar]
        if not inputs:
            return f"An external's body is a placeholder; {fixed}."
        return (
            f"An external's body is a placeholder: return {inputs[0]}, or a slice such as"
            f" {inputs[0]}[:2], for a length the input gives, or {fixed}."
        )
    return f"Write it in plain Python, or {fixed}."


# ── Versions ─────────────────────────────────────────────────────────────────

# __init__'s attribute order.
_COPIED = (
    "name",
    "implementation",
    "args",
    "results",
    "min_latency_ns",
    "nominal_latency_ns",
    "max_latency_ns",
    "latency_mode",
    "timeout_ns",
    "submit_policy",
    "deadline_ns",
    "link",
    "id",
    "custom_id",
    "uniform",
    "__doc__",
    "__wrapped__",
    "_error",
    "_shape",
    "_plan",
    "_n",
    "_made",
    "_fast",
    "_template",
    "_key",
    "base_name",
    "_length_rules",
)


def _unsized_names(template: ClassicalFunction[..., Any]) -> list[str]:
    shape = template._shape
    assert shape is not None
    return [shape.names[i] for i, p in enumerate(shape.plan) if p.__class__ is int]


def _key_items(key: Any) -> tuple[int, ...]:
    return (key,) if key.__class__ is int else tuple(key)


def _lengths_text(template: ClassicalFunction[..., Any], key: Any) -> str:
    return ", ".join(
        f"{name}={abs(n)}"
        for name, n in zip(_unsized_names(template), _key_items(key), strict=True)
    )


def _make(template: ClassicalFunction[..., Any], key: Any) -> ClassicalFunction[..., Any]:
    """The version of *template* for *key*: per unsized parameter its
    length, negative for a ``list[int]`` of numbers (i32) rather than
    measurements (levels)."""
    from ..kernel.ir_builder import CompilationError

    shape = template._shape
    assert shape is not None
    version: ClassicalFunction[..., Any] = object.__new__(ClassicalFunction)
    for attr in _COPIED:
        setattr(version, attr, getattr(template, attr))
    version._shape, version._plan, version._n, version._made = None, None, 0, None
    version._fast = _NO_VERSIONS
    items = iter(_key_items(key))
    args = list(shape.plan)
    for i, p in enumerate(args):
        if p.__class__ is int:
            n = next(items)
            args[i] = _sized(p, abs(n), n >= 0)
    version.args = tuple(args)
    text = _lengths_text(template, key)
    if not version.results:  # () only when a result's length is per version
        lengths = {n: p.count for n, p in zip(shape.names, args, strict=True) if not p.is_scalar}
        try:
            version.results = _result_specs(template, shape, lengths, text)
        except TypeError as error:
            raise CompilationError(str(error)) from None
    if not template.custom_id:
        suffix = "_".join(str(n) if n >= 0 else f"i{-n}" for n in _key_items(key))
        version.name = f"{template.name}__{suffix}"
        version.id = function_id(version.name)
    version._template = template
    version._key = key
    version.base_name = template.name
    return version


def _version(template: ClassicalFunction[..., Any], key: Any) -> ClassicalFunction[..., Any]:
    made = template._made
    assert made is not None
    version = made.get(key)
    if version is None:
        version = made[key] = _make(template, key)
    return version


def _enter_slow(builder: Any, template: ClassicalFunction[..., Any], key: Any) -> Any:
    """A submit whose version isn't made yet, or of a function with ``id=``
    (every one): ``builder._versions[template]`` maps each key to its
    version, in first use order. With ``id=`` the function has one version
    per program."""
    from ..kernel.ir_builder import CompilationError

    inner = builder._versions.get(template)
    if inner is None:
        inner = builder._versions[template] = {}
    else:
        entered = inner.get(key)
        if entered is not None:
            return entered
    if inner and template.custom_id:
        first = next(iter(inner))
        raise CompilationError(
            f"{template.name}.submit(): {template.name} has id={template.id:#x}, which names one"
            f" wire schema, but this program calls it with {_lengths_text(template, key)} here"
            f" and {_lengths_text(template, first)} earlier. Call it with one length, or drop"
            " id= to compile one version per length."
        )
    version = _version(template, key)
    inner[key] = version
    return version


def _handlers_for(
    functions: Sequence[ClassicalFunction[..., Any]],
    handlers: Mapping[str, Callable[..., Any]] | None,
) -> dict[str, Callable[..., Any]]:
    """What answers each function of a program, by name: a handler given
    for its name, else one given for its template's name (serving every
    version), else its body."""
    chosen: dict[str, Callable[..., Any]] = {
        f.name: f.implementation for f in functions if f.implementation is not None
    }
    for f in functions:
        handler = None
        if handlers:
            handler = handlers.get(f.name) or handlers.get(f.base_name)
        if handler is not None:
            chosen[f.name] = handler
    for name, handler in (handlers or {}).items():
        chosen.setdefault(name, handler)
    return chosen


class Pending[R](Protocol):
    """What ``fn.submit(...)`` returns: a future while a kernel is traced, an
    answer already taken when called outside a kernel.

    ``.result()`` gives the declared results; with ``timeout`` (an int number
    of program cycles, or a time with units) it gives ``(in_time, *results)``.
    """

    @overload
    def result(self, timeout: None = None) -> R: ...
    @overload
    def result[*Ts](self: Pending[tuple[*Ts]], timeout: int | Time) -> tuple[bool, *Ts]: ...
    @overload
    def result(self, timeout: int | Time) -> tuple[bool, R]: ...
    def result(self, timeout: int | Time | None = None) -> Any: ...


class Answer[R]:
    """The answer of a classical function called outside a kernel."""

    __slots__ = ("_value",)

    def __init__(self, value: R) -> None:
        self._value = value

    def done(self) -> bool:
        return True

    @overload
    def result(self, timeout: None = None) -> R: ...
    @overload
    def result[*Ts](self: Answer[tuple[*Ts]], timeout: int | Time) -> tuple[bool, *Ts]: ...
    @overload
    def result(self, timeout: int | Time) -> tuple[bool, R]: ...
    def result(self, timeout: int | Time | None = None) -> Any:
        if timeout is not None:
            if isinstance(self._value, tuple):
                return (True, *self._value)
            return (True, self._value)
        return self._value


class _Options(TypedDict, total=False):
    """The keyword options of ``pulse.classical`` and ``external``."""

    max_latency_ns: Required[float | Time]
    min_latency_ns: float | Time
    nominal_latency_ns: float | Time | None
    latency_mode: str
    timeout_ns: float | Time | None
    submit: str
    deadline_ns: float | Time | None
    link: int | str
    id: int | None
    name: str | None


@overload
def function[**Q, T](
    implementation: Callable[Q, T], /, **options: Unpack[_Options]
) -> ClassicalFunction[Q, T]: ...
@overload
def function[**Q, T](
    implementation: None = None, /, **options: Unpack[_Options]
) -> Callable[[Callable[Q, T]], ClassicalFunction[Q, T]]: ...
def function(implementation: Callable[..., Any] | None = None, /, **options: Any) -> Any:
    """Declare a classical function; ``pulse.classical`` is this decorator.

    The parameters' and result's type hints say what it takes and returns:
    ``float``, ``int`` and ``complex`` (one IQ point) take one value;
    ``pulse.Bits`` (measurements read as bits), ``list[int]`` (measured
    levels, or numbers), ``list[complex]`` (IQ points) and ``list[float]``
    take a list whose length comes from the call. Results are ``float``,
    ``int``, ``bool``, ``pulse.Bits``, ``list[float]``, ``list[int]``, a tuple
    of them or ``None``; a list result's length is inferred from the body.

    ``pulse.Length`` states a fixed length where one wire schema is wanted,
    as with ``id=``: ``pulse.Length[pulse.Bits, 8]`` takes eight measurements
    read as bits and ``pulse.Length[list[int], 8]`` eight numbers; a result
    ``-> pulse.Length[pulse.Bits, 2]`` answers two bits.

    Keyword options: ``max_latency_ns`` (required), ``min_latency_ns``,
    ``nominal_latency_ns``, ``latency_mode`` (``"bounded"``, ``"timeout"``
    with ``timeout_ns``, or ``"blocking"``), ``submit`` (``"eager"`` or
    ``"lazy"``), ``deadline_ns``, ``link`` (the link its requests travel on:
    a name from ``MachineModel.links``, such as ``"controller"``, or a link
    id; default 0; ``device`` is not used for it, because a cudaq-pulse
    device is the QPU), ``name`` and ``id``. A latency, timeout or deadline
    is a time (``400 * pulse.ns``) or a number of ns.

    The hints are read at the next compile, and a mistake in them is raised
    where the function is used.
    """
    if "args" in options or "results" in options:
        _removed(options)
    if implementation is not None:
        return _declare(implementation, options)

    def decorate(fn: Callable[..., Any]) -> ClassicalFunction[..., Any]:
        return _declare(fn, options)

    return decorate


def _declare(fn: Callable[..., Any], options: dict[str, Any]) -> ClassicalFunction[..., Any]:
    if not isinstance(fn, types.FunctionType) or fn.__name__ == "<lambda>":
        raise InvalidTypeError(
            f"@pulse.classical takes a function defined with def, whose type hints say what it"
            f" takes and returns; got {fn!r}"
        )
    declared: ClassicalFunction[..., Any] = ClassicalFunction(fn, **options)
    _PENDING.append(declared)
    return declared


def _removed(options: Mapping[str, Any]) -> typing.NoReturn:
    raise InvalidTypeError(
        "args= and results= are gone: a classical function's type hints say what it"  # remnant-ok
        " takes and returns, such as def decode(syndromes: pulse.Bits) -> bool"
    )


def external[**Q, T](
    **options: Unpack[_Options],
) -> Callable[[Callable[Q, T]], ClassicalFunction[Q, T]]:
    """Declare a classical function served elsewhere: a declaration whose
    hints give its schema and whose body is a placeholder for its result,
    which never runs::

        @pulse.classical.external(max_latency_ns=50, link="feedback")
        def table(index: pulse.Length[pulse.Bits, 3]) -> float:
            return float()

        @pulse.classical.external(max_latency_ns=900, link="gpu")
        def decode(syndrome: pulse.Bits) -> tuple[float, pulse.Bits]:
            return (float(), syndrome[:2])

    A placeholder is a value of the result's type: ``float()``,
    ``pulse.Bits()`` (its length stated), an input (``return outcomes``) or a
    slice of one, whose length the result takes; a ``None`` result needs no
    return. A list parameter of no stated length takes its length from the
    call, one version per length as for ``pulse.classical``; state the
    length the device takes, ``pulse.Length[pulse.Bits, 3]``, and a call of
    another length is a compile error.

    It takes the keyword options of ``pulse.classical``. Simulating a program
    that calls it needs a handler for it.
    """
    if "args" in options or "results" in options:
        _removed(options)

    def decorate(stub: Callable[Q, T]) -> ClassicalFunction[Q, T]:
        if not isinstance(stub, types.FunctionType):
            raise InvalidTypeError(
                f"@pulse.classical.external takes a stub defined with def, got {stub!r}"
            )
        declared: ClassicalFunction[Q, T] = ClassicalFunction(None, _signature=stub, **options)
        _resolve(declared)
        return declared

    return decorate


def random_index(
    n: int,
    /,
    *,
    name: str = "random_index",
    max_latency_ns: float | Time = 0.0,
    link: int | str = 0,
    **options: Any,
) -> ClassicalFunction[[], int]:
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

    def draw() -> int:
        return random.randrange(n)

    function = ClassicalFunction._declared(
        draw,
        (),
        (_I64,),
        name=name,
        max_latency_ns=max_latency_ns,
        link=link,
        **options,
    )
    function.uniform = n
    return function


_AnyFunction = ClassicalFunction[..., Any]
_IndexFunction = ClassicalFunction[[], int]


class _Declare(Protocol):
    @overload
    def __call__[**Q, T](
        self, implementation: Callable[Q, T], /, **options: Unpack[_Options]
    ) -> ClassicalFunction[Q, T]: ...
    @overload
    def __call__[**Q, T](
        self, implementation: None = None, /, **options: Unpack[_Options]
    ) -> Callable[[Callable[Q, T]], ClassicalFunction[Q, T]]: ...


class _External(Protocol):
    def __call__[**Q, T](
        self, **options: Unpack[_Options]
    ) -> Callable[[Callable[Q, T]], ClassicalFunction[Q, T]]: ...


class ClassicalDecorator(_Declare, Protocol):
    """What type checkers take ``pulse.classical`` to be: the decorator, with
    this module's names as attributes.

    A module can't be declared callable, so ``cudaq_pulse`` declares its
    ``classical`` attribute with this type for checkers only; at run time it
    is this module.
    """

    Bits: type[Bits]
    Link: type[Link]
    ClassicalFunction: type[_AnyFunction]
    Pending: type[Pending[Any]]
    function: _Declare
    external: _External
    function_id: Callable[[str], int]
    random_index: Callable[..., _IndexFunction]
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


def __getattr__(name: str) -> Any:
    if name == "Link":
        from ._link import Link

        globals()["Link"] = Link
        return Link
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


sys.modules[__name__].__class__ = _ClassicalModule
