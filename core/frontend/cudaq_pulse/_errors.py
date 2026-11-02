# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The errors cudaq-pulse raises, and the problems they carry."""

from __future__ import annotations

import numbers
import sys
import threading
from collections.abc import Callable, Iterable
from dataclasses import Field, dataclass
from typing import Any, ClassVar, Final

__all__ = [
    "CompilationError",
    "DeviceDefinitionError",
    "ExecutionError",
    "IndexOutOfRangeError",
    "InternalError",
    "InvalidTypeError",
    "InvalidValueError",
    "NotFoundError",
    "Problem",
    "PulseError",
    "PulseWarning",
    "TimingError",
    "UnsupportedError",
    "UpgradeNotice",
]

# How many problems an error's message lists; ``problems`` holds them all.
_SHOWN = 10
_ISSUES_URL = "https://github.com/NVIDIA/cudaq-pulse/issues"


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __reduce__(self) -> str:
        return "UNSET"


UNSET: Final[Any] = _Unset()


@dataclass(frozen=True, slots=True)
class Problem:
    """One thing that is wrong, where it is, and how to fix it.

    ``message`` says what is wrong; ``path`` where, in the terms of what the
    user wrote (``"layout[2]"``, ``"controllers[0].drive_lines"``);
    ``value`` is the offending value (``Problem.UNSET`` when there is none);
    ``fix`` what to do. ``file``, ``line`` and ``column`` locate it in a
    file, when it came from one; ``line`` and ``column`` count from 1.
    """

    UNSET: ClassVar[Any] = UNSET

    message: str
    path: str = ""
    value: Any = UNSET
    fix: str = ""
    file: str | None = None
    line: int | None = None
    column: int | None = None

    def __str__(self) -> str:
        text = f"{self.path}: {self.message}" if self.path else self.message
        if self.fix:
            text = f"{text}; {self.fix}"
        if self.file is None:
            return text
        where = self.file
        if self.line is not None:
            where += f":{self.line}"
            if self.column is not None:
                where += f":{self.column}"
        return f"{where}: {text}"


class PulseError(Exception):
    """Base class of every error cudaq-pulse raises on purpose.

    Each subclass also has the built-in base its kind of error has in
    Python, so ``except ValueError``, ``except KeyError`` or ``except
    RuntimeError`` catches it as it would the built-in error. Errors that
    the native extension raises while running MLIR passes are not
    subclasses yet, nor are some ``ValueError`` and ``TypeError`` for a
    malformed compile option or controller limit.

    An error that found several problems at once lists them in
    ``problems``, and its message shows the first ten.
    """

    problems: tuple[Problem, ...] = ()

    def __init__(self, *args: object, problems: Iterable[Problem] = ()) -> None:
        super().__init__(*args)
        if problems:
            self.problems = tuple(problems)

    def __reduce__(self) -> tuple[Any, ...]:
        from ._pickling import error_reduce

        return error_reduce(self)

    @property
    def location(self) -> Problem | None:
        """The first problem that names a file, if any."""
        return next((p for p in self.problems if p.file is not None), None)

    def _header(self) -> str:
        return Exception.__str__(self)

    def __str__(self) -> str:
        text = self._header()
        if not self.problems:
            return text
        lines = [str(p) for p in self.problems[:_SHOWN]]
        if len(self.problems) > _SHOWN:
            lines.append(f"... and {len(self.problems) - _SHOWN} more")
        return text + ":\n  " + "\n  ".join(lines)


class DeviceDefinitionError(PulseError, ValueError):
    """A device's definition is wrong: ``pulse.device()`` lists every problem
    of its config or file in ``problems``, each with its path in the config
    (and, for a file, its line and column)."""

    def __init__(self, message: str = "", *, problems: Iterable[Problem] = ()) -> None:
        problems = tuple(problems)
        if not message:
            message = f"{_count(len(problems), 'problem')} in the device definition"
        super().__init__(message, problems=problems)


class InvalidValueError(PulseError, ValueError):
    """An argument, option or setting has the right type but a value that
    cannot be used: a negative duration, an empty list, a shape that does
    not match."""


class InvalidTypeError(PulseError, TypeError):
    """An argument, option or callback result has a type that cannot be
    used, such as a string where a number is needed."""


class NotFoundError(PulseError, KeyError):
    """A name has nothing behind it: a calibration value, a handler, a
    parameter or a file that is not there. The message lists what there is."""


class UnsupportedError(PulseError, NotImplementedError):
    """What was asked is not available yet, or not on this platform or
    Python version. It is a ``NotImplementedError``, and so also a
    ``RuntimeError``."""


class CompilationError(PulseError, RuntimeError):
    """A kernel cannot be traced or compiled: an unsupported construct, an
    operation given the wrong kind of argument, or invalid pulse IR. The
    message says what to write instead, and a note gives the kernel line."""


class TimingError(CompilationError):
    """A program cannot meet a timing it declares: a deadline for an answer,
    or a limit on its length. ``problems`` names each one that is missed."""


class ExecutionError(PulseError, RuntimeError):
    """Running a compiled program failed: a simulation, a classical call or
    a GPU launch."""


class InternalError(PulseError, RuntimeError):
    """A bug in cudaq-pulse: a check that a valid program never fails. The
    message asks for a report with the kernel that raised it."""

    def _header(self) -> str:
        from ._build_config import PACKAGE_VERSION

        return (
            f"{Exception.__str__(self)}; this is a bug in cudaq-pulse {PACKAGE_VERSION}; "
            f"please report it at {_ISSUES_URL} with the kernel that raised it"
        )


class _CallTimeoutError(ExecutionError, TimeoutError):
    """A classical call got no answer in time."""


class IndexOutOfRangeError(PulseError, IndexError):
    """An index past the end of what it indexes, such as a qudit index past
    the qudits of a device or register. The message says how many there are."""


class PulseWarning(UserWarning):
    """Base class of the warnings cudaq-pulse issues, so that
    ``warnings.filterwarnings("error", category=pulse.PulseWarning)`` turns
    them all into errors."""


class UpgradeNotice(PulseWarning):
    """Something still works but has a newer form, such as a file in an
    older format; the message names the newer form. Issued once per cause."""


class _SpecializationWarning(PulseWarning, RuntimeWarning):
    """A kernel call compiled a new specialization of the kernel."""


class _CoarseStepWarning(PulseWarning, RuntimeWarning):
    """A simulation's ``dt_ns`` is too coarse for the frequencies it holds."""


def _positive_int(name: str, value: Any) -> int:
    """*value*, a count such as ``shots``, as an int of at least 1; otherwise
    the error saying what it got."""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise InvalidTypeError(f"{name} must be a positive int, got {value!r}")
    if value < 1:
        raise InvalidValueError(f"{name} must be a positive int, got {value!r}")
    return int(value)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _closest(name: str, candidates: Iterable[str]) -> str | None:
    """The candidate closest to *name*, if one is close: the one
    ``difflib.get_close_matches`` picks."""
    # difflib rejects a candidate whose characters shared with *name*,
    # counted with repeats, are too few for its cutoff of 0.6 (its
    # quick_ratio): 10 * shared < 3 * (len(name) + len(candidate)). Those are
    # dropped before difflib, and most names go without loading it. A
    # candidate's characters that occur in *name* bound its shared count; one
    # translate finds them for every candidate, and only the candidates that
    # bound lets through are counted exactly.
    candidates = list(candidates)
    size = len(name)
    near = candidates
    whole = "\0".join(candidates)
    if whole.count("\0") == len(candidates) - 1:
        foreign = "".join(set(whole).difference(name, "\0"))
        kept = whole.translate(str.maketrans("", "", foreign)).split("\0")
        lengths = zip(candidates, map(len, kept), strict=False)
        near = [c for c, k in lengths if 10 * k >= 3 * (size + len(c))]
    chars = tuple(set(name))
    counts = tuple(map(name.count, chars))
    near = [
        c for c in near if 10 * sum(map(min, map(c.count, chars), counts)) >= 3 * (size + len(c))
    ]
    if not near:
        return None
    import difflib

    close = difflib.get_close_matches(name, near, n=1)
    return close[0] if close else None


def _ranges(numbers: Iterable[int]) -> str:
    """``"0..3, 5, 7..9"`` for the sorted distinct *numbers*."""
    parts: list[str] = []
    run: list[int] = []
    for n in sorted(set(numbers)):
        if run and n == run[-1] + 1:
            run.append(n)
            continue
        if run:
            parts.append(str(run[0]) if len(run) == 1 else f"{run[0]}..{run[-1]}")
        run = [n]
    if run:
        parts.append(str(run[0]) if len(run) == 1 else f"{run[0]}..{run[-1]}")
    return ", ".join(parts)


# What ``dataclass(frozen=True)`` sets on a class whose body does not.
_DATACLASS_NAMES = (
    "__dataclass_params__",
    "__dataclass_fields__",
    "__init__",
    "__repr__",
    "__eq__",
    "__hash__",
    "__setattr__",
    "__delattr__",
    "__match_args__",
) + (("__replace__",) if sys.version_info >= (3, 13) else ())
_DATACLASS_LOCK = threading.RLock()


class _WhenUsed(tuple[type, str, Any]):
    """``(cls, name, field)``: a name ``dataclass`` is yet to set, or to
    change, on *cls*. Reading it, from the class or from an instance, makes
    the class the dataclass, and reads the name it then has. *field* is the
    ``field()`` the class body gave the name, if it gave one."""

    __slots__ = ()

    def __get__(self, instance: object, owner: type | None = None) -> Any:
        cls, name, _ = self
        with _DATACLASS_LOCK:
            if cls.__dict__.get(name) is self:
                for waiting in [v for v in cls.__dict__.values() if v.__class__ is _WhenUsed]:
                    if waiting[2] is None:
                        delattr(cls, waiting[1])
                    else:
                        setattr(cls, waiting[1], waiting[2])
                dataclass(frozen=True)(cls)
        value = cls.__dict__.get(name, self)
        if value is self:
            return getattr(owner or cls, name) if instance is None else getattr(instance, name)
        get = getattr(value.__class__, "__get__", None)
        return value if get is None else get(value, instance, owner)


def _dataclass_when_used(*, frozen: bool) -> Callable[[type], type]:
    """``dataclass(frozen=True)``, applied when the class is first used, for
    a class most programs never use. Every name the dataclass sets, and every
    field given a ``field()``, reads as it would, from the class or an
    instance (made, or unpickled), and makes the class the dataclass first:
    ``dataclasses.fields``, ``is_dataclass``, ``inspect.signature``,
    pickling and subclassing see the dataclass. Annotate with ``dataclass``
    under ``TYPE_CHECKING``."""
    if not frozen:
        raise InternalError("only a frozen dataclass is made when used")

    def wrap(cls: type) -> type:
        namespace = cls.__dict__
        for name in _DATACLASS_NAMES:
            if name not in namespace:
                setattr(cls, name, _WhenUsed((cls, name, None)))
        for name in namespace.get("__annotations__", ()):
            value = namespace.get(name)
            if value.__class__ is Field:
                setattr(cls, name, _WhenUsed((cls, name, value)))
        return cls

    return wrap
