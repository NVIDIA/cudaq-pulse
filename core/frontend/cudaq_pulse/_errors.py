# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The errors cudaq-pulse raises, and the problems they carry."""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, ClassVar, Final

__all__ = [
    "CompilationError",
    "DeviceDefinitionError",
    "ExecutionError",
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


class PulseWarning(UserWarning):
    """Base class of the warnings cudaq-pulse issues, so that
    ``warnings.filterwarnings("error", category=pulse.PulseWarning)`` turns
    them all into errors."""


class UpgradeNotice(PulseWarning):
    """Something still works but has a newer form, such as a file in an
    older format; the message names the newer form. Issued once per cause."""


class _SpecializationWarning(PulseWarning, RuntimeWarning):
    """A kernel call compiled a new specialization of the kernel."""


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _closest(name: str, candidates: Iterable[str]) -> str | None:
    """The candidate closest to *name*, if one is close."""
    close = difflib.get_close_matches(name, list(candidates), n=1)
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
