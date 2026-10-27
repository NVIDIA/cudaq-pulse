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

__all__ = ["DeviceDefinitionError", "Problem", "PulseError"]

# How many problems an error's message lists; ``problems`` holds them all.
_SHOWN = 10


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

    Each subclass also keeps its built-in base, so ``except RuntimeError``
    or ``except LookupError`` still catches what it caught before. Errors
    that the native extension raises while running MLIR passes, and
    ``ValueError`` or ``TypeError`` for a malformed argument, are not
    subclasses.

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

    def __str__(self) -> str:
        text = Exception.__str__(self)
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
