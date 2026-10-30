# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Closed loops between shots: the parameters a program runs with, and the
log of what was estimated.

A loop compiles its kernel once and runs it with the store's current
values each iteration, without compiling again. Amplitudes, phases and
frequencies can be float parameters of the kernel, passed each iteration. A
quantity that shapes the timeline, such as a wait, is read at run time
instead: ``store.reader("wait_ns")`` is a classical function that answers
with the store's current value (on hardware, a read of a controller
register), and a variable wait takes its length from the answer::

    store = ParameterStore(amplitude=0.157, wait_ns=2000.0)
    read_wait = store.reader("wait_ns", bounds=(0.0, 40000.0))

    @pulse.kernel
    def probe(q, amplitude: float):
        ...
        wait(d, read_wait.submit().result(), bounds=(0, 80000))

    log = EstimateLog()
    compiled = pulse.compile(probe, device=dev)
    for i in range(10_000):
        lowered = compiled.lower(q, store["amplitude"])
        outcome = ...  # run or simulate it
        estimate = update(...)  # the estimator, outside the kernel
        store.set("amplitude", estimate, iteration=i)
        log.record(i, "amplitude", estimate, uncertainty=..., truth=...)

The estimators of ``cudaq_pulse.classical.calibration`` serve both inside
kernels (as the bodies of classical functions) and in such loops.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from .._errors import InvalidValueError, NotFoundError
from . import ClassicalFunction, Float

if TYPE_CHECKING:
    from .._units import Time

__all__ = ["Estimate", "EstimateLog", "ParameterStore", "Update"]


@dataclass(frozen=True)
class Update:
    """One change of a parameter: its new value, the store's version after
    it, and the loop iteration that made it."""

    name: str
    value: float
    version: int
    iteration: int | None


class ParameterStore:
    """Named calibration parameters, versioned on every change.

    ``store[name]`` reads one; ``values(*names)`` reads several in order, for
    a kernel's float arguments; ``set`` and ``update`` change them and bump
    ``version``; ``history`` lists every change.
    """

    def __init__(self, values: Mapping[str, float] | None = None, /, **named: float) -> None:
        self._values: dict[str, float] = {}
        self.version = 0
        self.history: list[Update] = []
        for name, value in {**(values or {}), **named}.items():
            self.set(name, value)

    def __getitem__(self, name: str) -> float:
        try:
            return self._values[name]
        except KeyError:
            known = ", ".join(sorted(self._values)) or "none"
            raise NotFoundError(f"no parameter {name!r} (the store has {known})") from None

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def __len__(self) -> int:
        return len(self._values)

    def values(self, *names: str) -> tuple[float, ...]:
        return tuple(self[name] for name in names)

    def set(self, name: str, value: float, *, iteration: int | None = None) -> None:
        value = float(value)
        if not math.isfinite(value):
            raise InvalidValueError(f"{name} must be finite, got {value}")
        self._values[name] = value
        self.version += 1
        self.history.append(Update(name, value, self.version, iteration))

    def update(self, values: Mapping[str, float], *, iteration: int | None = None) -> None:
        for name, value in values.items():
            self.set(name, value, iteration=iteration)

    def reader(
        self,
        name: str,
        *,
        link: int | str = 0,
        max_latency_ns: float | Time = 0.0,
        bounds: tuple[float, float] | None = None,
    ) -> ClassicalFunction:
        """A classical function of no arguments that answers with the current
        value of *name* (in clock cycles if it is a wait's length), so a
        program compiled once reads each iteration's value at run time. With
        *bounds* the value is clamped to them. On hardware it is a read of a
        controller register named like the function, ``read_<name>``."""
        self[name]  # it must exist

        def read() -> float:
            value = self[name]
            if bounds is not None:
                value = min(max(value, bounds[0]), bounds[1])
            return value

        return ClassicalFunction(
            read,
            name=f"read_{name}",
            args=(),
            results=(Float(),),
            max_latency_ns=max_latency_ns,
            link=link,
        )

    def snapshot(self) -> dict[str, float]:
        """The current values (a copy)."""
        return dict(self._values)

    def __repr__(self) -> str:
        return f"ParameterStore({self._values!r}, version={self.version})"


@dataclass(frozen=True)
class Estimate:
    """One estimate of a quantity, with what was true if known."""

    iteration: int
    name: str
    value: float
    uncertainty: float | None = None
    truth: float | None = None
    time_s: float | None = None


class EstimateLog:
    """Every estimate a loop made, by quantity."""

    def __init__(self) -> None:
        self.entries: list[Estimate] = []

    def record(
        self,
        iteration: int,
        name: str,
        value: float,
        *,
        uncertainty: float | None = None,
        truth: float | None = None,
        time_s: float | None = None,
    ) -> None:
        self.entries.append(Estimate(iteration, name, float(value), uncertainty, truth, time_s))

    def __len__(self) -> int:
        return len(self.entries)

    def series(self, name: str) -> list[Estimate]:
        return [e for e in self.entries if e.name == name]

    def tracking_error(self, name: str) -> float:
        """The root-mean-square of estimate minus truth over the estimates of
        *name* that have a truth."""
        errors = [e.value - e.truth for e in self.series(name) if e.truth is not None]
        if not errors:
            raise InvalidValueError(f"no estimates of {name!r} with a truth")
        return math.sqrt(sum(x * x for x in errors) / len(errors))

    def to_json(self) -> str:
        return json.dumps([asdict(e) for e in self.entries])

    @classmethod
    def from_json(cls, text: str) -> EstimateLog:
        log = cls()
        log.entries = [Estimate(**e) for e in json.loads(text)]
        return log

    def extend(self, entries: Iterable[Estimate]) -> None:
        self.entries.extend(entries)

    def __repr__(self) -> str:
        names = sorted({e.name for e in self.entries})
        return f"<EstimateLog of {len(self.entries)} estimates of {', '.join(names) or 'nothing'}>"
