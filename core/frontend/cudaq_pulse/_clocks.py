# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Clocks: the program's, each controller's, and the grids between them.

A kernel counts time in cycles of the device's *program clock*,
``Device.clock_ghz``. A controller may run its own clock: at ``f_k`` against
the program's ``f_p`` it counts ``r = f_k / f_p`` of its cycles per program
cycle, and starts an operation only every ``grid`` program cycles, where
``r = a / grid`` in lowest terms. A line's operations start and last whole
multiples of its controller's grid; a sync of lines meets on the least
common multiple of theirs. A controller's program counts in its own cycles:
the compiler converts once, when it exports.

Without a shared reference clock, two controllers' starts are each known to
within one of their own cycles, which budgets across controllers add.
"""

from __future__ import annotations

import enum
import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from ._errors import InvalidValueError

if TYPE_CHECKING:
    from .controller import ControllerLimits

__all__ = ["Rounding", "cycles_on"]

# Clocks are given in GHz as floats; their ratios are fractions with
# denominators up to this.
_MAX_DENOMINATOR = 10_000
_TOLERANCE = 1e-9


class Rounding(enum.StrEnum):
    """How ``cycles_on`` rounds a length onto its line's grid."""

    UP = "up"
    DOWN = "down"
    NEAREST = "nearest"


def _definition_error(message: str) -> Exception:
    """The error for a device whose clocks do not fit together."""
    return ValueError(message)


def _ghz(value: float) -> str:
    return f"{value:g} GHz"


def clock_ratio(clock_ghz: float, program_ghz: float) -> Fraction:
    """``clock_ghz / program_ghz`` as a fraction; an error if it is not one."""
    ratio = Fraction(clock_ghz / program_ghz).limit_denominator(_MAX_DENOMINATOR)
    if abs(float(ratio) * program_ghz - clock_ghz) > _TOLERANCE * clock_ghz:
        raise _definition_error(
            f"a controller's {_ghz(clock_ghz)} clock has no whole-cycle ratio to the "
            f"{_ghz(program_ghz)} program clock; give the device a 'clock_ghz' both "
            "divide into"
        )
    return ratio


def _fraction(ghz: float) -> Fraction:
    return Fraction(ghz).limit_denominator(1_000_000)


def _lcm(clocks: Sequence[float]) -> float:
    """The slowest clock every one of *clocks* divides into."""
    values = [_fraction(c) for c in clocks]
    numerator = functools.reduce(math.lcm, (v.numerator for v in values))
    denominator = functools.reduce(math.gcd, (v.denominator for v in values))
    return numerator / denominator


def program_clock_ghz(
    declared: float | None, controller_clocks: Mapping[str, float], default: float
) -> float:
    """The program clock of a device: *declared* (its config's ``clock_ghz``)
    if given, else the fastest of its controllers' clocks when every other
    one divides it, else *default* when it has no controllers.

    Raises the device-definition error when the controllers' clocks do not
    divide evenly into one and none is declared, listing the clocks that
    work; it never picks one.
    """
    if declared is not None:
        if isinstance(declared, bool) or not isinstance(declared, (int, float)):
            raise _definition_error(f"the device's clock_ghz must be a number, got {declared!r}")
        if not declared > 0 or not math.isfinite(declared):
            raise _definition_error(f"the device's clock_ghz must be positive, got {declared}")
        for clock in controller_clocks.values():
            clock_ratio(clock, declared)
        return float(declared)
    if not controller_clocks:
        return float(default)
    fastest = max(controller_clocks.values())
    if all(clock_ratio(c, fastest).numerator == 1 for c in controller_clocks.values()):
        return float(fastest)
    common = _lcm(list(controller_clocks.values()))
    options = [f"{_ghz(common)}, which every controller's clock divides"]
    for name, clock in sorted(controller_clocks.items(), key=lambda item: -item[1]):
        grids = ", ".join(
            f"{other} every {clock_ratio(c, clock).denominator}"
            for other, c in sorted(controller_clocks.items())
            if clock_ratio(c, clock).denominator > 1
        )
        options.append(f"{_ghz(clock)} ({name}'s; then {grids} program cycles)")
    clocks = ", ".join(f"{name} at {_ghz(c)}" for name, c in sorted(controller_clocks.items()))
    raise _definition_error(
        f"the controllers' clocks ({clocks}) do not divide evenly into one, so the "
        "device needs a program clock: set 'clock_ghz' in its config to one of "
        + "; ".join(options)
    )


@dataclass(frozen=True, slots=True)
class ControllerClock:
    """One controller's clock against the program clock."""

    name: str
    clock_ghz: float
    # Own cycles per program cycle.
    ratio: Fraction

    @property
    def grid(self) -> int:
        """Program cycles between the instants it can start an operation."""
        return self.ratio.denominator

    def own(self, cycles: int) -> int:
        """*cycles* program cycles in its own, rounded up."""
        return -((-cycles * self.ratio.numerator) // self.ratio.denominator)


@dataclass(frozen=True, slots=True)
class ClockPlan:
    """The controllers of a compiled program and the lines each one plays."""

    program_ghz: float
    controllers: tuple[ControllerClock, ...]
    reference_clock_ghz: float | None = None
    # (role, qudit) -> index into controllers; absent: the first.
    owners: Mapping[tuple[str, int], int] | None = None

    def controller_of(self, role: str, qudit: int) -> ControllerClock:
        if self.owners is None:
            return self.controllers[0]
        return self.controllers[self.owners.get((role, qudit), 0)]

    def grid_of(self, role: str, qudit: int) -> int:
        return self.controller_of(role, qudit).grid

    @property
    def lcm(self) -> int:
        """Where every line can meet."""
        return functools.reduce(math.lcm, (c.grid for c in self.controllers), 1)

    def attributes(self, n_qudits: int) -> str | None:
        """The module attributes that put the scheduler on the grids."""
        if self.lcm == 1:
            return None
        if self.owners is None:
            return f"{{pulse.line_grid = {self.controllers[0].grid} : i64}}"
        arrays = []
        for role in ("drive", "readout"):
            grids = ", ".join(str(self.grid_of(role, q)) for q in range(n_qudits))
            arrays.append(f"pulse.{role}_line_grid = array<i64: {grids}>")
        return "{pulse.line_grid = 1 : i64, " + ", ".join(arrays) + "}"

    def start_uncertainty_ns(
        self, skew_cycles: Mapping[str, int] | None = None
    ) -> dict[str, float]:
        """How well each controller's start is known, in ns: its skew, plus one
        of its own cycles when no reference clock is shared."""
        extra = 0 if self.reference_clock_ghz is not None else 1
        skew = skew_cycles or {}
        return {c.name: (skew.get(c.name, 0) + extra) / c.clock_ghz for c in self.controllers}

    def budget_uncertainty_ns(
        self, a: str, b: str, skew_cycles: Mapping[str, int] | None = None
    ) -> float:
        """The start uncertainty a budget between controllers *a* and *b* adds."""
        if a == b:
            return 0.0
        each = self.start_uncertainty_ns(skew_cycles)
        return each[a] + each[b]


def _check_reference(reference_ghz: float, clocks: Mapping[str, float]) -> None:
    for name, clock in sorted(clocks.items()):
        multiple = clock / reference_ghz
        if abs(multiple - round(multiple)) > _TOLERANCE * multiple or round(multiple) < 1:
            raise _definition_error(
                f"the {name}'s {_ghz(clock)} clock is not a whole multiple of the "
                f"{_ghz(reference_ghz)} reference clock"
            )


def plan(
    limits: ControllerLimits | None, program_ghz: float, reference_clock_ghz: float | None = None
) -> ClockPlan | None:
    """The clocks of a program for *limits*' controller, or ``None`` when every
    controller runs at the program clock and no reference clock is given."""
    own = None if limits is None else limits.clock_ghz
    if reference_clock_ghz is None and (own is None or own == program_ghz):
        return None
    key = (own, program_ghz, reference_clock_ghz)
    found = _PLANS.get(key)
    if found is None:
        found = _PLANS[key] = _plan(own, program_ghz, reference_clock_ghz)
    return found


_PLANS: dict[tuple[float | None, float, float | None], ClockPlan] = {}


def _plan(own: float | None, program_ghz: float, reference: float | None) -> ClockPlan:
    clock = program_ghz if own is None else own
    if reference is not None:
        _check_reference(reference, {"controller": clock})
    controller = ControllerClock("controller", clock, clock_ratio(clock, program_ghz))
    return ClockPlan(program_ghz, (controller,), reference)


def off_grid(
    plan: ClockPlan, lines: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> list[str]:
    """A violation for each line event that does not start and last whole
    cycles of its controller."""
    found = []
    for event in events:
        line = event.get("line")
        if line is None or event["kind"] not in ("drive", "readout", "wait"):
            continue
        start, duration = event["start_cycles"], event["duration_cycles"]
        if start is None or duration is None:
            continue
        role, qudit = lines[line]["role"], lines[line]["qudit"]
        controller = plan.controller_of(role, qudit)
        grid = controller.grid
        if start % grid == 0 and duration % grid == 0:
            continue
        where = (
            f"the {role} line of qudit {qudit} (line {line}), on the {controller.name} at "
            f"{_ghz(controller.clock_ghz)}, which starts and times operations every {grid} "
            "program cycles"
        )
        problems = []
        if duration % grid:
            problems.append(
                f"lasts {duration} cycles; the nearest valid lengths are "
                f"{duration // grid * grid} and {-(-duration // grid) * grid} "
                f"(pulse.cycles_on(line, {duration}) rounds onto the grid)"
            )
        if start % grid:
            problems.append(
                "starts off the grid; the nearest valid starts are "
                f"{start // grid * grid} and {-(-start // grid) * grid}"
            )
        found.append(f"{where}: the {event['kind']} at {start} " + ", and ".join(problems))
    return found


_TIME_PARAMETERS = frozenset({"duration", "sigma", "beta", "risefall"})


def _scaled_waveform(node: Mapping[str, Any], ratio: Fraction) -> dict[str, Any]:
    out = dict(node)
    params = node.get("params")
    if params is not None:
        factor = float(ratio)
        out["params"] = {
            k: (v * factor if k in _TIME_PARAMETERS and v is not None else v)
            for k, v in params.items()
        }
    if "of" in node:
        out["of"] = [_scaled_waveform(child, ratio) for child in node["of"]]
    return out


def own_cycles(data: Any, controller: ControllerClock) -> Any:
    """*data* with every ``*_cycles`` time in *controller*'s own cycles,
    rounded up (line times are on its grid, so exact), and every waveform's
    time parameters scaled."""
    if isinstance(data, list):
        return [own_cycles(item, controller) for item in data]
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        if key.endswith("_cycles") and isinstance(value, int) and not isinstance(value, bool):
            out[key] = controller.own(value)
        elif key == "waveform" and isinstance(value, dict):
            out[key] = _scaled_waveform(value, controller.ratio)
        elif isinstance(value, (dict, list)):
            out[key] = own_cycles(value, controller)
        else:
            out[key] = value
    return out


def clock_keys(plan: ClockPlan, controller: ControllerClock) -> dict[str, float]:
    """The clock entries of a controller's export beside ``clock_ghz``."""
    keys: dict[str, float] = {}
    if controller.ratio != 1:
        keys["program_clock_ghz"] = plan.program_ghz
    if plan.reference_clock_ghz is not None:
        keys["reference_clock_ghz"] = plan.reference_clock_ghz
    return keys


def _round(n: float, grid: int, rounding: Rounding) -> int:
    if rounding is Rounding.UP:
        return -(-math.ceil(n - _TOLERANCE) // grid) * grid
    if rounding is Rounding.DOWN:
        return math.floor(n + _TOLERANCE) // grid * grid
    return math.floor(n / grid + 0.5) * grid


def cycles_on(line: Any, n: float, rounding: Rounding | str = Rounding.UP) -> int:
    """*n* program cycles rounded onto *line*'s grid: the nearest length (or
    start) its controller can time, ``up`` (default), ``down`` or ``nearest``.

    Call it in a kernel with a line from ``get_drive_line`` or
    ``get_readout_line``; *n* must be known while the kernel compiles. A line
    whose controller runs at the program clock takes every whole cycle, so
    ``cycles_on(line, n)`` is ``ceil(n)``.
    """
    raise RuntimeError("cycles_on() must be called inside a @cudaq_pulse.kernel function")


def round_on_grid(n: Any, grid: int, rounding: Rounding | str) -> int:
    """``cycles_on`` once its line's grid is known."""
    try:
        mode = Rounding(rounding)
    except ValueError:
        raise InvalidValueError(
            f"cycles_on() rounding must be one of {', '.join(m.value for m in Rounding)}, "
            f"got {rounding!r}"
        ) from None
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
        raise InvalidValueError(
            f"cycles_on() takes a finite, non-negative number of cycles, got {n!r}"
        )
    return _round(float(n), grid, mode)
