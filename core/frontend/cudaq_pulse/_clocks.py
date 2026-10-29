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
cycle, and may start operations only every ``cycles_per_timestep`` of them.
Its lines' timestep (their ``grid``) is the fewest whole program cycles
that are a whole number of those steps: ``b`` for ``r = a / b`` in lowest
terms and one cycle per step. A line's operations start and last whole
timesteps; a sync of lines meets on the least common multiple of theirs. A
controller's program counts in its own cycles: the compiler converts once,
when it exports.

Without a shared reference clock, two controllers' starts are each known to
within one of their own cycles, which budgets across controllers add.
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from ._errors import InvalidValueError
from .ops import RoundDirection, round_duration

if TYPE_CHECKING:
    from .controller import ControllerLimits

__all__ = ["RoundDirection", "round_duration"]

# Clocks are given in GHz as floats; their ratios are fractions with
# denominators up to this.
_MAX_DENOMINATOR = 10_000
_TOLERANCE = 1e-9


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
    # Its lines' timestep: program cycles between the instants it can start
    # an operation.
    grid: int = 0

    def __post_init__(self) -> None:
        if not self.grid:
            object.__setattr__(self, "grid", self.ratio.denominator)

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
    if limits is None:
        own, steps, samples = None, 1, None
    else:
        own, steps, samples = limits.clock_ghz, limits.cycles_per_timestep, limits.sample_rate_ghz
    if (
        reference_clock_ghz is None
        and steps == 1
        and (own is None or own == program_ghz)
        and (samples is None or _sample_step(samples, program_ghz) == 1)
    ):
        return None
    key = (own, program_ghz, reference_clock_ghz, steps, samples)
    found = _PLANS.get(key)
    if found is None:
        found = _PLANS[key] = _plan(own, program_ghz, reference_clock_ghz, steps, samples)
    return found


_PLANS: dict[tuple[float | None, float, float | None, int, float | None], ClockPlan] = {}


def _sample_step(sample_rate_ghz: float, program_ghz: float) -> int:
    """The program cycles of the shortest whole number of samples."""
    try:
        return made_for(sample_rate_ghz, program_ghz)[1]
    except ValueError:
        raise _definition_error(
            f"a controller's {_ghz(sample_rate_ghz)} sample rate has no whole-cycle ratio "
            f"to the {_ghz(program_ghz)} program clock"
        ) from None


def timestep(ratio: Fraction, cycles_per_timestep: int) -> int:
    """A timestep in program cycles: the fewest whole program cycles that
    are whole steps of *cycles_per_timestep* own cycles, at *ratio* own
    cycles per program cycle."""
    own = ratio.denominator * cycles_per_timestep
    return own // math.gcd(ratio.numerator, own)


def _plan(
    own: float | None,
    program_ghz: float,
    reference: float | None,
    steps: int,
    samples: float | None,
) -> ClockPlan:
    clock = program_ghz if own is None else own
    if reference is not None:
        _check_reference(reference, {"controller": clock})
    ratio = clock_ratio(clock, program_ghz)
    grid = timestep(ratio, steps)
    if samples is not None:
        grid = math.lcm(grid, _sample_step(samples, program_ghz))
    controller = ControllerClock("controller", clock, ratio, grid)
    return ClockPlan(program_ghz, (controller,), reference)


def sample_rate(limits: ControllerLimits, program_ghz: float) -> tuple[int, int] | None:
    """The samples a line of *limits*' controller plays per program cycle, as
    ``(n, d)``: ``n`` every ``d`` cycles; ``None`` when it is one."""
    if limits.sample_rate_ghz is not None:
        n, d = made_for(limits.sample_rate_ghz, program_ghz)
        return None if n == d else (n, d)
    if limits.clock_ghz is None or limits.clock_ghz == program_ghz:
        return None
    rate = clock_ratio(limits.clock_ghz, program_ghz)
    return None if rate == 1 else (rate.numerator, rate.denominator)


def made_for(sample_rate_ghz: Any, program_ghz: float) -> tuple[int, int]:
    """The ``(n, d)`` of samples made at *sample_rate_ghz*: ``n`` every ``d``
    program cycles. Raises ``ValueError`` when that is not a ratio of whole
    numbers."""
    if (
        isinstance(sample_rate_ghz, bool)
        or not isinstance(sample_rate_ghz, (int, float))
        or not math.isfinite(sample_rate_ghz)
        or not sample_rate_ghz > 0
    ):
        raise InvalidValueError(
            f"sample_rate_ghz must be a positive number, got {sample_rate_ghz!r}"
        )
    rate = Fraction(sample_rate_ghz / program_ghz).limit_denominator(_MAX_DENOMINATOR)
    if abs(float(rate) * program_ghz - sample_rate_ghz) > _TOLERANCE * sample_rate_ghz:
        raise InvalidValueError(
            f"sample_rate_ghz={sample_rate_ghz:g} has no whole-cycle ratio to the "
            f"{_ghz(program_ghz)} program clock"
        )
    return rate.numerator, rate.denominator


def line_delays(
    delays: Mapping[tuple[str, int], float],
    limits: ControllerLimits | None,
    plan: ClockPlan | None,
    program_ghz: float,
    n_qudits: int,
) -> tuple[str | None, dict[tuple[str, int], float]]:
    """The ``pulse.line_delays`` attribute that moves each delayed line by
    whole steps of its grid (``None`` when none moves), and each line's fine
    delay in ns, the rest. Raises ``ValueError`` for a rest its controller
    can neither add nor tolerate."""
    fine_max = 0.0 if limits is None else limits.max_fine_delay_ns
    tolerance = 0.0 if limits is None else limits.alignment_tolerance_cycles
    shifts: dict[tuple[str, int], int] = {}
    fine: dict[tuple[str, int], float] = {}
    for (role, qudit), ns in delays.items():
        controller = None if plan is None else plan.controller_of(role, qudit)
        grid = 1 if controller is None else controller.grid
        own_ghz = program_ghz if controller is None else controller.clock_ghz
        step_ns = grid / program_ghz
        steps = ns / step_ns
        whole = math.floor(steps + _TOLERANCE) if fine_max > 0 else round(steps)
        rest = ns - whole * step_ns
        if abs(rest) < _TOLERANCE:
            rest = 0.0
        if fine_max > 0 and rest > fine_max + _TOLERANCE:
            raise InvalidValueError(
                f"line_delays_ns: the {role} line of qudit {qudit} is delayed {ns:g} ns, "
                f"{rest:g} ns past a whole number of its {step_ns:g} ns timesteps, "
                f"and the controller adds at most {fine_max:g} ns of fine delay; raise its "
                f"max_fine_delay_ns to {rest:g}, or calibrate the delay to "
                f"{whole * step_ns:g} or {(whole + 1) * step_ns:g} ns"
            )
        if fine_max == 0 and abs(rest) * own_ghz > tolerance + _TOLERANCE:
            low = math.floor(steps + _TOLERANCE) * step_ns
            raise InvalidValueError(
                f"line_delays_ns: the {role} line of qudit {qudit} is delayed {ns:g} ns, "
                f"{abs(rest):g} ns ({abs(rest) * own_ghz:g} of its controller's cycles) from "
                f"a whole number of its {step_ns:g} ns timesteps, and the controller has no fine "
                f"delay; give it max_fine_delay_ns, raise its alignment_tolerance_cycles to "
                f"{abs(rest) * own_ghz:g}, or calibrate the delay to {low:g} or "
                f"{low + step_ns:g} ns"
            )
        fine[(role, qudit)] = rest if fine_max > 0 else 0.0
        if whole:
            shifts[(role, qudit)] = whole * grid
    if not shifts:
        return None, fine
    arrays = []
    for role in ("drive", "readout"):
        cycles = ", ".join(str(shifts.get((role, q), 0)) for q in range(n_qudits))
        arrays.append(f"{role} = array<i64: {cycles}>")
    return "{" + ", ".join(arrays) + "}", fine


def off_grid(
    plan: ClockPlan,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[dict[str, Any]],
    convert: bool = False,
    waveforms: bool = True,
    run_time: bool = False,
) -> list[str]:
    """A violation for each line event that does not start and last whole
    cycles of its controller; with *convert*, the events' times are then in
    their controller's own cycles, in place (and with *waveforms*, their
    waveforms' time parameters scaled), in the same pass. With *run_time*
    (the program computes values at run time), what the events compute at
    run time is converted too (``own_run_time``)."""
    found = []
    single = plan.controllers[0] if plan.owners is None else None
    controller = plan.controllers[0]
    a, b = controller.ratio.numerator, controller.ratio.denominator
    grid = controller.grid
    for event in events:
        # A submit's or an answer's "line" is its link.
        line = event["line"] if event["kind"] in _GRIDDED else None
        if single is None:
            controller = (
                plan.controllers[0]
                if line is None
                else plan.controller_of(lines[line]["role"], lines[line]["qudit"])
            )
            a, b = controller.ratio.numerator, controller.ratio.denominator
            grid = controller.grid
        start = event["start_cycles"]
        duration = event["duration_cycles"]
        if (
            grid > 1
            and line is not None
            and start is not None
            and duration is not None
            and (start % grid or duration % grid)
        ):
            found.append(_off_grid(controller, lines[line], line, event["kind"], start, duration))
        if convert and run_time and line is not None and (a != b or grid > 1):
            found.extend(own_run_time(controller, lines[line], event, waveforms))
        if convert and a != b:
            if start is not None:
                event["start_cycles"] = -((-start * a) // b)
            if duration is not None:
                event["duration_cycles"] = -((-duration * a) // b)
            end = event["end_cycles"]
            if end is not None:
                event["end_cycles"] = -((-end * a) // b)
            tree = event.get("waveform") if waveforms else None
            if tree is not None:
                _scale(tree, a / b)
    return found


_GRIDDED = frozenset({"drive", "readout", "wait"})


def own_run_time(
    controller: ControllerClock,
    line: Mapping[str, Any],
    event: dict[str, Any],
    waveforms: bool = True,
) -> list[str]:
    """Convert what a line event computes at run time to *controller*'s
    cycles, in place, by the rule for its fixed times: a variable wait's
    length, clamped to its bounds as the run clamps it, and its bounds; and
    (with *waveforms*) its waveform's operands computed at run time. A
    length becomes a ``pulse.own_cycles`` expression, which the controller
    fails at run time when the length is off the line's timesteps. Returns
    the violations: a length that is off them for a value it can be known
    to take now, or an operand the export cannot tell is a time."""
    a, b = controller.ratio.numerator, controller.ratio.denominator
    grid = controller.grid
    found: list[str] = []
    length = event.get("length")
    if length is not None:
        low = event.get("min_cycles")
        high = event.get("max_cycles")
        if low is not None:
            length = {"op": "arith.maxsi", "of": [length, {"const": low}]}
            event["min_cycles"] = low * a // b
        if high is not None:
            length = {"op": "arith.minsi", "of": [length, {"const": high}]}
            event["max_cycles"] = -((-high * a) // b)
        event["length"] = _own_length(length, a, b, grid)
        found.extend(_decided(controller, line, event, "length", length))
    tree = event.get("waveform")
    if tree is not None:
        _own_operands(controller, line, event, tree, a, b, waveforms, found)
    return found


def _own_length(cycles: Any, a: int, b: int, grid: int) -> dict[str, Any]:
    return {"op": "pulse.own_cycles", "of": [cycles], "num": a, "den": b, "timestep": grid}


def _own_operands(
    controller: ControllerClock,
    line: Mapping[str, Any],
    event: Mapping[str, Any],
    node: dict[str, Any],
    a: int,
    b: int,
    convert: bool,
    found: list[str],
) -> None:
    computed = node.get("runtime")
    if computed:
        shape = node["shape"]
        times = _TIMES_OF.get(shape, ())
        named = "params" in node
        for key, value in computed.items():
            if not named:
                if a != b:
                    found.append(_unconvertible(controller, line, event, node, key))
                continue
            if key == "duration":
                found.extend(_decided(controller, line, event, "waveform's duration", value))
                if convert:
                    computed[key] = _own_length(value, a, b, controller.grid)
            elif key in times and a != b and convert:
                factor = a / b
                computed[key] = {
                    "op": "arith.mulf",
                    "of": [value, {"const": int(factor) if factor.is_integer() else factor}],
                }
    for child in node.get("of", ()):
        _own_operands(controller, line, event, child, a, b, convert, found)


def _where(controller: ControllerClock, line: Mapping[str, Any], event: Mapping[str, Any]) -> str:
    return (
        f"line {line_name(line['role'], line['qudit'])}, on the {controller.name} at "
        f"{_ghz(controller.clock_ghz)}: the {event['kind']} that is event {event['index']}"
    )


def _decided(
    controller: ControllerClock,
    line: Mapping[str, Any],
    event: Mapping[str, Any],
    what: str,
    cycles: Any,
) -> list[str]:
    """The violation of a run-time length that can take a value off the
    line's timesteps, when the values it can take are known now."""
    grid = controller.grid
    values = _values(cycles)
    if grid == 1 or values is None:
        return []
    off = sorted(v for v in values if v % grid)
    if not off:
        return []
    shown = ", ".join(f"{v:g}" for v in off)
    return [
        f"{_where(controller, line, event)} has a {what}, computed at run time, of "
        f"{shown} cycles for some outcomes, which isn't a whole number of the line's "
        f"timesteps of {grid} cycles"
    ]


def _unconvertible(
    controller: ControllerClock,
    line: Mapping[str, Any],
    event: Mapping[str, Any],
    node: Mapping[str, Any],
    key: str,
) -> str:
    shape = node.get("name", node["shape"])
    return (
        f"{_where(controller, line, event)} plays a {shape!r} waveform whose operand {key} "
        "is computed at run time, and the export can't tell whether that operand is a time "
        f"to convert to the controller's cycles ({controller.ratio} of a program cycle "
        "each); compute it while the kernel compiles, or give it a waveform shape whose "
        "parameters are named"
    )


# The results of the ops a set of values passes through unchanged in kind.
_VALUE_OPS: Mapping[str, Any] = {
    "arith.addf": lambda x, y: x + y,
    "arith.addi": lambda x, y: x + y,
    "arith.subf": lambda x, y: x - y,
    "arith.subi": lambda x, y: x - y,
    "arith.mulf": lambda x, y: x * y,
    "arith.muli": lambda x, y: x * y,
    "arith.maxsi": max,
    "arith.minsi": min,
    "arith.fptosi": lambda x: int(x),
    "arith.sitofp": float,
    "pulse.duration_from_int": lambda x: x,
}


def _values(expression: Any, limit: int = 64) -> set[float] | None:
    """The values *expression* can take, when it computes from constants and
    selects between them alone and they are few; else None."""
    if isinstance(expression, (int, float)):
        return {expression}
    if not isinstance(expression, dict):
        return None
    if "const" in expression:
        return {expression["const"]}
    op: str = expression.get("op", "")
    of = expression.get("of", ())
    if op == "arith.select" and len(of) == 3:
        then, other = _values(of[1], limit), _values(of[2], limit)
        if then is None or other is None:
            return None
        return then | other
    fn = _VALUE_OPS.get(op)
    if fn is None:
        return None
    sets = [_values(x, limit) for x in of]
    if any(s is None for s in sets):
        return None
    out: set[float] = set()
    for args in itertools.product(*sets):  # type: ignore[arg-type]
        out.add(fn(*args))
        if len(out) > limit:
            return None
    return out


def off_grid_messages(
    controller: ControllerClock,
    lines: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    off: Sequence[tuple[int, int, int]],
) -> list[str]:
    """The violations of the line events the native walk found off
    *controller*'s grid, each (index, start, duration) in program cycles."""
    found = []
    for index, start, duration in off:
        event = events[index]
        line = event["line"]
        found.append(_off_grid(controller, lines[line], line, event["kind"], start, duration))
    return found


def _off_grid(
    controller: ControllerClock,
    line: Mapping[str, Any],
    index: int,
    kind: str,
    start: int,
    duration: int,
) -> str:
    return off_timestep(
        line_name(line["role"], line["qudit"]), controller.grid, kind, start, duration
    )


def line_name(role: str, qudit: int) -> str:
    """``d3`` for the drive line of qudit 3, ``r3`` for its readout line."""
    return f"{role[0]}{qudit}"


def off_timestep(name: str, grid: int, kind: str, start: int, duration: int) -> str:
    """The violation of a *kind* on line *name* that does not start and last
    whole timesteps of *grid* program cycles."""
    problems = []
    if duration % grid:
        problems.append(
            f"{duration} cycles isn't a whole number of timesteps (the {kind} at cycle "
            f"{start}; pulse.round_duration(line, {duration}) gives "
            f"{-(-duration // grid) * grid}, or {duration // grid * grid} rounding down)"
        )
    if start % grid:
        problems.append(
            f"the {kind} starts at cycle {start}, which isn't a whole number of timesteps "
            f"(the nearest starts are {start // grid * grid} and {-(-start // grid) * grid})"
        )
    return f"line {name} moves in timesteps of {grid} cycles; " + "; ".join(problems)


# The times of each kind of export record, in cycles; lower bounds round down.
EVENT_TIMES = ("start_cycles", "duration_cycles", "end_cycles")
# A run-time wait's bounds, which only some events have.
EVENT_BOUNDS = ("max_cycles",)
EVENT_LOWER_BOUNDS = ("min_cycles",)
EMULATION_EVENT_TIMES = ("start_cycles", "duration_cycles", "epoch_start_cycles")
EPOCH_TIMES = ("start_cycles", "max_shift_cycles")
EPOCH_LOWER_BOUNDS = ("min_shift_cycles",)
CALL_TIMES = (
    "submit_cycles",
    "submit_duration_cycles",
    "due_cycles",
    "deadline_cycles",
    "exposed_cycles",
    "first_use_cycles",
    "taken_cycles",
)


def to_own(
    records: Sequence[dict[str, Any]],
    keys: Sequence[str],
    controller: ControllerClock,
    lower_bounds: Sequence[str] = (),
) -> None:
    """Convert the *keys* times of *records* (each has every key) to
    *controller*'s own cycles, in place, rounded up (line times are on its
    grid, so exact), and the *lower_bounds* rounded down."""
    a, b = controller.ratio.numerator, controller.ratio.denominator
    for record in records:
        for key in keys:
            value = record[key]
            if value is not None:
                record[key] = -((-value * a) // b)
        for key in lower_bounds:
            value = record[key]
            if value is not None:
                record[key] = value * a // b


def _scale(node: dict[str, Any], factor: float) -> None:
    params = node.get("params")
    if params is not None:
        for key in _TIMES_OF.get(node["shape"], ()):
            value = params[key]
            if value is not None:
                params[key] = value * factor
    children = node.get("of")
    if children:
        for child in children:
            _scale(child, factor)


# The time parameters of each waveform shape, as ``controller._PARAMETERS``
# names them.
_TIMES_OF = {
    "square": ("duration",),
    "gaussian": ("duration", "sigma"),
    "gaussian_square": ("duration", "sigma", "risefall"),
    "drag": ("duration", "sigma", "beta"),
    "cosine": ("duration",),
    "tanh_ramp": ("duration", "sigma"),
    "custom": ("duration",),
}


def clock_keys(plan: ClockPlan, controller: ControllerClock) -> dict[str, float]:
    """The clock entries of a controller's export beside ``clock_ghz``."""
    keys: dict[str, float] = {}
    if controller.ratio != 1:
        keys["program_clock_ghz"] = plan.program_ghz
    if plan.reference_clock_ghz is not None:
        keys["reference_clock_ghz"] = plan.reference_clock_ghz
    return keys


def _round(n: float, grid: int, direction: RoundDirection) -> int:
    if direction is RoundDirection.UP:
        return -(-math.ceil(n - _TOLERANCE) // grid) * grid
    if direction is RoundDirection.DOWN:
        return math.floor(n + _TOLERANCE) // grid * grid
    return math.floor(n / grid + 0.5) * grid


def round_on_grid(n: Any, grid: int, direction: RoundDirection | str) -> int:
    """``round_duration`` once its line's timestep is known."""
    try:
        mode = RoundDirection(direction)
    except ValueError:
        raise InvalidValueError(
            "round_duration() direction must be one of "
            f"{', '.join(m.value for m in RoundDirection)}, got {direction!r}"
        ) from None
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0:
        raise InvalidValueError(
            f"round_duration() takes a finite, non-negative number of cycles, got {n!r}"
        )
    return _round(float(n), grid, mode)
