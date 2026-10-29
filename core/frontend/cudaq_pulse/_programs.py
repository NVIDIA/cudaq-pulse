# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The modules a one-controller program of float parameters never makes:
modules with list or qudit parameters (``_Placed``, bound per call), modules
on several clocks or with line delays (``_ClockedLowered``), and modules whose
controller limits each call's values are checked against (``_CheckedPerCall``).
``compiler`` imports it when it makes one of them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .compiler import Lowered, _bind_checked

if TYPE_CHECKING:
    from array import array

    from . import _clocks
    from ._device import Device
    from .compiler import CompileOptions

# Placements whose qudits a role remembers.
_PLACEMENTS = 4096
# Placed modules a role keeps for simulations: each is a copy of the module.
_PLACED_MODULES = 16

# Values whose controller limits were checked, kept per module.
_ACCEPTED_CALLS = 4096
_PACKERS: dict[int, Callable[..., bytes]] = {}


def _bits(values: tuple[float, ...]) -> bytes:
    """*values* as a key, by their bits, so that -0.0 and 0.0, or two NaNs,
    are told apart."""
    try:
        return _PACKERS[len(values)](*values)
    except KeyError:
        import struct

        pack = _PACKERS[len(values)] = struct.Struct(f"{len(values)}d").pack
        return pack(*values)


def _concrete_general(self: Lowered) -> Any:
    """``Lowered._concrete`` of a module with clocks or runtime qudits: the
    module with the values and carriers substituted, shared by every view
    with the same ones, or the module itself without them."""
    if not self.arguments and not self.carriers:
        return self._module
    if self._concrete_module is None:
        arguments: Sequence[float] = self.arguments
        for v in arguments:
            if v.__class__ is not float and v.__class__ is not int or v - v != 0.0:
                arguments = self._checked_arguments()
                break
        args = _schedule_args(self.options, self.clocks)
        with self._lock:
            if self.carriers:
                module = self._module.specialize_placed(
                    list(arguments),
                    [],
                    *args,
                    list(self.carriers),
                    _frequency_table(self.target),
                    self.target.fingerprint,
                )
            else:
                module = self._module.specialize(list(arguments), [], *args)
        self._concrete_module = module
    return self._concrete_module


class _ClockedLowered(Lowered):
    """A module with a clock plan or line delays. Its own class keeps them off
    every other module and every call's view, which stay the size they are
    on one clock."""

    __slots__ = ("clocks", "fine_delays")
    _concrete = _concrete_general


class _Role:
    """What the calls of a module with list or qudit parameters share: its
    symbols, which slots and static qudits get a line, and a content hash of
    the module without any call's values (``program_id``).

    ``symbols`` holds ``(name, kind, parameter, position)`` for each float
    argument of the module, in order (kind ``"float"``, or ``"vector"`` for a
    list element), then for each slot (kind ``"slot"``); *position* is the
    element's or qudit's index in its parameter, or ``None``.
    """

    __slots__ = (
        "symbols",
        "slots",
        "lines",
        "used",
        "every",
        "program_id",
        "dependencies",
        "placed",
        "views",
        "modules",
        "export_plan",
        "structured",
    )

    def __init__(self, symbols: tuple[tuple[str, str, str, int | None], ...]) -> None:
        self.symbols = symbols
        self.slots = 0
        self.lines: tuple[int, ...] = ()
        self.used: tuple[int, ...] = ()
        self.every: tuple[int, ...] = ()
        self.program_id = ""
        # The symbols each exported event's fields depend on, once exported.
        self.dependencies: list[dict[str, list[str]]] | None = None
        # The qudits a call uses, by its carriers (at most _PLACEMENTS).
        self.placed: dict[tuple[int, ...], tuple[int, ...]] = {}
        # A call without float values is its carriers and device: its view,
        # shared by the calls that repeat them (at most _PLACEMENTS).
        self.views: dict[tuple[tuple[int, ...], Device], Lowered] = {}
        # The module with a placement's carriers bound and its float
        # arguments kept, for simulations (the latest _PLACED_MODULES).
        self.modules: dict[tuple[tuple[int, ...], Device], Any] = {}
        # What every export shares, built by the first (controller._export_plan).
        self.export_plan: Any = None
        # The format /2 export's: each event's kind, trips and symbols.
        self.structured: list[tuple[str, int, dict[str, list[str]]]] | None = None

    def __reduce__(self) -> tuple[Any, ...]:
        # Without what its calls made, which holds native modules.
        return (
            _rebuilt_role,
            (self.symbols, self.slots, self.lines, self.used, self.every, self.program_id),
        )

    @staticmethod
    def identify(
        buffer: Any, name: str, parameters: Sequence[str], target: Device, options: CompileOptions
    ) -> str:
        import hashlib

        digest = hashlib.blake2b(digest_size=16)
        digest.update(memoryview(buffer).cast("B"))
        structure = target._structure or _structure(target)
        digest.update(repr((name, tuple(parameters), structure, options)).encode())
        return digest.hexdigest()


def _structure(device: Device) -> str:
    """``device._structure``: what a kernel whose qudits are runtime
    parameters depends on, not the carriers or their calibration, which are
    bound per call."""
    import hashlib

    content = repr((device.target.model, device.clock_ghz)).encode()
    made = device._structure = hashlib.blake2b(content, digest_size=16).hexdigest()
    return made


def _rebuilt_role(
    symbols: tuple[tuple[str, str, str, int | None], ...],
    slots: int,
    lines: tuple[int, ...],
    used: tuple[int, ...],
    every: tuple[int, ...],
    program_id: str,
) -> _Role:
    role = _Role(symbols)
    role.slots = slots
    role.lines = lines
    role.used = used
    role.every = every
    role.program_id = program_id
    return role


class _Placed(Lowered):
    """A module with list or qudit parameters, or a call's view of one."""

    __slots__ = ("_role", "carriers", "clocks", "fine_delays")
    _concrete = _concrete_general

    def __init__(self, *args: Any, role: _Role, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._role = role
        self.carriers = ()
        self.clocks = None
        self.fine_delays = None

    def bind(
        self,
        values: Sequence[float],
        carriers: tuple[int, ...] = (),
        target: Device | None = None,
    ) -> Lowered:
        """This module with values for its float parameters and, with
        *carriers*, its runtime qudits bound on *target*, sharing it. The
        values are checked as ``Lowered.bind`` checks them."""
        return _place(self, values, carriers, target)


def _placement(view: Lowered) -> Any:
    """*view*'s module with its carriers bound and its float arguments kept,
    shared by the views of one placement."""
    role = view._role
    assert role is not None and view.carriers
    key = (view.carriers, view.target)
    module = role.modules.get(key)
    if module is None:
        with view._lock:
            module = view._module.place_carriers(list(view.carriers), _frequency_table(view.target))
        if len(role.modules) >= _PLACED_MODULES:
            role.modules.clear()
        role.modules[key] = module
    return module


def _place(
    lowered: Lowered, values: Sequence[float], carriers: tuple[int, ...], target: Device | None
) -> _Placed:
    """``Lowered.bind`` of a module with list or qudit parameters."""
    try:
        fits = len(values) == len(lowered.parameters)
    except TypeError:
        fits = False
    if not fits and lowered._generic is None:
        values = _bind_checked(lowered.parameters, values)
    waits = lowered.waits
    if waits is not None:
        try:
            waits.check(values)
        except TypeError:
            _bind_checked(lowered.parameters, values)
            raise
    role = lowered._role
    assert role is not None
    view = _Placed.__new__(_Placed)
    view._module = lowered._module
    view._text = None
    view._lock = lowered._lock
    view._concrete_module = None
    view.target = lowered.target
    view.metrics = lowered.metrics
    view.parameters = ()
    view.arguments = tuple(values)
    view.qudits = lowered.qudits
    view.options = lowered.options
    view.scheduled = lowered.scheduled
    view._generic = lowered
    view.waits = waits
    view.classical = lowered.classical
    view.runtime_selects = lowered.runtime_selects
    view._role = role
    view.carriers = carriers
    view.clocks = lowered.clocks
    view.fine_delays = lowered.fine_delays
    if carriers:
        assert target is not None
        view.target = target
        placed = role.placed.get(carriers)
        if placed is None:
            qudits = {*role.used, *(carriers[k] for k in role.lines)}
            placed = tuple(sorted(qudits or {*role.every, *carriers}))
            if len(role.placed) < _PLACEMENTS:
                role.placed[carriers] = placed
        view.qudits = placed
    return view


class _CheckedPerCall(Lowered):
    """A module whose controller limits some float values exceed and others
    do not: each call's values are checked when they are bound, once per
    distinct values, so that no limit fails first at export."""

    __slots__ = ("_accepted", "clocks", "fine_delays")
    _concrete = _concrete_general

    def __init__(self, lowered: Lowered) -> None:
        super().__init__(
            lowered._module,
            target=lowered.target,
            metrics=lowered.metrics,
            parameters=lowered.parameters,
            qudits=lowered.qudits,
            options=lowered.options,
            scheduled=lowered.scheduled,
            waits=lowered.waits,
        )
        self.classical = lowered.classical
        self.runtime_selects = lowered.runtime_selects
        self.clocks = lowered.clocks
        self.fine_delays = lowered.fine_delays
        self._accepted: set[bytes] = set()

    def bind(self, values: Sequence[float]) -> Lowered:
        return self._accept(super().bind(values))

    def _with_values(self, values: Sequence[float]) -> Lowered:
        return self._accept(super()._with_values(values))

    def _accept(self, view: Lowered, placement: bytes = b"") -> Lowered:
        """*view*, once its values are within the limits (checked once per
        distinct values and *placement*). Its values are checked here, not
        where it is used: the limits need them."""
        arguments = view.arguments
        for v in arguments:
            if v.__class__ is not float and v.__class__ is not int or v - v != 0.0:
                arguments = tuple(view._checked_arguments())
                break
        key = _bits(arguments) + placement
        if key not in self._accepted:
            from .controller import _call_violations, _limit_error

            violations = _call_violations(view)
            if violations:
                raise _limit_error(view, violations)
            if len(self._accepted) >= _ACCEPTED_CALLS:
                self._accepted.clear()
            self._accepted.add(key)
        return view


class _CheckedPlaced(_CheckedPerCall):
    """A ``_CheckedPerCall`` module with list or qudit parameters."""

    __slots__ = ("_role", "carriers")

    def __init__(self, lowered: Lowered) -> None:
        super().__init__(lowered)
        self._role = lowered._role
        self.carriers = ()

    def bind(
        self,
        values: Sequence[float],
        carriers: tuple[int, ...] = (),
        target: Device | None = None,
    ) -> Lowered:
        view = _place(self, values, carriers, target)
        if not carriers:
            return self._accept(view)
        return self._accept(
            view, _bits(tuple(map(float, carriers))) + view.target.fingerprint.encode()
        )


def _schedule_args(
    options: CompileOptions, clocks: _clocks.ClockPlan | None
) -> tuple[str, int, int, int, int]:
    """The scheduler's arguments, with *clocks*' readout latency when its
    controllers count it in their own cycles."""
    args = options.schedule_args()
    if clocks is None or clocks.readout_latency_cycles is None:
        return args
    return (*args[:3], clocks.readout_latency_cycles, args[4])


def _frequency_table(device: Device) -> array[float]:
    """The 0-1 frequency of every carrier of *device*, by carrier index (0
    where the target has none), as a compiled module records it."""
    table = device._table
    if table is None:
        from array import array

        table = array("d", bytes(8 * (max(device.frequencies, default=-1) + 1)))
        for carrier, frequency in device.frequencies.items():
            table[carrier] = frequency
        device._table = table
    return table
