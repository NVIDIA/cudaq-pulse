# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Link``, the cost of a link to a classical quantum system, which
``cudaq_pulse.classical`` loads when it is first used."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._errors import InvalidTypeError, InvalidValueError
from .._units import Time

if TYPE_CHECKING:
    from .._system import Controller

__all__ = ["Link"]


@dataclass(frozen=True)
class Link:
    """What one link to a classical quantum system costs, for scheduling.

    ``fixed_ns`` is the fixed cost of sending a request and ``bytes_per_ns``
    its bandwidth (0 for unlimited): a request holds the link for
    ``fixed_ns + bytes / bytes_per_ns``, at least one clock cycle.
    ``feedback_ns`` is added to every latency: the time from an answer
    arriving to a pulse being able to use it. ``fixed_ns`` and
    ``feedback_ns`` are times (``100 * pulse.ns``) or numbers of ns. ``slots`` is how many
    requests the link holds at once (its slots), and ``slot_bytes`` the
    largest request or answer, its 24-byte header included.

    ``controllers`` are the controllers of a ``ControlSystem`` the link
    reaches (``None``: all of them): a call over it can send only their
    measurements, and only they can use its answers.

    A link is one service the controllers share. With ``per_controller``,
    each controller it reaches has an instance of its own, with its own
    slots, its own processor (``Processor.links``) and its own dispatch
    keys: a call over it runs on the instance of the controller that sends
    its measurements and uses its answer, which must be one controller, so
    no answer crosses between controllers.
    """

    fixed_ns: float | Time = 0.0
    bytes_per_ns: float = 0.0
    feedback_ns: float | Time = 0.0
    slots: int = 64
    slot_bytes: int = 1024
    controllers: Sequence[Controller] | None = None
    per_controller: bool = False

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
        if not isinstance(self.slots, int) or self.slots < 1:
            raise InvalidValueError(f"Link.slots must be at least 1, got {self.slots!r}")
        if not isinstance(self.slot_bytes, int) or self.slot_bytes <= 24:
            raise InvalidValueError(
                f"Link.slot_bytes must exceed the 24-byte header, got {self.slot_bytes!r}"
            )
        if not isinstance(self.per_controller, bool):
            raise InvalidTypeError(
                f"Link.per_controller is True or False, got {self.per_controller!r}"
            )

    def cost(self) -> dict[str, float | int]:
        """What the link costs, as JSON-serializable data (with
        ``per_controller`` when it has an instance per controller)."""
        data: dict[str, float | int] = {
            "fixed_ns": self.fixed_ns,
            "bytes_per_ns": self.bytes_per_ns,
            "feedback_ns": self.feedback_ns,
            "slots": self.slots,
            "slot_bytes": self.slot_bytes,
        }
        if self.per_controller:
            data["per_controller"] = True
        return data

    def attribute(self, uncertainty_ns: float = 0.0, name: str | None = None) -> str:
        """The link as an entry of the module attribute ``classical.links``;
        *uncertainty_ns* is how far apart the starts of the controllers it
        joins may be, which an answer taken on another controller than the
        one that sent the request adds to its budget. A per-controller link
        is marked so, with its *name*."""
        uncertainty = (
            f"start_uncertainty_ns = {float(uncertainty_ns)!r} : f64, " if uncertainty_ns else ""
        )
        instances = ""
        if self.per_controller:
            instances = f', per_controller, name = "{name or ""}"'
        return (
            f"{{fixed_ns = {float(self.fixed_ns)!r} : f64, "
            f"bytes_per_ns = {float(self.bytes_per_ns)!r} : f64, "
            f"feedback_ns = {float(self.feedback_ns)!r} : f64, "
            f"{uncertainty}"
            f"slots = {self.slots} : i64, "
            f"slot_bytes = {self.slot_bytes} : i64{instances}}}"
        )


# Pickles and reprs name the public module.
Link.__module__ = "cudaq_pulse.classical"
