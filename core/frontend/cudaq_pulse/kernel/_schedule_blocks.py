# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Per-kernel scheduling policies.

A kernel declared with ``@kernel(schedule="asap")`` or ``schedule="alap"``
and called from another kernel is traced into a ``pulse.schedule_block``: it
is scheduled on its own with its policy, then placed whole by the policy of
the kernel that calls it. A kernel without ``schedule=`` is inlined into its
caller and scheduled with it.

``compile(k, schedule=...)`` sets the policy of ``k`` only; the kernels it
calls keep theirs. ``compile(k, flatten_schedules=True)`` ignores every
called kernel's policy and schedules the program as one.
"""

from __future__ import annotations

from typing import Any

from .._errors import InvalidValueError, _closest
from .ir_builder import CompilationError

# Policies a called kernel can be scheduled with on its own.
BLOCK_SCHEDULES = frozenset({"asap", "alap"})
# Resource limits belong to the machine, so they apply to the whole program.
ROOT_ONLY_SCHEDULES = frozenset({"rcp", "alap_rcp"})
DEFAULT_SCHEDULE = "alap"


def validate_kernel_schedule(schedule: str | None) -> str | None:
    """Check a ``@kernel(schedule=...)`` value when the kernel is declared."""
    if schedule is None or schedule in BLOCK_SCHEDULES:
        return schedule
    if schedule in ROOT_ONLY_SCHEDULES:
        raise InvalidValueError(
            f"schedule={schedule!r} applies to a whole program, not to one kernel; "
            f"pass it to cudaq_pulse.compile(..., schedule={schedule!r}) instead"
        )
    close = _closest(str(schedule).lower(), BLOCK_SCHEDULES | ROOT_ONLY_SCHEDULES)
    if close in ROOT_ONLY_SCHEDULES:
        hint = (
            f" Did you mean {close!r}? It applies to a whole program; pass it to "
            f"cudaq_pulse.compile(..., schedule={close!r}) instead."
        )
    else:
        hint = f" Did you mean {close!r}?" if close else ""
    raise InvalidValueError(
        f"Unknown kernel schedule {schedule!r}.{hint} Choose from: "
        + ", ".join(sorted(BLOCK_SCHEDULES))
        + ", or None to schedule the kernel with its caller"
    )


def validate_max_duration(cycles: int | None) -> int | None:
    """Check a ``@kernel(max_duration_cycles=...)`` value."""
    if cycles is None:
        return None
    if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
        raise ValueError(
            f"max_duration_cycles must be a whole number of cycles > 0, got {cycles!r}"
        )
    return cycles


def block_budget(kernel: Any, builder: Any) -> int | None:
    """The ``max_duration_cycles`` of called *kernel*, which its block carries.

    Raises ``CompilationError`` when its calls have no length of their own
    to hold to it.
    """
    cycles = getattr(kernel, "max_duration_cycles", None)
    if cycles is None:
        return None
    name = kernel.__name__
    if getattr(kernel, "schedule", None) is None:
        raise CompilationError(
            f"{name}() has max_duration_cycles but no schedule=, so it is scheduled "
            "with its caller and its calls have no length of their own; give it "
            'schedule="asap" or schedule="alap"'
        )
    if not getattr(builder, "schedule_blocks", False):
        raise CompilationError(
            f"flatten_schedules=True schedules {name}() with its caller, so its "
            "max_duration_cycles cannot hold; compile without flatten_schedules"
        )
    return int(cycles)


def root_schedule(own: str | None, schedule: str | None) -> str:
    """The policy for a compiled kernel: ``compile``'s, else its own."""
    if schedule is not None:
        return schedule
    return own or DEFAULT_SCHEDULE


def check_root_schedule(schedule: str, builder: Any) -> None:
    """Reject resource-constrained scheduling of a program with blocks."""
    if schedule in ROOT_ONLY_SCHEDULES and builder.schedule_block_count:
        raise CompilationError(
            f"schedule={schedule!r} does not support kernels called with their own "
            "schedule= yet; remove schedule= from the called kernels, or compile "
            "with flatten_schedules=True"
        )
