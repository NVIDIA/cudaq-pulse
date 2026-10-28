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

from .._errors import InvalidValueError
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
    raise InvalidValueError(
        f"Unknown kernel schedule {schedule!r}. Choose from: "
        + ", ".join(sorted(BLOCK_SCHEDULES))
        + ", or None to schedule the kernel with its caller"
    )


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
