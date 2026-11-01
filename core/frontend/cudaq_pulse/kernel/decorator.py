# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, overload

if TYPE_CHECKING:
    from ._kernel import Kernel


class KernelDecorator(Protocol):
    """``@kernel(...)`` with options: makes a kernel of the function it decorates."""

    def __call__[**P](self, fn: Callable[P, Any], /) -> Kernel[P]: ...


@overload
def kernel[**P](fn: Callable[P, Any], /) -> Kernel[P]: ...
@overload
def kernel(
    fn: None = None,
    /,
    *,
    cache_size: int | None = None,
    schedule: str | None = None,
    unroll_threshold: int | None = None,
) -> KernelDecorator: ...


def kernel[**P](
    fn: Callable[P, Any] | None = None,
    /,
    *,
    cache_size: int | None = None,
    schedule: str | None = None,
    unroll_threshold: int | None = None,
) -> Kernel[P] | KernelDecorator:
    """Decorator that makes a Python function a pulse kernel.

    Calling the kernel compiles it on first use and runs it on the device of
    its qudit handles, returning what the kernel returns. Parameters annotated
    ``float`` stay symbolic in the compiled module; the kernel is
    specialized on every other parameter, and up to *cache_size*
    specializations are kept (``CUDAQ_PULSE_CACHE_SIZE``, 32 by default).

    Supported control flow inside the kernel:

    - ``for i in range(N)`` (bound known at compile time)
    - ``if`` / ``else`` on values known at compile time, or on a measurement

    Example::

        @cudaq_pulse.kernel
        def rabi(q, amp: float):
            d = get_drive_line(q)
            drive(d, gaussian(40, amp, 10.0))

        dev = cudaq_pulse.device("transmon_17q")
        rabi(dev.qudit_handles[0], 0.3)

    A kernel called from another kernel is scheduled with it, unless it
    sets its own policy: with ``schedule="asap"`` or ``schedule="alap"`` it
    is scheduled on its own, then placed whole by the caller's policy, so
    its internal timing is the same wherever it is called. Compiled or
    called on its own, the kernel uses its policy unless
    ``cudaq_pulse.compile(k, schedule=...)`` overrides it.

    *unroll_threshold* sets which ``for ... in range(n)`` loops are unrolled
    while tracing (see ``cudaq_pulse.compile``): a non-negative ``int``, 8 by
    default, and 0 never unrolls.
    """
    from ._bytecode_normalize import check_interpreter
    from ._kernel import Kernel
    from ._schedule_blocks import validate_kernel_schedule
    from ._unroll import resolve_unroll_threshold

    check_interpreter()
    validate_kernel_schedule(schedule)
    if unroll_threshold is not None:
        resolve_unroll_threshold(unroll_threshold)

    def wrap[**Q](f: Callable[Q, Any], /) -> Kernel[Q]:
        return Kernel(
            f,
            cache_size=cache_size,
            schedule=schedule,
            unroll_threshold=unroll_threshold,
        )

    return wrap if fn is None else wrap(fn)
