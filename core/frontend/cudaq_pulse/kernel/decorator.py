# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any, overload

if TYPE_CHECKING:
    from ._kernel import Kernel

_next_vid = 0


def _alloc_vid() -> int:
    global _next_vid
    vid = _next_vid
    _next_vid += 1
    return vid


class QuditRef:
    """A single qudit reference representing one quantum degree of freedom.

    Created via ``cudaq_pulse.qudit_ref()``.  Passed as an argument to
    ``@kernel`` functions to bind physical qubit resources.
    """

    __slots__ = ("_vid", "_index")

    def __init__(self, index: int | None = None) -> None:
        if index is not None and (not isinstance(index, int) or index < 0):
            raise ValueError("qudit index must be a non-negative integer")
        self._vid = _alloc_vid()
        self._index = index

    @property
    def index(self) -> int | None:
        """Requested physical index, or ``None`` for positional binding."""
        return self._index


class QvecRef:
    """A fixed-size vector of qudit references.

    Created via ``cudaq_pulse.qvec_ref(n)``.  Supports ``len()``, integer
    indexing and iteration over its ``QuditRef`` elements.
    """

    def __init__(self, n: int):
        if not isinstance(n, int) or n < 0:
            raise ValueError("qvec size must be a non-negative integer")
        self._n = n
        self._qudits = [QuditRef(index) for index in range(n)]

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> QuditRef:
        if not 0 <= idx < self._n:
            raise IndexError(f"qudit index {idx} out of range [0, {self._n})")
        return self._qudits[idx]

    def __iter__(self) -> Iterator[QuditRef]:
        return iter(self._qudits)


def qudit_ref(index: int | None = None) -> QuditRef:
    """Allocate a single qudit reference.

    Use as an argument when calling a ``@kernel`` function::

        q = cudaq_pulse.qudit_ref()
        my_kernel(q)

    Pass an index to bind a kernel argument to a specific physical qubit,
    for example ``cudaq_pulse.qudit_ref(4)``.

    Args:
        index: Optional non-negative physical qubit index. When omitted,
            qubits are assigned by argument position.

    Returns:
        A fresh ``QuditRef`` with a unique virtual ID.
    """
    return QuditRef(index)


def qvec_ref(n: int) -> QvecRef:
    """Allocate a vector of *n* qudit references.

    Example::

        qubits = cudaq_pulse.qvec_ref(4)
        my_kernel(qubits[0], qubits[1])

    Args:
        n: Number of qudits.

    Returns:
        A ``QvecRef`` containing *n* ``QuditRef`` elements.
    """
    return QvecRef(n)


@overload
def kernel(fn: Callable[..., Any], /) -> Kernel: ...
@overload
def kernel(
    fn: None = None, /, *, cache_size: int | None = None, schedule: str | None = None
) -> Callable[[Callable[..., Any]], Kernel]: ...


def kernel(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    cache_size: int | None = None,
    schedule: str | None = None,
) -> Kernel | Callable[[Callable[..., Any]], Kernel]:
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
            d, t = get_drive_line(q)
            drive(d, gaussian(40, amp, 10.0), t)

        rabi(cudaq_pulse.qudit_ref(), 0.3)

    A kernel called from another kernel is scheduled with it, unless it
    sets its own policy: with ``schedule="asap"`` or ``schedule="alap"`` it
    is scheduled on its own, then placed whole by the caller's policy, so
    its internal timing is the same wherever it is called. Compiled or
    called on its own, the kernel uses its policy unless
    ``cudaq_pulse.compile(k, schedule=...)`` overrides it.
    """
    from ._bytecode_normalize import check_interpreter
    from ._kernel import Kernel
    from ._schedule_blocks import validate_kernel_schedule

    check_interpreter()
    validate_kernel_schedule(schedule)

    def wrap(f: Callable[..., Any]) -> Kernel:
        return Kernel(f, cache_size=cache_size, schedule=schedule)  # type: ignore[arg-type]

    return wrap if fn is None else wrap(fn)
