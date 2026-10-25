# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Execution: the GPU runtime and its errors.

The names below come from ``cudaq_pulse.runtime.gpu``, which is imported on
first use so that simulating on the CPU never loads the GPU runtime discovery.
"""

import typing as _typing

if _typing.TYPE_CHECKING:
    from .gpu import GpuDeviceUnavailableError as GpuDeviceUnavailableError
    from .gpu import GpuRuntime as GpuRuntime
    from .gpu import GpuRuntimeError as GpuRuntimeError
    from .gpu import GpuRuntimeLoadError as GpuRuntimeLoadError
    from .gpu import GpuSupportNotBuiltError as GpuSupportNotBuiltError
    from .gpu import require_gpu_runtime as require_gpu_runtime

__all__ = [
    "GpuDeviceUnavailableError",
    "GpuRuntime",
    "GpuRuntimeError",
    "GpuRuntimeLoadError",
    "GpuSupportNotBuiltError",
    "require_gpu_runtime",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import gpu

    value = getattr(gpu, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    names = set(globals()) | set(__all__)
    return sorted(n for n in names if not n.startswith("_") or n.startswith("__"))
