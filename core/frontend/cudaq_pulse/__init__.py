# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""cudaq-pulse: pulse-level quantum programming on MLIR.

Usage::

    import cudaq_pulse as pulse

    @pulse.kernel
    def rabi(qubit, amplitude: float):
        drive_line, tone = get_drive_line(qubit)
        drive(drive_line, gaussian(64, amplitude, 16.0), tone)

    dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
    rabi(dev.qudit_handles[0], 0.5)            # compile on first call, then run
    lowered = rabi.lower(dev.qudit_handles[0], 0.5)
    print(lowered.mlir())

The kernel language (``drive``, ``gaussian``, ``get_drive_line``, ...) lives
in ``cudaq_pulse.ops``; inside a ``@pulse.kernel`` its names are used bare.
A bare int duration is a number of cycles of the device's clock; a time
with units, ``20 * pulse.ns`` (``pulse.us``, ``pulse.ms``, ``pulse.s``),
converts to cycles exactly when the kernel compiles, so one kernel runs on
devices with different clocks. ``pulse.round_duration(line, n)`` rounds a length
onto the grid of the line's controller, for controllers whose clock is
slower than the program's.
Compiler types (``CompileMetrics``, ``ScheduledEvent``, ``MachineModel``,
``DEFAULT_PASSES``) are in ``cudaq_pulse.compiler``, target descriptions and
calibrations in ``cudaq_pulse.targets``. ``pulse.classical`` declares the
classical functions a kernel calls while the QPU runs. ``pulse.controller``
exports a compiled kernel to a controller and states a controller's limits,
and ``pulse.backends`` finds the controller backends installed beside
cudaq-pulse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _build_config
from . import ops as ops
from ._clocks import RoundDirection as RoundDirection
from ._clocks import round_duration as round_duration
from ._device import Device as Device
from ._device import DeviceMismatchError as DeviceMismatchError
from ._device import NoDeviceError as NoDeviceError
from ._device import Qudit as Qudit
from ._device import Register as Register
from ._device import Target as Target
from ._device import device as device
from ._device import get_device as get_device
from ._errors import CompilationError as CompilationError
from ._errors import DeviceDefinitionError as DeviceDefinitionError
from ._errors import ExecutionError as ExecutionError
from ._errors import InternalError as InternalError
from ._errors import InvalidTypeError as InvalidTypeError
from ._errors import InvalidValueError as InvalidValueError
from ._errors import NotFoundError as NotFoundError
from ._errors import Problem as Problem
from ._errors import PulseError as PulseError
from ._errors import PulseWarning as PulseWarning
from ._errors import TimingError as TimingError
from ._errors import UnsupportedError as UnsupportedError
from ._errors import UpgradeNotice as UpgradeNotice
from ._units import Time as Time
from ._units import ms as ms
from ._units import ns as ns
from ._units import s as s
from ._units import us as us
from .compiler import Lowered as Lowered
from .compiler import Pass as Pass
from .compiler import compile as compile
from .kernel import kernel as kernel
from .kernel._kernel import CompiledKernel as CompiledKernel
from .kernel._kernel import Kernel as Kernel
from .targets._registry import NoTargetError as NoTargetError

# A module can't be declared callable, so checkers see the decorator's type.
if TYPE_CHECKING:
    from .classical import ClassicalDecorator

    classical: ClassicalDecorator
else:
    from . import classical as classical

__version__: str = _build_config.PACKAGE_VERSION

# Simulation, GPU execution and visualization load NumPy, the GPU runtime
# discovery or plotting, which compiling does not need: import them on first
# use.
if TYPE_CHECKING:
    from . import backends as backends
    from . import controller as controller
    from . import viz as viz
    from .runtime.gpu import GpuRuntimeError as GpuRuntimeError
    from .runtime.simulate import SimulationResult as SimulationResult
    from .runtime.simulate import simulate as simulate

_LAZY_EXPORTS = {
    "SimulationResult": ".runtime.simulate",
    "simulate": ".runtime.simulate",
    "GpuRuntimeError": ".runtime.gpu",
}
_LAZY_MODULES = {"viz": ".viz", "backends": ".backends", "controller": ".controller"}

__all__ = [
    "kernel",
    "Kernel",
    "compile",
    "CompiledKernel",
    "Lowered",
    "Pass",
    "simulate",
    "SimulationResult",
    "device",
    "get_device",
    "Device",
    "Qudit",
    "Register",
    "Target",
    "PulseError",
    "Problem",
    "DeviceDefinitionError",
    "InvalidValueError",
    "InvalidTypeError",
    "NotFoundError",
    "UnsupportedError",
    "CompilationError",
    "TimingError",
    "ExecutionError",
    "InternalError",
    "PulseWarning",
    "UpgradeNotice",
    "NoTargetError",
    "NoDeviceError",
    "DeviceMismatchError",
    "GpuRuntimeError",
    "round_duration",
    "RoundDirection",
    "Time",
    "ns",
    "us",
    "ms",
    "s",
    "viz",
    "classical",
    "controller",
    "backends",
]


def __getattr__(name: str) -> object:
    import importlib

    module = _LAZY_MODULES.get(name)
    if module is not None:
        value: object = importlib.import_module(module, __name__)
    else:
        source = _LAZY_EXPORTS.get(name)
        if source is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        value = getattr(importlib.import_module(source, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*__all__, "__version__"])
