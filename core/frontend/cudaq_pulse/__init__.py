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

The compiler recognizes the kernel DSL vocabulary (``drive``, ``gaussian``,
``get_drive_line``, etc.) as bare names inside ``@pulse.kernel`` functions.
"""

import typing as _typing

from . import _build_config

__version__: str = _build_config.PACKAGE_VERSION

# ── Infrastructure (accessed via pulse.*) ────────────────────────────────

from ._device import Device as Device
from ._device import DeviceMismatchError as DeviceMismatchError
from ._device import NoDeviceError as NoDeviceError
from ._device import Qudit as Qudit
from ._device import Register as Register
from ._device import Target as Target
from ._device import device as device
from ._device import get_device as get_device
from .compile import DEFAULT_PASSES as DEFAULT_PASSES
from .compile import CompileMetrics as CompileMetrics
from .compile import Lowered as Lowered
from .compile import MachineModel as MachineModel
from .compile import ScheduledEvent as ScheduledEvent
from .compile import compile as compile
from .kernel import kernel as kernel
from .kernel._kernel import CompiledKernel as CompiledKernel
from .kernel._kernel import Kernel as Kernel
from .kernel.decorator import QuditRef as QuditRef
from .kernel.decorator import QvecRef as QvecRef
from .kernel.decorator import qudit_ref as qudit_ref
from .kernel.decorator import qvec_ref as qvec_ref
from .kernel.ir_builder import CompilationError as CompilationError
from .kernel.ir_builder import Parameter as Parameter
from .ops import cosine as cosine
from .ops import custom as custom
from .ops import custom_samples as custom_samples
from .ops import drag as drag
from .ops import drive as drive
from .ops import gaussian as gaussian
from .ops import gaussian_square as gaussian_square

# ── Kernel DSL ops (injected as bare names into importer's namespace) ────
from .ops import get_drive_line as get_drive_line
from .ops import get_readout_line as get_readout_line
from .ops import readout as readout
from .ops import set_frequency as set_frequency
from .ops import set_phase as set_phase
from .ops import shift_frequency as shift_frequency
from .ops import shift_phase as shift_phase
from .ops import square as square
from .ops import sync as sync
from .ops import tanh_ramp as tanh_ramp
from .ops import wait as wait
from .ops import wf_add as wf_add
from .ops import wf_mul as wf_mul
from .ops import wf_neg as wf_neg
from .ops import wf_scale as wf_scale
from .ops import wf_sub as wf_sub
from .targets._registry import NoTargetError as NoTargetError

# Simulation and GPU execution load NumPy and the GPU runtime discovery, which
# compiling does not need: import them on first use.
if _typing.TYPE_CHECKING:
    from .runtime.simulate import SimulationResult as SimulationResult
    from .runtime.simulate import simulate as simulate

_LAZY_EXPORTS = {
    "SimulationResult": ".runtime.simulate",
    "simulate": ".runtime.simulate",
}


def __getattr__(name: str) -> object:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    # Private modules (_device, _native, _build_config) and helpers stay
    # importable but are not listed.
    names = set(globals()) | set(_LAZY_EXPORTS)
    return sorted(n for n in names if not n.startswith("_") or n.startswith("__"))


__all__: list[str] = [
    # infrastructure
    "kernel",
    "Kernel",
    "compile",
    "CompiledKernel",
    "Lowered",
    "CompileMetrics",
    "MachineModel",
    "ScheduledEvent",
    "DEFAULT_PASSES",
    "Parameter",
    "CompilationError",
    "device",
    "get_device",
    "Device",
    "Qudit",
    "Register",
    "Target",
    "NoDeviceError",
    "NoTargetError",
    "DeviceMismatchError",
    "SimulationResult",
    "simulate",
    "qudit_ref",
    "qvec_ref",
    "QuditRef",
    "QvecRef",
    # kernel DSL ops
]
