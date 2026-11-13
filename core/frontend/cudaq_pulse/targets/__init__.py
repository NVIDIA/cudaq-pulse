# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QPU target definitions: Hamiltonians, decoherence models, connectivity."""

from ._registry import UnknownQuantumSystemError, available_quantum_systems, register_quantum_system
from .base import (
    Coupling,
    DeviceDescription,
    DispersiveReadout,
    DragPulse,
    Drive,
    FluxCZ,
    MicrowaveDrive,
    PhysicalQudit,
    Readout,
    ReadoutPulse,
    ResidualZZ,
)
from .calibration import Calibration
from .models import SimulationModel, TransmonModel
from .transmon import transmon_17q, transmon_generic

__all__ = [
    "DeviceDescription",
    "PhysicalQudit",
    "Readout",
    "DispersiveReadout",
    "Drive",
    "MicrowaveDrive",
    "Coupling",
    "ResidualZZ",
    "DragPulse",
    "ReadoutPulse",
    "FluxCZ",
    "SimulationModel",
    "TransmonModel",
    "Calibration",
    "transmon_17q",
    "transmon_generic",
    "register_quantum_system",
    "available_quantum_systems",
    "UnknownQuantumSystemError",
]
