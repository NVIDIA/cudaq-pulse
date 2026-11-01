# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The kernel of the console's device-wide spectroscopy sweep."""

import cudaq_pulse as pulse


@pulse.kernel
def detuned_rabi(qubit, duration: int, amplitude: float, detuning_hz: float):
    """A square drive *detuning_hz* off the qudit's nominal frequency."""
    drive_line = get_drive_line(qubit)  # noqa: F821
    shift_frequency(drive_line.tone, detuning_hz)  # noqa: F821
    drive(drive_line, square(duration, amplitude))  # noqa: F821
