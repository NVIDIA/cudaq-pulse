# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The pulse MLIR evolve() would compile for a traced kernel call."""

import cudaq_pulse as pulse


def pulse_mlir(ir, *, qubit_freq_hz, clock_ghz=2.0):
    kernel, args = ir.source
    return pulse.compile(
        kernel,
        list(args),
        clock_ghz=clock_ghz,
        qubit_freq_hz=qubit_freq_hz,
        passes=("verify", "canonicalize", "virtual_z", "fusion"),
    ).mlir
