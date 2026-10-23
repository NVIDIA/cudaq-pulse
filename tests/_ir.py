# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The Python-object IR of a kernel call, for tests that inspect traced ops."""

from cudaq_pulse.kernel.bytecode_bridge import compile_kernel_bytecode
from cudaq_pulse.kernel.ir_builder import PythonIRBuilder


def python_ir(kernel, *args) -> PythonIRBuilder:
    return compile_kernel_bytecode(kernel.__wrapped__)(*args)
