# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


import cudaq_pulse as pulse
import pytest
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q
from tests._ir import python_ir


def test_kernel_basic_decoration():

    @pulse.kernel
    def my_kernel(q0):
        pass

    assert isinstance(my_kernel, pulse.Kernel)
    assert my_kernel.__wrapped__.__name__ == "my_kernel"


def test_kernel_caching():

    @pulse.kernel
    def my_kernel(q0):
        pass

    q = Q[0]
    first = my_kernel.lower(q)
    assert my_kernel.lower(q) is first
    assert my_kernel.cache_info().compiled == 1


def test_kernel_invocation():

    @pulse.kernel
    def echo(q0):
        pass

    q = Q[0]
    result = python_ir(echo, q)
    assert result is not None


def test_qudit_ref_is_not_a_kernel_op():
    @pulse.kernel
    def internal():
        q = qudit_ref()  # pyright: ignore[reportUndefinedVariable]  # noqa: F821
        d0, t0 = get_drive_line(q)

    with pytest.raises(CompilationError, match="unknown pulse op: qudit_ref"):
        python_ir(internal)
