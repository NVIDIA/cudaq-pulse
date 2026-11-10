# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The JIT resolves runtime symbols without adding libraries to the global scope.

The GPU runtime and the CUDA libraries it needs are loaded RTLD_LOCAL, so
they cannot interpose on or clash with other copies of the CUDA stack in the
process. The JIT finds the runtime's functions in the libraries it is given
and keeps them local too. These tests use the native extension, which Python
loads RTLD_LOCAL, as the library. On macOS every loaded image is searched
for the process's symbols, so only Linux has a global scope to check.
"""

from __future__ import annotations

import ctypes
import sys

import pytest
from cudaq_pulse._native import _cudaq_pulse_native as native

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="RTLD_LOCAL hides symbols only on Linux"
)

_SYMBOL = "PyInit__cudaq_pulse_native"
_ADDRESS_OF = f"""
llvm.func @{_SYMBOL}() -> !llvm.ptr
llvm.func @address() -> !llvm.ptr {{
  %f = llvm.mlir.addressof @{_SYMBOL} : !llvm.ptr
  llvm.return %f : !llvm.ptr
}}
"""


def _global_scope_has(name: str) -> bool:
    return hasattr(ctypes.CDLL(None), name)


def _local_address() -> int | None:
    return ctypes.cast(getattr(ctypes.CDLL(native.__file__), _SYMBOL), ctypes.c_void_p).value


@linux_only
def test_the_extension_is_loaded_locally():
    assert not _global_scope_has(_SYMBOL)


def test_external_functions_resolve_from_the_given_libraries():
    module = native.MLIRPipeline().jit_llvm_dialect(_ADDRESS_OF, [native.__file__])
    address = ctypes.CFUNCTYPE(ctypes.c_void_p)(module.lookup("address"))
    assert address() == _local_address()


@linux_only
def test_resolving_does_not_make_the_library_global():
    native.MLIRPipeline().jit_llvm_dialect(_ADDRESS_OF, [native.__file__]).lookup("address")
    assert not _global_scope_has(_SYMBOL)


@linux_only
def test_a_function_in_no_library_is_unresolved():
    module = native.MLIRPipeline().jit_llvm_dialect(_ADDRESS_OF, [])
    with pytest.raises(RuntimeError, match=_SYMBOL):
        module.lookup("address")


def test_a_library_that_cannot_be_opened_is_reported(tmp_path):
    bogus = tmp_path / "libnot-a-library.so"
    bogus.write_bytes(b"not a shared library")
    with pytest.raises(RuntimeError, match="libnot-a-library.so"):
        native.MLIRPipeline().jit_llvm_dialect(_ADDRESS_OF, [str(bogus)])


def test_a_function_no_library_defines_is_named_in_the_error():
    missing = _ADDRESS_OF.replace(_SYMBOL, "cudaq_pulse_missing_symbol")
    module = native.MLIRPipeline().jit_llvm_dialect(missing, [native.__file__])
    with pytest.raises(RuntimeError, match="libraries: cudaq_pulse_missing_symbol"):
        module.lookup("address")
