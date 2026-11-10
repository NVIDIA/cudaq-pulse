# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Python side of the JIT/runtime boundary, without a GPU."""

from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import sys

import numpy as np
import pytest
from cudaq_pulse.runtime import _jit as jit
from cudaq_pulse.runtime import gpu


def test_scalars_and_arrays_marshal_to_ctypes():
    values, keep_alive = jit._marshal_args([3, np.int32(4), 0.5, np.float32(1.5)])
    assert [type(v) for v in values] == [ctypes.c_int64] * 2 + [ctypes.c_double] * 2
    assert [v.value for v in values] == [3, 4, 0.5, 1.5]
    assert keep_alive == []


def test_arrays_are_passed_contiguous_and_kept_alive():
    strided = np.arange(12, dtype=np.float64)[::3]
    (pointer,), keep_alive = jit._marshal_args([strided])
    (copy,) = keep_alive
    assert copy.flags.c_contiguous
    assert pointer.value == copy.ctypes.data
    np.testing.assert_array_equal(copy, strided)


@pytest.mark.parametrize(
    "value, message",
    [
        (object(), "object is not a JIT argument"),
        ("text", "str is not a JIT argument"),
        ([1, 2], "list is not a JIT argument"),
        (True, "booleans are not JIT arguments"),
        (np.zeros(3, dtype=np.float32), "arrays must be float64, complex128 or int64"),
    ],
)
def test_unsupported_arguments_are_rejected(value, message):
    with pytest.raises(TypeError, match=message):
        jit._marshal_args([1, value])


def test_results_own_their_data():
    assert [f.name for f in dataclasses.fields(jit.JITResult)] == ["shape", "dtype", "_array"]
    data = np.arange(4, dtype=np.complex128)
    result = jit.JITResult(shape=(2, 2), dtype=data.dtype, _array=data)
    out = result.to_numpy()
    assert out.shape == (2, 2)
    out[0, 0] = 99
    assert data[0] == 0


class _FakeRuntime:
    """Stands in for the loaded GPU runtime library."""

    def __init__(self, result: bytes):
        self.result = result

    def cudm_last_result_size(self):
        return len(self.result)

    def cudm_last_result_copy(self, destination, size):
        ctypes.memmove(destination, self.result, size)
        return 0

    def cudm_last_error_message(self):
        return b"boom"


def _run(result: bytes, **kwargs):
    compiler = jit.JITCompiler.__new__(jit.JITCompiler)
    compiler._runtime = gpu.GpuRuntime(library=_FakeRuntime(result), path=None, device_count=1)
    entry = ctypes.CFUNCTYPE(None)(lambda: None)

    class Module:
        def lookup(self, name):
            return ctypes.cast(entry, ctypes.c_void_p).value

    return compiler.load_and_run(Module(), **kwargs)


def test_a_density_matrix_result_is_shaped():
    state = np.eye(4, dtype=np.complex128) / 4
    (result,) = _run(state.tobytes(), n_qubits=2)
    np.testing.assert_allclose(result.to_numpy(), state)


def test_a_state_vector_result_is_shaped():
    state = np.arange(4, dtype=np.complex128)
    (result,) = _run(state.tobytes(), n_qubits=2)
    assert result.shape == (4,)
    np.testing.assert_array_equal(result.to_numpy(), state)


@pytest.mark.parametrize("values", [4 * 4 + 1, 5, 4 * 4 - 1, 1])
def test_results_of_another_size_are_rejected(values):
    wrong = np.zeros(values, dtype=np.complex128).tobytes()
    with pytest.raises(RuntimeError, match="not a complex128 state vector or density matrix"):
        _run(wrong, n_qubits=2)


def test_results_of_a_partial_value_are_rejected():
    with pytest.raises(RuntimeError, match="returned 24 bytes"):
        _run(b"\0" * 24, n_qubits=0)


def test_the_state_dimension_overrides_the_qubit_count():
    (result,) = _run(np.zeros(9 * 9, dtype=np.complex128).tobytes(), n_qubits=2, state_dim=9)
    assert result.shape == (9, 9)
    (result,) = _run(np.zeros(3, dtype=np.complex128).tobytes(), n_qubits=2, state_dim=3)
    assert result.shape == (3,)


def test_an_empty_result_reports_the_runtime_error():
    with pytest.raises(RuntimeError, match="cuDensityMat execution failed: boom"):
        _run(b"", n_qubits=1)


def test_a_library_without_the_runtime_api_is_rejected():
    libc = "libc.so.6" if sys.platform.startswith("linux") else ctypes.util.find_library("c")
    with pytest.raises(gpu.GpuRuntimeLoadError, match="is not a cudaq-pulse GPU runtime"):
        gpu._load(libc, [])
