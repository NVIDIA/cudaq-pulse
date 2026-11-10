# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process JIT for lowered MLIR modules targeting the cuDensityMat runtime.

Lowering, code generation, and linking all happen inside the native
extension through MLIR's ExecutionEngine; no external compiler or LLVM tool
is invoked.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .._errors import ExecutionError, InvalidTypeError
from .._native._cudaq_pulse_native import JITModule, MLIRPipeline
from .gpu import GpuRuntime, require_gpu_runtime


@dataclass(frozen=True)
class JITResult:
    """A state returned from JIT-compiled code, owned by this object."""

    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    _array: npt.NDArray[Any] = field(repr=False, compare=False)

    def to_numpy(self) -> npt.NDArray[Any]:
        """A copy of the state as a NumPy array of ``shape``."""
        result: npt.NDArray[Any] = self._array.reshape(self.shape).copy()
        return result


class JITCompiler:
    """Compiles pulse MLIR through the full lowering pipeline and executes.

    Pipeline: pulse MLIR -> qop -> cudm -> LLVM -> ORC JIT -> execute
    """

    def __init__(self) -> None:
        self._runtime: GpuRuntime = require_gpu_runtime()
        self._pipeline = MLIRPipeline()

    def compile_pulse_mlir(self, pulse_mlir: str) -> JITModule:
        """Lower pulse MLIR and JIT-compile it against the GPU runtime."""
        return self._pipeline.jit_pulse(pulse_mlir, [str(self._runtime.path)])

    def load_and_run(
        self,
        module: JITModule,
        entry: str = "main",
        args: Sequence[Any] | None = None,
        n_qubits: int = 1,
        state_dim: int | None = None,
    ) -> list[JITResult]:
        """Invoke a JIT-compiled entry point and collect its result state.

        ``state_dim`` is the Hilbert-space dimension (``2**n_qubits`` by
        default); a result that is neither a state vector nor a density
        matrix of that dimension is rejected.
        """
        runtime = self._runtime.library
        c_args, keep_alive = _marshal_args(args or [])
        func_type = ctypes.CFUNCTYPE(None, *(type(a) for a in c_args))
        func = func_type(module.lookup(entry))
        func(*c_args)
        del keep_alive

        hilbert_dim = state_dim if state_dim is not None else 2**n_qubits
        itemsize = np.dtype(np.complex128).itemsize
        byte_count = runtime.cudm_last_result_size()
        if byte_count <= 0:
            raw_message = runtime.cudm_last_error_message()
            message = (
                raw_message.decode("utf-8", errors="replace")
                if raw_message
                else "unknown cuDensityMat error"
            )
            raise ExecutionError(f"cuDensityMat execution failed: {message}")
        shapes = {
            itemsize * hilbert_dim: (hilbert_dim,),
            itemsize * hilbert_dim * hilbert_dim: (hilbert_dim, hilbert_dim),
        }
        shape = shapes.get(byte_count)
        if shape is None:
            raise ExecutionError(
                f"cuDensityMat returned {byte_count} bytes, which is not a complex128 state "
                f"vector or density matrix of dimension {hilbert_dim}"
            )
        buffer = (ctypes.c_byte * byte_count)()
        status = runtime.cudm_last_result_copy(buffer, byte_count)
        if status != 0:
            raise ExecutionError("could not copy the cuDensityMat result buffer")
        array = np.frombuffer(buffer, dtype=np.complex128).copy()
        return [JITResult(shape=shape, dtype=np.dtype(np.complex128), _array=array)]


_ARRAY_DTYPES = (np.dtype(np.float64), np.dtype(np.complex128), np.dtype(np.int64))


def _marshal_args(args: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """Convert Python arguments to ctypes values.

    Integers become ``int64``, floats ``double``, and float64, complex128 or
    int64 NumPy arrays a pointer to C-contiguous data. Returns the ctypes
    values and the arrays they point into, which the caller keeps alive for
    the call.
    """
    c_args: list[Any] = []
    keep_alive: list[Any] = []
    for index, a in enumerate(args):
        if isinstance(a, (bool, np.bool_)):
            raise InvalidTypeError(f"argument {index}: booleans are not JIT arguments")
        if isinstance(a, (int, np.integer)):
            c_args.append(ctypes.c_int64(int(a)))
        elif isinstance(a, (float, np.floating)):
            c_args.append(ctypes.c_double(float(a)))
        elif isinstance(a, np.ndarray):
            if a.dtype not in _ARRAY_DTYPES:
                raise InvalidTypeError(
                    f"argument {index}: arrays must be float64, complex128 or int64, got {a.dtype}"
                )
            contiguous = np.ascontiguousarray(a)
            keep_alive.append(contiguous)
            c_args.append(contiguous.ctypes.data_as(ctypes.c_void_p))
        else:
            raise InvalidTypeError(
                f"argument {index}: {type(a).__name__} is not a JIT argument; "
                "use an int, a float or a NumPy array"
            )
    return c_args, keep_alive
