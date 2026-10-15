# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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

from .._native._cudaq_pulse_native import JITModule, MLIRPipeline
from .gpu import GpuRuntime, require_gpu_runtime


@dataclass(frozen=True)
class JITResult:
    """Container for values returned from JIT-compiled code."""

    raw_ptr: int
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    _array: npt.NDArray[Any] | None = field(default=None, repr=False, compare=False)

    def to_numpy(self) -> npt.NDArray[Any]:
        """Interpret the raw device-to-host copy as a NumPy array."""
        result: npt.NDArray[Any]
        if self._array is not None:
            result = self._array.reshape(self.shape).copy()
            return result
        n_elements = 1
        for s in self.shape:
            n_elements *= s
        buf = (ctypes.c_double * (n_elements * 2)).from_address(self.raw_ptr)
        flat = np.ctypeslib.as_array(buf).view(np.complex128)
        result = flat.reshape(self.shape).copy()
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

    def compile_module(self, mlir_text: str) -> JITModule:
        """JIT-compile an LLVM-dialect MLIR module against the GPU runtime."""
        return self._pipeline.jit_llvm_dialect(mlir_text, [str(self._runtime.path)])

    def load_and_run(
        self,
        module: JITModule,
        entry: str = "main",
        args: Sequence[Any] | None = None,
        n_qubits: int = 1,
    ) -> list[JITResult]:
        """Invoke a JIT-compiled entry point and collect its result state."""
        runtime = self._runtime.library
        runtime.cudm_last_result_size.argtypes = []
        runtime.cudm_last_result_size.restype = ctypes.c_int64
        runtime.cudm_last_result_copy.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        runtime.cudm_last_result_copy.restype = ctypes.c_int
        runtime.cudm_last_error_message.argtypes = []
        runtime.cudm_last_error_message.restype = ctypes.c_char_p

        c_args = _marshal_args(args or [])
        func_type = ctypes.CFUNCTYPE(None, *(type(a) for a in c_args))
        func = func_type(module.lookup(entry))
        func(*c_args)

        byte_count = runtime.cudm_last_result_size()
        if byte_count <= 0:
            raw_message = runtime.cudm_last_error_message()
            message = (
                raw_message.decode("utf-8", errors="replace")
                if raw_message
                else "unknown cuDensityMat error"
            )
            raise RuntimeError(f"cuDensityMat execution failed: {message}")
        if byte_count % np.dtype(np.complex128).itemsize:
            raise RuntimeError(
                f"cuDensityMat returned {byte_count} bytes, which is not a complex128 state buffer"
            )
        buffer = (ctypes.c_byte * byte_count)()
        status = runtime.cudm_last_result_copy(buffer, byte_count)
        if status != 0:
            raise RuntimeError("could not copy the cuDensityMat result buffer")
        array = np.frombuffer(buffer, dtype=np.complex128).copy()
        hilbert_dim = 2**n_qubits
        if array.size == hilbert_dim:
            shape: tuple[int, ...] = (hilbert_dim,)
        elif array.size == hilbert_dim * hilbert_dim:
            shape = (hilbert_dim, hilbert_dim)
        else:
            shape = (array.size,)
        return [
            JITResult(
                raw_ptr=array.ctypes.data, shape=shape, dtype=np.dtype(np.complex128), _array=array
            )
        ]

    def compile_and_run(
        self,
        program: Any,
        *,
        entry: str = "main",
        n_qubits: int | None = None,
    ) -> list[JITResult]:
        """Compile a pulse ``Program`` or Pulse-MLIR string and execute it."""
        if isinstance(program, str):
            pulse_mlir = program
        else:
            from ..passes.ir_types import Program
            from ..passes.to_pulse_mlir import program_to_pulse_mlir

            if not isinstance(program, Program):
                raise TypeError("expected a pulse Program or Pulse-MLIR string")
            pulse_mlir = program_to_pulse_mlir(program)
            if n_qubits is None:
                n_qubits = len(program.qubit_freq_hz) or 1
        module = self.compile_pulse_mlir(pulse_mlir)
        return self.load_and_run(module, entry=entry, n_qubits=n_qubits or 1)


def _marshal_args(args: Sequence[Any]) -> list[Any]:
    """Convert Python arguments to ctypes-compatible values."""
    c_args: list[Any] = []
    for a in args:
        if isinstance(a, (int, np.integer)):
            c_args.append(ctypes.c_int64(int(a)))
        elif isinstance(a, (float, np.floating)):
            c_args.append(ctypes.c_double(float(a)))
        elif isinstance(a, np.ndarray):
            c_args.append(a.ctypes.data_as(ctypes.c_void_p))
        else:
            c_args.append(ctypes.c_void_p(id(a)))
    return c_args


def compile_and_run(
    module_text: str,
    args: Sequence[Any] | None = None,
    *,
    entry: str = "main",
    n_qubits: int = 1,
) -> list[JITResult]:
    """One-shot: JIT-compile an LLVM-dialect MLIR module and execute it.

    Raises a ``GpuRuntimeError`` subclass if GPU execution is unavailable.
    """
    compiler = JITCompiler()
    module = compiler.compile_module(module_text)
    return compiler.load_and_run(module, entry=entry, args=args or [], n_qubits=n_qubits)


def compile_and_run_pulse(
    pulse_mlir: str,
    *,
    entry: str = "main",
    n_qubits: int = 1,
) -> list[JITResult]:
    """One-shot: compile pulse MLIR through the full pipeline and execute.

    Runs pulse -> qop -> cudm -> llvm -> JIT -> execute. Raises a
    ``GpuRuntimeError`` subclass if GPU execution is unavailable.
    """
    compiler = JITCompiler()
    module = compiler.compile_pulse_mlir(pulse_mlir)
    return compiler.load_and_run(module, entry=entry, n_qubits=n_qubits)
