..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Compilation Pipeline
====================

CUDA-Q Pulse compiles ``@pulse.kernel`` functions through a multi-stage pipeline
that goes from Python source to GPU-executable code.

.. code-block:: text

   @pulse.kernel Python function
         |
         | bytecode tracing
         v
   Packed int64 buffer (numpy)
         |
         | zero-copy FFI (single call)
         v
   In-memory MLIR ModuleOp (Pulse dialect)
         |
         | C++ passes of passes= (DEFAULT_PASSES by default)
         v
   Optimized Pulse-dialect MLIR
         |
         | schedule of the policy (asap, alap, rcp, alap_rcp)
         v
   Scheduled Pulse-dialect MLIR  -->  CompiledKernel
         |
         | (optional) dialect lowering
         v
   Pulse -> QOp -> CuDensityMat -> LLVM
         |
         | JIT compile + execute
         v
   GPU simulation via cuDensityMat

Stage 1: Bytecode Tracing
--------------------------

The ``@pulse.kernel`` decorator captures the decorated function's CPython
bytecode. When a kernel compiles, a ``PackedIRBuilder`` traces
the bytecode and writes each operation into a flat ``numpy.ndarray``
of ``int64`` values. This packed buffer encodes operation types,
arguments, and waveform attributes in a compact binary format.

Stage 2: FFI to C++
--------------------

The packed buffer is sent to the C++ ``PulseModuleBuilder`` in a single
zero-copy FFI call via nanobind. The builder iterates the buffer and
constructs typed MLIR operations (``pulse.drive``, ``pulse.gaussian``,
etc.) directly into an ``mlir::ModuleOp``.

Stage 3: MLIR Passes
---------------------

The in-memory MLIR module is then run through the C++ passes named in
``passes=`` via ``mlir::PassManager``; pattern passes in a row run as one
rewrite. See :doc:`mlir_passes` for details on each pass.

Stage 4: Scheduling
--------------------

The schedule pass of the policy (``pulse-schedule-alap`` by default)
assigns ``start_cycles`` and ``duration_cycles`` attributes to every timed
operation, respecting data dependencies, sync constraints, and operation
durations; a module that declares classical calls is then checked with
``pulse-classical-check``. A pulse waits for
its line and for its tone: pulses that share a tone play in the order the
tone is threaded through them.

``float`` parameters stay function arguments of the scheduled module. A call
with values binds them on a copy, folds it, reruns ``fold_phases`` and
``fuse_pulses`` where the values open them, and schedules it again.

Stage 5: Lowering (Optional)
-----------------------------

For GPU execution, three dialect conversion passes lower the IR:

1. **Pulse -> QOp**: Converts pulse operations to quantum operator
   algebra (Hamiltonians, Lindbladians)
2. **QOp -> CuDensityMat**: Maps operators to cuDensityMat API calls
3. **CuDensityMat -> LLVM**: Final lowering to LLVM IR with runtime
   library calls

``mlir(level)`` on a scheduled kernel gives the program's text at the Pulse
level or at any of the three levels below it. This block lowers a small kernel
for one qudit and prints each level:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def rabi(qubit):
       drive_line = get_drive_line(qubit)
       drive(drive_line, gaussian(20 * pulse.ns, 0.3, 5 * pulse.ns))

   qs = pulse.quantum_system("transmon_surface17")
   scheduled = rabi.lower(qs.qudits[0])
   print(scheduled.mlir("pulse"))  # the scheduled Pulse dialect
   print(scheduled.mlir("qop"))    # quantum operators: the Hamiltonian and dissipators
   print(scheduled.mlir("cudm"))   # cuDensityMat operators and evolution
   print(scheduled.mlir("llvm"))   # the LLVM dialect, ready to compile

The execution lowering supports transmon models with any number of levels in
a rotating frame. A program that measures is lowered one interval at a time
along a measurement-outcome path, and the intervals are joined by projecting
the state. Systems without a simulation model (a quantum system whose ``model`` is
``None`` or has no simulator) and unspecialized waveforms are rejected
instead of silently approximated. See
:doc:`../user_guide/gpu_execution` for the complete model and limitations.

:example:`09_compilation_pipeline.py` walks through ``pulse.compile()`` and
``lower()``, and :example:`transmon/transmon_surface17_full_pipeline.py`
prints each stage of the lowering: the Pulse MLIR, the QOp form and the LLVM
IR.
