..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Compilation
===========

``pulse.compile()`` is the single public entry point for compiling
a ``@pulse.kernel`` function into a scheduled, optimized MLIR module.

Basic Usage
-----------

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def my_kernel(qubit):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(40, 0.3, 10.0), tone)

   compiled_kernel = pulse.compile(my_kernel, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5.0e9})

.. :spellcheck-enable:

The ``compile()`` Function
--------------------------

.. autofunction:: cudaq_pulse.compile
   :noindex:

Parameters
~~~~~~~~~~

``kernel_fn``
   A ``@pulse.kernel``-decorated function.

``args``
   Positional arguments -- typically ``pulse.qudit_ref()`` objects matching
   the kernel's parameters.

``clock_ghz`` *(float, default 2.0)*
   System clock frequency in GHz. Determines the physical time
   corresponding to one clock cycle.

``qubit_freq_hz`` *(dict[int, float], optional)*
   Mapping from qubit index to qubit frequency in Hz. Used for
   rotating-frame calculations and scheduling.

``schedule`` *(str, default "alap")*
   Scheduling policy. Options: ``"asap"``, ``"alap"``, ``"rcp"``,
   ``"alap_rcp"``. ASAP and ALAP preserve data and physical-line
   dependencies, including the order of pulses that play on the same tone;
   RCP variants also enforce ``MachineModel`` resource limits.

``passes`` *(sequence of str, optional)*
   Optimization passes to run. Defaults to ``("verify", "virtual_z", "fusion")``.
   Pass an empty tuple ``()`` to skip all passes.

   Available passes: ``"verify"``, ``"canonicalize"``, ``"virtual_z"``,
   ``"fusion"``, ``"licm"``.

``machine`` *(MachineModel, optional)*
   Resource limits for the RCP schedules: ``pulse.MachineModel(
   max_concurrent_drives=4, max_concurrent_readouts=2, readout_latency_vtu=0,
   line_switch_penalty_vtu=0)``.

CompiledKernel
--------------

``pulse.compile()`` returns a ``CompiledKernel`` object:

.. autoclass:: cudaq_pulse.CompiledKernel
   :noindex:
   :members:
   :undoc-members:

Key properties and methods:

``.mlir``
   The Pulse-dialect MLIR text representation (lazily rendered).

``.module``
   The in-memory ``PulseModule`` (MLIR ``ModuleOp``).

``.metrics``
   A ``CompileMetrics`` dataclass with per-stage timing.

``.lower_to_llvm()``
   Run the full MLIR lowering pipeline (Pulse -> QOp -> CuDensityMat -> LLVM).

``.run()``
   JIT-compile and execute the unitary, two-level lowering on GPU via
   cuDensityMat. Use :func:`cudaq_pulse.evolve` for target-aware T1/T2,
   coupling, crosstalk, and drive-calibration metadata.

CompileMetrics
--------------

.. autoclass:: cudaq_pulse.CompileMetrics
   :noindex:
   :members:
   :undoc-members:

Fields (all in milliseconds):

- ``trace_ms`` -- bytecode tracing and packed buffer construction
- ``ffi_ms`` -- C++ MLIR module construction from packed buffer
- ``passes_ms`` -- optimization pass execution time
- ``schedule_ms`` -- scheduling pass time
- ``total_ms`` -- sum of all stages
- ``op_count`` -- number of MLIR operations in the module

Threads
-------

The native stages (module construction, passes, scheduling, specialization,
lowering and JIT compilation) release the GIL, so kernels can be compiled,
evaluated at different parameter values, and lowered from several Python
threads at once. Calls on one ``CompiledKernel`` are serialized; tracing a
kernel is Python and holds the GIL.

Long Sessions and Memory
------------------------

Compiled kernels share an MLIR context, which keeps every attribute it has
created (constants, qubit frequencies, timing) until it is freed. A context
is retired, and freed once its last kernel is gone, after it has built a
number of modules or a number of packed program words, whichever comes
first. The limits are read from the environment each time a kernel is
compiled:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Variable
     - Default
     - Meaning
   * - ``CUDAQ_PULSE_CONTEXT_MAX_MODULES``
     - 256
     - Modules built in one context before it is retired.
   * - ``CUDAQ_PULSE_CONTEXT_MAX_WORDS``
     - 1048576
     - Packed program words (8 bytes each) built in one context before it
       is retired.

Each must be a whole number from 1 to 2\ :sup:`40`; any other value makes
``compile()`` raise ``ValueError`` naming the variable. Lower values bound
the memory a long session keeps (a loop that recompiles a large program
with new qubit frequencies, for example) at the cost of creating contexts
more often, about 0.4 ms each. ``python -m cudaq_pulse.doctor`` prints the
limits in effect.

Customizing the Pass Pipeline
-----------------------------

To run a custom set of passes:

.. :spellcheck-disable:

.. code-block:: python

   compiled_kernel = pulse.compile(
       my_kernel,
       [pulse.qudit_ref()],
       qubit_freq_hz={0: 5e9},
       passes=("verify", "canonicalize", "virtual_z", "fusion", "licm"),
   )

.. :spellcheck-enable:

To skip passes entirely (useful for benchmarking):

.. :spellcheck-disable:

.. code-block:: python

   compiled_kernel = pulse.compile(
       my_kernel,
       [pulse.qudit_ref()],
       qubit_freq_hz={0: 5e9},
       passes=(),
   )

.. :spellcheck-enable:
