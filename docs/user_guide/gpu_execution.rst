..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

GPU Execution
=============

.. warning::

   GPU execution is a **research preview**. It is not production software and
   carries no API, numerical, compatibility, or stability guarantee.

cudaq-pulse contains an experimental lowering and runtime path intended to
connect compiled kernels to NVIDIA cuDensityMat. This path remains active
research and should not be treated as a production simulator.

Installation
------------

.. :spellcheck-disable:

.. code-block:: bash

   pip install "cudaq-pulse[cu12]"   # CUDA 12 libraries
   pip install "cudaq-pulse[cu13]"   # CUDA 13 libraries

.. :spellcheck-enable:

The Linux wheels include a cuDensityMat runtime library for each of CUDA 12
and CUDA 13. The ``cu12`` and ``cu13`` extras install cuDensityMat and the
CUDA libraries for that major from PyPI; ``gpu`` is an alias for ``cu12``.
The two extras write the same files, so an environment may contain only one
of them. macOS wheels are compiler-only.

Execution needs an NVIDIA GPU with compute capability 7.0 or newer and a
Linux driver new enough for the installed CUDA major. No CUDA toolkit is
used.

.. list-table:: Driver requirements
   :header-rows: 1

   * - Extra
     - Minimum NVIDIA driver
   * - ``[cu12]`` (or ``[gpu]``)
     - 525.60.13
   * - ``[cu13]``
     - 580

cudaq-pulse picks the runtime that matches the installed extra. To choose it
explicitly, set ``CUDAQ_PULSE_CUDA`` to ``12`` or ``13``. ``CUDM_RUNTIME_LIB``
points at a specific runtime library instead, for development builds.

Troubleshooting
---------------

Run the diagnostics first:

.. :spellcheck-disable:

.. code-block:: bash

   python -m cudaq_pulse.doctor

.. :spellcheck-enable:

It prints the version, the wheel platform, whether the GPU runtime is built
in, which CUDA extra is installed, each library it loads with its path or the
load error, and the driver version and device count. It exits non-zero when
GPU execution does not work; ``--allow-no-gpu`` accepts a working
compiler-only installation.

From Python, the same check is:

.. :spellcheck-disable:

.. code-block:: python

   from cudaq_pulse.runtime.gpu import require_gpu_runtime

   require_gpu_runtime()   # returns the loaded runtime or raises

.. :spellcheck-enable:

Source builds enable the runtime with ``-DCUDAQ_PULSE_ENABLE_GPU=ON``, which
makes CUDA and cuDensityMat required: configuration fails instead of
producing a compiler-only build. See ``CONTRIBUTING.md`` for the commands.

.. _gpu-errors:

GPU Runtime Errors
------------------

Every execution path calls ``require_gpu_runtime()`` first. It either returns
a runtime backed by a usable device or raises one of three errors, all
subclasses of ``cudaq_pulse.runtime.gpu.GpuRuntimeError``:

``GpuSupportNotBuiltError``
   This installation was built without the GPU runtime: a macOS wheel or a
   source build configured without ``CUDAQ_PULSE_ENABLE_GPU``. Reinstall a
   Linux wheel, or rebuild with GPU support.

``GpuRuntimeLoadError``
   The runtime is built in but its libraries failed to load, usually because
   no CUDA extra is installed or both ``cu12`` and ``cu13`` are. The message
   names the library that failed and any CUDA libraries that are not
   installed as Python packages.

``GpuDeviceUnavailableError``
   The libraries loaded, but no NVIDIA driver or device is available, or the
   driver is older than the installed CUDA major requires. The message names
   the driver version needed.

Compilation, transformation, and IR inspection never require the runtime, so
they work in every installation.

Pipeline Overview
-----------------

The target-aware public execution path is:

1. Call a ``@pulse.kernel`` to produce traced pulse IR.
2. ``pulse.evolve(..., target=...)`` verifies, optimizes, and schedules it.
3. The native pipeline lowers Pulse -> QOp -> CuDensityMat -> LLVM IR.
4. MLIR's ExecutionEngine compiles the LLVM IR in process and executes it
   against the cuDensityMat runtime library; no external compiler or LLVM
   tool is involved.

``pulse.compile()`` also returns a ``CompiledKernel`` whose
``lower_to_llvm()`` and ``run()`` methods expose the native pipeline directly.
That path has no ``Target`` parameter and therefore represents unitary,
two-level execution with pulse amplitudes already expressed in radians per
nanosecond. Use ``pulse.evolve`` when target calibration, coupling, or T1/T2
metadata matters.

MLIR Lowering
-------------

To inspect the lowered LLVM IR:

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   compiled_kernel = pulse.compile(my_kernel, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5e9})
   llvm_ir = compiled_kernel.lower_to_llvm()
   print(llvm_ir)

.. :spellcheck-enable:

The lowering passes through three dialect conversions:

**Pulse -> QOp**
   Converts supported drive operations and waveforms into backend-agnostic
   Hamiltonian and Lindblad operator algebra. A drive in a retained loop
   becomes one pulse-train coefficient. A program that measures is lowered
   one segment at a time along a chosen outcome for each measurement; see
   :ref:`measurement-simulation`.

**QOp -> CuDensityMat**
   Maps operator algebra to cuDensityMat API calls (state creation,
   operator construction, time evolution).

**CuDensityMat -> LLVM**
   Lowers cuDensityMat operations to LLVM IR with runtime library calls.

GPU Simulation
--------------

To execute on a GPU (requires cuDensityMat runtime):

.. :spellcheck-disable:

.. code-block:: python

   import math
   import cudaq_pulse as pulse
   from cudaq_pulse.targets import Qubit, Target

   target = Target(
       name="one-qubit-demo",
       qubits={
           0: Qubit(
               index=0,
               frequency_hz=5.0e9,
               anharmonicity_hz=-200.0e6,
               t1_us=50.0,
               t2_star_us=30.0,
               drive_params={"amplitude_scale_rad_per_ns": 1.0},
           )
       },
   )

   @pulse.kernel
   def x_gate(qubit):
       drive_line, tone = get_drive_line(qubit)
       # 40 virtual units at 2 GHz is 20 ns.
       drive(drive_line, square(40, math.pi / 20.0), tone)

   ir = x_gate(pulse.qudit_ref())
   result = pulse.evolve(
       ir,
       target=target,
       t_start=0.0,
       t_end=20.0,
       num_steps=200,
       integrator="rk4",
   )
   print(result.final_state)

.. :spellcheck-enable:

``t_start`` and ``t_end`` are in nanoseconds. The ``integrator`` argument
selects the time-evolution scheme used by the cuDensityMat runtime path:

- ``rk1``, ``rk2``, ``rk4`` -- fixed-step explicit Runge-Kutta methods.
- ``magnus`` -- a fixed-step Magnus expansion evaluated as a truncated Taylor
  series of the propagator; preserves structure well for smoothly varying
  drives.
- ``crank_nicolson`` -- a fixed-step implicit predictor-corrector scheme that
  is more stable for stiff dynamics.

All five schemes reuse the same boundary-safe sampling that keeps piecewise
constant pulse segments from being sampled across a discontinuity.

Physical Model
--------------

The current native execution lowering is a two-level transmon model in one
rotating frame per qubit. Drive envelopes are converted to X/Y Hamiltonian
coefficients in radians per nanosecond; tone frequency and phase determine the
rotating-frame detuning and quadrature. A target may supply
``amplitude_scale_rad_per_ns`` explicitly, or pulse infers a scale from its
calibrated Gaussian/DRAG pi-pulse parameters when available.

Target relaxation and dephasing data produce Lindblad collapse operators.
Coupling edges are modeled as always-on XX terms and residual crosstalk as
always-on ZZ terms. ``anharmonicity_hz`` is retained as calibration metadata
but is not part of this two-level model; leakage requires a future multilevel
lowering.

Unsupported Execution Features
------------------------------

The execution lowering fails explicitly for readout/acquisition, observables,
neutral-atom targets, multilevel models, unspecialized numeric parameters,
arbitrary Python waveform callbacks, and waveform-algebra nodes. Built-in
waveforms and ``custom_samples`` are supported. The callback runtime currently
supports at most 128 drive operations per compiled module.

The GPU test suite (``pytest tests -m gpu``, or the ``check-pulse-gpu`` target
in a source build) covers single-qubit drives, T1 decay, idle evolution,
two-qubit coupling, frame and I/Q modulation, an eight-qubit ladder register,
closed-system physics validation, quantum-algorithm building blocks,
integrator parity across ``rk4``/``magnus``/``crank_nicolson``, and the public
compile/JIT path. Source builds also run a cuDensityMat link check as part of
``check-pulse``. Selecting the GPU tests where the runtime is unusable is an
error, not a skip. These tests provide research regression coverage; they do
not establish production numerical accuracy.

The ``run()`` method and returned state representation are experimental and
will evolve with the lowering and runtime implementation.

.. _measurement-simulation:

Measurement and Feedback
------------------------

``pulse.simulate()`` evolves a program that measures, and branches on the
results, in segments between measurements. A readout measures its qudit in
the computational basis when it completes: every level ``k`` is a separate
outcome, and each continues along its own path, on which a branch on that
measurement takes the arm the outcome selects. The unnormalized states of
all paths sum to the ensemble

.. math::

   \rho' = \sum_m U_m P_m \rho P_m^\dagger U_m^\dagger

with no sampling error. Requesting ``shots`` also samples per-shot records
of the outcomes from the same branch probabilities.

.. :spellcheck-disable:

.. code-block:: python

   result = pulse.simulate(active_reset(pulse.qudit_ref()), target=target,
                           shots=1000, backend="cpu")
   result.density_matrix   # ensemble state at the end
   result.branches         # [(outcomes, probability), ...]
   result.records          # one row of outcomes per shot

.. :spellcheck-enable:

``backend="cpu"`` integrates the same lowered operator with a NumPy RK4
reference; ``backend="gpu"`` evolves each segment with cuDensityMat, starting
from the projected state. ``levels`` gives qudits more than two levels; a
transmon target then includes its anharmonicity. ``pulse.evolve()`` uses
the GPU path for programs that measure and returns the ensemble density
matrix.
