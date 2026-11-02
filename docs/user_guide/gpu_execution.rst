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

.. code-block:: bash

   RELEASE=https://github.com/NVIDIA/cudaq-pulse/releases/expanded_assets/v0.1.0
   pip install "cudaq-pulse[cu12]==0.1.0" --find-links "$RELEASE"   # CUDA 12 libraries
   pip install "cudaq-pulse[cu13]==0.1.0" --find-links "$RELEASE"   # CUDA 13 libraries

The Linux wheels include a cuDensityMat runtime library for each of CUDA 12
and CUDA 13. The ``cu12`` and ``cu13`` extras install cuDensityMat and the
CUDA libraries for that major from PyPI; ``gpu`` is an alias for ``cu12``.
The two extras write the same files, so an environment may contain only one
of them. macOS wheels have no GPU runtime; they compile and simulate on the
CPU (``backend="cpu"``).

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

.. code-block:: bash

   python -m cudaq_pulse.doctor

It prints the version, the wheel platform, whether the GPU runtime is built
in, which CUDA extra is installed, each library it loads with its path or the
load error, and the driver version and device count. It exits non-zero when
GPU execution does not work; ``--allow-no-gpu`` accepts a working
compiler-only installation.

From Python, the same check is:

.. code-block:: python

   from cudaq_pulse.runtime import require_gpu_runtime

   require_gpu_runtime()   # returns the loaded runtime or raises

Source builds enable the runtime with ``-DCUDAQ_PULSE_ENABLE_GPU=ON``, which
makes CUDA and cuDensityMat required: configuration fails instead of
producing a compiler-only build. See ``CONTRIBUTING.md`` for the commands.

.. _gpu-errors:

GPU Runtime Errors
------------------

Every execution path calls ``require_gpu_runtime()`` first. It either returns
a runtime backed by a usable device or raises one of three errors, all
subclasses of ``cudaq_pulse.runtime.GpuRuntimeError``:

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

The public execution path is:

1. ``kernel.lower(*args)`` verifies, optimizes and schedules the kernel for
   the device of its qudit handles.
2. ``pulse.simulate(lowered, backend="gpu")`` annotates the module with the
   device's calibration (drive scales, couplings, crosstalk, T1/T2) and the
   native pipeline lowers it Pulse -> QOp -> CuDensityMat -> LLVM IR.
3. MLIR's ExecutionEngine compiles the LLVM IR in process and executes it
   against the cuDensityMat runtime library; no external compiler or LLVM
   tool is involved.

Float arguments stay symbolic through the Pulse -> QOp lowering; the GPU
runtime takes constants, so the GPU path substitutes the call's values into
a private copy of the module first.

MLIR Lowering
-------------

To inspect the lowered LLVM IR:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def my_kernel(qubit):
       drive_line = get_drive_line(qubit)
       duration = 20 * pulse.ns
       sigma = 5 * pulse.ns
       drive(drive_line, gaussian(duration, 0.3, sigma))

   dev = pulse.device("transmon_17q")
   llvm_ir = my_kernel.lower(dev.qudit_handles[0]).mlir("llvm")
   print(llvm_ir)

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

.. code-block:: python

   import math
   import cudaq_pulse as pulse
   from cudaq_pulse.targets import Qubit, TargetDescription

   target = TargetDescription(
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
   def x_gate(qubit, duration):
       drive_line = get_drive_line(qubit)
       drive(drive_line, square(duration, math.pi / 20.0))

   dev = pulse.device({"target": target})
   pi_time = 20 * pulse.ns
   lowered = x_gate.lower(dev.qudit_handles[0], pi_time)
   result = pulse.simulate(lowered, backend="gpu", t_end=20.0, dt_ns=0.1, integrator="rk4")
   print(result.density_matrix)

``t_end`` and ``dt_ns`` are in nanoseconds; ``t_end`` defaults to the end of
the schedule. ``dt_ns`` defaults to 0.05 ns, or to 20 steps per period of the
fastest frequency the simulation holds, rounded to a whole fraction of a
clock cycle: a coupling between the program's qudits turns at their
detuning (0.024 ns for a pair 2.1 GHz apart, where 0.05 ns drifts by about
3e-5 per microsecond), and a qudit's levels at the spread of their energies
in its frame (0.015 ns for a transmon with 6 levels and a 330 MHz
anharmonicity, whose levels spread over 3.3 GHz). A
``dt_ns`` you pass that takes fewer than 20 steps per period of the fastest
frequency the simulation holds warns with a ``cudaq_pulse.PulseWarning``
naming that frequency and a step that resolves it: with three or more levels,
a qudit's coherence of levels 0 and 2 turns at its anharmonicity, which a
0.5 ns step (10 steps per period at 200 MHz) damps by about 15 % over a
200 ns shaped pulse. A step so coarse that the integrator diverges, about two
steps per period or fewer (1 ns for 4 levels at a 184 MHz anharmonicity,
which spread over 552 MHz), raises a ``cudaq_pulse.ExecutionError`` naming
the same and a step that resolves it, rather than returning a state that is
not finite. The ``integrator`` argument selects the time-evolution scheme
used by the cuDensityMat runtime path:

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

The simulated space holds only the qudits the program uses: those it gets a
drive or readout line on, one mode each in increasing order of device index.
A pair at qudits 13 and 3 of the 17-qudit ``transmon_17q`` device is a 3 x 3 problem
with three levels, not a 3^14 one. Couplings and crosstalk between those
qudits keep their device values; the other qudits are left out, with their
couplings, as classical spectators in their ground state. To simulate an
idle qudit, get a line on it. ``result.qudits`` lists the device qudit of
each mode: ``result.qudits[i]`` has ``result.dims[i]`` levels and is digit
``i`` of the state index, least significant first. An ``initial_state``
uses the same layout, and measurements report device qudits.

The native execution lowering is a transmon model in one rotating frame per
qudit, with the levels the target's simulation model declares for it (three
for a ``TransmonModel``) unless ``levels`` sets them. Drive envelopes are converted to X/Y Hamiltonian
coefficients in radians per nanosecond; tone frequency and phase determine the
rotating-frame detuning and quadrature. A target may supply
``amplitude_scale_rad_per_ns`` explicitly, or pulse infers a scale from its
calibrated Gaussian/DRAG pi-pulse parameters when available.

Target relaxation and dephasing data produce Lindblad collapse operators.
A qudit whose ``thermal_population`` p is nonzero starts in its thermal
mixture (the excited level holds p, and higher levels follow the same
Boltzmann ratio p / (1 - p) per level) unless ``initial_state`` is given,
and relaxes towards it: down at (1 - p) / T1 and up at p / T1, so
populations relax at the measured 1 / T1. Qudits dephase at their Ramsey
T2*, or at their Hahn-echo T2 with ``simulate(..., dephasing="t2_echo")``.
A coupling edge is an exchange coupling g (a^dagger b + a b^dagger); in the
qudits' own frames it turns at their detuning, so detuned qudits exchange
little and coupled transmons with three or more levels acquire the
dispersive ZZ shift. Counter-rotating terms are dropped. A residual crosstalk
entry of rate zeta (E11 - E10 - E01 + E00 = zeta) adds zeta n_a n_b. With
three or more levels, a qudit's drive acts through
its ladder operator (X = a + a^dagger, Y = i(a^dagger - a)) and
``anharmonicity_hz`` adds (alpha / 2) n (n - 1), so leakage out of the qubit
subspace is modeled; with two levels the anharmonicity is calibration
metadata only.

Unsupported Execution Features
------------------------------

The execution lowering fails explicitly for observables, systems with
no simulation model, products of two float arguments, arbitrary Python waveform
callbacks, and waveform-algebra nodes. Built-in waveforms and
``custom_samples`` are supported. The callback runtime holds 1024
time-dependent coefficients per evolution: two per drive op (a drive
repeated by a loop needs only one pair) and four per coupling between
detuned qudits. A single evolution cannot contain a measurement: a
program that measures is lowered and evolved one segment at a time (see
:ref:`measurement-simulation`).

The GPU test suite (``pytest tests -m gpu``, or the ``check-pulse-gpu`` target
in a source build) covers single-qubit drives, T1 decay, idle evolution,
two-qubit coupling, frame and I/Q modulation, an eight-qubit ladder register,
closed-system physics validation, quantum-algorithm building blocks,
integrator parity across ``rk4``/``magnus``/``crank_nicolson``, and the public
lowering and JIT path. Source builds also run a cuDensityMat link check as part of
``check-pulse``. Selecting the GPU tests where the runtime is unusable is an
error, not a skip. These tests provide research regression coverage; they do
not establish production numerical accuracy.

The returned state representation is experimental and may change with the
lowering and runtime implementation.

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

.. code-block:: python

   @pulse.kernel
   def active_reset(qubit):
       drive_line = get_drive_line(qubit)
       readout_line = get_readout_line(qubit)
       readout_time = 500 * pulse.ns
       pi_time = 20 * pulse.ns
       result = readout(readout_line, square(readout_time, 0.05))
       sync(drive_line, readout_line)
       if result:
           drive(drive_line, square(pi_time, math.pi / 20.0))

   lowered = active_reset.lower(dev.qudit_handles[0])
   result = pulse.simulate(lowered, shots=1000, backend="cpu")
   result.density_matrix   # ensemble state at the end
   result.level_population(1)   # P(1) of the one qudit; level_population(1, qudit=q) for one of several
   result.level_populations()   # [P(0), P(1), P(2)] for a three-level qudit
   result.branches         # [Branch(levels=(0,), probability=0.5, calls=()), ...]
   result.records          # one row of outcomes per shot

``backend="cpu"`` integrates the same lowered operator with a NumPy RK4
reference; ``backend="gpu"`` evolves each segment with cuDensityMat, starting
from the projected state. By default each qudit has the levels its target's
model declares (three for a transmon); ``levels=2`` simulates qubits, an int
sets every qudit's levels and a dict by device qudit index sets those it
names. With three or more levels a transmon includes its anharmonicity.
