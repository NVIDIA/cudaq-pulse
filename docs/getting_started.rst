..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Getting Started
===============

.. warning::

   CUDA-Q Pulse is **research-preview software**, not production software or a
   product-supported CUDA-Q feature. APIs, IR, runtime behavior, build options,
   and numerical behavior may change incompatibly or be removed without
   notice. No stability or production-readiness guarantee is provided.

Installation
------------

CUDA-Q Pulse requires Python 3.12 or newer and does not depend on CUDA-Q. Its
wheels contain a statically linked MLIR, so ``pip`` is the only installer
needed. Install it from PyPI; ``pip`` picks the wheel for your platform:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

   # Compiler and CPU simulation (Linux and macOS)
   pip install cudaq-pulse

For GPU simulation on Linux, also install one CUDA extra, never both:

.. code-block:: bash

   # CUDA 12 libraries: NVIDIA driver 525.60.13 or newer
   pip install "cudaq-pulse[cu12]"

   # CUDA 13 libraries instead: NVIDIA driver 580 or newer
   # pip install "cudaq-pulse[cu13]"

Then check what was installed and whether it works:

.. code-block:: bash

   python -m cudaq_pulse.doctor

``cudaq-pulse-opt``, the MLIR driver, comes along automatically: it is in
``cudaq-pulse-tools``, a wheel of its own, one per platform for every Python,
which ``cudaq-pulse`` requires at the same version. The other dependencies,
CUDA libraries included, come from PyPI too. Each release on the `GitHub
releases page <https://github.com/NVIDIA/cudaq-pulse/releases>`_ also carries
the wheels and the source distribution.

``[gpu]`` is an alias for ``[cu12]``. Each CUDA extra installs cuDensityMat
and the CUDA runtime libraries it needs from PyPI; the two extras write the
same files, so install only one of them per environment.

.. list-table:: Supported platforms
   :header-rows: 1

   * - Platform
     - Python
     - Capabilities
   * - Linux x86_64 and aarch64, glibc 2.28 or newer
     - 3.12+
     - Compiler and CPU simulation; GPU simulation with ``[cu12]`` or
       ``[cu13]``
   * - macOS 11 or newer, arm64
     - 3.12+
     - Compiler and CPU simulation

GPU simulation additionally requires an NVIDIA GPU with compute capability 7.0
or newer and a Linux driver of at least 525.60.13 for ``[cu12]`` or 580 for
``[cu13]``. The driver is the only system requirement: no CUDA toolkit or
cuQuantum SDK installation is needed. See :doc:`user_guide/gpu_execution`.

Every installation can trace, compile, transform, inspect and simulate
kernels, the last with the NumPy reference, and has the
``cudaq-pulse-opt`` MLIR driver on ``PATH``. Only GPU simulation needs the GPU, and it
never degrades silently; see :ref:`gpu-errors`. If something does not
work, run ``python -m cudaq_pulse.doctor`` first. Without a CUDA extra it
ends with ``Result: OK, compilation and CPU simulation work; GPU simulation
is not set up.`` and names the extra and driver that enable GPU simulation.

The doctor takes at most one of ``--allow-no-gpu``, ``--expect-no-device``,
``--setup-editor`` or ``--system``, and none is needed for the check above.
``--json`` goes with ``--system``; ``--project`` and ``--yes`` go with
``--setup-editor``.

.. list-table:: Doctor options
   :header-rows: 1

   * - Option
     - What it does
     - See
   * - ``--allow-no-gpu``
     - Succeed when compilation and CPU simulation work, even if GPU
       simulation is set up but unavailable because the build has no GPU
       runtime or no usable device or driver exists.
     - :doc:`user_guide/gpu_execution`
   * - ``--expect-no-device``
     - Succeed only if every GPU library loads and the sole problem is
       that no usable GPU device exists.
     - :doc:`user_guide/gpu_execution`
   * - ``--setup-editor``
     - Show the mypy and ruff configuration the kernel language needs,
       and write it when confirmed (pyright-based editors need none).
     - :doc:`user_guide/editors`
   * - ``--system FILE``
     - Check the system file ``FILE`` as ``pulse.QuantumSystem.load(FILE)``
       does: every problem, its controllers and links, and whether their
       backends are installed.
     - :doc:`user_guide/quantum_systems`
   * - ``--json``
     - With ``--system``, print the report as JSON.
     - :doc:`user_guide/quantum_systems`
   * - ``--project PROJECT``
     - Project directory for ``--setup-editor`` (default: the current
       directory).
     - :doc:`user_guide/editors`
   * - ``--yes``
     - With ``--setup-editor``, write without asking.
     - :doc:`user_guide/editors`

Hello World
-----------

Define a pulse kernel, run it by calling it, and inspect the generated MLIR:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def rabi_oscillation(qubit, amplitude: float, duration, sigma):
       drive_line = get_drive_line(qubit)
       readout_line = get_readout_line(qubit)
       drive(drive_line, gaussian(duration, amplitude, sigma))
       sync(drive_line, readout_line)
       return readout(readout_line, readout_line.get_calibration(pulse.gates.READOUT))

   qs = pulse.quantum_system("transmon_surface17")
   qubit = qs.qudits[0]
   duration = 32 * pulse.ns
   sigma = 8 * pulse.ns
   # A call compiles the kernel and simulates one shot on the CPU, no GPU needed.
   outcome = rabi_oscillation(qubit, 0.5, duration, sigma)
   print(outcome)   # the readout's outcome: 1 in about 96 shots of 100
   scheduled = rabi_oscillation.lower(qubit, 0.5, duration, sigma)
   print(scheduled.mlir())
   print(f"Compiled in {scheduled.metrics.total_ms:.1f} ms")

A call returns what the kernel returns, here the outcome of its readout. The
pulse lasts 32 ns, 64 cycles of the quantum system's 2 GHz clock. A duration is
an ``int`` number of cycles or a time with units, and the kernel takes it as
an argument, like the pulse's other values; see :ref:`durations`.

For populations, many shots or other options, ``pulse.simulate`` takes the
kernel and the same arguments and returns the result:

.. code-block:: python

   result = pulse.simulate(rabi_oscillation, qubit, 0.5, duration, sigma)
   print(f"P(1) = {result.level_population(1):.3f}")   # P(1) = 0.962

   shots = pulse.simulate(rabi_oscillation, qubit, 0.5, duration, sigma, shots_count=1000, seed=7)
   print(int((shots.records == 1).sum()), "of 1000 shots read 1")   # 955 of 1000 shots read 1

The Compiler Pipeline
---------------------

CUDA-Q Pulse is a Python-first compiler pipeline with four stages:

1. **Write a kernel in Python** -- the ``@pulse.kernel`` DSL
   (``get_drive_line``, ``drive``, ``gaussian``, ``wait``, ``sync``, ...).
2. **Compile to MLIR** -- calling a kernel compiles it on first use and runs
   it; ``kernel.lower(*args)`` returns the compiled module, whose
   ``.mlir()`` is the scheduled Pulse dialect, and ``pulse.compile(kernel)``
   compiles ahead of the call. See :doc:`user_guide/compilation`.
3. **Extend the pipeline** -- check a compiled module's ``schedule`` in
   Python, or load your own MLIR passes into ``cudaq-pulse-opt`` as plugins.
   See :doc:`user_guide/custom_passes`.
4. **Lower and simulate** -- ``pulse.simulate(kernel, *args)`` compiles a
   call and evolves it with the NumPy reference, and
   ``pulse.simulate(kernel, *args, backend="gpu")`` on the GPU, through the native
   Pulse -> QOp -> CuDensityMat -> LLVM lowering (``scheduled.mlir("llvm")``).
   A program compiled already is simulated the same way without compiling
   it again: ``pulse.simulate(program, *values)`` binds the values of its
   run-time float parameters as ``program.bind(values)`` does.

For how the compiler works inside, from bytecode tracing to scheduling and
lowering, see :doc:`architecture/pipeline`.

GPU simulation via NVIDIA cuDensityMat is a preview capability; see
:doc:`user_guide/gpu_execution`.

IDE Setup
---------

None is needed. Pylance (in VS Code and VS Code-based editors), pyright and
basedpyright know the operations used by bare name inside ``@pulse.kernel``
functions, with
completion, hover documentation and checked arguments, as soon as
cudaq-pulse is installed. See :doc:`user_guide/editors`.

Running the Tests
-----------------

The test suite lives in the source repository rather than the wheel, and it
imports whichever ``cudaq_pulse`` is installed, so it checks the installation
you actually have:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cudaq-pulse.git
   cd cudaq-pulse
   pip install --group test        # pytest, hypothesis, lit (pip 25.1+)

   pytest tests -m "not gpu"       # any installation
   pytest tests                    # GPU installations: also runs the GPU suite

The GPU tests exercise single-qubit drive evolution, T1 decay, two-qubit XX
coupling, integrator parity, and the public compile/JIT path. Selecting them
on an installation that cannot run them stops pytest immediately and reports
which of the three GPU errors applies, rather than skipping them. They are
research regression tests, not a production numerical-accuracy
qualification.

Building from Source
--------------------

Contributors build with CMake against a pinned, checksum-verified MLIR
toolchain that a script downloads; nothing else is compiled from source. The
complete workflow, including GPU builds, linting, benchmarks, and building
this documentation, is described in ``CONTRIBUTING.md`` at the repository
root. In short:

.. code-block:: bash

   uv sync
   python3 tools/fetch-mlir-toolchain.py   # optional; configuring fetches it too
   cmake --preset dev
   cmake --build --preset check        # builds, then runs check-pulse

To build this documentation, run ``cmake --preset docs`` and
``cmake --build --preset docs``; the HTML is written to
``build/docs/docs/html``.
