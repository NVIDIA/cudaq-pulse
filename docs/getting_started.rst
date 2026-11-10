..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Getting Started
===============

.. warning::

   CUDA-Q pulse is **research-preview software**, not production software or a
   product-supported CUDA-Q feature. APIs, IR, runtime behavior, build options,
   and numerical behavior may change incompatibly or be removed without
   notice. No stability or production-readiness guarantee is provided.

Installation
------------

cudaq-pulse requires Python 3.12 or newer and does not depend on CUDA-Q. Its
wheels contain a statically linked MLIR, so ``pip`` is the only installer
needed. Releases are published on the `GitHub releases page
<https://github.com/NVIDIA/cudaq-pulse/releases>`_, not on PyPI; ``pip``
picks the wheel for your platform from the release:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   RELEASE=https://github.com/NVIDIA/cudaq-pulse/releases/expanded_assets/v0.1.0

   # Compiler and CPU simulation (Linux and macOS)
   pip install "cudaq-pulse==0.1.0" --find-links "$RELEASE"

   # Plus GPU simulation (Linux), with CUDA 12 or CUDA 13 libraries
   pip install "cudaq-pulse[cu12]==0.1.0" --find-links "$RELEASE"
   pip install "cudaq-pulse[cu13]==0.1.0" --find-links "$RELEASE"

   # Report what was installed and whether it works
   python -m cudaq_pulse.doctor

Keep the version pin: it stops ``pip`` from choosing a different package of
the same name from PyPI. The dependencies, CUDA libraries included, come from
PyPI. A wheel downloaded from the release installs the same way, as
``pip install ./cudaq_pulse-0.1.0-<platform>.whl`` (or
``"./cudaq_pulse-0.1.0-<platform>.whl[cu12]"``).

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
kernels, the last with the NumPy reference, and ships the
``cudaq-pulse-opt`` MLIR driver. Only GPU simulation needs the GPU, and it
never degrades silently; see :ref:`gpu-errors`. If something does not
work, run ``python -m cudaq_pulse.doctor`` first.

Hello World
-----------

Define a pulse kernel, compile it, simulate it on the CPU, and inspect the
generated MLIR:

.. code-block:: python

   import cudaq_pulse as pulse

   cal = {"duration": 32 * pulse.ns, "sigma": 8 * pulse.ns}

   @pulse.kernel
   def rabi_oscillation(qubit, amplitude: float):
       drive_line = get_drive_line(qubit)
       drive(drive_line, gaussian(cal["duration"], amplitude, cal["sigma"]))

   dev = pulse.device("transmon_17q")
   qubit = dev.qudit_handles[0]
   result = pulse.simulate(rabi_oscillation, qubit, 0.5)    # NumPy reference, no GPU needed
   print(f"P(1) = {result.level_population(1):.3f}")   # P(1) = 0.972
   lowered = rabi_oscillation.lower(qubit, 0.5)
   print(lowered.mlir())
   print(f"Compiled in {lowered.metrics.total_ms:.1f} ms")

The pulse lasts 32 ns, 64 cycles of the device's 2 GHz clock. A duration is
an ``int`` number of cycles or a time with units; see :ref:`durations`.

The Compiler Pipeline
---------------------

cudaq-pulse is a Python-first compiler pipeline with four stages:

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
   Pulse -> QOp -> CuDensityMat -> LLVM lowering (``lowered.mlir("llvm")``).
   A program compiled already is simulated the same way without compiling
   it again: ``pulse.simulate(program, *values)`` binds the values of its
   run-time float parameters as ``program.bind(values)`` does.

GPU simulation via NVIDIA cuDensityMat is a preview capability; see
:doc:`user_guide/gpu_execution`.

IDE Setup
---------

None is needed. Pylance, Cursor, pyright and basedpyright know the
operations used by bare name inside ``@pulse.kernel`` functions, with
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
