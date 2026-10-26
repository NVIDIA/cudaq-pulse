..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

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

.. :spellcheck-disable:

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

.. :spellcheck-enable:

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

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def rabi_oscillation(qubit, amplitude: float):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(64, amplitude, 16.0), tone)

   dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
   lowered = rabi_oscillation.lower(dev.qudit_handles[0], 0.5)
   result = pulse.simulate(lowered)    # NumPy reference, no GPU needed
   print(f"P(1) = {result.density_matrix[1, 1].real:.3f}")   # P(1) = 0.973
   print(lowered.mlir())
   print(f"Compiled in {lowered.metrics.total_ms:.1f} ms")

.. :spellcheck-enable:

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
4. **Lower and simulate** -- ``pulse.simulate(lowered)`` evolves the
   compiled module with the NumPy reference, and
   ``pulse.simulate(lowered, backend="gpu")`` on the GPU, through the native
   Pulse -> QOp -> CuDensityMat -> LLVM lowering (``lowered.mlir("llvm")``).

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

.. :spellcheck-disable:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cudaq-pulse.git
   cd cudaq-pulse
   pip install --group test        # pytest, hypothesis, lit (pip 25.1+)

   pytest tests -m "not gpu"       # any installation
   pytest tests                    # GPU installations: also runs the GPU suite

.. :spellcheck-enable:

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

.. :spellcheck-disable:

.. code-block:: bash

   uv sync
   python3 tools/fetch-mlir-toolchain.py   # optional; configuring fetches it too
   cmake -S . -B build/dev -G Ninja \
     -DCUDAQ_PULSE_BUILD_TESTS=ON \
     -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
   cmake --build build/dev --target check-pulse

.. :spellcheck-enable:

To build this documentation, configure with ``-DCUDAQ_PULSE_BUILD_DOCS=ON``
and build the ``pulse-docs`` target; the HTML is written to
``build/dev/docs/html``.
