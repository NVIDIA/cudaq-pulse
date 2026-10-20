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
needed.

.. :spellcheck-disable:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate

   # Compiler only (Linux and macOS)
   pip install cudaq-pulse

   # Compiler plus GPU simulation (Linux), with CUDA 12 or CUDA 13 libraries
   pip install "cudaq-pulse[cu12]"
   pip install "cudaq-pulse[cu13]"

   # Report what was installed and whether it works
   python -m cudaq_pulse.doctor

.. :spellcheck-enable:

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
     - Compiler; GPU execution with ``[cu12]`` or ``[cu13]``
   * - macOS 11 or newer, arm64
     - 3.12+
     - Compiler only

GPU execution additionally requires an NVIDIA GPU with compute capability 7.0
or newer and a Linux driver of at least 525.60.13 for ``[cu12]`` or 580 for
``[cu13]``. The driver is the only system requirement: no CUDA toolkit or
cuQuantum SDK installation is needed. See :doc:`user_guide/gpu_execution`.

Every installation can trace, compile, transform, and inspect kernels, and
ships the ``cudaq-pulse-opt`` MLIR driver. Only execution needs the GPU, and
it never degrades silently; see :ref:`gpu-errors`. If something does not
work, run ``python -m cudaq_pulse.doctor`` first.

Hello World
-----------

Define a pulse kernel, compile it, and inspect the generated MLIR:

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def rabi_oscillation(qubit):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(64, 0.5, 16.0), tone)

   compiled_kernel = pulse.compile(rabi_oscillation, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5.0e9})
   print(compiled_kernel.mlir)
   print(f"Compiled in {compiled_kernel.metrics.total_ms:.1f} ms")

.. :spellcheck-enable:

The Compiler Pipeline
---------------------

cudaq-pulse is a Python-first compiler pipeline with four stages:

1. **Write a kernel in Python** -- the ``@pulse.kernel`` DSL
   (``get_drive_line``, ``drive``, ``gaussian``, ``wait``, ``sync``, ...).
2. **Compile to MLIR** -- ``pulse.compile()`` returns a ``CompiledKernel``
   whose ``.mlir`` is the scheduled Pulse dialect. See :doc:`user_guide/compilation`.
3. **Write transform passes in Python and apply them** -- passes are plain
   ``Program -> Program`` functions; compose the built-ins or author your own.
   See :doc:`user_guide/passes`.
4. **Emit** -- lower the transformed program back to MLIR with
   ``program_to_pulse_mlir``.

GPU simulation via NVIDIA cuDensityMat is a preview capability; see
:doc:`user_guide/gpu_execution`.

IDE Setup
---------

For the best experience, add a ``pyrightconfig.json`` to your project root:

.. :spellcheck-disable:

.. code-block:: json

   {
       "reportUndefinedVariable": "warning"
   }

.. :spellcheck-enable:

This downgrades bare-name diagnostics inside ``@pulse.kernel`` functions
from errors to warnings. See :doc:`user_guide/kernels` for details.

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
