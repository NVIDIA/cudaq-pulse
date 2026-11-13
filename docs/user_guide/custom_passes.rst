..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Writing Custom Passes
=====================

Compiling a kernel runs the built-in pipeline (the passes of ``passes=`` and
a schedule) on an in-memory Pulse-dialect MLIR module. You can extend it in
two ways, each shown by a runnable example in ``examples/custom_passes/``.

Checks in Python
----------------

The module of a kernel call exposes its schedule as a list of
:class:`~cudaq_pulse.compiler.ScheduledEvent` records (kind, line, qubit, start and
duration), so a pass that inspects the schedule is a plain Python function:

.. code-block:: python

   import cudaq_pulse as pulse

   def total_drive_time(compiled: pulse.Lowered) -> int:
       return sum(e.end - e.start for e in compiled.schedule if e.kind == "drive")

``examples/custom_passes/python_pass.py`` enforces a concurrent-drive limit
the scheduler does not know about and shows the ``rcp`` schedule satisfying
it.

Transforms in C++
-----------------

Passes that rewrite the IR are MLIR passes. ``cudaq-pulse-opt`` loads them
from a plugin library with ``--load-pass-plugin``:

.. code-block:: bash

   cudaq-pulse-opt --load-pass-plugin=PulseExamplePlugin.so \
       --pass-pipeline='builtin.module(pulse-example-drive-count)' kernel.mlir

``examples/custom_passes/cpp/DriveCount.cpp`` is a complete plugin: a pass
class plus the ``mlirGetPassPluginInfo`` entry point that registers it. Its
``CMakeLists.txt`` builds it against the same MLIR toolchain as cudaq-pulse,
either as part of the cudaq-pulse build or on its own. Built on its own, it
loads into the ``cudaq-pulse-opt`` a wheel installs, as
``tools/ci/opt-plugin-test.sh`` checks; the wheels carry no MLIR headers, so
build plugins against the toolchain ``tools/fetch-mlir-toolchain.py``
fetches. Get a kernel's MLIR from ``Lowered.mlir()`` to run plugin passes on
it.
