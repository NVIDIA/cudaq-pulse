..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

cudaq-pulse
===========

**cudaq-pulse** is a pulse-level MLIR dialect and programming model for
quantum control. Its Python DSL drives an MLIR-based compiler pipeline:

1. Write a pulse kernel with the ``@pulse.kernel`` DSL.
2. Call it, or compile it to Pulse-dialect MLIR with ``kernel.lower()`` or
   ``pulse.compile()``.
3. Inspect the schedule, or extend the pipeline with custom passes.
4. Simulate it with the NumPy reference on any machine, or on an NVIDIA GPU.

.. warning::

   CUDA-Q pulse is **research-preview software**, not production software or a
   product-supported CUDA-Q feature. APIs, dialects, runtime interfaces,
   numerical behavior, build options, and file layout may change incompatibly
   or be removed without notice. No stability, compatibility, performance, or
   production-readiness guarantee is provided. Expect the work to evolve as
   the research matures.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/kernels
   user_guide/operations
   user_guide/compilation
   user_guide/devices
   user_guide/custom_passes
   user_guide/editors
   user_guide/classical_calls
   user_guide/controller_backends

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/pipeline
   architecture/dialects
   architecture/mlir_passes

.. toctree::
   :maxdepth: 1
   :caption: Resources

   examples

.. toctree::
   :maxdepth: 1
   :caption: Preview / Experimental

   user_guide/gpu_execution


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
