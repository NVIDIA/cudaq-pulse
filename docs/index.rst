..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

cudaq-pulse
===========

**cudaq-pulse** puts the pulses, the measurements and the classical calls
between them in one Python program, compiled to open MLIR, then simulated,
emulated, or exported to your controller:

1. Write a pulse kernel with the ``@pulse.kernel`` DSL, calling classical
   functions (``@pulse.classical``) where it needs their answers.
2. Compile it with ``pulse.compile()`` and schedule it with ``lower()``;
   inspect the schedule and the MLIR, or extend the pipeline with custom
   passes.
3. Send the one compiled program to any of three destinations: simulate it
   (``pulse.simulate``, with the NumPy reference on any machine or on an
   NVIDIA GPU), emulate it in virtual time against its classical services
   (``classical.emulation.emulate``), or export it to a controller backend
   (``export_to_controller()``).

Project status
--------------

.. note::

   CUDA-Q pulse is an early-stage research project. We are releasing it now
   because the pulse layer is where much of the open work in quantum control
   happens, and we want to build it in the open with the people who use it.
   It is under active development: expect bugs, incomplete features, and
   incompatible changes to the Python APIs, MLIR dialects and file formats
   between releases. It is not a supported CUDA-Q product feature. We invite
   you to try it with your own quantum systems and help shape its direction.

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
   user_guide/quantum_systems
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
