..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Tutorials
=========

Each tutorial builds one idea from a few short snippets you can run on a
laptop, and ends by pointing to the example files that take it further.
Each page's code is also one Python script you can download from the page.
Start with the first; the rest can be read in any order once you have it.

Most tutorials run on the built-in ``transmon_surface17`` quantum system.
:doc:`../user_guide/transmon_surface17` shows its layout, frequencies and
calibrated gates.

.. toctree::
   :maxdepth: 1
   :caption: Basics

   first_pulse_program
   coherence
   measure_and_react
   several_qubits
   idle_replacement
   own_quantum_system

.. toctree::
   :maxdepth: 1
   :caption: Real-time control

   classical_calls
   adaptive_calibration
   realtime_decoding
   t_state_injection

.. toctree::
   :maxdepth: 1
   :caption: GPUs and controllers

   gpu_simulation
   kernel_to_controller
   control_systems

.. toctree::
   :maxdepth: 1
   :caption: Extending CUDA-Q Pulse

   isa_lowering
   isa_limits
