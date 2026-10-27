..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Examples
========

The examples in the ``examples/`` directory cover pulse-level programming
from single-qubit Rabi oscillations to multi-qubit error-correction cycles.
Every example except ``transmon/krinner_evolve_gpu.py`` runs on a CPU-only
machine, Linux or macOS, and the test suite runs each of them. Those marked
*(simulates)* also evolve the program with the NumPy reference
(``pulse.simulate``); the others compile kernels and print their MLIR,
schedules or metrics. Numbers 11 and 15 were removed and are not reused.

All examples use the canonical import convention:

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

.. :spellcheck-enable:

Core Examples
-------------

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - #
     - File
     - Description
   * - 01
     - ``01_single_qubit_rabi.py``
     - Rabi oscillation: an amplitude sweep that compiles once *(simulates)*
   * - 02
     - ``02_two_qubit_cross_resonance.py``
     - Two-qubit CNOT via echoed cross-resonance driving
   * - 03
     - ``03_t1_t2_dissipator.py``
     - T1 decay against its prediction; Ramsey and Hahn-echo kernels *(simulates)*
   * - 04
     - ``04_hahn_echo.py``
     - Canonical Hahn spin-echo sequence (pi/2 - tau - pi - tau - pi/2)
   * - 05
     - ``05_waveform_gallery.py``
     - All 8 waveform constructors and 5 algebraic combinators
   * - 06
     - ``06_phase_frequency_control.py``
     - Phase and frequency manipulation primitives on tone channels
   * - 07
     - ``07_multi_qubit_sync.py``
     - Multi-qubit programs with sync and registers of qudit handles
   * - 08
     - ``08_readout_and_branching.py``
     - Readout channels and measurement-conditioned branching *(simulates)*
   * - 09
     - ``09_compilation_pipeline.py``
     - ``pulse.compile()`` and ``lower()`` walkthrough of the compiler stages
   * - 10
     - ``10_scheduling_comparison.py``
     - Side-by-side ASAP, ALAP, and RCP scheduling on a 4-qubit kernel
   * - 12
     - ``12_error_detection.py``
     - ``pulse-verify`` diagnostics for intentionally malformed Pulse MLIR
   * - 13
     - ``13_dynamical_decoupling.py``
     - XY4, Uhrig, and CPMG dynamical decoupling sequences
   * - 14
     - ``14_randomized_benchmarking.py``
     - Randomized benchmarking: random Cliffords and their inverse *(simulates)*
   * - 16
     - ``16_visualization.py``
     - Schedule timelines; matplotlib plots with the ``[viz]`` extra
   * - 17
     - ``17_ghz_state_prep.py``
     - N-qubit GHZ state preparation using echoed cross-resonance CNOTs
   * - 18
     - ``18_per_kernel_schedules.py``
     - Per-kernel ASAP and ALAP policies in a stabilizer cycle with
       dynamical decoupling, compared with one flat schedule
   * - 19
     - ``19_classical_calls.py``
     - Classical calls at QPU runtime: real-time decoding with feed-forward,
       call slack and hidden latency *(simulates)*
   * - 20
     - ``20_adaptive_calibration.py``
     - Adaptive calibration with variable waits, tracking a drifting device
       *(simulates)*

Custom Passes
-------------

``examples/custom_passes/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``python_pass.py``
     - A schedule check written in Python over ``Lowered.schedule``
   * - ``cpp/DriveCount.cpp``
     - An MLIR pass plugin loaded with ``cudaq-pulse-opt --load-pass-plugin``

See :doc:`user_guide/custom_passes`.

Controller Backend
------------------

``examples/controller_backend/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``toy_controller/__init__.py``
     - A toy controller backend: its limits, ``emit`` and ``run``
   * - ``pyproject.toml``
     - Its distribution, registering it in the ``cudaq_pulse.backends``
       entry-point group

See :doc:`user_guide/controller_backends`.

Hardware Target Examples
------------------------

**Transmon** (``examples/transmon/``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``krinner_rabi.py``
     - Rabi oscillation on the Krinner 17-qubit transmon target
   * - ``krinner_full_pipeline.py``
     - Each stage of the native lowering: Pulse MLIR, QOp, and LLVM IR
   * - ``krinner_evolve_gpu.py``
     - ``pulse.simulate(..., backend="gpu")`` on the Krinner target (needs an
       NVIDIA GPU and the ``[cu12]`` or ``[cu13]`` extra)
   * - ``krinner_surface_code_cycle.py``
     - Timeline of one error correction cycle from the Krinner lattice and CZ schedule

**Neutral Atom** (``examples/neutral_atom/``). The neutral-atom targets
compile only; simulating them is not supported yet.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``rydberg_blockade_demo.py``
     - Two-atom Rydberg blockade interaction
   * - ``rydberg_chain_adiabatic.py``
     - Adiabatic detuning sweep on a 1D Rydberg atom chain

Running Examples
----------------

.. :spellcheck-disable:

.. code-block:: bash

   # From a clone of the repository, with cudaq-pulse installed.
   python examples/01_single_qubit_rabi.py

.. :spellcheck-enable:
