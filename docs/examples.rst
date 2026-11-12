..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Examples
========

The examples in the ``examples/`` directory cover pulse-level programming
from single-qubit Rabi oscillations to multi-qubit error-correction cycles.
Every example except ``transmon/transmon_surface17_evolve_gpu.py`` runs on a CPU-only
machine, Linux or macOS, and the test suite runs each of them. Those marked
*(simulates)* also evolve the program with the NumPy reference
(``pulse.simulate``); the others compile kernels and print their MLIR,
schedules or metrics. Numbers 11 and 15 were removed and are not reused.

All examples use the canonical import convention:

.. code-block:: python

   import cudaq_pulse as pulse

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
     - Adaptive calibration with variable waits, tracking a drifting quantum system
       *(simulates)*
   * - 21
     - ``21_calibration_protocol.py``
     - A millisecond calibration chain run on the controller: readout errors,
       Ramsey frequency, pi and pi/2 amplitudes, and T1 by the analytical
       decay estimator and adaptive tracking, each step's answer setting the
       next step's pulses; the frequency binary search; and what each
       estimator costs the controller, after Marciniak et al.,
       "Millisecond-Scale Calibration and Benchmarking of Superconducting
       Qubits", arXiv:2602.11912 (2026), and Berritta et al., "Real-time
       adaptive tracking of fluctuating relaxation rates in superconducting
       qubits", arXiv:2506.09576 (2025) *(simulates)*
   * - 22
     - ``22_runtime_qudits.py``
     - One compile for every qudit of a quantum system: the qudit is bound at each
       call, and a frequency the kernel needs, read from the quantum system on the
       host, is an ordinary float parameter
   * - 23
     - ``23_define_a_qpu.py``
     - Define a QPU: two controllers loaded from an installed description
       with the installation's wiring, a "feedback" link that takes a readout
       of one and answers the other, the compile-time checks against the
       control system, and the QPU saved and loaded as one system file
   * - 24
     - ``24_pipelined_calibration.py``
     - Closed-loop calibration with a compile pipeline: the kernel is
       compiled once, worker processes bind and export the pickled program,
       and the next batch is ready when the QPU finishes the last
       *(simulates)*
   * - 25
     - ``25_precompiled_branches.py``
     - Every next program of an adaptive experiment compiled ahead in worker
       processes and looked up by the check's outcome, against compiling on
       demand *(simulates)*
   * - 26
     - ``26_drift_following.py``
     - Following a drifting qudit: a background process recompiles when the
       quantum system fingerprint changes, and the host swaps programs in between
       batches *(simulates)*
   * - 27
     - ``27_realtime_decoding.py``
     - Real-time QEC decoding with a deadline and a fallback table: a
       decoder configured in CUDA-Q QEC, called with CUDA-Q QEC's own
       functions; needs cudaq-qec (Linux) and an NVIDIA GPU *(simulates)*

Examples 24 to 26 send compiled programs between processes, which works
because programs pickle; their worker processes use the ``spawn`` start
method.

Custom Passes
-------------

``examples/custom_passes/``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``python_pass.py``
     - A schedule check written in Python over ``ScheduledKernel.schedule``
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

Transmon Quantum System Examples
--------------------------------

**Transmon** (``examples/transmon/``):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``transmon_surface17_rabi.py``
     - Rabi oscillation on the 17-qubit ``transmon_surface17`` quantum system
   * - ``transmon_surface17_full_pipeline.py``
     - Each stage of the native lowering: Pulse MLIR, QOp, and LLVM IR
   * - ``transmon_surface17_evolve_gpu.py``
     - ``pulse.simulate(..., backend="gpu")`` on the 17-qubit ``transmon_surface17`` quantum system (needs an
       NVIDIA GPU and the ``[cu12]`` or ``[cu13]`` extra)
   * - ``transmon_surface17_surface_code_cycle.py``
     - Timeline of one error correction cycle from the 17-qubit lattice and CZ schedule
   * - ``transmon_surface17_multiplexed_readout.py``
     - One readout feedline shared by the four X-check ancillas: the tones it plays at
       once, and each readout's tone slot in the controller export

Running Examples
----------------

.. code-block:: bash

   # From a clone of the repository, with cudaq-pulse installed.
   python examples/01_single_qubit_rabi.py
