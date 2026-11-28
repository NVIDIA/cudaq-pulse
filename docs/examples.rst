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

The :doc:`tutorials <tutorials/index>` explain the same ideas step by step,
and each one ends by naming the examples that take it further. If you are
new to cudaq-pulse, start with :doc:`tutorials/first_pulse_program`.

All examples use the canonical import convention:

.. code-block:: python

   import cudaq_pulse as pulse

Core Examples
-------------

.. list-table:: Core examples
   :header-rows: 1
   :widths: 10 25 65

   * - #
     - File
     - Description
   * - 01
     - :example:`01_single_qubit_rabi.py <01_single_qubit_rabi.py>`
     - Rabi oscillation: an amplitude sweep that compiles once *(simulates)*
   * - 02
     - :example:`02_two_qubit_cross_resonance.py <02_two_qubit_cross_resonance.py>`
     - Two-qubit CNOT via echoed cross-resonance driving
   * - 03
     - :example:`03_t1_t2_dissipator.py <03_t1_t2_dissipator.py>`
     - T1 decay against its prediction; Ramsey and Hahn-echo kernels *(simulates)*
   * - 04
     - :example:`04_hahn_echo.py <04_hahn_echo.py>`
     - Canonical Hahn spin-echo sequence (pi/2 - tau - pi - tau - pi/2)
   * - 05
     - :example:`05_waveform_gallery.py <05_waveform_gallery.py>`
     - All 8 waveform constructors and 5 algebraic combinators
   * - 06
     - :example:`06_phase_frequency_control.py <06_phase_frequency_control.py>`
     - Phase and frequency manipulation primitives on tone channels
   * - 07
     - :example:`07_multi_qubit_sync.py <07_multi_qubit_sync.py>`
     - Multi-qubit programs with sync and registers of qudit handles
   * - 08
     - :example:`08_readout_and_branching.py <08_readout_and_branching.py>`
     - Readout channels and measurement-conditioned branching *(simulates)*
   * - 09
     - :example:`09_compilation_pipeline.py <09_compilation_pipeline.py>`
     - ``pulse.compile()`` and ``lower()`` walkthrough of the compiler stages
   * - 10
     - :example:`10_scheduling_comparison.py <10_scheduling_comparison.py>`
     - Side-by-side ASAP, ALAP, and RCP scheduling on a 4-qubit kernel
   * - 12
     - :example:`12_error_detection.py <12_error_detection.py>`
     - ``pulse-verify`` diagnostics for intentionally malformed Pulse MLIR
   * - 13
     - :example:`13_dynamical_decoupling.py <13_dynamical_decoupling.py>`
     - XY4, Uhrig, and CPMG dynamical decoupling sequences
   * - 14
     - :example:`14_randomized_benchmarking.py <14_randomized_benchmarking.py>`
     - Randomized benchmarking: random Cliffords and their inverse *(simulates)*
   * - 16
     - :example:`16_visualization.py <16_visualization.py>`
     - Schedule timelines; matplotlib plots with the ``[viz]`` extra
   * - 17
     - :example:`17_ghz_state_prep.py <17_ghz_state_prep.py>`
     - N-qubit GHZ state preparation using echoed cross-resonance CNOTs
   * - 18
     - :example:`18_per_kernel_schedules.py <18_per_kernel_schedules.py>`
     - Per-kernel ASAP and ALAP policies in a stabilizer cycle with
       dynamical decoupling, compared with one flat schedule
   * - 19
     - :example:`19_classical_calls.py <19_classical_calls.py>`
     - Classical calls at QPU runtime: real-time decoding with feed-forward,
       call slack and hidden latency *(simulates)*
   * - 20
     - :example:`20_adaptive_calibration.py <20_adaptive_calibration.py>`
     - Adaptive calibration with variable waits, tracking a drifting quantum system
       *(simulates)*
   * - 21
     - :example:`21_calibration_protocol.py <21_calibration_protocol.py>`
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
     - :example:`22_runtime_qudits.py <22_runtime_qudits.py>`
     - One compile for every qudit of a quantum system: the qudit is bound at each
       call, and a frequency the kernel needs, read from the quantum system on the
       host, is an ordinary float parameter
   * - 23
     - :example:`23_define_a_qpu.py <23_define_a_qpu.py>`
     - Define a QPU: two controllers loaded from an installed description
       with the installation's wiring, a "feedback" link that takes a readout
       of one and answers the other, the compile-time checks against the
       control system, and the QPU saved and loaded as one system file
   * - 24
     - :example:`24_pipelined_calibration.py <24_pipelined_calibration.py>`
     - Closed-loop calibration with a compile pipeline: the kernel is
       compiled once, worker processes bind and export the pickled program,
       and the next batch is ready when the QPU finishes the last
       *(simulates)*
   * - 25
     - :example:`25_precompiled_branches.py <25_precompiled_branches.py>`
     - Every next program of an adaptive experiment compiled ahead in worker
       processes and looked up by the check's outcome, against compiling on
       demand *(simulates)*
   * - 26
     - :example:`26_drift_following.py <26_drift_following.py>`
     - Following a drifting qudit: a background process recompiles when the
       quantum system fingerprint changes, and the host swaps programs in between
       batches *(simulates)*
   * - 27
     - :example:`27_realtime_decoding.py <27_realtime_decoding.py>`
     - Real-time QEC decoding with a deadline and a fallback table: a
       decoder configured in CUDA-Q QEC, called with CUDA-Q QEC's own
       functions; needs cudaq-qec (Linux) and an NVIDIA GPU *(simulates)*
   * - 28
     - :example:`28_t_state_injection.py <28_t_state_injection.py>`
     - T-state injection with the virtual-Z ``pulse.gates.T``, three-axis
       tomography, and a heralded repeat-until-success loop on the
       controller, post-selected on the herald's answer with
       ``result.postselect`` *(simulates)*
   * - 29
     - :example:`29_waveform_envelopes.py <29_waveform_envelopes.py>`
     - The samples each built-in envelope plays, from
       ``cudaq_pulse.controller.sample_waveform`` of the controller export,
       printed and plotted; the operations guide's waveform explorer is the
       interactive version
   * - 30
     - :example:`30_control_systems_compared.py <30_control_systems_compared.py>`
     - One QPU under five control systems, each defined in the file: the
       length of an active reset and of one surface-code round, the reset's
       ground-state population and a data qubit's idle error, side by side;
       the tutorial :doc:`tutorials/control_systems` builds it step by step
       *(simulates)*

Examples 24 to 26 send compiled programs between processes, which works
because programs pickle; their worker processes use the ``spawn`` start
method.

Custom Passes
-------------

``examples/custom_passes/``:

.. list-table:: Custom pass examples
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - :example:`python_pass.py <custom_passes/python_pass.py>`
     - A schedule check written in Python over ``ScheduledKernel.schedule``
   * - :example:`cpp/DriveCount.cpp <custom_passes/cpp/DriveCount.cpp>`
     - An MLIR pass plugin loaded with ``cudaq-pulse-opt --load-pass-plugin``

See :doc:`user_guide/custom_passes`.

Controller Backend
------------------

``examples/controller_backend/``:

.. list-table:: Controller backend examples
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - :example:`toy_controller/__init__.py <controller_backend/toy_controller/__init__.py>`
     - A toy controller backend: its limits, ``emit`` and ``run``
   * - :example:`pyproject.toml <controller_backend/pyproject.toml>`
     - Its distribution, registering it in the ``cudaq_pulse.backends``
       entry-point group
   * - :example:`text_backend.py <controller_backend/text_backend.py>`
     - A small backend that plays a program as text, registered in the
       running process; :doc:`tutorials/kernel_to_controller` shows it a
       piece at a time

See :doc:`user_guide/controller_backends`.

ISA Lowering
------------

``examples/isa_lowering/``:

.. list-table:: ISA lowering example
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - :example:`toy_isa.py <isa_lowering/toy_isa.py>`
     - A lowering to a toy controller instruction set, its controllers
       described as data: the timestep, a waveform table, oscillators, a
       shared feedline, feedback branches within and across controllers, a
       decoder call, and emulated toy controllers

See :doc:`tutorials/isa_lowering` and :doc:`tutorials/isa_limits`
(experimental).

Transmon Quantum System Examples
--------------------------------

**Transmon** (``examples/transmon/``), on the built-in chip that
:doc:`user_guide/transmon_surface17` describes:

.. list-table:: Transmon examples
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - :example:`transmon_surface17_rabi.py <transmon/transmon_surface17_rabi.py>`
     - Rabi oscillation on the 17-qubit ``transmon_surface17`` quantum system
   * - :example:`transmon_surface17_full_pipeline.py <transmon/transmon_surface17_full_pipeline.py>`
     - Each stage of the native lowering: Pulse MLIR, QOp, and LLVM IR
   * - :example:`transmon_surface17_evolve_gpu.py <transmon/transmon_surface17_evolve_gpu.py>`
     - ``pulse.simulate(..., backend="gpu")`` on the 17-qubit ``transmon_surface17`` quantum system (needs an
       NVIDIA GPU and the ``[cu12]`` or ``[cu13]`` extra)
   * - :example:`transmon_surface17_surface_code_cycle.py <transmon/transmon_surface17_surface_code_cycle.py>`
     - Timeline of one error correction cycle from the 17-qubit lattice and CZ schedule
   * - :example:`transmon_surface17_multiplexed_readout.py <transmon/transmon_surface17_multiplexed_readout.py>`
     - One readout feedline shared by the four X-check ancillas: the tones it plays at
       once, and each readout's tone slot in the controller export

Challenges
----------

The `challenges/ folder <https://github.com/NVIDIA/cudaq-pulse/tree/main/challenges>`_
has starter problems you can solve and score locally:

- **Find the pi pulse:** Halve the length of a calibrated X pulse.
- **Mystery qubit:** Measure the T1 of a qubit from shots alone.
- **Benchmark it yourself:** Find a pulse's error by randomized benchmarking.
- **Schedule golf:** Make a five-qubit program finish sooner without changing what it does.
- **Fix the kernel:** Find the five mistakes in a kernel, by what it plays.
- **Reset one qubit:** Bring a qubit in an unknown state to its ground state, in time.

Running Examples
----------------

.. code-block:: bash

   # From a clone of the repository, with cudaq-pulse installed.
   python examples/01_single_qubit_rabi.py
