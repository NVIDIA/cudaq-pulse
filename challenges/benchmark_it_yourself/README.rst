..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Benchmark it yourself
=====================

Level: Starter, at the top of the band.

How good is a qubit's calibrated X90 pulse? Measure its error by randomized benchmarking.

The scorer can check which pulses you play, not how you chose them: any train of the calibrated
X90 and Y90 is within the rules. Randomized benchmarking is what the challenge is there to teach,
and what the rungs are tuned for.

Write one file, ``my_rb.py``, with a function ``error_per_gate(lab, seed)`` that returns the error
of one X90 pulse, and score it:

.. code-block:: console

   python challenges/benchmark_it_yourself/score.py my_rb.py

``lab.qubit`` is the qubit: its frequency, anharmonicity and calibrated pulses are those of a
qubit like Z4 of ``transmon_surface17``, and its T1 and T2* are hidden. ``lab.run(kernel,
lab.qubit, ...)`` plays your kernel once and returns a result with ``level_population(i)``: exact
numbers, nothing is sampled.

The scorer calls your function for five hidden qubits. After each estimate it prints the exact
error of the pulse on that qubit, then ``BRONZE``, ``SILVER``, ``SCORE`` (your worst deviation
and your largest number of sequences) and ``SEED``. Each run draws new qubits; give the seed as a
second argument to repeat a run.

Limits
------

- A sequence plays only the qubit's calibrated X90 and Y90, back to back from the start (a frame
  change before a pulse is allowed, a wait is not), then at most one calibrated readout, at once.
- One call of ``error_per_gate`` plays at least 2 sequences and 20 pulses, and at most 6,000
  pulses.
- A run with a seed given repeats known qubits, so it prints its rungs but no score.

Rungs
-----

.. list-table::
   :header-rows: 1

   * - Rung
     - Pass line
     - Hint
   * - Bronze
     - Every estimate within a factor 1.5 of the exact error.
     - In example 14, play the calibrated pulses and fit ``survival = A p^depth + 1/2``. The error
       per Clifford is ``(1 - p)/2``; a Clifford is 37/12 pulses on average.
   * - Silver
     - Every estimate within 10 %, from at most 8 sequences.
     - Two sequences of one depth differ by tens of pulses. What does the decay really depend on?

Expect 45 to 60 minutes for bronze after the tutorials.
