..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Challenges
==========

Small problems to solve with cudaq-pulse. Each folder holds the statement (``README.rst``), a
scorer (``score.py``) and reference solutions (``reference/``). You write one file, run the scorer
on it and read the result:

.. code-block:: console

   python challenges/pi_pulse/score.py my_pulse.py

.. list-table::
   :header-rows: 1

   * - Challenge
     - Level
     - What you do
   * - `Find the pi pulse <pi_pulse/README.rst>`_
     - Starter
     - Halve the length of a calibrated X pulse.
   * - `Mystery qubit <mystery_qubit/README.rst>`_
     - Starter
     - Measure the T1 of a qubit from shots alone.
   * - `Benchmark it yourself <benchmark_it_yourself/README.rst>`_
     - Starter
     - Find a pulse's error by randomized benchmarking.

Rungs
-----

Each challenge has a bronze and a silver rung. Bronze is reached from the tutorials; silver needs
one idea more. Starter challenges have no ranking: post your ``SCORE`` line on the challenge's
issue, and say what in the documentation slowed you down.

How scoring works
-----------------

Your file never runs in the scorer's process. The scorer starts a second Python process that loads
your file and compiles your kernels, and receives from it only numbers: the pulses your kernel
plays and when. The scorer checks those numbers against the challenge's limits and plays them
itself, on a qubit only it knows where the challenge has a hidden one. Where a qubit is hidden, each
run of the scorer draws new ones and prints a ``SEED`` to repeat the run; a repeated run shows its
rungs but is not a score. The second process is a separate process, not a sandbox: it keeps
honest work honest, and the rest is the honour rule.

The reference solutions are there to read after you have tried.
