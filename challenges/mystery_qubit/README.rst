..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Mystery qubit
=============

Level: Starter.

A qubit arrives with no T1 on its datasheet. Measure it.

Write one file, ``solution.py``, with a function ``estimate_t1(lab)`` that returns T1 in
microseconds, and score it:

.. code-block:: console

   python challenges/mystery_qubit/score.py solution.py

``lab.qubit`` is the qubit. ``lab.run(kernel, lab.qubit, ..., shots_count=n)`` plays your kernel
``n`` times and returns the shots, one row per shot, as ``pulse.simulate(...).records`` does. Shots
are all you get. T1 is between 20 and 200 microseconds, and the qubit is cold (thermal population
0).

The scorer tries 20 hidden qubits and prints each estimate beside the true value, then ``BRONZE``,
``SILVER``, ``SCORE`` (the total number of shots; lower is better) and ``SEED``. Each run draws
new qubits; give the seed as a second argument to repeat a run.

Limits
------

- A kernel plays drive pulses and waits, and ends in exactly one readout.
- The readout is a square pulse, as the calibrated one is.
- At most 100,000 shots in 200 runs per qubit.
- A kernel has at most 64 drive pulses and lasts at most 2 ms.
- A run with a seed given repeats known qubits, so it prints its rungs but no score.

Rungs
-----

.. list-table::
   :header-rows: 1

   * - Rung
     - Pass line
     - Hint
   * - Bronze
     - Every estimate within 5 %.
     - Start from the Coherence tutorial's T1 sweep, and count shots instead of reading a
       population.
   * - Silver
     - Bronze, with at most 10,000 shots per qubit on average.
     - A shot says most about T1 at one particular wait. Find it roughly, then spend your shots
       there.

Expect about 40 minutes for bronze after the Coherence tutorial.
