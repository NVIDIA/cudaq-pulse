..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Find the pi pulse
=================

Level: Starter.

The calibrated X pulse of qubit Z4 on ``transmon_surface17`` takes 40 ns. Make one that takes
20 ns.

Write one file, ``my_pulse.py``, with a kernel ``pi_pulse(qudit)`` that only plays drive pulses,
and score it:

.. code-block:: console

   python challenges/pi_pulse/score.py my_pulse.py

The scorer starts Z4 in level 0, plays your pulse and prints ``BRONZE`` and ``SILVER``, pass or
fail, and a ``SCORE`` line with ``P1`` and ``P2``: the populations of levels 1 and 2 when the
pulse ends.

Limits
------

- At most 20 ns in all, waits included.
- Every sample at most 0.5 in magnitude (the shipped X pulse peaks at 0.13).
- Every pulse within 50 MHz of the qubit's 0-1 frequency.
- At most 64 pulses.

The scorer plays your pulse as the samples a controller would send at the system's 2 GHz clock:
one value every 0.5 ns. Your own ``simulate`` run follows a smooth shape more finely, so the two
can differ: in the fourth or fifth decimal place for a pulse as wide as the references', and by
more for one only a few samples wide. The scorer's number is the one that counts.

Rungs
-----

.. list-table::
   :header-rows: 1

   * - Rung
     - Pass line
     - Hint
   * - Bronze
     - ``P1 >= 0.98``
     - Shorten the Gaussian of "Your First Pulse Program" and sweep its amplitude.
   * - Silver
     - ``P1 >= 0.999`` and ``P2 <= 0.0002``
     - Amplitude alone will not do it: look at what stays in level 0 and what reaches level 2,
       and try ``drag()`` or a small detuning.

Getting started
---------------

.. code-block:: python

   import cudaq_pulse as pulse

   system = pulse.quantum_system("transmon_surface17")
   qubit = system.qudit("Z4")
   result = pulse.simulate(my_kernel, qubit, initial_state=[1, 0, 0])
   print(result.level_population(1))

Expect 20 to 30 minutes for bronze after the first tutorial.
