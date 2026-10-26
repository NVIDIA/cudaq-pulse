..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Pulse Operations
================

All operations listed here are used as bare names inside ``@pulse.kernel``
functions. They are intercepted during bytecode tracing and lowered
to MLIR operations in the Pulse dialect.

Channel Access
--------------

.. function:: get_drive_line(qubit) -> (line, tone)

   Obtain the drive channel for a qubit. Returns a ``(line, tone)`` pair:
   the *line* is passed to ``drive()``, ``wait()``, and ``sync()``; the
   *tone* is passed to phase and frequency ops.

   A qubit has one drive channel for the whole program. The first call
   acquires it; every later call, in the kernel or in a kernel it calls,
   returns the channel's current line and tone. Pulses on it play one after
   another, and phase and frequency changes carry over: ``x90(q); x90(q)``,
   where ``x90`` calls ``get_drive_line(q)`` itself, plays two pulses in
   sequence. A loop whose body acquires the channel of a qubit that exists
   before the loop acquires it once, before the loop, and stays rolled.

.. function:: get_readout_line(qubit) -> (line, tone)

   Obtain the readout channel for a qubit. Same return convention, and the
   same one channel per qubit.

Scheduling Operations
---------------------

.. function:: drive(line, waveform, tone)

   Play a waveform envelope on a drive line at the given tone frequency.

.. function:: readout(line, waveform, tone)

   Acquire a measurement through a readout line.

.. function:: wait(line, duration)

   Insert an idle delay of *duration* clock cycles on a line.

.. function:: sync(line1, line2, ...)

   Synchronize two or more lines to a common time point.
   All lines are padded to the latest time among them.

Phase and Frequency Control
---------------------------

These operations modify the rotating frame of a tone channel.

.. function:: shift_phase(tone, phase)

   Advance the tone's frame phase by *phase* (radians): a virtual Z rotation
   by *phase*, in the convention of McKay et al., PRA 96, 022330 (2017). A
   pulse played after it turns about an axis rotated by ``-phase``, so
   ``shift_phase(t, pi / 2)`` followed by an X90 pulse is an X90 about the
   -Y axis. It takes no time, and the virtual-Z pass folds it into the next
   drive.

.. function:: set_phase(tone, phase)

   Set the tone's frame phase (radians), with the same sign as
   ``shift_phase``.

.. function:: shift_frequency(tone, frequency)

   Add a relative frequency offset (Hz) to the tone.

.. function:: set_frequency(tone, frequency)

   Set the absolute frequency (Hz) of the tone. The phase carries over
   continuously.

In the qudit's frame, which rotates at its 0-1 frequency, a tone at
``w_tone`` drives with ``H = (A/2)(E a^dagger + E^* a)`` and
``E(t) = envelope(t) exp(-i theta(t))``, where the frame phase
``theta(t) = theta_0 + (w_tone - w_q) t``: a tone above the qudit runs ahead of
its frame. The 1-2 transition of a transmon is driven resonantly by a tone at
``w_q + alpha``.

Waveform Constructors
---------------------

Each constructor returns a waveform value that can be passed to
``drive()`` or combined with waveform arithmetic.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Function
     - Description
   * - ``gaussian(duration, amplitude, sigma)``
     - Gaussian envelope with standard deviation *sigma*
   * - ``square(duration, amplitude)``
     - Constant-amplitude (flat-top) envelope
   * - ``drag(duration, amplitude, sigma, beta)``
     - DRAG pulse for leakage suppression: ``G + i beta dG/dt`` for the
       Gaussian ``G``, with *beta* in clock cycles like *sigma*. Leakage of a
       transmon is smallest near ``beta = -1 / alpha``, for the
       anharmonicity ``alpha`` in radians per clock cycle (``2 pi f_alpha /
       f_clock``, negative for a transmon).
   * - ``cosine(duration, amplitude)``
     - Raised-cosine envelope
   * - ``tanh_ramp(duration, amplitude, sigma)``
     - Hyperbolic tangent ramp from zero to the requested amplitude
   * - ``gaussian_square(duration, amplitude, sigma, width)``
     - Flat-top pulse with Gaussian rise/fall edges
   * - ``custom(duration, envelope_fn)``
     - Named user-defined envelope callback for compiler experimentation
   * - ``custom_samples(samples)``
     - Waveform from a non-empty sequence of at most 253 real-valued samples

For ``gaussian_square``, ``width`` is the flat-top width in virtual time units;
``duration - width`` must be a positive even integer so the two edges have the
same integer duration.

The GPU execution lowering supports the built-in constructors and
``custom_samples``. Arbitrary ``custom`` callbacks must be specialized to
samples by a future pass and are currently rejected before execution.

Waveform Arithmetic
-------------------

Waveforms can be combined algebraically inside kernels:

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def combined_waveform(qubit):
       drive_line, tone = get_drive_line(qubit)
       envelope_a = gaussian(64, 0.5, 16.0)
       envelope_b = gaussian(64, 0.3, 10.0)
       combined = wf_add(envelope_a, envelope_b)
       scaled = wf_scale(envelope_a, 0.5)
       inverted = wf_neg(envelope_a)

.. :spellcheck-enable:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Operation
   * - ``wf_add(left, right)``
     - ``left + right``
   * - ``wf_sub(left, right)``
     - ``left - right``
   * - ``wf_mul(left, right)``
     - ``left * right`` (element-wise)
   * - ``wf_scale(waveform, scalar)``
     - ``scalar * waveform``
   * - ``wf_neg(waveform)``
     - ``-waveform``

The operands of ``wf_add``, ``wf_sub`` and ``wf_mul`` must have the same
duration; a mismatch is a compile error. For simulation and execution, a
sum, difference, scaling or negation lowers exactly: the drive Hamiltonian
is linear in the envelope, so each term plays as its own envelope. A
product is not linear and is sampled once per clock cycle into
piecewise-constant samples, as ``custom_samples`` would be.
