..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Idle Replacement: Echoing a Readout's Crosstalk
===============================================

When an ancilla qubit is read out, the photons in its resonator can shift
a data qubit beside it. The data qubit then turns by an angle that depends
on the ancilla's state. No gate played before the readout's outcome is
known removes that conditional phase; a frame change conditioned on the
outcome could, once it arrives (:doc:`measure_and_react`). This tutorial
declares such a crosstalk on ``transmon_surface17``, measures the
conditional phase it leaves on the data qubit, and fills the data qubit's
idle time during the readout with echo pulses that mostly cancel it,
without moving any other operation of the program.

What you'll learn:

- how to declare readout crosstalk on a quantum system;
- how to measure a data qubit's conditional phase with two Ramsey
  sequences;
- how ``compiled.replace_idle(rule)`` finds a qubit's idle windows and
  fills them;
- why the spacing of echo pulses matters when the shift is not constant.

All the code on this page is also one Python script,
:download:`idle_replacement.py <scripts/idle_replacement.py>`.

The crosstalk
-------------

Qubit Z2's readout drives its resonator with a few photons. Here, each
photon is said to move data qubit D5's frequency by -20 kHz. That number
is an assumption made for this page: ``ReadoutCrosstalk`` states it, and the
simulator works out the photons themselves from Z2's ``DispersiveReadout``
and ``READOUT`` calibration, as the readout plays:

.. code-block:: python

   import math

   import cudaq_pulse as pulse
   from cudaq_pulse.quantum_systems import ReadoutCrosstalk

   system = pulse.quantum_system("transmon_surface17")
   d5, z2 = system.qudit("D5"), system.qudit("Z2")
   system = system.replace(
       readout_crosstalk=[ReadoutCrosstalk(resonator_of=z2.index, qudit=d5.index, chi_hz=-20e3)]
   )
   d5, z2 = system.qudit("D5"), system.qudit("Z2")
   print(system.readout_crosstalk)

.. code-block:: text
   :class: output

   (ReadoutCrosstalk(resonator_of=14, qudit=4, chi_hz=-20000.0),)

The photons ring up while the readout plays and ring down after it, in
about ``1 / kappa``, 13 ns for Z2; they ring up about half as fast. Played
at the midpoint between the frequencies Z2's two states pull its resonator
to, the readout fills it with the same photons whichever state Z2 is in.
D5's shift is then the same too, and a frame change could take it out.
This page plays the readout 2.9 MHz higher, on the resonator as Z2's
ground state pulls it. There the resonator holds 4.9 photons with Z2 in
its ground state and 2.6 with Z2 excited, so D5's shift depends on Z2's
state.

The conditional phase
---------------------

The kernel puts D5 in a superposition with a quarter turn, reads out Z2
beside it, and turns D5 again. ``flip`` excites Z2 first, and ``quarter``
shifts the phase of D5's second turn by 90 degrees:

.. code-block:: python

   @pulse.kernel
   def data_beside_readout(data, ancilla, flip: int, quarter: int):
       data_drive = get_drive_line(data)
       ancilla_drive = get_drive_line(ancilla)
       ancilla_readout = get_readout_line(ancilla)
       if flip:
           drive(ancilla_drive, ancilla_drive.get_calibration(pulse.gates.X))
       sync(data_drive, ancilla_drive, ancilla_readout)
       drive(data_drive, data_drive.get_calibration(pulse.gates.X90))
       sync(data_drive, ancilla_readout)
       shift_frequency(ancilla_readout.tone, 2.9e6)
       readout(ancilla_readout, ancilla_readout.get_calibration(pulse.gates.READOUT))
       wait(ancilla_readout, 120)
       sync(data_drive, ancilla_readout)
       for _ in range(quarter):
           shift_phase(data_drive.tone, math.pi / 2)
       drive(data_drive, data_drive.get_calibration(pulse.gates.X90))

D5's excited population with and without the quarter turn gives the angle
D5 turned through. The conditional phase is that angle with Z2 excited less
the angle with Z2 in its ground state:

.. code-block:: python

   def conditional_phase(compiled):
       phases = []
       for flip in (0, 1):
           excited = []
           for quarter in (0, 1):
               program = compiled.lower(d5, z2, flip, quarter)
               result = pulse.simulate(program, dt_ns=0.02, levels=2)
               excited.append(result.level_populations(d5)[1])
           phases.append(math.atan2(2 * excited[1] - 1, 2 * excited[0] - 1))
       return math.degrees(math.remainder(phases[1] - phases[0], 2 * math.pi))


   compiled = pulse.compile(data_beside_readout, quantum_system=system)
   print(f"no fill: {conditional_phase(compiled):+.2f} degrees")

.. code-block:: text
   :class: output

   no fill: +2.15 degrees

``dt_ns=0.02`` fixes the simulation's step at about the one it would pick
by itself: D5's coupling to Z2 turns at their 2.3 GHz detuning, which a
coarser step would not resolve.

D5's idle window
----------------

Between its two quarter turns, D5's drive line plays nothing. A rule
without a fill only lists that stretch:

.. code-block:: python

   listed = compiled.replace_idle(pulse.IdleReplacement(None, threshold=160 * pulse.ns, qudits=[d5]))
   for window in listed.lower(d5, z2, 0, 0).idle_windows or ():
       print(f"qudit {window.qudit}: cycles {window.start} to {window.end}; {window.result}")

.. code-block:: text
   :class: output

   qudit 4: cycles 80 to 600; not filled: the rule has no fill

The window is 260 ns long: the 200 ns readout and the 60 ns wait after it,
during which Z2's resonator rings down.

Filling it with echoes
----------------------

A π pulse in the middle of the window flips D5, so that the phase D5 gains
in the second half of the window undoes the phase it gained in the first,
as far as the shift is the same in both halves. A fill is a kernel of the
idle qudit and its window's length in clock cycles. This one spaces
``count`` π pulses evenly over the window; a window must hold all of them,
so the rule's threshold is 160 ns, four 40 ns pulses:

.. code-block:: python

   flip_cycles = round(
       system.physical_qudits[d5.index].get_calibration(pulse.gates.X).duration_ns
       * system.clock_ghz
   )


   def evenly_spaced(count):
       @pulse.kernel
       def flips(qudit, window: int):
           line = get_drive_line(qudit)
           flip = line.get_calibration(pulse.gates.X)
           gap = (window - count * flip_cycles) // count
           wait(line, gap // 2)
           for i in range(count):
               drive(line, flip)
               if i < count - 1:
                   wait(line, gap)
           wait(line, window - count * flip_cycles - (count - 1) * gap - gap // 2)

       return flips


   for count in (1, 2, 4):
       rule = pulse.IdleReplacement(evenly_spaced(count), threshold=160 * pulse.ns, qudits=[d5])
       phase = conditional_phase(compiled.replace_idle(rule))
       print(f"{count} flip{'s' if count > 1 else ''}: {phase:+.2f} degrees")

.. code-block:: text
   :class: output

   1 flip: +0.29 degrees
   2 flips: -1.51 degrees
   4 flips: -0.10 degrees

.. themed-figure:: figures/conditional_phase.svg
   :alt: D5's conditional phase from Z2's readout, in degrees, for each fill of its idle window: no fill +2.15, 1 flip +0.29, 2 flips -1.51, 4 flips -0.10. One flip cancels most of it, two flips overshoot, four cancel it again.
   :width: 100%

   D5's conditional phase with each fill of its idle window.

Every other operation keeps its start: ``replace_idle`` writes the fill into
the window and checks that nothing moves.

A single echo cancels most of the conditional phase. What it leaves comes
from where the shift sits in the window: the readout fills the first 200
ns, and the photons have rung down within about 40 ns of its end, so the
shift weighs on the window's first part. Two evenly spaced flips do worse
than one: they set the middle 130 ns, where the readout plays at full
strength, against a first stretch that holds its ring-up and a last one
that holds almost no shift. Here, four flips follow the shift closely
enough to cancel it again. How many to use, and where, depends on how the
shift changes over the window, which the simulator gives from the
resonator's equation.
