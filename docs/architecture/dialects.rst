..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

MLIR Dialects
=============

cudaq-pulse defines four MLIR dialects. Three form a layered IR stack. A kernel
compiles to the ``pulse`` dialect; simulation lowers it through ``qop`` to
``cudm`` and then to LLVM calls into the cuDensityMat runtime. The fourth, ``classical``, is a
classical real-time dialect for co-processing at QPU runtime, used beside
``pulse``. Every op of
each dialect is listed in its reference page, generated from the ODS
definitions in ``core/mlir/include/cudaq-pulse/Dialect/``.

.. toctree::
   :maxdepth: 1

   dialects/pulse
   dialects/classical
   dialects/qop
   dialects/cudm

Pulse
-----

The pulse program as written: qudits, the lines and tones that drive and read
them out, waveforms, and timing. Lines and tones are linear values, consumed
once and returned updated by each op that uses them, so the order of the
pulses on a line is explicit in the IR and ``pulse-verify`` can check it.
Waveforms are ordinary values. Durations are in clock cycles of the quantum system,
and scheduling records each timed op's ``start_cycles`` and ``duration_cycles``
(and ``period_cycles`` in loops) as integers in the op's properties, printed
in a ``<{...}>`` dictionary before its other attributes. A function's
``pulse.schedule_state`` says how far it is scheduled: without one it is
``unscheduled``; ``#pulse.schedule_state<timed>`` once every timed op that
runs has its start and length; ``#pulse.schedule_state<placed>`` once each
line's calibrated delay is applied to its ops' starts as well. Scheduling
writes ``placed`` when a line delay moves some op, ``timed`` otherwise.

A module describes the quantum system it plays on in ``pulse.quantum_system``,
lists indexed by qudit, every field optional:

.. code-block:: text

   module attributes {pulse.quantum_system = #pulse.quantum_system<
       qudit_frequency_hz = [5.0e+09, 5.1e+09], level_counts = [3, 3],
       anharmonicity_hz = [-2.0e+08, -2.1e+08], t1_ns = [5.0e+04, 4.0e+04],
       t2_ns = [3.0e+04, 2.5e+04], drive_scale_rad_per_ns = [1.0, 1.0],
       static_terms = [#pulse.static_term<zz, qudits = [0, 1], rate_hz = 4.5e+04>]>} {
   }

A compiled module gives ``qudit_frequency_hz``, the frequencies it is compiled
for. ``pulse.simulate`` gives the whole simulated system, whose frequencies are
the qudits' own: on a system other than the one compiled for, or one whose
parameters drift, they differ from the tones the program plays, and the
simulation sees the detuning. ``pulse-to-qop`` puts a qudit the attribute gives
no frequency for at its drive line's tone. The simulation choices stay
attributes of their own: ``pulse.modes`` (the qudits simulated),
``pulse.thermal_populations``, ``pulse.coupling_pairs`` and
``pulse.coupling_strength_hz``.

A module describes the control system it plays on in ``pulse.control_system``:
the program clock, as its period in femtoseconds, an exact fraction
(``clock_period_fs = 1250000/3`` is 2.4 GHz, ``500000`` is 2 GHz); its lines
(``#pulse.line``: ``drive`` or ``readout``, the qudits it reaches, the most
tones it plays at once, its delay in program cycles and the index of its
controller); and its controllers (``#pulse.controller``: the program cycles
between the instants it can start an operation, ``grid_cycles``; the outcome
latency to and the start spread from each other controller and its start
uncertainty; whether its frames persist between shots and whether it blocks
on calls; and the longest the program may be, ``max_duration_cycles``):

.. code-block:: text

   module attributes {pulse.control_system = #pulse.control_system<
       clock_period_fs = 500000,
       lines = [#pulse.line<drive, qudits = [0], delay_cycles = 3>,
                #pulse.line<readout, qudits = [0, 1, 2, 3], max_concurrent_tones = 4,
                            controller = 1>],
       controllers = [
         #pulse.controller<outcome_latency_cycles = [0, 6], spread_cycles = [0, 6],
                           uncertainty_ns = 1.0>,
         #pulse.controller<grid_cycles = 2, outcome_latency_cycles = [6, 0],
                           spread_cycles = [6, 0], uncertainty_ns = 2.0,
                           blocks_on_calls>]>} {
   }

Fields at their default are left out: a compiled module without a controller
has only its clock. A line the attribute does not list has no delay and, with
one controller, is that controller's; with several, such as the line of a
qudit a call binds at run time, it may be on any of them and moves on every
one's grid.

- Types: ``!pulse.qudit``, ``!pulse.drive_line``, ``!pulse.readout_line``,
  ``!pulse.tone``, ``!pulse.waveform``, ``!pulse.measurement``,
  ``!pulse.duration``, ``!pulse.timepoint``, ``!pulse.iq_data``.
- Lines: ``pulse.qudit_alloc``, ``pulse.get_drive_line``,
  ``pulse.get_readout_line``.
- Playing and timing: ``pulse.drive``, ``pulse.readout``, ``pulse.wait``,
  ``pulse.sync``, ``pulse.schedule_block``.
- Frames: ``pulse.shift_phase``, ``pulse.set_phase``,
  ``pulse.shift_frequency``, ``pulse.set_frequency``.
- Waveforms: ``pulse.square``, ``pulse.gaussian``, ``pulse.drag``,
  ``pulse.cosine``, ``pulse.tanh_ramp``, ``pulse.gaussian_square``,
  ``pulse.custom_samples``, and the element-wise ``pulse.add``,
  ``pulse.sub``, ``pulse.mul``, ``pulse.scale`` and ``pulse.neg``.
- Measurements: ``pulse.is_set`` and ``pulse.outcome``, which feedback
  branches on with ``scf.if``.

Frames
~~~~~~

A tone is a frame: a carrier frequency and a frame phase, threaded from op
to op like a line.

- A frame change (``pulse.shift_phase``, ``pulse.set_phase``,
  ``pulse.shift_frequency``, ``pulse.set_frequency``) takes no time and takes
  effect at the next pulse played on the tone. A change after the last pulse
  on a tone plays nothing; the controller export still lists it as a frame
  update.
- A wait does not advance a tone's frame phase, unlike OpenPulse, where a
  delay advances the frame. The carrier runs on through the wait: a tone off
  its qudit's frequency by a detuning is that detuning times the elapsed time
  further round in the qudit's frame, whether the time passed in a wait or in
  pulses.
- A frame rotates through ``pulse.sync``: a change made before a sync
  applies to the next pulse on the tone after it, however long the sync holds
  the line.
- ``pulse.readout`` plays its pulse and gives a measurement (the level
  ``pulse.is_set`` and ``pulse.outcome`` read); ``pulse.iq_acquire`` gives the
  raw IQ point instead. Both return their line and tone, so a frame change
  before either applies to it. Only ``pulse.readout`` is scheduled: the
  scheduler gives ``pulse.iq_acquire`` no start or duration, and the
  controller export has no event for it.

``test/Conversion/pulse-to-qop-frames.mlir`` and
``test/Dialect/Pulse/readout_iq_acquire.mlir`` show each rule.

Classical
---------

A classical real-time dialect for co-processing at QPU runtime: the GPU,
host or FPGA services a pulse program calls while the QPU runs, for decoding,
calibration and data transfer. ``classical.callable`` declares a function,
with the types of its arguments and results and its latency. A call is two
timed events: ``classical.submit`` sends the request on a link, a linear value
like a line, and ``classical.wait`` is where the answer is needed. Pulses
whose parameters or branch conditions come from the answer start after it is
due; others are not delayed.

- Types: ``!classical.link``, ``!classical.future``.
- Ops: ``classical.callable``, ``classical.get_link``, ``classical.submit``,
  ``classical.wait``.

:doc:`../user_guide/classical_calls` describes the Python interface, and
``examples/19_classical_calls.py`` (see :doc:`../examples`) uses it end to
end: ``@pulse.classical`` functions on two links, a decoder with a
timeout fallback, an adaptive T1 estimator that sets a variable wait, the
call report, simulation, and the emulation export.

QOp
---

Backend-agnostic quantum operator algebra. ``pulse-to-qop`` builds one
``qop.lindblad`` per program: a Hamiltonian of ``qop.make_sum`` terms, each a
``qop.make_product`` of ``qop.spin`` or ``qop.matrix_leaf`` operators with a
``qop.const_scalar`` or, for a drive envelope, a ``qop.callback_scalar``
coefficient, and the collapse operators of the quantum system's T1 and T2. The
bosonic, fermionic, commutator and super-operator ops are planned surface for
later backends; nothing produces or lowers them yet.

CuDensityMat
------------

The cuDensityMat API as IR. ``qop-to-cudm`` turns the Lindbladian into a
handle, states, elementary operators, operator terms and a composite operator,
then a ``cudm.evolve`` with the chosen integrator. ``cudm-to-llvm`` lowers each
op to a call into the cudaq-pulse cuDensityMat runtime (``cudm_init``,
``cudm_evolve`` and so on), which the JIT links when a program is simulated
on the GPU.
