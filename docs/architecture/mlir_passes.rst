..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

MLIR Passes
===========

cudaq-pulse implements its optimizations, checks and schedules as C++ MLIR
passes on the in-memory ``mlir::ModuleOp``. Each is registered under the
name below, so ``cudaq-pulse-opt`` runs it on its own:

.. code-block:: bash

   cudaq-pulse-opt --pulse-fold --pulse-virtual-z kernel.mlir
   cudaq-pulse-opt '--pulse-schedule-rcp=max-drives=2 alap=true' kernel.mlir

``pulse.compile`` runs the passes named in ``passes=``, then the schedule of
its policy. Each pass name maps to these passes:

==================  ======================================
``passes=`` name    Passes
==================  ======================================
``"verify"``        ``pulse-verify``
``"virtual_z"``     ``pulse-fold``, ``pulse-virtual-z``
``"fusion"``        ``pulse-fold``, ``pulse-fusion``
``"canonicalize"``  ``pulse-fold``, ``pulse-canonicalize``
``"licm"``          ``loop-invariant-code-motion``
==================  ======================================

Optimization passes
-------------------

``pulse-fold``, ``pulse-canonicalize``, ``pulse-virtual-z`` and
``pulse-fusion`` are pattern passes: each is one greedy rewrite of a
function. Pattern passes in a row in ``passes=`` run as a single rewrite with
all their patterns, to the point where none applies, so splitting the work
into passes costs no extra walk of the module.

pulse-fold
   Applies the folders of the ops (a frame change by zero, a scale by one, a
   double negation), merges equal constants, and removes dead ops and
   unreachable blocks. The other pattern passes do none of this on their
   own; the ``passes=`` names run it with them.

pulse-canonicalize
   Removes a ``pulse.sync`` of one line and a sync of the lines another sync
   just produced; merges waits in a row on one line, each of a constant
   length and neither variable, into one; inlines a ``pulse.schedule_block``
   with no drives, readouts or waits; branches on ``pulse.is_set``; and
   inlines or drops ``scf.for`` loops that run once or never.

pulse-virtual-z
   Folds a ``pulse.shift_phase`` by a constant into the ``frame_phase_offset``
   of the drive that follows on its tone, after moving it past an
   ``scf.if`` whose arms both pass it through unchanged, into a schedule block that
   takes the tone next, or out of the end of a block. A frame rotation then
   costs no operation of its own.

pulse-fusion
   Merges two drives in a row on one line and tone, whose waveforms are
   square pulses of the same amplitude and of constant durations, with the
   same phase offset and no frame change folded into the second, into one
   drive of the summed duration. A wait for an answer or for a blocking
   classical call between them keeps them apart.

loop-invariant-code-motion
   MLIR's pass: hoists waveform construction and other loop-invariant
   operations out of ``scf.for`` bodies.

When a call binds a kernel's ``float`` parameters, the module with the
values substituted is folded with MLIR's canonicalization patterns, and the
virtual-Z and fusion patterns run again, if the module ran them, on the frame
changes and square pulses the values reach: a phase or an amplitude that was
a parameter folds or fuses once it is a number.

Checks
------

pulse-verify
   Checks that waveform durations are positive, that each drive and readout
   line is consumed once at each step, and, on a scheduled module, that the
   intervals on each physical line neither overlap nor move backwards.

pulse-classical-check
   On a scheduled module that declares classical calls: checks slot sizes,
   deadlines, answer timing, link capacity and the order of calls on each
   link. Scheduling such a module runs it, also for each call's values;
   simulation does not.

Schedules
---------

Each schedule assigns ``start_cycles`` and ``duration_cycles`` to every timed
operation, respecting data dependencies, syncs, the order of pulses on each
tone and the policies of schedule blocks. A wait for an answer or for a
blocking call holds back only the lines of its timeline, which it opens a new
epoch for; see :doc:`../user_guide/classical_calls`.

pulse-schedule-asap, pulse-schedule-alap
   Place each operation as early, or as late, as its dependencies allow.

pulse-schedule-rcp
   Resource-constrained: at most ``max-drives`` drives and ``max-readouts``
   readouts play at once, a readout's line and result are ready
   ``readout-latency`` cycles after it ends, and a unit stays busy
   ``switch-penalty`` cycles after each operation. ``alap=true`` gives the
   ``alap_rcp`` policy. ``pulse.compile`` takes these from ``MachineModel``.

pulse-inline-schedule-blocks
   Inlines every ``pulse.schedule_block``, ignoring its policy
   (``flatten_schedules=True``).

pulse-unroll-classical-loops
   Fully unrolls loops that hold classical calls; simulation runs it on its
   copy of the module before scheduling it.

Lowering
--------

``pulse-to-qop``, ``qop-to-cudm`` and ``cudm-to-llvm`` lower a scheduled
module for GPU execution; see :doc:`pipeline`.
