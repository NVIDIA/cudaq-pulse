..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

MLIR Dialects
=============

cudaq-pulse defines three MLIR dialects that form a layered IR stack. A kernel
compiles to the ``pulse`` dialect; simulation lowers it through ``qop`` to
``cudm`` and then to LLVM calls into the cuDensityMat runtime. Every op of
each dialect is listed in its reference page, generated from the ODS
definitions in ``core/mlir/include/cudaq-pulse/Dialect/``.

.. toctree::
   :maxdepth: 1

   dialects/pulse
   dialects/qop
   dialects/cudm

Pulse
-----

The pulse program as written: qudits, the lines and tones that drive and read
them out, waveforms, and timing. Lines and tones are linear values, consumed
once and returned updated by each op that uses them, so the order of the
pulses on a line is explicit in the IR and ``pulse-verify`` can check it.
Waveforms are ordinary values. Durations are in clock cycles of the target,
and scheduling records each timed op's ``start_vtu`` and ``duration_vtu``.

- Types: ``!pulse.qref``, ``!pulse.drive_line``, ``!pulse.readout_line``,
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

QOp
---

Backend-agnostic quantum operator algebra. ``pulse-to-qop`` builds one
``qop.lindblad`` per program: a Hamiltonian of ``qop.make_sum`` terms, each a
``qop.make_product`` of ``qop.spin`` or ``qop.matrix_leaf`` operators with a
``qop.const_scalar`` or, for a drive envelope, a ``qop.callback_scalar``
coefficient, and the collapse operators of the target's T1 and T2. The
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
