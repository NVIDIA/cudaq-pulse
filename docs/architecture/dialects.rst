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
Waveforms are ordinary values. Durations are in clock cycles of the target,
and scheduling records each timed op's ``start_cycles`` and ``duration_cycles``.

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
