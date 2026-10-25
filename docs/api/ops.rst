..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Pulse Operations API
====================

Channel Access
--------------

.. autofunction:: cudaq_pulse.ops.get_drive_line

.. autofunction:: cudaq_pulse.ops.get_readout_line

Scheduling
----------

.. autofunction:: cudaq_pulse.ops.drive

.. autofunction:: cudaq_pulse.ops.readout

.. autofunction:: cudaq_pulse.ops.wait

.. autofunction:: cudaq_pulse.ops.sync

Phase and Frequency
-------------------

.. autofunction:: cudaq_pulse.ops.shift_phase

.. autofunction:: cudaq_pulse.ops.set_phase

.. autofunction:: cudaq_pulse.ops.shift_frequency

.. autofunction:: cudaq_pulse.ops.set_frequency

Waveform Constructors
---------------------

.. autofunction:: cudaq_pulse.ops.gaussian

.. autofunction:: cudaq_pulse.ops.square

.. autofunction:: cudaq_pulse.ops.drag

.. autofunction:: cudaq_pulse.ops.cosine

.. autofunction:: cudaq_pulse.ops.tanh_ramp

.. autofunction:: cudaq_pulse.ops.gaussian_square

.. autofunction:: cudaq_pulse.ops.custom

.. autofunction:: cudaq_pulse.ops.custom_samples

Waveform Arithmetic
-------------------

.. autofunction:: cudaq_pulse.ops.wf_add

.. autofunction:: cudaq_pulse.ops.wf_sub

.. autofunction:: cudaq_pulse.ops.wf_mul

.. autofunction:: cudaq_pulse.ops.wf_scale

.. autofunction:: cudaq_pulse.ops.wf_neg
