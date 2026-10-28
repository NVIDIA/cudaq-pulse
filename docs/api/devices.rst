..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Device API
==========

.. autofunction:: cudaq_pulse.device

.. autofunction:: cudaq_pulse.get_device

.. autoclass:: cudaq_pulse.Target
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.Device
   :members:

.. autoclass:: cudaq_pulse.Qudit
   :members:

.. autoclass:: cudaq_pulse.Register
   :members:

Clocks
------

A device's ``clock_ghz`` is its program clock. A controller with a slower
clock, or several cycles per timestep, starts operations only every few
program cycles, its lines' :ref:`timestep <timestep>`; ``round_duration``
rounds a length to whole timesteps in a kernel.

.. autofunction:: cudaq_pulse.round_duration

.. autoclass:: cudaq_pulse.RoundDirection
   :members:
   :undoc-members:

Times with units
----------------

A bare int duration is a number of cycles of the program clock.
``20 * pulse.ns`` is a time: the compiler converts it to cycles exactly,
for the device the kernel compiles for, and rejects a time that is not a
whole number of cycles. A plain float in a duration is an error, since it
could mean either.

.. autoclass:: cudaq_pulse.Time

One nanosecond, microsecond, millisecond and second:

.. autodata:: cudaq_pulse.ns
   :no-value:

.. autodata:: cudaq_pulse.us
   :no-value:

.. autodata:: cudaq_pulse.ms
   :no-value:

.. autodata:: cudaq_pulse.s
   :no-value:

Errors are on :doc:`errors`; the targets a device is built from are on
:doc:`targets`.

Calibrations and the transmon circuit model
-------------------------------------------

.. autoclass:: cudaq_pulse.targets.Calibration
   :members:

.. automodule:: cudaq_pulse.targets.transmon_model
   :members: TransmonCircuit, ChargeCoupledPair, anchored_charge_coupling_hz
