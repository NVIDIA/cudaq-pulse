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
clock starts operations only every few program cycles (see
:doc:`../user_guide/controller_backends`); ``cycles_on`` rounds a length
onto a line's grid in a kernel.

.. autofunction:: cudaq_pulse.cycles_on

.. autoclass:: cudaq_pulse.Rounding
   :members:
   :undoc-members:

Errors are on :doc:`errors`; the targets a device is built from are on
:doc:`targets`.

Calibrations and the transmon circuit model
-------------------------------------------

.. autoclass:: cudaq_pulse.targets.Calibration
   :members:

.. automodule:: cudaq_pulse.targets.transmon_model
   :members: TransmonCircuit, ChargeCoupledPair, anchored_charge_coupling_hz
