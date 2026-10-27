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

Errors are on :doc:`errors`; the targets a device is built from are on
:doc:`targets`.

Calibrations and the transmon circuit model
-------------------------------------------

.. autoclass:: cudaq_pulse.targets.Calibration
   :members:

.. automodule:: cudaq_pulse.targets.transmon_model
   :members: TransmonCircuit, ChargeCoupledPair, anchored_charge_coupling_hz
