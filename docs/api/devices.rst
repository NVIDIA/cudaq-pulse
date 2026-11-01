..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Device API
==========

.. autofunction:: cudaq_pulse.device

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

Control systems
---------------

A device's ``control_system`` is the controllers that play it, their wiring
and the links beside them. ``pulse.controller("vendor.model", drive_lines=...,
readout_lines=...)`` loads a ``Controller`` from an installed description (or
a TOML file, by path) with the installation's wiring; each file is parsed
once. The wiring includes ``peer_latency_cycles``: the latency of an
outcome the controller reads to each receiving controller, counted in the
sending controller's own cycles and keyed by the receiver's index in the
control system (one number for every receiver; a receiver not listed: 0).
``Device.as_dict()`` writes the whole QPU as one system file, format
``cudaq-pulse.system/1``, which ``pulse.device()`` reads back, raising a
``DeviceDefinitionError`` that lists every problem with its line.
``python -m cudaq_pulse.doctor --system FILE`` checks a system file.

A controller's ``processors`` run the classical functions called on their
links. Compiling checks each function there (the operations it uses, its
size, its state and its compute time at the processor's ``cycles_per_op``
against its ``max_latency_ns``) and each processor's totals; a processor
without ``"loop"`` runs loops unrolled, and their size is counted so.
``lowered.controller_usage`` is the report: ``print`` it for a table, or
read ``.within_limits`` and ``.controllers[i].processors[j]``. In a
description or a system file, a processor is a ``[[processors]]`` table,
with ``[processors.dispatch_keys]`` beneath it.

.. autoclass:: cudaq_pulse.Controller
   :members: as_dict, supports_capability

.. autoclass:: cudaq_pulse.ControlSystem
   :members: as_dict

.. autoclass:: cudaq_pulse.Processor
   :members:

.. autoclass:: cudaq_pulse.Capability
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.Link
   :no-index:

Calibrations and the transmon circuit model
-------------------------------------------

.. autoclass:: cudaq_pulse.targets.Calibration
   :members:

.. automodule:: cudaq_pulse.targets.transmon_model
   :members: TransmonCircuit, ChargeCoupledPair, anchored_charge_coupling_hz
