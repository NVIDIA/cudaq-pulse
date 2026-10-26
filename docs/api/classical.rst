..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Classical Calls API
===================

See :doc:`../user_guide/classical_calls` for how calls are scheduled,
simulated and emulated.

Functions and links
-------------------

.. autofunction:: cudaq_pulse.classical.function

.. autofunction:: cudaq_pulse.classical.external

.. autoclass:: cudaq_pulse.classical.ClassicalFunction
   :members: submit

.. autoclass:: cudaq_pulse.classical.Pending
   :members: result

.. autoclass:: cudaq_pulse.classical.Link
   :members:

.. autoclass:: cudaq_pulse.compiler.MachineModel
   :members: link_id, link_name, link
   :no-index:

Types
-----

.. autoclass:: cudaq_pulse.classical.Bits
.. autoclass:: cudaq_pulse.classical.Levels
.. autoclass:: cudaq_pulse.classical.IQ
.. autoclass:: cudaq_pulse.classical.Float
.. autoclass:: cudaq_pulse.classical.Integer
.. autoclass:: cudaq_pulse.classical.Byte
.. autoclass:: cudaq_pulse.classical.Floats
.. autoclass:: cudaq_pulse.classical.Integers
.. autoclass:: cudaq_pulse.classical.Bytes

.. autofunction:: cudaq_pulse.classical.random_index

Reports
-------

.. autoclass:: cudaq_pulse.compiler.ClassicalCall
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.compiler.Epoch
   :members:
   :undoc-members:

Wire format
-----------

.. automodule:: cudaq_pulse.classical.wire
   :members: schema, encode_request, decode_request, encode_answer, decode_answer, native

QEC decoders
------------

.. automodule:: cudaq_pulse.classical.qec
   :members: Decoder, enqueue_syndromes, enqueue_syndromes_mapped, get_corrections, reset_decoder

Emulation
---------

.. automodule:: cudaq_pulse.classical.emulation
   :members: export, manifest, emulate, Transport, InProcess, UdpChannel, serve_udp, EmulationResult, CallTrace, RealtimeChannel, device_call_text

Calibration
-----------

.. automodule:: cudaq_pulse.classical.calibration
   :members:

Loops between shots
-------------------

.. automodule:: cudaq_pulse.classical.loop
   :members:

Drifting devices
----------------

.. automodule:: cudaq_pulse.runtime.drift
   :members:
