..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Classical Calls API
===================

See :doc:`../user_guide/classical_calls` for how calls are scheduled,
simulated and emulated.

Functions and links
-------------------

.. autofunction:: cudaq_pulse.classical.function

.. autofunction:: cudaq_pulse.classical.external

.. autofunction:: cudaq_pulse.classical.dispatch_key

.. autoclass:: cudaq_pulse.classical.ClassicalDecorator

.. autoclass:: cudaq_pulse.classical.ClassicalFunction
   :members: submit, specialize

.. autoclass:: cudaq_pulse.classical.Pending
   :members: result

.. autoclass:: cudaq_pulse.classical.Link
   :members:

Measurements and randomness
---------------------------

.. autoclass:: cudaq_pulse.classical.Bits

.. autodata:: cudaq_pulse.Length
   :no-value:

   ``typing.Annotated``, re-exported to state a fixed list length:
   ``pulse.Length[pulse.Bits, 8]`` takes or answers eight bits and
   ``pulse.Length[list[int], 8]`` eight numbers. The length is a positive
   whole number.

.. autofunction:: cudaq_pulse.classical.random_index

Reports
-------

.. autoclass:: cudaq_pulse.compiler.ClassicalCall
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.compiler.Segment
   :members:
   :undoc-members:

Wire format
-----------

.. automodule:: cudaq_pulse.classical.wire
   :members: schema, encode_request, decode_request, encode_answer, decode_answer, native, HEADER_BYTES, REQUEST_MAGIC, RESPONSE_MAGIC

CUDA-Q QEC decoders
-------------------

``pulse.classical`` also takes CUDA-Q QEC's real-time decoding functions,
``cudaq_qec.qec.enqueue_syndromes``, ``get_corrections`` and
``reset_decoder``, with the options above (``max_latency_ns`` is 0 unless
given for the calls that answer nothing); see :ref:`the user guide
<cudaq-qec-decoders>`.

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
