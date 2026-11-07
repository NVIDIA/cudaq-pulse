..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Controller Backends API
=======================

See :doc:`../user_guide/controller_backends` for the exported program and
how backends are found, and :doc:`errors` for ``ControllerLimitError`` and
the ``BackendError`` exceptions.

Export and limits
-----------------

.. automethod:: cudaq_pulse.Lowered.export_controller
   :no-index:

.. autodata:: cudaq_pulse.controller.FORMAT

.. autoclass:: cudaq_pulse.controller.DriveLine
   :members: qudit, delay_ns

.. autoclass:: cudaq_pulse.controller.ReadoutLine
   :members: qudit, delay_ns

.. autoclass:: cudaq_pulse.controller.WaveformMemoryScope
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.Event
   :members:
   :undoc-members:

.. autofunction:: cudaq_pulse.controller.check

.. autofunction:: cudaq_pulse.controller.export

.. autofunction:: cudaq_pulse.controller.unroll

.. autofunction:: cudaq_pulse.controller.sample_waveform

.. autofunction:: cudaq_pulse.controller.evaluate_expression

.. autoclass:: cudaq_pulse.controller.ExpressionTerm
   :members:
   :undoc-members:

.. autofunction:: cudaq_pulse.controller.shape_key

What classical functions cost
-----------------------------

.. autofunction:: cudaq_pulse.controller.resources

.. autofunction:: cudaq_pulse.controller.analyze

.. autoclass:: cudaq_pulse.controller.ResourceReport
   :members:

.. autoclass:: cudaq_pulse.controller.FunctionResources
   :members:

.. autoclass:: cudaq_pulse.controller.LinkUsage

.. autoclass:: cudaq_pulse.controller.FunctionCost
   :members:

.. autoclass:: cudaq_pulse.controller.OpClass
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.controller.OpCounts
   :members: total, replace

.. autoclass:: cudaq_pulse.controller.OpCycles
   :members: replace

What a program uses of each processor
-------------------------------------

``lowered.controller_usage`` is :func:`~cudaq_pulse.controller.usage` of a
lowered program. A program over a processor's limits does not compile: its
``ControllerLimitError`` carries the report as ``usage``, with each limit
exceeded and by how much in ``usage.exceeded``.

.. autofunction:: cudaq_pulse.controller.usage

.. autoclass:: cudaq_pulse.controller.UsageReport
   :members:

.. autoclass:: cudaq_pulse.controller.LimitExceeded
   :members:

.. autoclass:: cudaq_pulse.controller.ControllerUsage

.. autoclass:: cudaq_pulse.controller.ProcessorUsage

.. autoclass:: cudaq_pulse.controller.FunctionUsage
   :members:

Backends
--------

.. autodata:: cudaq_pulse.backends.GROUP

.. autoclass:: cudaq_pulse.backends.Backend
   :members:

.. autofunction:: cudaq_pulse.backends.available

.. autofunction:: cudaq_pulse.backends.get

.. autofunction:: cudaq_pulse.backends.register

.. autofunction:: cudaq_pulse.backends.unregister

.. autofunction:: cudaq_pulse.backends.validate
