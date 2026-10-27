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

.. autoclass:: cudaq_pulse.controller.ControllerLimits
   :members: as_dict

.. autofunction:: cudaq_pulse.controller.check

.. autofunction:: cudaq_pulse.controller.export

.. autofunction:: cudaq_pulse.controller.sample_waveform

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

.. autodata:: cudaq_pulse.controller.OP_CLASSES

Backends
--------

.. autodata:: cudaq_pulse.backends.GROUP

.. autoclass:: cudaq_pulse.backends.Backend
   :members:

.. autofunction:: cudaq_pulse.backends.available

.. autofunction:: cudaq_pulse.backends.get

.. autofunction:: cudaq_pulse.backends.register

.. autofunction:: cudaq_pulse.backends.unregister
