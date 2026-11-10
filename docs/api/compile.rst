..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Compilation API
===============

.. autofunction:: cudaq_pulse.compile

.. autoclass:: cudaq_pulse.CompiledKernel
   :members:

.. autoclass:: cudaq_pulse.Lowered
   :members:

.. autoclass:: cudaq_pulse.Pass
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.compiler.IRLevel
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.compiler.ScheduledEvent
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.compiler.CompileMetrics
   :members:
   :undoc-members:

.. autodata:: cudaq_pulse.compiler.DEFAULT_PASSES
   :no-value:

   The passes ``pulse.compile`` runs when ``passes`` is not given, in order.
