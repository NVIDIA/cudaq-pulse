..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Compilation API
===============

This page documents ``pulse.compile`` and ``pulse.load`` and the objects
they produce: the compiled and scheduled kernels, the passes that
transform them, the idle windows and exposure a scheduled kernel reports,
and the compile metrics. See :doc:`../user_guide/compilation`
for how calling a kernel compiles it and reuses the compilation.

.. autofunction:: cudaq_pulse.compile

.. autofunction:: cudaq_pulse.load

.. autoclass:: cudaq_pulse.CompiledKernel
   :members:

.. autoclass:: cudaq_pulse.ScheduledKernel
   :members:

.. autoclass:: cudaq_pulse.Pass
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.IdleReplacement

.. autoclass:: cudaq_pulse.compiler.IdleWindow
   :members:

.. autoclass:: cudaq_pulse.compiler.IdleWindows

.. autoclass:: cudaq_pulse.compiler.QuditExposure
   :members:

.. autoclass:: cudaq_pulse.compiler.Exposure

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
