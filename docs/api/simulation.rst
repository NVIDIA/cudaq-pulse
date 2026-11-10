..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Simulation API
==============

.. warning::

   Simulation is a research-preview interface with no stability or
   numerical-accuracy guarantee. See :doc:`../user_guide/gpu_execution` for
   the physical model and unsupported features.

.. autofunction:: cudaq_pulse.simulate

.. autoclass:: cudaq_pulse.SimulationResult
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.BranchCoverage
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.runtime.simulate.Measurement
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.runtime.simulate.Branch
   :members:
   :undoc-members:

GPU runtime
-----------

The GPU runtime is loaded on first use; its errors are on :doc:`errors`.

.. autofunction:: cudaq_pulse.runtime.require_gpu_runtime

.. autoclass:: cudaq_pulse.runtime.GpuRuntime
   :members:
