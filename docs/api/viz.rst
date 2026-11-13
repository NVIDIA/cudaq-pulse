..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0


Visualization API
=================

Plotting needs matplotlib, from the ``[viz]`` extra:
``pip install "cudaq-pulse[viz]"``. ``ScheduledKernel.timeline(figsize=, title=)``
calls ``plot_schedule`` on a compiled module.

.. autofunction:: cudaq_pulse.viz.plot_schedule

.. autofunction:: cudaq_pulse.viz.save_schedule
