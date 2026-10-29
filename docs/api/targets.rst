..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.


Targets API
===========

A target describes a device's qudits, couplings and decoherence; a device
(:doc:`devices`) pairs one with a qudit layout.

.. autoclass:: cudaq_pulse.targets.TargetDescription
   :members:

.. autoclass:: cudaq_pulse.targets.Qubit
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.targets.Coupling
   :members:
   :undoc-members:

.. autoclass:: cudaq_pulse.targets.CrosstalkEntry
   :members:
   :undoc-members:

Simulation models
-----------------

A target's ``model`` says how the simulator treats its qudits and how many
levels each has. Each model class declares its levels; an instance sets a
qudit's own, keyed by the target's qudit index:

.. code-block:: python

   TargetDescription(name="chip", qubits=qubits, model=TransmonModel(levels={0: 4}))

The model also gives a drive line its transitions: ``line.tones`` has a tone
for each ``(lower, upper)`` pair of the qudit's levels, at the frequency
``transitions_hz`` gives it, and the line's own tone, ``line.tone``, is the
one of ``default_transition``. A model numbers its levels from 0; one whose
levels have names says in its docstring which number each name has.

.. autoclass:: cudaq_pulse.targets.SimulationModel
   :members: levels, default_transition, levels_of, energies_hz, transitions_hz

.. autoclass:: cudaq_pulse.targets.TransmonModel

Shipped targets
---------------

.. autofunction:: cudaq_pulse.targets.transmon_krinner_17q

.. autofunction:: cudaq_pulse.targets.transmon_generic

Registering targets
-------------------

.. autofunction:: cudaq_pulse.targets.register_target

.. autofunction:: cudaq_pulse.targets.available_targets
