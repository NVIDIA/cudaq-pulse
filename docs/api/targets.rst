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

Drive and readout
-----------------

A qudit's ``drive`` and ``readout`` are components of their modality: a
``MicrowaveDrive`` and a ``DispersiveReadout`` for a transmon. They are
read-only values; ``component.replace(field=value)`` makes a changed copy.
Simulation picks its model by the component's type and refuses one it has no
model for. In a system file each is a mapping with its ``kind``
(``"microwave"``, ``"dispersive"``) and its fields.

.. autoclass:: cudaq_pulse.targets.Drive
   :members:

.. autoclass:: cudaq_pulse.targets.MicrowaveDrive
   :members:

.. autoclass:: cudaq_pulse.targets.Readout
   :members:

.. autoclass:: cudaq_pulse.targets.DispersiveReadout
   :members:

Couplings
---------

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

   from cudaq_pulse.targets import Qubit, TargetDescription, TransmonModel

   qubits = {
       i: Qubit(index=i, frequency_hz=f, anharmonicity_hz=-2.0e8, t1_us=50.0, t2_star_us=30.0)
       for i, f in enumerate([5.0e9, 5.2e9])
   }
   # Qudit 0 has four levels; qudit 1 has the model's default.
   chip = TargetDescription(name="chip", qubits=qubits, model=TransmonModel(levels={0: 4}))

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

.. autofunction:: cudaq_pulse.targets.transmon_17q

.. autofunction:: cudaq_pulse.targets.transmon_generic

Registering targets
-------------------

.. autofunction:: cudaq_pulse.targets.register_target

.. autofunction:: cudaq_pulse.targets.available_targets
