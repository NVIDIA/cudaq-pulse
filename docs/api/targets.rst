..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0


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
``MicrowaveDrive`` and a ``DispersiveReadout`` for a transmon. They hold the
qudit's physics only; the pulses it is tuned for are its calibrations
(below). They are read-only values; ``component.replace(field=value)`` makes
a changed copy.
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

.. autoclass:: cudaq_pulse.targets.ResidualZZ
   :members:
   :undoc-members:

Gate calibrations
-----------------

The pulses a device is tuned for are calibrations, by gate
(``pulse.gates``): a qudit's ``calibrations`` hold its ``X`` (and, if
published, ``X90``, ``Y``, ``Y90``) as a ``DragPulse`` and its ``READOUT`` as
a ``ReadoutPulse``; a coupling's hold its ``CZ`` as a ``FluxCZ``. A key
``(gate, (lower, upper))`` holds another transition's. ``get_calibration``
reads them, deriving what is not published:

.. code-block:: python

   import cudaq_pulse as pulse
   from cudaq_pulse.targets import DragPulse, Qubit, ReadoutPulse

   qubit = Qubit(
       0, 5.0e9, -2.0e8, 50.0, 30.0,
       calibrations={
           pulse.gates.X: DragPulse(duration_ns=40.0, sigma_ns=10.0),
           pulse.gates.READOUT: ReadoutPulse(duration_ns=300.0, integration_ns=400.0),
       },
   )
   qubit.get_calibration(pulse.gates.X90)   # X at half the amplitude of a pi rotation

In a system file a qudit's or coupling's ``calibrations`` is a list of
entries, each its ``gate``, its ``transition`` if not ``[0, 1]``, and its
fields. A qudit's ``metadata`` is the user's own data (JSON values): the
library never reads it, it does not make two devices differ, and system
files keep it.

.. automodule:: cudaq_pulse.gates
   :members: Gate

.. autoclass:: cudaq_pulse.targets.DragPulse
   :members:

.. autoclass:: cudaq_pulse.targets.ReadoutPulse
   :members:

.. autoclass:: cudaq_pulse.targets.FluxCZ
   :members:

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

.. autodata:: cudaq_pulse.targets.transmon.CZ_LAYERS
   :no-value:

.. autofunction:: cudaq_pulse.targets.transmon_generic

Registering targets
-------------------

.. autofunction:: cudaq_pulse.targets.register_target

.. autofunction:: cudaq_pulse.targets.available_targets
