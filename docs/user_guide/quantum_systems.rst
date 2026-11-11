..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

Quantum systems
===============

A *quantum system* is the calibrated physics of a machine (qudit
frequencies and anharmonicities, couplings, residual ZZ, T1/T2, the pulses
its gates are calibrated for, and the pulse clock) together with a
*layout*, which maps the qudit numbering a program uses to its carriers.
It is read-only.

.. code-block:: python

   import cudaq_pulse as pulse

   qs = pulse.quantum_system("transmon_surface17")                  # shipped quantum system
   chip = pulse.quantum_system("transmon_surface17", layout=[4, 5, 6])
   chip = pulse.QuantumSystem.load("chip.json")              # the same, from a system file

   q = chip.qudits[0]          # a handle naming carrier 4
   pair = chip.qudits[0:2]     # a Register: carriers 4 and 5
   some = chip.qudits[[0, 2]]  # carriers 4 and 6

   chip.qudit_freq[0]                 # 0-1 frequency of qudit 0, in Hz
   chip.transition_freq[0, 1, 2]      # 1-2 frequency of qudit 0
   chip.qudit_freq[0] = 5.01e9        # ValueError: assignment destination is read-only

``pulse.quantum_system`` takes a registered name, or builds a quantum system
from ``name=`` and its ``qudits=``, ``couplings=``, ``residual_zz=``,
``model=`` and ``attribution=``; either way optionally ``layout=``,
``calibration=`` and ``cooldown=``, ``control_system=`` and ``clock_ghz=``.
A string is always a name: ``pulse.QuantumSystem.load(path)`` reads a system file
(as ``QuantumSystem.as_dict()`` writes one, and ``json.dumps`` saves), and
``pulse.QuantumSystem.from_dict(data)`` its data, with the same keys.
``layout[i]`` is the carrier that qudit ``i`` names; by default every
carrier, in order. The ``model`` (``TransmonModel`` by default) decides the
operator of the couplings, for transmons the exchange
g (a^dagger b + a b^dagger):

.. code-block:: python

   from cudaq_pulse.quantum_systems import Coupling, FluxCZ, PhysicalQudit

   cz = FluxCZ(duration_ns=98.0)
   qpu = pulse.quantum_system(
       name="pair",
       qudits={i: PhysicalQudit(i, f, -2.0e8, 50.0, 30.0) for i, f in enumerate([4.8e9, 6.9e9])},
       couplings=[Coupling((0, 1), strength_hz=5.7e6, calibrations={pulse.gates.CZ: cz})],
   )

A coupling's ``strength_hz`` is g / 2 pi; ``None`` says it is not measured
yet, and such a system compiles, schedules and exports while
``pulse.simulate`` refuses it, naming the coupling. ``residual_zz`` lists
any ZZ between two qudits beyond what the couplings produce, as
``ResidualZZ((0, 1), rate_hz=-2.0e4)``, rate zeta / 2 pi with
zeta = E11 - E10 - E01 + E00. The values are checked when the system is
built, and one ``QuantumSystemDefinitionError`` lists every problem at its path
(``couplings[1].qudits[0]``).

``qs.replace(...)`` makes a variant of a quantum system, as
``dataclasses.replace`` does: the same build keywords, each defaulting to
the system's own value (``qudits`` to its ``physical_qudits``). Passing a
``QuantumSystem`` to ``pulse.quantum_system`` is a ``TypeError``.

.. code-block:: python

   q0 = qs.physical_qudits[0]
   q0 = q0.replace(frequency_hz=q0.frequency_hz + 1e6)
   detuned = qs.replace(qudits={**qs.physical_qudits, 0: q0})
   first_two = qs.replace(layout=[0, 1])

Arguments or a system file with problems raise ``QuantumSystemDefinitionError``, a
``ValueError``, which lists every problem rather than the first. Each is a
``pulse.Problem`` with its path (the argument, or its place in the file),
the offending value and a fix; for a file, the line and column too:

.. code-block:: text

   QuantumSystemDefinitionError: 2 problems in quantum system file lab.json:
     lab.json:3:20: layout[2]: carrier 99 is not in quantum system 'transmon_surface17' (it has 0..16)
     lab.json:4:15: cooldown: 'x' is given without a calibration; add "calibration", a shipped name or a file path, or remove "cooldown"

``python -m cudaq_pulse.doctor --system lab.json`` runs the same checks
without a script, one problem per line; ``--json`` prints them as JSON, and
it exits 1 when the file has a problem.

Qudit handles
-------------

A ``Qudit`` names a carrier of one quantum system; it is identity, not location. A
kernel called with handles runs on their quantum system. Passing handles of two
quantum systems to one call, or handles of another quantum system to a kernel compiled for
one, raises ``QuantumSystemMismatchError``, and passing the same qudit twice raises
``TypeError``.

Quantum systems compare by content: two quantum systems built from the same values are
equal and share compiled kernels, but each is its own quantum system, so their
handles do not mix in one call. A layout is part of the content, and an
empty one is a ``QuantumSystemDefinitionError``: leave ``layout`` out for every
carrier.

To change a frequency, edit the quantum system, or pass the frequency to the kernel
as a ``float`` argument and set it on the tone with ``set_frequency``; the
argument stays symbolic, so a frequency sweep compiles once.

Gate calibrations
-----------------

The pulses a quantum system is tuned for are its gate calibrations, named by
``pulse.gates`` (``X``, ``X90``, ``Y``, ``Y90``, ``READOUT``, ``CZ``). In a
kernel a line gives its qudit's, ready to play; outside one, a qudit handle
and a coupling give the same values:

.. code-block:: python

   qs = pulse.quantum_system("transmon_surface17")

   @pulse.kernel
   def x_then_measure(q):
       d = get_drive_line(q)
       r = get_readout_line(q)
       x = d.get_calibration(pulse.gates.X)        # a DragPulse
       drive(d, x)
       drive(d, x.replace(amplitude=x.amplitude / 4))
       sync(d, r)
       readout(r, r.get_calibration(pulse.gates.READOUT))

   qs.qudits[4].get_calibration(pulse.gates.X).amplitude         # 0.1313: a pi rotation
   a, d = qs.qudit("X1"), qs.qudit("D2")
   qs.coupling(a, d).get_calibration(pulse.gates.CZ).duration_ns   # 98.0

What a quantum system does not publish is derived: an ``X`` without an amplitude
gets the one that makes its area a pi rotation at the qudit's drive scale,
``X90`` is ``X`` at half the amplitude, and ``Y`` and ``Y90`` are ``X`` and
``X90`` at +90 degrees. A ``CZ`` or ``READOUT`` the quantum system does not publish
is a ``NotFoundError`` naming the coupling or qudit, and so is another
transition's (``transition=(1, 2)``) calibration. A ``FluxCZ`` does not play:
flux lines are not modeled yet, so a kernel leaves the qudits idle for its
``duration_ns``.

The process quantum system
--------------------------

A script can leave the choice of quantum system to the command line with a *process
target*:

.. code-block:: bash

   CUDAQ_PULSE_QUANTUM_SYSTEM=transmon_surface17 python script.py
   python -m cudaq_pulse --quantum-system transmon_lattice:4 script.py --script-args

The launcher sets ``CUDAQ_PULSE_QUANTUM_SYSTEM`` for the script and passes the
script's own arguments through unparsed. ``pulse.quantum_system()`` with no
argument returns the process quantum system's quantum system, whose handles the script then
passes to its kernels; a kernel call with no qudit handles always raises
``NoQuantumSystemError``.

Registered names are ``transmon_surface17`` and ``transmon_lattice:N``
(``transmon_lattice:N:ring`` and ``transmon_lattice:N:grid`` for the other
topologies), and a system's ``name`` is the name it is registered under:
``transmon_lattice(4).name`` is ``"transmon_lattice:4"``.
``cudaq_pulse.quantum_systems.register_quantum_system`` adds more.

``transmon_surface17`` is the distance-three surface-code quantum system of
Krinner et al., Nature 605, 669 (2022), built from the values the paper
publishes. Its qudits 0-16 are D1-D9, X1-X4 and Z1-Z4 (``PhysicalQudit.label``), and
its 24 couplings are the auxiliary-data pairs of the stabilizers, each with
its CZ's duration, buffers and interaction frequency (its ``CZ``
calibration, below).
``cudaq_pulse.quantum_systems.transmon.CZ_LAYERS`` gives the CZ time steps of a
cycle, each a tuple of (auxiliary, data) qudit pairs.
``python -m cudaq_pulse.doctor`` lists them and reports the process quantum system.

The paper's J / 2 pi ~ 7 MHz is the ``|11> <-> |20>`` coupling of the CZ at
each pair's interaction point (Lacroix, PhD thesis, ETH Zurich (2025), on the
same quantum system); X2-D4's is measured, 7.4 MHz at 4.4 GHz (Hellings et al.,
arXiv:2503.04610). The simulated exchange comes from the
quantum system's calibration (below): one device-wide charge coupling g_c, and the
exchange J_qq = g_c <0|n_a|1> <0|n_d|1> at the idle points is 5.67-6.08 MHz.
The idle ZZ of the charge-coupled pairs is -4.5 to -7.3 kHz
(``cudaq_pulse.quantum_systems.transmon.transmon_surface17_idle_zz_hz``), below the
paper's "below about 8 kHz"; simulated with three levels, the couplings
reproduce it within a few percent. Each qubit also records its Hahn-echo T2
and thermal excited-state population from Tab. I (``QuantumSystem.t2_echo_times``,
``QuantumSystem.thermal_populations``): a simulation starts each qubit in its
thermal mixture and relaxes it there, and dephases at the Hahn-echo T2 with
``simulate(..., dephasing="t2_echo")``.

Calibrations
------------

A calibration holds the values that are fitted rather than published, apart
from the quantum system: for the ``transmon_surface17`` quantum system, the shipped
``transmon_surface17`` calibration holds each qubit's circuit (E_C, E_J,max and the SQUID asymmetry d, fitted to Tab. I and the
other sweet spot of Fig. 1c), each pair's charge coupling, the flux-line
model of Hellings et al., the in-model CZ knobs of all 24 pairs, crosstalk,
defect bands and feedline groups. Every value records its
``kind`` (``published``, ``digitized`` or ``fitted``), ``source``,
``uncertainty`` and ``cooldown``, one of the file's cooldowns.

.. code-block:: python

   qs = pulse.quantum_system("transmon_surface17", calibration="transmon_surface17", cooldown="2022-main")
   cal = qs.calibration
   cal.entry("qudits", "D4", "ec")        # value with its provenance
   cal.exchange_hz("X2-D4")               # J_qq at the idle points
   cal.exchange_hz("X2-D4", data_hz=4.4e9)  # ... with D4 tuned to 4.4 GHz
   cal.idle_zz_hz("X2-D4")
   cal.flux_for_detuning("D4", 500e6)     # flux quanta for a +500 MHz detuning
   cal.detuning_for_flux("X2", 0.25)

``calibration=`` is a shipped name, the path of a calibration file or a
``cudaq_pulse.quantum_systems.Calibration``; the ``transmon_surface17`` quantum system uses the
``transmon_surface17`` calibration by default. Its couplings replace the system's, so quantum systems with different
calibrations compile separately. Detunings are in Hz; for qubits whose
circuit, including d, is calibrated (``Calibration.flux_qudits``, all 17
here) ``flux_for_detuning`` and ``detuning_for_flux`` convert to and from
flux quanta. Every value in the shipped file is published, digitized from a
published figure, or fitted, and cites its source; fitted values also name
the fitting method.

The coupling is one device-wide charge coupling, g_c = 3.12 MHz
(``coupling.charge_coupling_g_c.calibration_A``), fitted so that X2-D4's
``|11> <-> |20>`` coupling is Hellings et al.'s measured 7.4 MHz at 4.4 GHz.
At each pair's own interaction point it gives J_2 = 7.3-9.6 MHz, the idle
exchange is 5.67-6.08 MHz and the idle ZZ -4.5 to -7.3 kHz. The CZ knobs
are fitted with the same coupling, so the simulated quantum system and its gates
agree.

The CZ knobs (``cz.knobs.per_pair``) are those of the sudden-net-zero pulse
the quantum system uses (Negirneac et al., PRL 126, 220502 (2021)): the plateau
duration t_p, the 2.5 ns transition section t_mid, its amplitude B and the
amplitude offset, fitted in a closed two-transmon model (five levels each,
no rotating-wave approximation, ideal flux line, no spectators) for a
conditional phase of 180 degrees and the least leakage. Their mean
duration, 2 t_p + t_mid, is 65.0 +- 4.8 ns against the published 68(7) ns.
Each record names the coupling it was fitted with in ``coupling_used``,
the device-wide g_c. Each record keeps its predicted phase and leakage and its ``flags``:
nine pairs leak at least 1e-4, a floor set by the 0.417 ns time grid, and
Z2-D7 defaults to an unphysical B = 1.934 (its data qubit moves only 261
MHz), with the in-range alternatives in ``alternatives``.
