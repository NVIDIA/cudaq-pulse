..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Devices and Targets
===================

A *target* is the calibrated physics of a machine: qudit frequencies and
anharmonicities, couplings, crosstalk, T1/T2 and drive calibration, and the
pulse clock. A *device* is a target together with a *layout*, which maps
the qudit numbering a program uses to the target's carriers. Both are
read-only.

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)   # shipped target
   chip = pulse.device({"target": "transmon_krinner_17q", "layout": [4, 5, 6]})
   chip = pulse.device("chip.json")                            # the same, from a file

   q = chip.qudit_handles[0]          # a handle naming carrier 4
   pair = chip.qudit_handles[0:2]     # a Register: carriers 4 and 5
   some = chip.qudit_handles[[0, 2]]  # carriers 4 and 6

   chip.qubit_freq[0]                 # 0-1 frequency of qudit 0, in Hz
   chip.transition_freq[0, 1, 2]      # 1-2 frequency of qudit 0
   chip.qubit_freq[0] = 5.01e9        # ValueError: assignment destination is read-only

.. :spellcheck-enable:

A config is a mapping (or a JSON file of one) with ``"target"``, a
registered target name or a ``cudaq_pulse.targets.Target``, and optionally
``"layout"``. ``layout[i]`` is the carrier that qudit ``i`` names; by
default every carrier of the target, in order.

Qudit handles
-------------

A ``Qudit`` names a carrier of one device; it is identity, not location. A
kernel called with handles runs on their device. Passing handles of two
different devices to one call raises ``DeviceMismatchError``, and passing
the same qudit twice raises ``TypeError``.

Devices compare by content: two devices built from the same config are
equal, share compiled kernels, and their handles can be mixed. A layout is
part of the content.

To change a frequency, edit the config, or pass the frequency to the kernel
as a ``float`` argument and set it on the tone with ``set_frequency``; the
argument stays symbolic, so a frequency sweep compiles once.

The process target
------------------

Scripts that do not name a device can run on a *process target*, chosen
outside the script:

.. :spellcheck-disable:

.. code-block:: bash

   CUDAQ_PULSE_TARGET=transmon_krinner_17q python script.py
   python -m cudaq_pulse --target transmon_generic:4 script.py --script-args

.. :spellcheck-enable:

The launcher sets ``CUDAQ_PULSE_TARGET`` for the script and passes the
script's own arguments through unparsed. A kernel call with no qudit handles
uses the process target; without one it raises ``NoDeviceError``.
``pulse.get_device()`` with no argument returns the process target's device.

Registered names are ``transmon_krinner_17q``, ``transmon_generic:N``,
``rydberg_chain:N`` and ``rydberg_square:RxC``;
``cudaq_pulse.targets.register_target`` adds more.

``transmon_krinner_17q`` is the distance-three surface-code device of
Krinner et al., Nature 605, 669 (2022), built from the values the paper
publishes. Its qudits 0-16 are D1-D9, X1-X4 and Z1-Z4 (``Qubit.label``), and
its 24 couplings are the auxiliary-data pairs of the stabilizers, each with
its CZ time step and interaction frequency.
``python -m cudaq_pulse.doctor`` lists them and reports the process target.

The paper's J / 2 pi ~ 7 MHz is the |11> <-> |20> coupling of the CZ,
sqrt(2) times the exchange coupling (Lacroix, PhD thesis, ETH Zurich (2025),
on the same device). The target simulates the exchange J_qq = 7 MHz /
sqrt(2) = 4.95 MHz on every pair, and the measured |11> <-> |20> coupling of
7.4 MHz on X2-D4 (Hellings et al., arXiv:2503.04610); each coupling records
its |11> <-> |20> value in ``gate_params["coupling_11_20_hz"]``. With three
levels the couplings give an idle ZZ of -2.8 to -4.9 kHz at second order
(``cudaq_pulse.targets.transmon.krinner_idle_zz_hz``), below the paper's
"below about 8 kHz". Each qubit
also records its Hahn-echo T2 and thermal excited-state population from
Tab. I (``Target.t2_echo_times``, ``Target.thermal_populations``); the
simulation does not use them yet.
