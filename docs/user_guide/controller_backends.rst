..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Controller Backends
===================

A controller backend runs compiled kernels on a quantum controller: it
lowers a scheduled program to the controller's own instructions and
waveform memory, and runs it there. Backends live outside cudaq-pulse, in
their own distributions, and cudaq-pulse finds the installed ones by an
entry point. ``examples/controller_backend/`` is a complete toy backend.

Exporting a program
-------------------

``Lowered.export_controller()`` returns a compiled kernel as JSON-serializable
data, format ``cudaq-pulse.controller/1``. Its ``lines`` give each line's
role (``drive`` or ``readout``) and qudit, and its ``events`` every drive,
readout, wait and classical call in program order, once per loop iteration
and both arms of a branch:

``start_vtu``, ``duration_vtu``, ``epoch``
    The final schedule, in clock cycles. A backend places every event at its
    start and does not schedule again.
``waveform``, ``amplitude``
    The waveform's expression (``{"shape": "gaussian", "params":
    {"duration": ..., "amplitude": ..., "sigma": ...}}``, with ``scale``,
    ``neg``, ``add``, ``sub`` and ``mul`` nodes over ``of``) and its overall
    complex amplitude. With ``export_controller(sample_rate_ghz=...)``, a
    whole multiple of the clock, each event also carries the envelope's
    ``samples``; ``cudaq_pulse.controller.sample_waveform`` computes them for
    one waveform.
``frequency_hz``, ``phase_rad``
    The tone's frequency and frame phase as the event starts, with every
    frame change and virtual Z before it applied, or ``null`` where a branch
    leaves them different.
``conditions``
    The measurement-conditioned branches the event lies in: the branch, the
    arm, and the readout events and answers the condition is computed from.
    ``test`` is ``"is_set"`` for one readout's outcome and ``"answer"`` for
    one answer value.
``function``, ``measurements``, ``wait``, ``timeout_ns``
    For a classical call's ``submit``, the function and the readout events
    whose measurements it sends; for its ``answer``, whether the wait is
    ``bounded`` (by the latency budget), ``blocking`` or has a ``timeout``.

``epochs``, ``calls``, ``links`` and ``manifest`` are those of
``Lowered.export_emulation()`` (see :doc:`classical_calls`).

Controller limits
-----------------

``ControllerLimits`` states what a controller can play: its clock, how
closely two operations on one line can follow each other, its waveform
memory and tones per line, which acquirer reads each qudit and how long a
readout holds it, the least feedback latency from a readout to an operation
that branches on it, and whether it can time out a wait for an answer.
Given to the machine model, they make every compilation check the scheduled
program, so that a program the controller cannot play fails at compile time
with ``ControllerLimitError``:

.. :spellcheck-disable:

.. code-block:: python

   from cudaq_pulse.compiler import MachineModel
   from cudaq_pulse.controller import ControllerLimits

   limits = ControllerLimits(clock_ghz=1.0, issue_vtu=2, min_wait_vtu=2,
                             acquirers={0: 0, 1: 0}, acquirer_hold_vtu=10,
                             answer_timeouts=False)
   machine = MachineModel(max_concurrent_readouts=1, controller=limits)
   lowered = pulse.compile(kernel, device=dev, machine=machine).lower(q)

.. :spellcheck-enable:

``cudaq_pulse.controller.check(lowered, limits)`` returns the violations
without raising. Without ``controller``, compiling checks nothing more.

Writing a backend
-----------------

A backend is an object with a ``name`` and three methods, the
``cudaq_pulse.backends.Backend`` protocol: ``machine_model()`` returns the
machine model to compile for its controller, ``emit(program)`` takes an
exported program and returns the controller's, and ``run(artifact,
shots=...)`` runs it. Its distribution registers it under its name in the
``cudaq_pulse.backends`` entry-point group, naming the backend or a callable
that makes one:

.. :spellcheck-disable:

.. code-block:: toml

   [project.entry-points."cudaq_pulse.backends"]
   toy = "toy_controller:ToyBackend"

.. code-block:: python

   from cudaq_pulse import backends

   backends.available()            # ["toy"]: installed, none imported
   backend = backends.get("toy")   # loaded on first use
   lowered = pulse.compile(kernel, device=dev, machine=backend.machine_model()).lower(q)
   result = backend.run(backend.emit(lowered.export_controller()), shots=100)

.. :spellcheck-enable:

``backends.register(name, factory)`` adds a backend in the running process
without a distribution, and takes precedence over an installed one of the
same name. ``backends.get`` of a name no backend has raises
``BackendNotFoundError`` with the installed names.
