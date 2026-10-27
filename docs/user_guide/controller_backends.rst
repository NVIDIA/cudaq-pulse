..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Writing a Controller Backend
============================

A controller backend runs compiled kernels on a quantum controller: it
lowers a scheduled program to the controller's own instructions and
waveform memory, and runs it there. Backends live outside cudaq-pulse, in
their own distributions, and cudaq-pulse finds the installed ones by a
standard Python entry point. This page shows the program a backend
receives, how a backend states what its controller can play, and a complete
minimal backend; ``examples/controller_backend/`` is the same backend as an
installable distribution.

Exporting a program
-------------------

``Lowered.export_controller()`` returns a compiled kernel as JSON-serializable
data. Its ``format`` is ``"cudaq-pulse.controller/1"``
(``cudaq_pulse.controller.FORMAT``): the schema's name and, after the slash,
its major version. Within one version keys and event kinds may be added, but
none is removed or changes meaning, so a backend ignores keys it does not
know and refuses event kinds it does not know; a change that would break a
backend gets a new version.

At the top level, ``target`` and ``clock_ghz`` name the device and its
clock, ``schedule`` the schedule policy, ``qudits`` the device qudits the
program uses, ``lines`` each line's index, ``role`` (``"drive"`` or
``"readout"``) and qudit, and ``limits`` the controller limits it was
checked against (``null`` without). ``events`` lists every drive, readout,
wait, sync and classical call in program order, once per loop iteration and
both arms of a branch. ``epochs``, ``calls``, ``links`` and ``manifest`` are
those of ``Lowered.export_emulation()`` (see :doc:`classical_calls`). Each
event has:

``index``, ``kind``
    Its position in ``events`` and one of ``"drive"``, ``"readout"``,
    ``"wait"``, ``"sync"``, ``"submit"`` and ``"answer"``.
``line``, ``qudit``
    The line it plays on and that line's qudit; ``null`` for a sync. For a
    classical call, ``line`` is its link and ``qudit`` is -1.
``start_cycles``, ``duration_cycles``, ``end_cycles``
    The final schedule as integers, in clock cycles. A backend places every
    event at its start and does not schedule again.
``iteration``
    The iteration of each loop around it, outermost first.
``epoch``
    The epoch it starts in. A wait whose length is an answer, and a blocking
    wait for an answer, end one epoch and start the next; everything in a
    later epoch shifts by the run-time lengths before it.
``conditions``
    The measurement-conditioned branches the event lies in, outermost
    first: the ``branch``, the ``arm`` (``true`` for the ``if`` arm), and
    the ``readouts`` and ``answers`` (event indices) the condition is
    computed from. ``test`` is ``"is_set"`` when the condition is one
    readout's outcome, ``"answer"`` when it is one answer value, else
    ``"expression"``. Events in the two arms of one branch may overlap
    in time; only one arm plays.

Drives and readouts also have:

``waveform``, ``amplitude``
    The waveform's expression, ``{"shape": "gaussian", "params":
    {"duration": ..., "amplitude": ..., "sigma": ...}}``, with ``scale``,
    ``neg``, ``add``, ``sub`` and ``mul`` nodes over ``of``, and
    ``samples`` for ``custom_samples`` (which ``custom()`` envelopes become
    when the kernel is traced); and its overall complex amplitude,
    ``{"re": ..., "im": ...}``. Times in parameters are in cycles.
``samples``
    With ``export_controller(sample_rate_ghz=...)``, a whole multiple of the
    clock, the envelope sampled at that rate, ``{"re": [...], "im":
    [...]}``. ``cudaq_pulse.controller.sample_waveform`` computes the same
    samples for one waveform.
``frequency_hz``, ``phase_rad``
    The tone's frequency and its frame phase as the event starts, with every
    frame change and virtual Z before it applied and the phase a detuned
    tone gains over time, or ``null`` where the two arms of a branch leave
    them different.

A sync has ``lines``, the lines it joins, and lasts 0 cycles: no event on
those lines before it ends after its start, and none after it starts
before.

A classical call is a ``submit``, while the request holds its link, and an
``answer``, when the answer is taken. Both have ``link`` and ``function``;
a submit has ``measurements``, the readout events whose outcomes it sends,
and an answer has ``wait``, ``"bounded"`` (by the latency budget),
``"blocking"`` or ``"timeout"``, with ``timeout_ns``.

Controller limits
-----------------

``ControllerLimits`` states what a controller can play: its clock, how
closely two operations on one line can follow each other, its waveform
memory and tones per line, which acquirer reads each qudit and how long a
readout holds it, the least feedback latency from a readout to an operation
that branches on it, whether it can stop waiting for an answer after a
timeout, and the most drives and readouts in one program. Given
to the machine model as ``MachineModel(controller=...)``, they make every
compilation check the scheduled program, so that a program the controller
cannot play fails at compile time with ``ControllerLimitError``, which lists
each violated limit. ``cudaq_pulse.controller.check(lowered, limits)``
returns the violations without raising. Without ``controller``, compiling
checks nothing more.

A minimal backend
-----------------

A backend is an object with a ``name`` and three methods, the
``cudaq_pulse.backends.Backend`` protocol: ``machine_model()`` returns the
machine model to compile for its controller, ``emit(program)`` takes an
exported program and returns the controller's, and ``run(artifact,
shots=...)`` runs it. ``formats``, if present, lists the export formats it
reads. This one emits one line of text per drive and readout:

.. code-block:: python

   from collections.abc import Mapping
   from typing import Any

   from cudaq_pulse.compiler import MachineModel
   from cudaq_pulse.controller import FORMAT, ControllerLimits


   class TextBackend:
       name = "text"
       formats = (FORMAT,)

       def machine_model(self) -> MachineModel:
           limits = ControllerLimits(clock_ghz=1.0, issue_cycles=2, min_wait_cycles=2)
           return MachineModel(controller=limits)

       def emit(self, program: Mapping[str, Any]) -> list[str]:
           played = []
           for event in program["events"]:
               if event["kind"] in ("wait", "sync"):
                   continue
               if event["kind"] not in ("drive", "readout") or event["conditions"]:
                   raise ValueError(f"cannot play event {event['index']}")
               played.append(
                   f"{event['start_cycles']} {event['kind']} line {event['line']} "
                   f"{event['waveform']['shape']} at {event['frequency_hz']:.6g} Hz"
               )
           return played

       def run(self, artifact: Any, *, shots: int = 1) -> list[list[str]]:
           return [list(artifact) for _ in range(shots)]

Its distribution registers it under its name in the
``cudaq_pulse.backends`` entry-point group, naming the backend class or any
callable without arguments that returns one:

.. code-block:: toml

   [project.entry-points."cudaq_pulse.backends"]
   text = "text_controller:TextBackend"

Using a backend
---------------

``backends.available()`` lists the installed backends without importing
any, and ``backends.get(name)`` loads one on first use and keeps it.
``backends.register(name, factory)`` adds a backend in the running process
without a distribution, and takes precedence over an installed one of the
same name:

.. code-block:: python

   import cudaq_pulse as pulse
   from cudaq_pulse import backends
   from cudaq_pulse.targets import Qubit, TargetDescription

   backends.register("text", TextBackend)
   backend = backends.get("text")
   dev = pulse.device(
       {
           "target": TargetDescription(
               name="one-qubit",
               clock_ghz=1.0,
               qubits={0: Qubit(0, 5.0e9, -2.0e8, 50.0, 30.0)},
           )
       }
   )


   @pulse.kernel
   def drive_and_measure(q):
       drive_line, drive_tone = get_drive_line(q)
       readout_line, readout_tone = get_readout_line(q)
       drive(drive_line, gaussian(40, 0.5, 10.0), drive_tone)
       sync(drive_line, readout_line)
       readout(readout_line, square(200, 0.05), readout_tone)


   compiled = pulse.compile(drive_and_measure, device=dev, machine=backend.machine_model())
   program = compiled.lower(dev.qudit_handles[0]).export_controller()
   print(backend.run(backend.emit(program), shots=1))
   # [['0 drive line 0 gaussian at 5e+09 Hz', '40 readout line 1 square at 5e+09 Hz']]

Errors
------

``backends.get`` raises ``BackendNotFoundError`` (a ``LookupError``) for a
name no backend has, listing the installed ones, and
``IncompatibleBackendError`` (a ``TypeError``) for a backend that fails to
import, is not a ``Backend``, or does not read this version's ``FORMAT``;
both are ``BackendError``. Compiling a program that exceeds the backend's
limits raises ``ControllerLimitError``, a ``CompilationError``.

Cost
----

Importing cudaq-pulse neither imports the backend modules nor looks for
installed backends; that happens on the first call to ``backends``. A
machine model without ``controller`` adds nothing to compilation. Exporting
and checking a program take time linear in its events.
