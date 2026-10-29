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
``"readout"``) and qudit (and, with line delays, its ``fine_delay_ns``;
see below), and ``limits`` the controller limits it was
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
    wait for an answer, end one epoch of the lines that wait for it and
    start the next; an event shifts by the run-time lengths of its epoch's
    barrier and of the barriers of the epochs it follows. Each epoch follows
    the one before it, or the epoch its ``after`` names.
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
    when the kernel is traced), at the line's own rate (its controller's
    ``sample_rate_ghz``); and its overall complex amplitude,
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
    them different or a frame change is known only at run time.
``tone``, ``frame_updates``, ``phase_offset_rad``
    Only on a tone whose frame depends on run time: one with a frame change
    computed from a readout or an answer, or made in a branch. The tone's
    id (such tones are numbered in the order they are first played), and
    the frame changes on that tone since its previous drive or readout, in
    program order: each with ``id``, ``op`` (``"shift_phase"``,
    ``"set_phase"``, ``"shift_frequency"`` or ``"set_frequency"``),
    ``value`` (a number, or an expression) and ``conditions``, the branch
    arms it is on the path of. One change pending before a branch whose arm
    plays on the tone is listed again, with the other arm's condition, at
    the next drive after the branch; the same ``id`` names it both times.
    A controller tracks such a tone's frame itself, applying these instead
    of reading ``frequency_hz`` and ``phase_rad``, each ``id`` once. A drive
    whose own phase offset does not change the frame has ``phase_offset_rad``.

A waveform parameter computed from a readout or an answer is ``null`` in
``params`` and has an expression in the node's ``runtime``, by parameter
name. An expression is ``{"const": x}``, ``{"answer": i, "result": k}``
(result ``k`` of the answer event ``i``), ``{"readout": i}`` (the outcome of
the readout event ``i``), ``{"op": name, "of": [...]}`` for an MLIR
operation (``"arith.mulf"``, ``"arith.fptosi"``, ...; comparisons add
``predicate``) over expressions, or ``{"unknown": name}``. A variable
wait's ``duration_cycles`` is its least length, and it also has ``length``,
the expression of its length in cycles, and ``min_cycles`` and
``max_cycles``, its bounds. A program that computes nothing from its
readouts and answers exports none of these keys.

A sync has ``lines``, the lines it joins, and lasts 0 cycles: no event on
those lines before it ends after its start, and none after it starts
before.

A classical call is a ``submit``, while the request holds its link, and an
``answer``, when the answer is taken. Both have ``link`` and ``function``;
a submit has ``measurements``, the readout events whose outcomes it sends,
and ``operands``, its arguments in the order of the function's argument
descriptors (one per scalar or element, and one ``{"readout": i}`` per
measurement), each an expression; an answer has ``wait``,
``"bounded"`` (by the latency budget), ``"blocking"`` or ``"timeout"``,
with ``timeout_ns``.

The events of one operation, and of operations with the same waveform,
share their ``waveform`` and ``amplitude`` objects; copy one before
changing it.

The structured format
---------------------

``export_controller(version=2)`` gives format
``"cudaq-pulse.controller/2"``, which grows with a program's operations
rather than with its executions, as a sequencer runs it. It has the keys of
``/1``, except that:

``events``
    Each event's ``kind`` is one of ``cudaq_pulse.Event``, named for what a
    controller does whatever the modality: ``"play"`` (a ``/1`` drive),
    ``"acquire"`` (a readout), ``"wait"``, ``"sync"``, ``"loop"``,
    ``"call"`` (a ``/1`` submit) and ``"answer"``. An event in a branch
    keeps its kind and lists its ``conditions``, and phase and frequency
    changes are in the ``tones`` of the events that play them, so neither
    has a kind of its own. A backend refuses a kind it does not know.

    A tree. A loop whose iterations repeat (each event moves by one period,
    and every tone keeps its frequency) is one ``loop`` event: ``trips``
    iterations from ``first_iteration`` (omitted when 0), one
    ``period_cycles`` apart, its ``start_cycles`` and ``duration_cycles``
    spanning them, and its ``body``, the events of its first iteration. An
    event in the body of loops starts ``start_cycles`` plus, for each loop
    around it, the iterations since its first times its period. A first
    iteration that differs from the rest is exported before its loop; a
    loop that does not repeat is exported once per iteration, as in ``/1``.
``waveforms``
    Each distinct waveform expression once; a play or acquire gives the
    index of its own in ``waveform``. With ``sample_rate_ghz``, ``samples``
    holds each one's samples, in the same order. Each also has ``id``, 16
    hex digits that depend on its expression alone, so they stay the same
    from one compile, call, line or controller to the next, and
    ``shape_id``, equal for waveforms that differ only in overall amplitude
    and phase (``shape_key``), the ones a controller that scales what it
    plays stores once.
``parameters``
    For a module with ``float`` parameters, each one's ``value`` in this
    call and what it ``reaches``: ``"amplitude"`` (an overall factor of a
    waveform), ``"phase"``, ``"frequency"``, ``"shape"`` (another waveform
    parameter), ``"timing"`` (a length) or ``"other"``. A parameter that
    reaches only amplitudes and phases leaves every ``shape_id`` the same
    for every value. Omitted without parameters.
``controller``
    In place of ``limits``: the controller, with the fields
    ``instruction_cycles``, ``min_idle_cycles``, ``detectors``,
    ``detector_recovery_cycles``,
    ``waveform_memory_samples``, ``max_tones_per_line``, ``max_pulses`` and
    ``supports_timeouts`` for ``ControllerLimits.issue_cycles``,
    ``min_wait_cycles``, ``acquirers``, ``acquirer_hold_cycles``,
    ``waveform_samples``, ``tones_per_line``,
    ``max_events`` and ``answer_timeouts``; the others keep their names.
    What it computes is in ``processors``, empty when no compute limit is
    set, else one processor with ``links``, ``clock_ghz``,
    ``cycles_per_op`` (the cost of each class of operation it computes),
    ``unsupported`` (the classes it cannot compute, by name; empty, and
    ``max_concurrent_calls``, 1, as ``ControllerLimits`` has no field to
    change them),
    ``fixed_cost_only``, ``max_ops``, ``max_state_bytes``,
    ``max_calls_per_shot`` and ``max_bytes_per_shot``, for
    ``compute_links``, ``compute_clock_ghz``, ``op_cycles``,
    ``controller_form``, ``max_ops``, ``state_bytes``,
    ``max_calls_per_shot`` and ``buffer_bytes``.
``quantum_system``
    In place of ``target``: the name of the quantum system the program is
    for.
``sample_rate_ghz``
    Beside ``clock_ghz``, always a number: the rate at which the
    controller's converters play waveform samples, so that the program
    states its own timing. A waveform of ``n`` samples lasts
    ``n * clock_ghz / sample_rate_ghz`` cycles. The ratio may be
    fractional (2.4 GHz on a 1 GHz clock), but every waveform lasts a whole
    number of samples, and the export raises ``ValueError`` for one that
    does not rather than round. The rate is the controller's, else the
    export's ``sample_rate_ghz``, which must then be the same, else
    ``clock_ghz`` (one sample per cycle). Unlike in ``/1``, it need not
    be a whole multiple of the clock, and ``samples`` are present only when
    the export is given ``sample_rate_ghz``.
``links``
    Each link with ``send_ns``, ``bandwidth_bytes_per_ns``,
    ``answer_delay_ns`` and ``slots`` for ``Link.fixed_ns``,
    ``bytes_per_ns``, ``feedback_ns`` and ``max_in_flight``.
``lines``
    Each line's ``qudits``, a list.
Played events
    ``tones``, a list of the tones the event plays, each with its
    ``frequency_hz``, its ``phase_rad`` in the loops' first iterations, and
    ``phase_step_rad``, the phase added per iteration of each loop around
    it, outermost first.
Left out
    What follows from the rest or is the default: an event's ``index`` (its
    place in a pre-order walk of the tree, loops included, which is what
    references give), its ``qudit`` (its line's), its ``end_cycles``, an
    empty ``iteration`` (``null`` for a loop kept rolled), an ``epoch`` of
    0, empty ``conditions``, ``readouts`` and ``answers``, the ``line`` of
    a sync or a classical call, a ``timeout_ns`` of ``null``, and the
    ``amplitude``, which follows from the waveform. A reference is to the
    latest execution of the event it names.
``programs``, ``bindings``
    Reserved for programs that several controllers share and the tables
    that bind them to each; empty for now.

``cudaq_pulse.controller.unroll(program)`` turns a ``/2`` program into the
``/1`` program of the same kernel, for a backend that needs every
execution; phases agree up to rounding and branches are numbered in the
order they first appear.
Clocks
------

A kernel counts time in cycles of the device's program clock,
``Device.clock_ghz`` (a device config's ``"clock_ghz"``, else the
target's); a duration written as a time converts to them when the kernel
compiles (see :ref:`durations`). A controller may run its own clock,
``ControllerLimits.clock_ghz``, at a whole-cycle ratio of it; may start
and time operations only every ``ControllerLimits.cycles_per_timestep`` of
its own cycles (default 1); and may play samples at
``ControllerLimits.sample_rate_ghz`` (default one per cycle of its clock),
at any ratio of whole numbers to its clock.

.. _timestep:

A line's **timestep** is the shortest time, in program cycles, at whose
multiples its controller can start and end operations: the fewest whole
program cycles that are a whole number of the controller's steps and of
its samples. On the program clock with one sample per cycle it is
``cycles_per_timestep``; at ``a / b`` of the program clock (in lowest
terms), with one cycle per step, it is ``b``; at 2.4 GHz samples on a
1 GHz clock it is 5 cycles, 12 samples. A length off the samples is an
error like any other off its timesteps; it is never rounded.

The scheduler places every operation on whole timesteps of its line and
meets a sync on a multiple of every synced line's timestep; a length that
is not a whole number of timesteps fails to compile with
``ControllerLimitError``, which names the line and the fix:
``line d3 moves in timesteps of 4 cycles; 22 cycles isn't a whole number of
timesteps``. ``pulse.round_duration(line, duration)`` in a kernel rounds a
length to whole timesteps (``direction="up"`` by default, ``"down"`` or
``"nearest"``, or a ``pulse.RoundDirection``), so a kernel can be written
for any controller::

    drive(line, gaussian(pulse.round_duration(line, 41), 0.1, 10.0))

The export counts the controller's own cycles: ``clock_ghz`` is the
controller's clock, and when that is not the program clock the export also
gives ``program_clock_ghz``, converts every time to the controller's cycles
(exactly for operations on its lines; rounded up for classical calls and
epochs) and scales each waveform's time parameters (``duration``,
``sigma``, ``beta``, ``risefall``) with them. A program for a controller at
the program clock is unchanged. ``ControlSystem(reference_clock_ghz=...)``
(or ``MachineModel(reference_clock_ghz=...)`` without a control system)
states a clock the controllers share; every controller's clock must be a
whole multiple of it, and the export gives it as ``reference_clock_ghz``.
Without one, each controller's start is known only to within one of its own
cycles, which budgets between controllers count.

Each ``pulse.Controller`` of a control system may run its own clock.
Without a declared ``"clock_ghz"`` the device's program clock is the
fastest controller clock when it is a whole multiple of every other; clocks
that are not are a ``DeviceDefinitionError`` listing the program clocks
that would work, never one picked for you. Every line is then scheduled on
the timesteps of the controller that plays it, and checked against that
controller's limits in its own cycles. A sync, and a wait whose length
comes from a classical answer, meet where every line's timesteps do. So do
the lines of a runtime qudit, so that the kernel's one schedule fits
whichever qudits a call binds. A line's delay is split by its own
controller's fine delay, and ``rcp`` keeps a branch on a readout for the
reading controller's ``feedback_latency_cycles`` converted to program
cycles. ``custom(duration, f)`` samples ``f`` at the rate of the line that
plays it, the waveforms made from it too; samples given to
``custom_samples`` play only on lines of the rate they were made for, and
on a runtime qudit's line, whose rate depends on the call, ``custom()`` is
a compile error when the controllers' rates differ.

Each controller gets its own program: ``lowered.export_controller(controller=k)``
is controller ``k``'s part, in its own cycles, with ``"controller": k``,
its ``lines`` and ``limits``, and the events it takes part in: those on its
lines, a sync with only its lines, the submit of a call whose measurements
it reads and the answer it takes, whether to branch on or to compute with.
Event and line ids stay the program's, so a value computed from an answer
names the answer by the same ``index`` in every controller's program. A
program that plays the lines of more than one controller needs
``controller=``; one on a single controller exports it without, and one
that plays no lines is exported by the controller that makes its calls.
A controller the program does not use exports a valid empty program: no
lines, events or calls. ``export_emulation(controller=k)`` gives the same
part for an emulator.

A link is shared by default: every controller it reaches sends its
requests to one instance, which holds ``max_in_flight`` requests for all
of them. A link that each controller has its own instance of, a decoder
beside each controller for example, is declared per controller:

.. code-block:: python

   import cudaq_pulse as pulse

   links = {"decoder": pulse.Link(fixed_ns=10, max_in_flight=1, per_controller=True)}

or ``per_controller = true`` in the link's table of a system file
(``[control_system.links.decoder]``). Each controller's calls over it then
use that controller's slots, so calls on two controllers are scheduled
without waiting for each other's slots, and each controller exports only
its own calls, with the call's ``"controller"`` in its submit, answer and
``calls`` entries. A call never crosses between controllers, so its
answer is due without the start uncertainty of a second controller. The
call's measurements and the pulses that use its answer must be on one
controller, or the program fails to compile, naming both controllers.
When the scheduler cannot tell a call's controller (it reads no
measurement and its answer is unused, or leaves the kernel), the link's
calls share one set of slots, as on a shared link. Each controller may
place the link on one of its own processors; ``controller_usage`` counts
each instance's calls on its controller's processor, and
``resources(lowered).links`` has a row per instance.

Times between controllers are budgeted with how well each one's start is
known: its ``clock_skew_cycles``, plus one of its own cycles without a
``reference_clock_ghz``. A pulse that branches on, or computes with,
another controller's readout starts once the outcome can reach it: the
reader's ``feedback_latency_cycles`` and its ``peer_latency_cycles`` to the
user, in the reader's cycles, and both controllers' start uncertainty.
``rcp`` and ``alap_rcp`` wait that long for each pair of reader and user,
and a pulse on the reader's own controller waits its feedback latency
alone; a program that starts the pulse sooner fails to compile with the
cycle it may start at. An answer taken on another controller than the one
that sent the request is budgeted with the two controllers' start
uncertainty, and ``due_cycles`` includes it; one controller sending and
taking it adds none. When the scheduler cannot tell which controllers they
are (the request carries no measurement, or the answer leaves the kernel),
the budget adds the start uncertainty of the two controllers the link
reaches that are known least well.

A sync of lines on several controllers lines up the first pulse after it
on each of its lines. Each of those lands within its controller's start
uncertainty of where it is meant to, plus how far its line delay is
rounded to its grid when the controller has no fine delay, all in the
controller's cycles, and that must fit the controller's
``alignment_tolerance_cycles``; otherwise the program fails to compile,
suggesting a shared ``reference_clock_ghz`` or the tolerance it needs.
Under ``rcp`` and ``alap_rcp``, ``max_concurrent_drives`` and
``max_concurrent_readouts`` count pulses on every controller: a pulse that
follows another controller's on the same lane starts their start
uncertainty after it, so the two never overlap.

What the program computes at run time is converted by the same rule. A
variable wait's ``length`` is clamped to its bounds, as the run clamps it,
and then wrapped in ``{"op": "pulse.own_cycles", "of": [x], "num": a,
"den": b, "timestep": g}``: the length ``x`` program cycles is ``x * a / b``
of the controller's, and the controller must fail the run when ``x`` is not
a whole number of the line's timesteps of ``g`` program cycles. A
controller with the program's clock but longer timesteps gets the wrapper
too, with ``num`` and ``den`` 1, as that check. ``min_cycles`` rounds down
and ``max_cycles`` up. A run-time ``duration`` is wrapped the same way, and
a run-time ``sigma``, ``beta`` or ``risefall`` is multiplied by ``a / b``
(``arith.mulf``). A run-time length whose values are all known while the
kernel compiles (it selects between constants) and include one off the
timesteps is a ``ControllerLimitError`` then. A run-time operand of a
waveform whose parameters the export does not name can't be converted, so
on a controller with its own clock it is a ``ControllerLimitError`` too,
never exported in program cycles.

Line delays
-----------

Cables and electronics delay each line differently. The calibrated delays
belong to the system, not to kernels: ``ControlSystem(line_delays_ns=...)``
(or ``MachineModel(line_delays_ns=...)`` without a control system) gives
them by line, ``("drive", qudit)`` or ``("readout", qudit)``, as times
(``2.3 * pulse.ns``) or numbers of ns, and every operation on a delayed line starts that much later than scheduled.
Whole :ref:`timesteps <timestep>` of the line move its operations, which
are then checked against the controller's limits like any others; the rest,
less than one timestep, is the line's ``fine_delay_ns`` in the export, for
a controller that can delay a line's output by less than that
(``ControllerLimits.max_fine_delay_ns``). A controller without fine delay
rounds the delay to the nearest timestep if that is within its
``alignment_tolerance_cycles``, and otherwise the program fails to compile
with an error naming the line, the rest and the fixes::

    machine = MachineModel(
        controller=ControllerLimits(clock_ghz=1.0, max_fine_delay_ns=1.0),
        line_delays_ns={("drive", 0): 2.3, ("readout", 0): 41.0},
    )
    # The drive line of qudit 0 moves 2 ns (4 cycles of a 2 GHz program
    # clock) and plays with fine_delay_ns = 0.3.

Compiling a kernel again with only other delays, such as a control system
calibrated again, reuses its trace and optimized module and schedules it
again, with the same result, byte for byte, as compiling it anew. A delay
of a line no controller of the system plays, or of a qudit the quantum
system lacks, is a ``DeviceDefinitionError``.

Waveform memory
---------------

``ControllerLimits(waveform_memory=WaveformMemory(...))`` describes the
memory a controller plays waveforms from, in samples:

.. code-block:: python

   from cudaq_pulse.controller import ControllerLimits, WaveformMemory

   memory = WaveformMemory(
       samples=65536,             # what it holds
       granule=32,                # each waveform rounds up to whole granules
       shared=True,               # one memory for every line
       synthesized=frozenset({"square", "gaussian"}),  # generated, not stored
       upload_samples_per_cycle=4.0,
       upload_while_playing=True,
   )
   limits = ControllerLimits(sample_rate_ghz=2.0, waveform_memory=memory)

Compiling checks that every waveform of a program fits at once: per line, or
once over all lines when ``shared``, each distinct waveform takes its
samples rounded up to ``granule``. With ``scaling`` (the default), the
controller plays one stored shape at any amplitude and phase, so waveforms
that differ only in those share memory, even when a ``float`` parameter sets
the amplitude; without it, only equal waveforms do. Shapes in
``synthesized``, and with ``hold`` a waveform that is constant, take none.
The upload rate, whether uploads overlap playing, and ``persistent`` (the
memory keeps its waveforms from one program to the next) are not checked;
they reach a backend, or a planner that places waveforms in memory, through
the export's ``controller``, with the waveform ids and parameter roles of
``/2``. ``waveform_samples=n`` is ``waveform_memory=WaveformMemory(samples=n)``.

Controller limits
-----------------

``ControllerLimits`` states what a controller can play, in its own clock
cycles: its clock, how
closely two operations on one line can follow each other, its waveform
memory and tones per line, which acquirer reads each qudit and how long a
readout holds it, the least feedback latency from a readout to an operation
that branches on it, whether it can stop waiting for an answer after a
timeout, and the most drives and readouts in one program. Given
to the machine model as ``MachineModel(controller=...)``, they make every
compilation check the scheduled program, so that a program the controller
cannot play fails at compile time with ``ControllerLimitError``, which lists
each violated limit. A limit that only some values of a ``float``
parameter exceed, such as the tones a frequency shift adds or the memory
a waveform width takes, is checked when a call supplies the values, once
per distinct values, so a call that compiles also exports.
``cudaq_pulse.controller.check(lowered, limits)`` returns the violations
without raising. Without ``controller``, compiling checks nothing more.

A minimal backend
-----------------

A backend is an object with a ``name`` and three methods, the
``cudaq_pulse.backends.Backend`` protocol: ``machine_model()`` returns the
machine model to compile for its controller, ``emit(program)`` takes an
exported program and returns the controller's, and ``run(artifact,
shots=...)`` runs it. ``formats``, if present, lists the export formats it
reads. A program it cannot emit it refuses with
``IncompatibleBackendError``.

Two more methods are optional, and only ``backends.validate`` uses them
(see `Validating a backend`_). ``events(artifact)`` gives an emitted
artifact back as one list of events per controller program, each a mapping
with its ``kind`` (``cudaq_pulse.Event``, or the ``/1`` name), ``line``
(the link of a call or answer), ``start_cycles`` and optionally
``duration_cycles``, every execution in program order.
``emulate(artifact, shots=...)`` runs it on the backend's emulator and
gives each shot's measurement outcomes, in the order the measurements end.

This one emits one line of text per drive and readout:

.. code-block:: python

   from collections.abc import Mapping
   from typing import Any

   from cudaq_pulse.backends import IncompatibleBackendError
   from cudaq_pulse.compiler import MachineModel
   from cudaq_pulse.controller import FORMAT, ControllerLimits


   class TextBackend:
       name = "text"
       formats = (FORMAT,)

       def machine_model(self) -> MachineModel:
           limits = ControllerLimits(clock_ghz=1.0, issue_cycles=2, min_wait_cycles=2)
           return MachineModel(controller=limits)

       def emit(self, program: Mapping[str, Any]) -> list[str]:
           if program["format"] != FORMAT:
               raise IncompatibleBackendError(f"cannot read {program['format']}")
           played = []
           for event in program["events"]:
               if event["kind"] in ("wait", "sync"):
                   continue
               if event["kind"] not in ("drive", "readout") or event["conditions"]:
                   raise IncompatibleBackendError(f"cannot play event {event['index']}")
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
               qubits={0: Qubit(0, 5.0e9, -2.0e8, 50.0, 30.0)},
           ),
           "clock_ghz": 1.0,
       }
   )
   cal = {"x_duration": 40 * pulse.ns, "x_sigma": 10 * pulse.ns, "readout_duration": 200 * pulse.ns}


   @pulse.kernel
   def drive_and_measure(q):
       drive_line = get_drive_line(q)
       readout_line = get_readout_line(q)
       drive(drive_line, gaussian(cal["x_duration"], 0.5, cal["x_sigma"]))
       sync(drive_line, readout_line)
       readout(readout_line, square(cal["readout_duration"], 0.05))


   compiled = pulse.compile(drive_and_measure, device=dev, machine=backend.machine_model())
   program = compiled.lower(dev.qudit_handles[0]).export_controller()
   print(backend.run(backend.emit(program), shots=1))
   # [['0 drive line 0 gaussian at 5e+09 Hz', '40 readout line 1 square at 5e+09 Hz']]

Validating a backend
--------------------

``backends.validate(backend, controller=...)`` runs the export test corpus,
which ships with cudaq-pulse, through a backend: a program for each kind of
event (``cudaq_pulse.Event``) and one with a branch, compiled for the
controller (by default the backend's machine model's) and exported in each
format the backend reads. Each must be emitted or refused with
``IncompatibleBackendError``, and a program with an event kind no format
has, or in a newer format, must be refused. A program that exceeds the
controller's limits is skipped.

.. code-block:: python

   from cudaq_pulse import backends
   from cudaq_pulse.controller import ControllerLimits

   report = backends.validate("text", controller=ControllerLimits(clock_ghz=1.0))
   print(report)
   # PASS play
   # ...
   # REFUSED call: cannot play event 1
   # ...
   # PASS unsupported_event
   # PASS newer_format
   # backend 'text': 7 passed, 3 refused
   print(report.ok)  # True

A backend that accepted the unknown event would instead report ``FAIL
unsupported_event: emit() accepted 'forward_outcome'; it should raise
IncompatibleBackendError``.

Two optional methods make the checks stronger. With ``events(artifact)``,
which gives the emitted programs back, one list of events per controller
(each with its ``kind``, ``line``, ``start_cycles`` and optionally
``duration_cycles``, every execution in program order), the number of
programs and the timing and order of plays, acquisitions, calls and answers
on each line must match the controller program, and the pulses must respect
the controller's limits. With ``emulate(artifact, shots=...)``, which gives
each shot's measurement outcomes in the order they end, the fraction of 1s
of each must match ``cudaq_pulse.simulate`` within ``tolerance`` (0.05 by
default) plus three standard errors of the ``shots``. Validating imports
nothing until it is called.

Errors
------

``backends.get`` raises ``BackendNotFoundError`` (a ``LookupError``) for a
name no backend has, listing the installed ones, and
``IncompatibleBackendError`` (a ``TypeError``) for a backend that fails to
import, is not a ``Backend``, or reads none of the formats this version
writes;
both are ``BackendError``. Compiling a program that exceeds the backend's
limits raises ``ControllerLimitError``, a ``CompilationError``.

Cost
----

Importing cudaq-pulse neither imports the backend modules nor looks for
installed backends; that happens on the first call to ``backends``. A
machine model without ``controller`` adds nothing to compilation. The
export and the limit checks are native. Exporting in ``/1`` takes time and
memory linear in a program's executions; in ``/2``, and for the checks, a
loop whose iterations repeat costs as much as its first two iterations and
its last, whatever its trip count. A violation inside it is still listed
once per iteration, each message built when it is read.
