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
wait, sync, classical call and frame update in program order, once per loop
iteration and both arms of a branch. ``segments``, ``calls``, ``links`` and
``manifest`` are those of ``Lowered.export_emulation()`` (see
:doc:`classical_calls`).

A kernel with list or qudit parameters compiles once for every call, and
its export says what each call binds: ``program_id`` is the same for every
call of that compile, and ``symbols`` lists the call's values, each with
its ``symbol``, ``kind`` and ``value``: a float parameter or element
(``"float"``), a runtime qudit's slot (``"slot"``, ``s0``, the carrier
bound), and the native frequencies of the slot's lines (``"frequency"``):
``s0.f01``, its 0-1 frequency, for a drive line, and ``s0.f_ro``, the
``frequency_hz`` of its readout (else the 0-1 frequency), for a readout
line. Each event names in its ``symbols`` the slot of its line
(``qudit``) and, by field (``frequency_hz``, ``phase_rad``,
``waveform``), the symbols its value depends on; a tone on a slot's line
depends on that line's native frequency. Each event has:

``index``, ``kind``
    Its position in ``events`` and one of ``"drive"``, ``"readout"``,
    ``"wait"``, ``"sync"``, ``"submit"``, ``"answer"`` and
    ``"frame_update"``.
``line``, ``qudit``
    The line it plays on and that line's qudit; ``null`` for a sync. For a
    classical call, ``line`` is its link and ``qudit`` is -1.
``start_cycles``, ``duration_cycles``, ``end_cycles``
    The final schedule as integers, in clock cycles. A backend places every
    event at its start and does not schedule again.
``iteration``
    The iteration of each loop around it, outermost first.
``segment``
    The segment it starts in. A wait whose length is an answer, and a blocking
    wait for an answer, end one segment of the lines that wait for it and
    start the next; an event shifts by the run-time lengths of its segment's
    barrier and of the barriers of the segments it follows. Each segment follows
    the one before it, or the segment its ``after`` names.
``conditions``
    The measurement-conditioned branches the event lies in, outermost
    first: the ``branch``, the ``arm`` (``true`` for the ``if`` arm), and
    the ``readouts`` and ``answers`` (event indices) the condition is
    computed from. ``test`` is ``"is_set"`` when the condition is whether
    one readout is set, ``"answer"`` when it is one answer value, else
    ``"expression"``, and then ``expression`` is the condition, a bool
    expression (see `Run-time expressions`_) over those readouts and
    answers. The event plays when every condition is its ``arm``. Events
    in the two arms of one branch may overlap in time; only one arm plays.

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
    ``value`` (a number, or an expression), ``start_cycles``, where it is
    made on the tone's timeline (see below), from which the frame gains
    phase at its new frequency even where the value is known only later,
    and ``conditions``, the branch arms it is on the path of. One change
    pending before a branch whose arm
    plays on the tone is listed again, with the other arm's condition, at
    the next drive after the branch; the same ``id`` names it both times.
    A controller tracks such a tone's frame itself, applying these instead
    of reading ``frequency_hz`` and ``phase_rad``, each ``id`` once. A drive
    whose own phase offset does not change the frame has ``phase_offset_rad``.

A program with such tones has ``runtime_tones`` (after ``events`` and
``frame_state``): by each
tone's id, as a string, its ``line`` (a line's own tone's line, else the line
it first plays on) and the frame every shot starts it in,
``reset_frequency_hz`` and ``reset_phase`` (in radians), to which a
controller tracking the tone applies its ``frame_updates``. On the line of a
runtime qudit, the entry's ``symbols`` name the symbol its reset frequency
is, ``s0.f01`` on a drive line and ``s0.f_ro`` on a readout line. A
program without such tones has no ``runtime_tones``.

Every shot starts from the reset frame: each tone at its own frequency, with
phase 0. A frame change (``shift_phase``, ``set_phase``, ``shift_frequency``,
``set_frequency``) is a point on the timeline: in the scheduled IR it has
``start_cycles`` and a ``duration_cycles`` of 0. The next drive or readout on
its tone takes it in its ``frequency_hz`` and ``phase_rad``; in a loop, the
next iteration's do. In a loop whose iterations do not play its tone, a
change is at the start of each iteration. A change takes no time: every
drive, readout, wait and loop, and the program, starts and lasts as it does
without it.

The compiler keeps every change the program makes, and where no later
drive or readout takes a change on some path, the export gives that
execution as a ``"frame_update"`` event of its own, decided from the
program alone:

- A change nothing takes is a frame update where it is made, in every
  iteration of a loop around it; a loop kept rolled stays rolled, the frame
  update in its body.
- A change a loop carries out of its last iteration, which nothing after
  the loop takes, is a frame update after the loop, at its place in the
  last iteration.
- A change made before a branch that only one arm takes is a frame update
  at the start of the other arm (or of the missing ``else``), with that
  arm's ``conditions``; the arm that takes it plays with it as before.
- A change a loop carries through a branch, or out of a loop whose
  iterations do not play its tone, is a frame update at every execution;
  a drive or readout that takes it on another path has it in its frame as
  well.

A frame update has ``op`` and ``value``, as in ``frame_updates``, its
``start_cycles`` (its place) and a ``duration_cycles`` of 0, and the tone's
``frequency_hz`` and ``phase_rad`` with the change applied; on a tone whose
frame depends on run time, ``tone`` and the ``frame_updates`` it applies,
as a drive has. A controller tracking that frame applies the event's
``frame_updates``, each ``id`` once, as it does a drive's; the last of them
is the event's own change, which its ``op`` and ``value`` restate, so it is
not applied again. Its ``line`` is its tone's, else the line the tone last
played on, else ``null``. A controller whose frames persist from one shot to the next
declares ``Controller(persistent_frames=True)``; such changes then
have no event, and the program's ``frame_state`` (after ``events``) gives
the frame each tone ends the shot in, by line: its ``line``, and its
``frequency_hz`` and ``phase_rad`` at the end of the shot, or, for a tone
whose frame depends on run time, its ``tone`` and the ``frame_updates`` not
yet taken.

A waveform parameter computed from a readout or an answer is ``null`` in
``params`` and has an expression (see `Run-time expressions`_) in the
node's ``runtime``, by parameter name. A variable
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

Run-time expressions
--------------------

What a program computes from its readouts and answers (a run-time waveform
parameter, a wait's ``length``, a frame update's ``value``, a submit's
``operands``, a branch's ``expression``) is an expression in a closed
vocabulary of terms, whatever operations the compiler computes it with:
``y = x if x > 0.1`` (else ``0.1``) and ``y = x if 0.1 < x`` are both
``max``. An expression is one of:

``{"const": x}``
    A number, or ``true`` or ``false``.
``{"readout": i}``
    The level the readout event ``i`` observed, an integer; 0 is the
    ground state, and the outcome is set for any other level.
``{"answer": i, "result": k}``
    Result ``k`` of the answer event ``i``: a float for a ``float``
    result, an integer for an ``int`` one, a bool for a bit.
``{"op": term, "type": t, "of": [...]}``
    A term over the expressions in ``of``, computed in the type ``t`` of
    its operands, ``"int"``, ``"float"`` or ``"bool"``. A ``const`` in a
    float term is that float, so ``{"const": 3}`` there is 3.0.

``cudaq_pulse.controller.EXPRESSION_TERMS`` holds every term. A value
computed any other way is an ``UnsupportedError`` when the program is
exported, never a term a backend has to guess at;
``cudaq_pulse.controller.evaluate_expression(expression, readouts,
answers)`` computes an expression as the controller must, which is the
reference for a backend's own evaluation. Integers are 64-bit two's
complement and floats IEEE 754 binary64, rounded to nearest, ties to even.

``add``, ``sub``, ``mul``
    ``a + b``, ``a - b`` and ``a * b``. Integers wrap modulo 2\ :sup:`64`;
    floats overflow to ±inf.
``div``
    ``a / b``. An integer quotient is truncated toward zero; a float
    divided by zero is ±inf, and 0 / 0 is NaN.
``floor_div``, ``rem``
    Integers only: the quotient rounded toward −inf, and the remainder
    ``a - b * div(a, b)``, which has the sign of ``a``. An integer ``div``,
    ``floor_div`` or ``rem`` by 0, or of −2\ :sup:`63` by −1, has no value,
    and the controller fails the run.
``neg``
    ``-a``. The integer −2\ :sup:`63` wraps to itself; a float changes
    sign, 0 and NaN too.
``max``, ``min``
    ``max(a, b)`` is ``a`` when ``a > b``, else ``b``, and ``min(a, b)`` is
    ``a`` when ``a < b``, else ``b``. For integers that is the usual
    maximum and minimum, which the export writes with the operands of the
    comparison it comes from. For floats the order is part of the value,
    as the kernel's comparison was: ``max(NaN, 1)`` is 1 and ``max(1,
    NaN)`` NaN, ``max(-0.0, 0.0)`` is 0.0 and ``max(0.0, -0.0)`` −0.0.
``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge``
    A bool: ``a == b``, ``a != b``, ``a < b``, ``a <= b``, ``a > b``, ``a
    >= b``. Integers compare signed. Floats compare ordered, so every
    comparison with a NaN is false but ``ne``, which is true. Bools have
    only ``eq`` and ``ne``.
``select``
    ``of`` is ``[c, a, b]``: ``a`` when the bool ``c`` is true, else ``b``;
    ``type`` is the type of ``a`` and ``b``.
``to_float``
    The integer ``a`` as the nearest float (``type`` ``"int"``).
``to_int``
    The float ``a`` truncated toward zero (``type`` ``"float"``); NaN, ±inf
    and floats outside the 64-bit integers have no value, and the
    controller fails the run.
``controller_cycles``
    A length of ``a`` program cycles in the controller's cycles, ``a *
    num / den``, with ``num``, ``den`` and ``timestep`` beside ``of`` (see
    `Clocks`_); the controller fails the run when ``a`` is
    not a whole number of timesteps.

A frontend computes an integer answer used as a time in floats, so
``wait(d, n + 4)`` is ``to_int(add(add(to_float(n), 4), 0.5))``. Whether
a readout is set is ``ne(readout, 0)``, and a bool used as an integer a
``select`` of 1 (or −1) and 0. A comparison has its constant second:
``3 < n`` is ``gt(n, 3)``. Bools combine without terms of their own:
``a and b`` is ``select(a, b, false)``, ``a or b`` is ``select(a, true,
b)``, ``a != b`` (exclusive or) is ``ne(a, b)``, and ``not a`` is
``ne(a, true)``, except that ``not`` of an integer comparison is the
opposite comparison (``not (n <= 3)`` is ``gt(n, 3)``) and of ``eq`` or
``ne`` the other; a float ``not (x <= 0.1)`` stays as it is, since it holds
of NaN and ``x > 0.1`` does not. There is no JSON schema; this section and
``evaluate_expression`` define the expressions.

The structured format
---------------------

``export_controller(version=2)`` gives format
``"cudaq-pulse.controller/2"``, which grows with a program's operations
rather than with its executions, as a sequencer runs it. It is written for a
controller on the program clock, in :ref:`timesteps <timestep>` of any
length; for one on its own clock, or with line delays, export
``/1``. It has the keys of ``/1``, except that:

``events``
    Each event's ``kind`` is one of ``cudaq_pulse.Event``, named for what a
    controller does whatever the modality: ``"play"`` (a ``/1`` drive),
    ``"acquire"`` (a readout), ``"wait"``, ``"sync"``, ``"loop"``,
    ``"call"`` (a ``/1`` submit), ``"answer"`` and ``"frame_update"``. An
    event in a branch keeps its kind and lists its ``conditions``, and phase
    and frequency changes are in the ``tones`` of the events that play
    them; a change no event plays is a ``"frame_update"``, with its frame
    in ``tones``. A backend refuses a kind it does not know.

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
    In place of ``limits``: the controller, its fields under the names
    ``limits`` gives them. What it computes is in ``processors``, always
    listed (empty for a controller without processors), each one as
    ``Processor.as_dict()`` gives it: ``links``, ``clock_ghz``,
    ``cycles_per_op`` (the cost of each class of operation it computes),
    ``unsupported`` (the classes it cannot compute, by name),
    ``max_concurrent_calls``, ``max_ops_per_function``,
    ``max_ops_total``, ``max_state_bytes_per_function``,
    ``max_state_bytes_total``, ``max_calls_per_shot``,
    ``max_bytes_per_shot``, ``fixed_cost_only`` and ``dispatch_keys``.
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
Played events and frame updates
    ``tones``, a list of the tones the event plays (a frame update's
    frame), each with its ``frequency_hz``, its ``phase_rad`` in the loops'
    first iterations, and ``phase_step_rad``, the phase added per iteration
    of each loop around it, outermost first.
Segments in a rolled loop
    An event's ``segment`` is that of the loops' first iterations; in a loop
    whose body ends segments (a blocking wait for an answer, or a wait whose
    length is an answer), ``segment_step`` is the segments added per iteration
    of each loop around it, outermost first, omitted where every step is 0.
Values known at run time
    As in ``/1``, and only where a value is computed from a readout or an
    answer: ``tone``, ``frame_updates`` and ``phase_offset_rad`` on a play
    or acquire, ``length``, ``min_cycles`` and ``max_cycles`` on a wait, and
    ``operands`` on a call; an expression's ``readout`` and ``answer`` are
    places, like other references, and resolve to the latest execution of
    that place before the event, so a rolled body refers to its own
    events, and a frame change's ``start_cycles`` is given as an event's
    is. A waveform parameter known only at run time is its expression
    in the entry of the program's ``waveforms`` the event plays, where
    ``/1`` has ``null`` and the node's ``runtime``. The program's
    ``runtime_tones`` and ``frame_state`` are as in ``/1``. A loop whose body changes or plays a tone whose frame
    is tracked at run time is exported once per iteration, so that each
    frame change keeps its own ``id``; a loop whose run-time values repeat
    with its iterations stays one loop.
Symbols
    A kernel with list or qudit parameters has ``program_id`` and
    ``symbols`` as in ``/1``, and each event the ``symbols`` its fields
    depend on; an event in a loop's body names those of any of its
    iterations.
Left out
    What follows from the rest or is the default: an event's ``index`` (its
    place in a pre-order walk of the tree, loops included, which is what
    references give), its ``qudit`` (its line's), its ``end_cycles``, an
    empty ``iteration`` (``null`` for a loop kept rolled), a ``segment`` of
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
``Device.clock_ghz`` (``pulse.device(..., clock_ghz=)``, else the
target's); a duration written as a time converts to them when the kernel
compiles (see :ref:`durations`). A controller may run its own clock,
``Controller.clock_ghz``, at a whole-cycle ratio of it; may start
and time operations only every ``Controller.cycles_per_timestep`` of
its own cycles (default 1); and may play samples at
``Controller.sample_rate_ghz`` (default one per cycle of its clock),
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
segments) and scales each waveform's time parameters (``duration``,
``sigma``, ``beta``, ``risefall``) with them. A program for a controller at
the program clock is unchanged. ``ControlSystem(reference_clock_ghz=...)``
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
variable wait's ``length`` is clamped to its bounds, as the run clamps it
(an integer ``max`` and ``min``), and then wrapped in ``{"op":
"controller_cycles", "type": "int", "of": [x], "num": a, "den": b,
"timestep": g}``: the length ``x`` program cycles is ``x * a / b``
of the controller's, and the controller must fail the run when ``x`` is not
a whole number of the line's timesteps of ``g`` program cycles. A
controller with the program's clock but longer timesteps gets the wrapper
too, with ``num`` and ``den`` 1, as that check. ``min_cycles`` rounds down
and ``max_cycles`` up. A run-time ``duration`` is wrapped the same way, and
a run-time ``sigma``, ``beta`` or ``risefall`` is multiplied by ``a / b``
(a float ``mul``). A run-time length whose values are all known while the
kernel compiles (it selects between constants) and include one off the
timesteps is a ``ControllerLimitError`` then. A run-time operand of a
waveform whose parameters the export does not name can't be converted, so
on a controller with its own clock it is a ``ControllerLimitError`` too,
never exported in program cycles.

Line delays
-----------

Cables and electronics delay each line differently. The calibrated delays
belong to the system, not to kernels: ``ControlSystem(line_delays_ns=...)``
gives them by line, ``("drive", qudit)`` or ``("readout", qudit)``, as times
(``2.3 * pulse.ns``) or numbers of ns, and every operation on a delayed line starts that much later than scheduled.
Whole :ref:`timesteps <timestep>` of the line move its operations, which
are then checked against the controller's limits like any others; the rest,
less than one timestep, is the line's ``fine_delay_ns`` in the export, for
a controller that can delay a line's output by less than that
(``Controller.max_fine_delay_ns``). A controller without fine delay
rounds the delay to the nearest timestep if that is within its
``alignment_tolerance_cycles``, and otherwise the program fails to compile
with an error naming the line, the rest and the fixes::

    control_system = pulse.ControlSystem(
        controllers=[pulse.Controller(clock_ghz=1.0, max_fine_delay_ns=1.0)],
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

Two plain values of ``Controller`` describe the memory a controller
plays waveforms from, the same in Python and in a controller description
in TOML (``waveform_memory_samples = 65536``, ``waveform_memory = "shared"``):

.. code-block:: python

   import cudaq_pulse as pulse

   limits = pulse.Controller(
       sample_rate_ghz=2.0,
       waveform_memory_samples=65536,     # what one memory holds, in samples
       waveform_memory="shared",   # one memory for every line; "per_line" by default
   )

Compiling checks that every waveform of a program fits at once: in the
memory of its line, or once over all lines when ``"shared"``. The
controller plays one stored shape at any amplitude and phase, so waveforms
that differ only in those share memory, even when a ``float`` parameter sets
the amplitude. A limit that only some ``float`` values exceed is checked once
when a call binds them, never per shot. A shared memory reaches the export's
``controller`` as ``waveform_memory``, with the waveform ids and parameter
roles of ``/2``; a memory per line leaves the key out.

Controller limits
-----------------

``Controller`` states what a controller can play, in its own clock
cycles: its clock, how
closely two operations on one line can follow each other, its waveform
memory and tones per line, which detector reads each qudit and how long a
readout holds it, the least feedback latency from a readout to an operation
that branches on it, whether it can stop waiting for an answer after a
timeout, and the most drives and readouts in one program. Given to
``pulse.compile(control_system=...)``, alone or in its ``ControlSystem``,
it makes every compilation check the scheduled program, so that a program the controller
cannot play fails at compile time with ``ControllerLimitError``, which lists
each violated limit. A limit that only some values of a ``float``
parameter exceed, such as the tones a frequency shift adds or the memory
a waveform width takes, is checked when a call supplies the values, once
per distinct values, so a call that compiles also exports.
``cudaq_pulse.controller.check(lowered, limits)`` returns the violations
without raising. A control system without controllers checks nothing
more.

A controller that plays nothing while one of its classical calls is out
declares ``Controller(blocks_on_calls=True)``. Each call then holds every
line from its submission until its answer is due: the scheduler sends it
once the lines before it are done and starts nothing after it before then,
in every schedule, and a program played on such a controller is checked
for it, naming each pulse that plays while a call is out. A call in a
kernel with its own ``schedule=``, which is placed whole, does not compile
for it. The export lists ``blocks_on_calls`` among the limits only when it
is set. In a control system of several controllers, a call holds each
controller that blocks on calls and issues it (whose instance of a
per-controller link runs it, else those that send its measurements, else,
over a shared link, every controller that plays), reads a measurement the
call sends or plays a pulse that uses its answer (branches on it, or
computes with it); the lines of the others play on while it is out.

A minimal backend
-----------------

A backend is an object with a ``name`` and three methods, the
``cudaq_pulse.backends.Backend`` protocol: ``control_system()`` returns the
control system to compile for (a ``ControlSystem``, or its one
``Controller``), ``emit(program)`` takes an
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

   from cudaq_pulse import Controller
   from cudaq_pulse.backends import IncompatibleBackendError
   from cudaq_pulse.controller import FORMAT


   class TextBackend:
       name = "text"
       formats = (FORMAT,)

       def control_system(self) -> Controller:
           return Controller(clock_ghz=1.0, instruction_cycles=2, min_idle_cycles=2)

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
       TargetDescription(name="one-qubit", qubits={0: Qubit(0, 5.0e9, -2.0e8, 50.0, 30.0)}),
       clock_ghz=1.0,
   )
   cal = {"x_duration": 40 * pulse.ns, "x_sigma": 10 * pulse.ns, "readout_duration": 200 * pulse.ns}


   @pulse.kernel
   def drive_and_measure(q):
       drive_line = get_drive_line(q)
       readout_line = get_readout_line(q)
       drive(drive_line, gaussian(cal["x_duration"], 0.5, cal["x_sigma"]))
       sync(drive_line, readout_line)
       readout(readout_line, square(cal["readout_duration"], 0.05))


   compiled = pulse.compile(drive_and_measure, device=dev, control_system=backend.control_system())
   program = compiled.lower(dev.qudit_handles[0]).export_controller()
   print(backend.run(backend.emit(program), shots=1))
   # [['0 drive line 0 gaussian at 5e+09 Hz', '40 readout line 1 square at 5e+09 Hz']]

Validating a backend
--------------------

``backends.validate(backend, controller=...)`` runs the export test corpus,
which ships with cudaq-pulse, through a backend: a program for each kind of
event (``cudaq_pulse.Event``), one with a branch and one with values known
only at run time, compiled for the controller (by default the backend's
control system's) and exported in each format the backend reads. Each must be emitted or refused with
``IncompatibleBackendError``, and a program with an event kind no format
has, or in a newer format, must be refused. A program that exceeds the
controller's limits is skipped.

.. code-block:: python

   from cudaq_pulse import Controller, backends

   report = backends.validate("text", controller=Controller(clock_ghz=1.0))
   print(report)
   # PASS play
   # ...
   # REFUSED call: cannot play event 1
   # ...
   # PASS unsupported_event
   # PASS newer_format
   # backend 'text': 7 passed, 5 refused
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
control system without controllers adds nothing to compilation. The
export and the limit checks are native. Exporting in ``/1`` takes time and
memory linear in a program's executions; in ``/2``, and for the checks, a
loop whose iterations repeat costs as much as its first two iterations and
its last, whatever its trip count. A violation inside it is still listed
once per iteration, each message built when it is read.
