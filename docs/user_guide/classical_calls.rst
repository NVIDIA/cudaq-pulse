..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Classical Calls at QPU Runtime
==============================

``classical`` is a classical real-time dialect for co-processing at QPU
runtime. A pulse program calls GPU, host or FPGA services while the QPU runs:
a decoder turns syndromes into corrections, an estimator turns outcomes into
the next pulse parameters, a calibration service updates a frequency. The
answer is used later in the same shot, to set an amplitude, a phase, a
frequency, the length of a wait or the branch taken.

A call is two timed events. The request is *submitted* on a link, and the
answer is *taken* where the program needs it. The scheduler budgets the
function's worst-case latency between them, so pulses that depend on the
answer start after it is due and pulses that do not are not delayed. It
reports, for every call, how much slack the answer has.

``examples/19_classical_calls.py`` uses everything on this page end to end,
and ``examples/20_adaptive_calibration.py`` calibrates against a drifting
device with the estimators described under "Calibrating while the QPU runs";
see :doc:`../examples`.

Declaring a function
--------------------

``@pulse.classical`` declares a function from its body: the type hints of its
parameters and result say what it takes and returns, and the options give
its latency and the link its requests travel on. The body is the reference
implementation: simulation and emulation run it in place of the service.

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.classical(
       min_latency_ns=200,
       nominal_latency_ns=300,
       max_latency_ns=400,
       link="gpu",
   )
   def decode(syndrome: pulse.Bits) -> bool:
       return bool(syndrome[0])

A parameter is one of:

- ``pulse.Bits``: measurement results read as bits, as in ``decode.submit([m])``;
- ``list[int]``: measurement results read as levels (``0``, ``1``, ``2``, ...
  for a qudit), or a list of integers;
- ``list[complex]``: the IQ points of readouts, or ``complex`` for one, with I
  as the real part and Q as the imaginary part;
- ``float``, ``int`` or ``list[float]``.

A result is ``float``, ``int``, ``bool``, ``pulse.Bits``, ``list[float]``,
``list[int]``, a tuple of them, or ``None``. A ``bool`` or ``pulse.Bits``
result gives values a branch can test.

**List lengths.** A list parameter takes its length from the call. A
function called with lists of two lengths has two *versions*, each with its
own wire id and schema, named by its lengths: ``flip.submit([m0, m1, m2])``
and ``flip.submit([m0])`` call ``flip__3`` and ``flip__1``, and reports, the
call manifest and ``controller.resources`` name the version. The length of a
list result is read from the body, for each version:

.. code-block:: python
   :class: fragment

   @pulse.classical(max_latency_ns=100)
   def flip(outcomes: pulse.Bits) -> pulse.Bits:
       return [b ^ 1 for b in outcomes]  # flip__3 answers with 3 bits

Lengths are read from the function's bytecode, not its source: from
slices, concatenation, comprehensions, ``for`` loops, ``while`` loops the
lengths end, branches the lengths decide, a table of equal-length lists, and
counts that cancel (the set bits of a ``pulse.Bits`` and the items a
comprehension keeps, padded back to an input's length). Where the values
decide it (a comprehension with a condition, a ``while`` loop the values
end, two returns of different lengths), compiling says so and names the
line, and ``lengths=`` states it: ``{"return": 4}``, ``{"return": "outcomes"}`` for the
length of an input, or for a tuple result one entry per item, ``None`` for a
scalar, as in ``{"return": (None, "syndrome")}``. ``lengths={"outcomes": 3}``
pins a parameter to one length; a function whose list parameters are all
pinned has one schema under its own name, so its list results need stated
lengths too. ``flip.specialize(outcomes=3)`` is the version of one length, for
``wire.schema`` or ``wire.native``. A function given ``id=`` has one version,
and calling it with a second length is an error.

The hints are read when a program that calls the function is first
compiled, and a mistake in them is raised there, naming the function. The
body is a ``def``: a lambda has no hints.

``pulse.classical.external`` declares a function that is served elsewhere,
with no body to run, from a stub whose hints give its schema. The device
that serves it fixes its layout, so every list length is stated:

.. code-block:: python
   :class: fragment

   @pulse.classical.external(max_latency_ns=50, lengths={"index": 2})
   def table(index: pulse.Bits) -> float:
       raise NotImplementedError

The stub's body never runs; ``raise NotImplementedError`` is the body that
mypy and pyright both accept for it.

The latency has a minimum, a nominal value and a maximum. The scheduler
budgets the maximum. Simulation and emulation use a triangular distribution
over ``[min, max]`` with its mode at the nominal value. ``latency_mode``
chooses what happens when an answer could be late: ``"bounded"`` (the
default) trusts the maximum, ``"timeout"`` gives up after ``timeout_ns``, and
``"blocking"`` stops the lines that wait for the answer until it arrives
(see the barriers below). ``submit="lazy"``
sends a request as late as the budget allows rather than as soon as its
arguments are ready, and ``deadline_ns`` bounds how long after its submission
an answer may be due.

Outside a kernel, calling the function runs its body, and ``fn.submit(...)``
returns an answer whose ``result()`` is ready at once. Between-shot code can
use the same functions as kernels.

Links
-----

A link is a path to a classical device with a cost: a fixed time per
request, a bandwidth, a feedback time from an answer arriving to a pulse
using it, a number of requests it holds at once and the largest request it
takes. They belong to the machine model, named or by id:

.. code-block:: python
   :class: fragment

   from cudaq_pulse.classical import Link
   from cudaq_pulse.compiler import MachineModel

   machine = MachineModel(
       links={
           "gpu": Link(fixed_ns=100, bytes_per_ns=10, feedback_ns=50),
           "controller": Link(fixed_ns=4, feedback_ns=8, max_in_flight=4),
       }
   )
   lowered = pulse.compile(kernel, device=dev, machine=machine).lower(q)

A function's ``link=`` names a link of the machine model, or gives a link id.
Named links get the lowest ids no other link has, in the order given, so the
first named link of a model without ids is link 0, the default. Compiling a
function that names a link the machine model does not have is an error.

Calling from a kernel
---------------------

In a kernel, ``fn.submit(...)`` sends the request and returns a future, and
``future.result()`` takes the answer:

.. code-block:: python
   :class: fragment

   @pulse.kernel
   def feed_forward(q):
       d = get_drive_line(q)
       r = get_readout_line(q)
       readout_time = 500 * pulse.ns
       decode_deadline = 350 * pulse.ns
       pi_time = 20 * pulse.ns
       pi_amplitude = 0.16  # square(pi_time, pi_amplitude) is a pi pulse
       m = readout(r, square(readout_time, 0.05))
       in_time, correction = decode.submit([m]).result(timeout=decode_deadline)
       sync(d, r)
       if in_time:
           if correction:
               drive(d, square(pi_time, pi_amplitude))
       else:
           if m:
               drive(d, square(pi_time, pi_amplitude))

With ``timeout`` the result starts with a flag that is true when the
answer came in time; the other values are zero when it did not, and the
program takes its fallback. ``timeout`` is a duration like any other: an
``int`` number of program cycles, or a time such as ``350 * pulse.ns``,
which converts once when the kernel compiles. A future is taken in the loop iteration that
submitted it. It may be passed to a kernel the program calls, or returned
from one, and taken there; an answer taken inside a kernel called with its
own ``schedule=`` is used only inside it, so return the future instead when
the caller needs the answer. Answers may be carried from one iteration to
the next, as an estimator's state is.

A wait may take its length from an answer, between bounds. An answer is a
plain number, a number of cycles; multiply it by a unit for an answer in
nanoseconds:

.. code-block:: python
   :class: fragment

   shortest_wait = 100 * pulse.ns
   longest_wait = 20 * pulse.us
   alpha, beta, tau = update_t1.submit([m], alpha, beta, tau).result()
   wait(d, tau * pulse.ns, bounds=(shortest_wait, longest_wait))

Such a variable wait is a barrier for the lines that depend on it, and it
opens a new *epoch* for them. The static schedule places it at its lower
bound, and what follows it on those lines shifts by the wait's actual length,
which is clamped to the bounds. ``Lowered.epochs()`` lists the epochs. A
variable wait may not be inside a ``schedule_block`` or a branch; a blocking
wait is a barrier in the same way.

A barrier holds back the lines of its *timeline*: lines share one when one
waits for the other, through a ``sync`` naming both, a measurement or an
answer used on both, a branch on one around the other, a tone or a value
carried between them, or a kernel with its own ``schedule=`` that plays on
both. They sync before the barrier; lines on other timelines keep their time,
so qubits that wait on answers of their own run side by side. Timelines also
merge where their schedules could not move independently: when a link has
fewer slots than the timelines that use it, under ``rcp`` and ``alap_rcp``
when there are more lines than lanes, and for a blocking wait whose answer
nothing uses, which holds back every line. Timelines that share a link each
get a share of its slots and budget each request as sent after the others',
so a request on a shared link is due a few cycles later than alone.

A length may also add ``float`` kernel parameters to an answer, as in
``wait(d, (answer + tau) * pulse.ns, bounds=(4 * pulse.ns, 500 * pulse.ns))``. The kernel still compiles once:
``tau`` is taken when the kernel is called, and each call checks that the
length with its answers taken as 0 (here ``tau``) is finite and at most the
upper bound, raising ``ValueError`` otherwise; it may be below the lower
bound, since the answer adds to it. The wait stays a barrier, and its whole
length is clamped to the bounds at run time.

Scheduling and the report
-------------------------

Every schedule (``asap``, ``alap``, ``rcp``, ``alap_rcp``) places calls the
same way:

- A request is submitted once its arguments are ready (eager, the default)
  or as late as the budget allows (lazy). It holds its link for
  ``fixed_ns + bytes / bytes_per_ns``, and a link holds ``max_in_flight``
  requests at once: one more waits until a slot is free, so a program never
  fails for having too many requests in flight.
- Its answer is due after the budget: the maximum latency (or the timeout),
  plus the link's feedback time.
- An operation whose amplitude, phase, frequency, duration or branch
  condition comes from the answer starts no earlier than that.
- A kernel called with its own ``schedule=`` keeps its calls where its own
  schedule puts them, eager submissions included, and moves with them: it
  is placed where its requests fit the slots its caller's requests leave.
- A blocking wait, like a variable wait, starts once every line of its
  timeline is done.

``Lowered.classical_report()`` gives one ``ClassicalCall`` per execution of a
call, in clock cycles: when it is submitted (``submit_cycles``), when its answer
is due (``due_cycles``) and taken, the first operation that uses it, its slack
(``slack_cycles``, how much later the answer could arrive without delaying a
pulse) and, with ``deadline_ns``, how far inside its deadline it is.
``exposed_cycles`` is how long the first use waits for the answer beyond when it
could start anyway: after its line is free and after the readout resonator
has depleted (``Qubit.depletion_ns``, from ``readout_params["depletion_ns"]``
or from ``kappa_eff_hz``). The rest of the budget, ``hidden_cycles``, overlaps
work that had to happen anyway, so a latency within the depletion time costs
nothing. The
timeline (``Lowered.timeline()``) draws each link as a row of its own,
labelled with its name.

After scheduling, ``pulse-classical-check`` verifies that every request and
answer fits its link's slots, that deadlines hold, that no answer is taken
before it is due, that no link holds more requests than ``max_in_flight``,
and that the submissions on a link go out one after another, in program
order.

Simulation
----------

``pulse.simulate`` runs a program with calls per measurement path. When a
call is due, its handler runs on the path's outcomes and numbers, the answer
is bound, and the path is scheduled again, so variable waits take their
actual lengths on each path. Branches share the state rather than copying
it.

.. code-block:: python
   :class: fragment

   result = pulse.simulate(lowered, initial_state=plus, latency_branching=True)
   for branch in result.branches:
       print(branch.levels, branch.probability, branch.answers("decode"))

- ``handlers={"decode": fn}`` replaces a function's body in every version
  of it, or provides one for a function declared with
  ``pulse.classical.external``; a handler named by a version, such as
  ``"flip__3"``, serves that version only.
- ``latency_branching=True`` branches a timeout on whether the answer came
  in time, with the probability of the latency model; with ``shots`` that
  outcome is sampled.
- ``readout="model"`` reports outcomes through each qubit's assignment
  errors, and ``"model_with_decay"`` adds relaxation during the readout.
  Calls and branches see the reported outcome; the state is projected on the
  true one.
- A function that takes IQ points (``complex`` or ``list[complex]``) gets
  them synthesized per shot from a
  dispersive readout model, at the true level of each readout, from the
  frequency of its tone and the amplitude of its pulse. It needs ``shots``.
- Each of ``result.branches`` lists its path's ``calls`` (inputs, answer,
  and whether it came in time); ``branch.answers("decode")`` gives the
  answers of one function, in call order.

A whole program with unanswered calls does not lower to LLVM:
``Lowered.mlir("llvm")`` points to ``pulse.simulate``, which lowers each
segment with its answers bound, on the CPU or the GPU.

``plant=`` simulates another target description than the device the program
was compiled for, keeping the program as compiled: its qubits' frequencies,
T1, T2, drive strength and readout parameters apply. That is how a drifting
device is simulated (see below).

Native and QEC handlers
-----------------------

``cudaq_pulse.classical.wire`` encodes requests and answers in the CUDA-Q
real-time format: a 24-byte header, then the arguments packed by type.
``wire.native(library, symbol, fn)`` makes a handler from a C function with
the CUDA-Q host-call signature, ``void handler(const void *request, void
*answer, size_t slot_bytes)``, so the same compiled handler serves
simulation, emulation and a real dispatcher.

``cudaq_pulse.classical.qec.Decoder`` declares the calls of a CUDA-QX
real-time decoder (enqueue syndromes, get corrections, reset) with their wire
ids, so a program built with it talks to a CUDA-QX decoding server
unchanged; in simulation, a Python function decodes each path's syndromes.

Emulation
---------

``Lowered.export_emulation()`` writes what a controller or an emulator runs
the program from, as JSON-ready data: the static schedule, each event's
epoch, the epochs, the calls, the links and the call manifest (every
function's wire id, link, payload sizes, latency and slots).

``cudaq_pulse.classical.emulation.emulate`` runs the program shot by shot in
virtual time without physics. Outcomes come from a source you give; each
call goes out as real wire bytes, in process or over UDP to a server from
``serve_udp``; and its answer arrives after a modelled transport delay in
each direction plus the service's measured time. A late answer takes the
timeout arm.

.. code-block:: python
   :class: fragment

   from cudaq_pulse.classical import emulation

   run = emulation.emulate(
       lowered,
       shots=200,
       transport={"gpu": emulation.Transport(fixed_ns=120, jitter_ns=40)},
       seed=1,
   )
   print(run.late_fraction, run.calls_per_second)

Calibrating while the QPU runs
------------------------------

``cudaq_pulse.classical.calibration`` holds estimators whose state goes in and
comes back out, so the same function is the body of a classical function a
kernel calls between measurements and runs between shots:

- ``t1_update`` and ``t1_wait``: a gamma posterior over the decay rate,
  updated from one "excite, wait, measure" shot, and the next wait.
- ``frequency_update`` and ``ramsey_setting``: a Gaussian posterior over the
  detuning, updated from one Ramsey shot, and the next wait and phase. The
  estimators' Ramsey shot reports 1 with probability
  ``(1 - cos(2 pi d t + phase)) / 2``; two pi/2 pulses with the second's
  tone shifted by ``pi - phase`` give it.
- ``three_point_decay`` and ``three_point_phase``: a decay time and a phase
  error from three points, whatever the amplitude and offset.
- ``golden_section_step`` and ``nelder_mead_step``: ask-and-tell searches;
  each step takes the cost at the point the last one asked for and returns
  the next point and whether the search has converged. Their states pack
  into a ``list[float]`` for calls.

.. code-block:: python
   :class: fragment

   from cudaq_pulse.classical import calibration as cal

   @pulse.classical(max_latency_ns=200, link="controller")
   def t1_estimator(
       survived: pulse.Bits, alpha: float, beta: float, wait_ns: float
   ) -> tuple[float, float, float]:
       alpha, beta = cal.t1_update(alpha, beta, bool(survived[0]), wait_ns)
       return alpha, beta, min(max(cal.t1_wait(alpha, beta), 100), 10000)

**The converged pattern.** A kernel's trip counts are fixed when it
compiles. To stop early, the estimator carries a converged flag in its state,
keeps answering with its final point once converged, and the kernel gates its
probe pulses on the flag; values cannot leave the arms of an ``if``, so the
call stays outside it:

.. code-block:: python
   :class: fragment

   for _ in range(iterations):
       if done < 0.5:
           shift_frequency(d.tone, x)
           drive(d, probe)
           shift_frequency(d.tone, -x)
       m = readout(r, ro)
       state, x, done = golden_step.submit([m], state).result()

**Choosing among settings.** An integer answer selects among arms with
``if i == 0: ... elif i == 1: ...``, or a balanced split, ``if i < 2:`` with
nested tests, which needs fewer comparisons for many arms. Every arm is
scheduled to end together, after the longest.

**Randomness.** ``cudaq_pulse.classical.random_index(n)`` is an on-device
random source: an answer drawn uniformly from ``0 .. n - 1`` by the
controller (the manifest marks it ``"random"``). Simulation without shots
follows every value with probability 1/n; with shots each shot draws one.
Answers can be passed to later calls, for example to compute a
randomized-benchmarking sequence's recovery gate.

**Between shots.** A loop compiles once and runs with the current parameters
each iteration. Amplitudes, phases and frequencies can be float parameters of
the kernel. A quantity that shapes the timeline, such as a wait, is read at
run time: ``ParameterStore.reader(name)`` (``cudaq_pulse.classical.loop``) is
a classical function that answers with the store's current value, on
hardware a read of a controller register, and a variable wait takes its
length from it. ``EstimateLog`` records each estimate with what was true,
when it is known.

.. code-block:: python
   :class: fragment

   store = ParameterStore(wait_ns=2000.0)
   read_wait = store.reader("wait_ns", link="controller", bounds=(0.0, 20000.0))

   @pulse.kernel
   def shot(q, amplitude: float):
       ...
       longest_wait = 20 * pulse.us
       wait(d, read_wait.submit().result() * pulse.ns, bounds=(0, longest_wait))

   compiled = pulse.compile(shot, device=dev, machine=machine)
   for i in range(10_000):
       lowered = compiled.lower(q, amplitude)   # no compiling after the first
       ...
       store.set("wait_ns", next_wait, iteration=i)

**A drifting device.** ``cudaq_pulse.runtime.drift.DriftingPlant`` moves one
qubit's T1 (``TelegraphT1``, switching between two values), frequency and
drive amplitude (``OrnsteinUhlenbeck``) as time advances, and records every
state in ``plant.truth``. ``plant.target()`` is the device now, for
``pulse.simulate(..., plant=...)``; ``p_survive``, ``p_ramsey`` and
``p_rabi`` give the standard experiments' outcome probabilities directly, for
loops of many iterations.

A calibration chain on the controller
-------------------------------------

``examples/21_calibration_protocol.py`` runs the millisecond calibration
chain of

   M. A. Marciniak, R. T. Birke, J. B. Severin, F. Berritta, D. Kjær,
   F. Nilsson, S. N. Themadath, S. Kallatt, J. L. Webb, K. Bentsen, T. Madsen,
   Z. Sun, S. Krøjer, C. W. Warren, J. Hastrup and M. Kjaergaard,
   "Millisecond-Scale Calibration and Benchmarking of Superconducting Qubits",
   arXiv:2602.11912 (2026),

as one program whose estimators are classical functions on the controller's
link, so nothing goes to a host or a GPU between steps: readout assignment
errors, the drive frequency from Ramsey shots at three phases (sparse phase
estimation), the pi and pi/2 amplitudes from pulse trains at three amplitude
scales, T1 from the analytical decay estimator at delays spaced by the last
estimate, and T1 again by the adaptive tracker of

   F. Berritta, J. Benestad, J. A. Krzywda, O. Krause, M. A. Marciniak,
   S. Krøjer, C. W. Warren, E. Hogedal, A. Nylander, I. Ahmad, A. Osman,
   J. Biznárová, M. Rommel, A. F. Roudsari, J. Bylander, G. Tancredi, J. Danon,
   J. Hastrup, F. Kuemmeth and M. Kjaergaard, "Real-time adaptive tracking of
   fluctuating relaxation rates in superconducting qubits", arXiv:2506.09576
   (2025),

whose prior is centred on the previous estimate and whose likelihood uses
the measured assignment errors. Each step is a kernel that returns its
estimator's answer, and the next step plays its pulses with it: the drive
tone is shifted by the measured detuning, the trains use the corrected
amplitudes, the delays come from the T1 estimate.

The estimators in ``cudaq_pulse.classical.calibration`` that the chain uses
(``adaptive_t1_update``, ``binary_search_update``, ``decay_time``,
``sparse_phase`` and their helpers) are in *controller form*: a fixed
number of scalars of state, closed-form arithmetic, branches and loops of a
trip count fixed when the program compiles, and no allocation or
exceptions. An estimate that cannot be formed is a result
(``decay_time`` returns whether it formed one), not an error.

Compile the chain with ALAP scheduling (the default). A pulse just before a
variable wait then ends where the wait starts, so the qubit decays for the
answer's length; under ASAP the excitation plays as early as it can, and
the qubit also decays through the call's latency.

A program that measures hundreds of times in one shot is simulated one
path per shot: ``pulse.simulate(lowered, shots=n, branches="sampled")``. Idle
waits and square pulses are propagated exactly, so a shot's cost does not
grow with its waits.

What it costs the controller
----------------------------

``cudaq_pulse.controller.resources(lowered)`` reads each classical
function's body once and reports its operations by class on its longest
path, how many it holds, its state in bytes, its compute time against its
``max_latency_ns``, and its calls, slots and bytes per shot. An expression
of constants costs nothing, as it folds when the program compiles, and
dividing by a constant counts as a multiply. The limits come
from ``ControllerLimits``, for the functions on its ``compute_links``:

.. :spellcheck-disable:

.. code-block:: python
   :class: fragment

   from cudaq_pulse import controller

   limits = controller.ControllerLimits(
       op_cycles={"div": 8, "sqrt": 8, "exp": 12, "log": 12, "trig": 12},
       compute_clock_ghz=0.25,       # 4 ns per cycle
       controller_form=True,
       max_ops=256,
       state_bytes=512,
   )
   machine = MachineModel(links={"controller": Link(fixed_ns=4)}, controller=limits)
   lowered = pulse.compile(chain, device=dev, machine=machine).lower(q)
   print(controller.resources(lowered))

.. :spellcheck-enable:

The report is a table: for the chain's adaptive T1 update, 47 operations
(one ``exp``, one ``log``, eight divisions) and 0.5 us at these costs, called
50 times per shot. A function that computes past its budget, holds too many
operations or bytes, or (with ``controller_form=True``) uses a ``while``
loop, a list or an exception fails to compile with ``ControllerLimitError``
naming it, the limit and the line. Without any of these limits, compiling
does not read function bodies.

Running on hardware
-------------------

A controller that runs these programs needs, beyond its pulse sequencer: a
way to send a request on a link at a scheduled time and to receive the
answer into registers; branching on an answer; frequency, phase and
amplitude updates from an answer; and, for variable waits, a wait whose
length comes from a register, after which the rest of the epoch is released
together. The emulation artifact gives every event its epoch and its time
from the epoch's start for that purpose. An epoch holds the lines of its
barrier's timeline, so epochs of different timelines move independently: an
epoch's shift adds to that of the epoch it follows, which is the one before
it unless the epoch names another in ``after``.

Not available yet: a channel through the CUDA-Q Realtime host library
(``emulation.RealtimeChannel``), exporting calls as CUDA-Q
``cc.device_call`` operations (``emulation.device_call_text``), and lowering
links to a controller dialect. The wire format, the manifest and the
handler signature are those of CUDA-Q Realtime, so these slot in without
changing programs.

Questions
---------

**Can a parameter be a** ``Sequence`` **rather than a** ``list``**?** Yes:
``Sequence[int]``, ``Sequence[float]`` and ``Sequence[complex]``, from
``collections.abc`` or ``typing``, mean what the ``list`` forms do. A type
checker then also accepts a tuple, or a ``pulse.Bits``, where ``list[int]``
takes only a list; this guide writes ``list``.
