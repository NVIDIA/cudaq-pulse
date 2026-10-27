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

``@pulse.classical`` declares a function with the types of its arguments and
results, its latency and the link its requests travel on. The body is the
reference implementation: simulation and emulation run it in place of the
service.

.. code-block:: python

   import cudaq_pulse as pulse
   from cudaq_pulse.classical import Bits, Float

   @pulse.classical(
       args=(Bits(1),),
       results=(Bits(1),),
       min_latency_ns=200,
       nominal_latency_ns=300,
       max_latency_ns=400,
       link="gpu",
   )
   def decode(syndrome):
       return [syndrome[0]]

Argument types are ``Bits(n)`` and ``Levels(n)`` (measurement results),
``IQ(n)`` (the IQ points of ``n`` readouts), ``Float()``, ``Integer()``,
``Byte()`` and the arrays ``Floats(n)``, ``Integers(n)`` and ``Bytes(n)``.
Results use the same types; a ``Bits`` or ``Levels`` result gives booleans or
integers a branch can test.

The latency has a minimum, a nominal value and a maximum. The scheduler
budgets the maximum. Simulation and emulation use a triangular distribution
over ``[min, max]`` with its mode at the nominal value. ``latency_mode``
chooses what happens when an answer could be late: ``"bounded"`` (the
default) trusts the maximum, ``"timeout"`` gives up after ``timeout_ns``, and
``"blocking"`` stops every line until the answer arrives. ``submit="lazy"``
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

   @pulse.kernel
   def feed_forward(q):
       d, t = get_drive_line(q)
       r, rt = get_readout_line(q)
       m = readout(r, square(1000, 0.05), rt)
       in_time, correction = decode.submit([m]).result(timeout_ns=350)
       sync(d, r)
       if in_time:
           if correction[0]:
               drive(d, square(40, PI_AMP), t)
       else:
           if m:
               drive(d, square(40, PI_AMP), t)

With ``timeout_ns`` the result starts with a flag that is true when the
answer came in time; the other values are zero when it did not, and the
program takes its fallback. A future is taken in the kernel that submitted
it, and in the same loop iteration. Answers may be carried from one iteration
to the next, as an estimator's state is.

A wait may take its length from an answer, between bounds:

.. code-block:: python

   alpha, beta, tau = update_t1.submit([m], alpha, beta, tau).result()
   wait(d, tau, bounds=(200, 40000))

Such a variable wait is a barrier: every line syncs before it, and it opens a
new *epoch*. The static schedule places it at its lower bound, and everything
after it shifts by the wait's actual length, which is clamped to the bounds.
``Lowered.epochs()`` lists the epochs. A variable wait may not be inside a
``schedule_block`` or a branch; a blocking wait is a barrier in the same way.

A length may also add ``float`` kernel parameters to an answer, as in
``wait(d, answer + tau, bounds=(8, 1000))``. The kernel still compiles once:
``tau`` is taken when the kernel is called, and each call checks the length
with its answers taken as 0 (here ``tau``) against the bounds, raising
``ValueError`` outside them. The wait stays a barrier, and its whole length
is clamped to the bounds at run time.

Scheduling and the report
-------------------------

Every schedule (``asap``, ``alap``, ``rcp``, ``alap_rcp``) places calls the
same way:

- A request is submitted once its arguments are ready (eager, the default)
  or as late as the budget allows (lazy). It holds its link for
  ``fixed_ns + bytes / bytes_per_ns``, and a link holds ``max_in_flight``
  requests at once.
- Its answer is due after the budget: the maximum latency (or the timeout),
  plus the link's feedback time.
- An operation whose amplitude, phase, frequency, duration or branch
  condition comes from the answer starts no earlier than that.

``Lowered.classical_report()`` gives one ``ClassicalCall`` per execution of a
call, in clock cycles: when it is submitted (``submit_vtu``), when its answer
is due (``due_vtu``) and taken, the first operation that uses it, its slack
(``slack_vtu``, how much later the answer could arrive without delaying a
pulse) and, with ``deadline_ns``, how far inside its deadline it is.
``exposed_vtu`` is how long the first use waits for the answer beyond when it
could start anyway: after its line is free and after the readout resonator
has depleted (``Qubit.depletion_ns``, from ``readout_params["depletion_ns"]``
or from ``kappa_eff_hz``). The rest of the budget, ``hidden_vtu``, overlaps
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

   result = pulse.simulate(lowered, initial_state=plus, latency_branching=True)
   for calls, (outcomes, p) in zip(result.calls, result.branches):
       print(outcomes, p, [c.answer for c in calls])

- ``handlers={"decode": fn}`` replaces a function's body, or provides one
  for a function declared with ``pulse.classical.external``.
- ``latency_branching=True`` branches a timeout on whether the answer came
  in time, with the probability of the latency model; with ``shots`` that
  outcome is sampled.
- ``readout="model"`` reports outcomes through each qubit's assignment
  errors, and ``"model_with_decay"`` adds relaxation during the readout.
  Calls and branches see the reported outcome; the state is projected on the
  true one.
- A function that takes ``IQ`` points gets them synthesized per shot from a
  dispersive readout model, at the true level of each readout, from the
  frequency of its tone and the amplitude of its pulse. It needs ``shots``.
- ``SimulationResult.calls`` lists each path's calls: inputs, answer, and
  whether it came in time.

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
  into ``Floats`` for calls.

.. code-block:: python

   from cudaq_pulse.classical import calibration as cal

   @pulse.classical(args=(Bits(1), Float(), Float(), Float()),
                    results=(Float(), Float(), Float()),
                    max_latency_ns=200, link="controller")
   def t1_estimator(survived, alpha, beta, wait_vtu):
       alpha, beta = cal.t1_update(alpha, beta, bool(survived[0]), wait_vtu / 2.0)
       return alpha, beta, min(max(cal.t1_wait(alpha, beta) * 2.0, 200), 20000)

**The converged pattern.** A kernel's trip counts are fixed when it
compiles. To stop early, the estimator carries a converged flag in its state,
keeps answering with its final point once converged, and the kernel gates its
probe pulses on the flag; values cannot leave the arms of an ``if``, so the
call stays outside it:

.. code-block:: python

   for _ in range(iterations):
       if done < 0.5:
           shift_frequency(t, x)
           drive(d, probe, t)
           shift_frequency(t, -x)
       m = readout(r, ro, rt)
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

   store = ParameterStore(wait_vtu=4000.0)
   read_wait = store.reader("wait_vtu", link="controller", bounds=(0.0, 40000.0))

   @pulse.kernel
   def shot(q, amplitude: float):
       ...
       wait(d, read_wait.submit().result(), bounds=(0, 40000))

   compiled = pulse.compile(shot, device=dev, machine=machine)
   for i in range(10_000):
       lowered = compiled.lower(q, amplitude)   # no compiling after the first
       ...
       store.set("wait_vtu", next_wait, iteration=i)

**A drifting device.** ``cudaq_pulse.runtime.drift.DriftingPlant`` moves one
qubit's T1 (``TelegraphT1``, switching between two values), frequency and
drive amplitude (``OrnsteinUhlenbeck``) as time advances, and records every
state in ``plant.truth``. ``plant.target()`` is the device now, for
``pulse.simulate(..., plant=...)``; ``p_survive``, ``p_ramsey`` and
``p_rabi`` give the standard experiments' outcome probabilities directly, for
loops of many iterations.

Running on hardware
-------------------

A controller that runs these programs needs, beyond its pulse sequencer: a
way to send a request on a link at a scheduled time and to receive the
answer into registers; branching on an answer; frequency, phase and
amplitude updates from an answer; and, for variable waits, a wait whose
length comes from a register, after which the rest of the epoch is released
together. The emulation artifact gives every event its epoch and its time
from the epoch's start for that purpose.

Not available yet: a channel through the CUDA-Q Realtime host library
(``emulation.RealtimeChannel``), exporting calls as CUDA-Q
``cc.device_call`` operations (``emulation.device_call_text``), and lowering
links to a controller dialect. The wire format, the manifest and the
handler signature are those of CUDA-Q Realtime, so these slot in without
changing programs.
