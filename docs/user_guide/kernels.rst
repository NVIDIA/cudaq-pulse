..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Pulse Kernels
=============

The ``@pulse.kernel`` decorator is the entry point for writing
pulse-level quantum programs. It captures the decorated function's
Python bytecode and traces it into an intermediate representation
that can be compiled to MLIR.

Import Convention
-----------------

A single import gives you everything:

.. code-block:: python

   import cudaq_pulse as pulse

Inside ``@pulse.kernel`` functions, DSL operations (``drive``,
``gaussian``, ``get_drive_line``, etc.) are used as bare names, which
editors complete and check with no configuration (see :doc:`editors`).
Infrastructure stays behind the ``pulse.`` prefix:

.. code-block:: python

   @pulse.kernel
   def rabi_oscillation(qubit, duration, sigma):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(duration, 0.5, sigma), tone)

   dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
   rabi_oscillation(dev.qudit_handles[0], 32 * pulse.ns, 8 * pulse.ns)

Defining a Kernel
-----------------

.. code-block:: python

   import cudaq_pulse as pulse

   cal = {
       "x_duration": 20 * pulse.ns,
       "x_sigma": 5 * pulse.ns,
       "x_beta": 0.25 * pulse.ns,
       "readout_duration": 500 * pulse.ns,
   }

   @pulse.kernel
   def my_kernel(qubit):
       drive_line, tone = get_drive_line(qubit)
       waveform = gaussian(cal["x_duration"], 0.3, cal["x_sigma"])
       drive(drive_line, waveform, tone)

The kernels on this page read their pulse durations from this ``cal``; see
:ref:`durations`.

Qudit arguments are **qudit handles** of a device: ``dev.qudit_handles[i]``
for one qudit, and ``dev.qudit_handles[0:4]`` or ``dev.qudit_handles[[0, 2]]``
for a register, which a kernel indexes and iterates like a list. The kernel
runs on the device of its handles; see :doc:`devices`. How arguments bind,
which of them the kernel is specialized on, and what a call returns are
described in :doc:`compilation`.

.. code-block:: python

   @pulse.kernel
   def two(q0, q1, amplitude: float):
       ...

   reg = dev.qudit_handles[0:2]
   two(reg[0], reg[1], 0.3)

Qudits always come from a device: a kernel takes them as parameters, and a
call passes the device's handles (``dev.qudit_handles[i]``, or a register
``dev.qudit_handles[0:4]``). A call without handles raises
``NoDeviceError``, unless the kernel was compiled for a device with
``pulse.compile(k, device=dev)``.

The kernel language (``get_drive_line``, ``drive``, ``gaussian``, ...) lives
in ``cudaq_pulse.ops``; inside a kernel its names are used bare, as above.

.. _durations:

Durations: cycles or times
--------------------------

A kernel counts time in cycles of the device's clock (``dev.clock_ghz``).
A duration (a waveform's length or flat top, a ``wait`` and its bounds, a
``pulse.round_duration`` length) is either an ``int`` number of cycles or a time
with units: ``20 * pulse.ns``, ``4 * pulse.us``, also ``pulse.ms`` and
``pulse.s``. A time converts to cycles exactly when the kernel compiles,
for the device it compiles for, so a kernel whose durations are times
runs unchanged on devices with different clocks. Shape parameters
(``sigma``, DRAG ``beta``) take cycles, fractional too, or times.

Keep durations out of the kernel body: pass them as arguments, or read
them from a calibration the kernel closes over.

.. code-block:: python

   cal = {"x_duration": 20 * pulse.ns, "x_sigma": 5 * pulse.ns, "x_amp": 0.2}

   @pulse.kernel
   def x_then_wait(qubit, gap):
       d, t = get_drive_line(qubit)
       x = gaussian(cal["x_duration"], cal["x_amp"], cal["x_sigma"])
       drive(d, x, t)
       wait(d, gap)
       drive(d, x, t)

   fast = pulse.device({"target": "transmon_krinner_17q", "clock_ghz": 2.0})
   slow = pulse.device({"target": "transmon_krinner_17q", "clock_ghz": 1.0})
   x_then_wait.lower(fast.qudit_handles[0], 100 * pulse.ns)  # 40, 200, 40 cycles
   x_then_wait.lower(slow.qudit_handles[0], 100 * pulse.ns)  # 20, 100, 20 cycles

- A plain float in a duration is an error, since it could be cycles or
  nanoseconds; the error shows both spellings::

     gaussian(): duration is a float (20.0). Durations are an int number of cycles, or a time with units:
         gaussian(20, ...)               # 20 cycles
         gaussian(20 * pulse.ns, ...)    # 20 nanoseconds

- A time that is not a whole number of cycles is an error that names its
  cycles; ``pulse.round_duration(line, 10.3 * pulse.ns)`` rounds it to whole
  :ref:`timesteps <timestep>` of the line instead.
- A time times a number, a time plus a time, a time over a number and minus
  a time are times. A time plus a number of cycles, and a time times a
  time, are errors. ``0.1 * pulse.us`` is exactly 100 ns: a time holds its
  value as a fraction, from the digits it is written with.
- A ``float`` parameter is a plain number, a number of cycles where it sets
  a length. Write ``wait(d, tau * pulse.ns)`` for a ``tau`` in nanoseconds:
  the length stays a runtime parameter, converted with one multiply. A time
  passed to a ``float`` parameter is a ``TypeError``. The answer of a
  classical call converts the same way: ``wait(d, f.result() * pulse.ns,
  bounds=(0, 4 * pulse.us))``.
- ``result(timeout=...)`` takes cycles or a time, as every duration does.
  Declaration options such as ``max_latency_ns=350`` stay plain numbers in
  nanoseconds.

Control Flow
------------

Kernels support a subset of Python control flow that can be captured
at trace time:

**For loops** with compile-time integer bounds:

.. code-block:: python

   @pulse.kernel
   def echo_sequence(qubit, gap):
       drive_line, tone = get_drive_line(qubit)
       for i in range(5):
           drive(drive_line, gaussian(cal["x_duration"], 0.3, cal["x_sigma"]), tone)
           wait(drive_line, gap)

A loop whose induction variable is never read, like the one above, is
captured once as a structured ``scf.for`` in the kernel IR, so the IR size
does not depend on the trip count, unless it is small: when the trip count
times the number of calls in the body is at most the unroll threshold, the
loop is unrolled while tracing (see below). Loops that read the induction variable,
or that carry Python state from one iteration to the next, are unrolled
exactly at trace time. Compiling keeps the same loops as an ``scf.for`` that
carries the lines and tones its body advances; the scheduler times every
iteration. A loop bound given by a ``float`` argument specializes the kernel
on its value; ``while`` loops are rejected.

**The unroll threshold** is ``pulse.compile(k, unroll_threshold=n)``, else
``@pulse.kernel(unroll_threshold=n)``, else 8; 0 keeps every such loop
rolled. Both forms are the same program: without passes their
schedules are identical, and with the default passes ``fuse_pulses`` may merge
back-to-back pulses of the unrolled form, with the same evolution. A kernel
or compiled kernel keeps the threshold it was given; compile it again for
another.

The default comes from timing both forms, on one transmon, of loops of 1 to
512 iterations whose bodies hold 2, 3 or 6 calls: compiling (trace, lowering
to MLIR, passes and schedule, a fresh ``pulse.compile`` each time) and
simulating on the CPU backend. A rolled loop costs about 0.3-0.6 ms to
compile at any size; unrolled, compiling costs about 7 us per call and
simulating grows with the number of separate pulses (a rolled loop lowers to
one pulse train). The unrolled form is up to 10 % faster below an estimated
size of 6, even at 8-12, and slower beyond: 1.2-1.4x at 24, 2-3x at 96 and
more than 20x at 1500, where simulation dominates. The default, 8, is the
crossover; ``benchmarks/compiler/unroll_threshold.py`` repeats the measurement.

**For loops over Python values** known when the kernel is traced: lists,
tuples, dictionaries, ``enumerate()``, ``zip()``, ``reversed()``, a qubit
vector, or any other finite iterable. These loops are always unrolled, one
traced iteration per element, with ``break``, ``continue`` and ``else``
behaving as in Python. Lines and tones held in lists stay threaded through
the loop:

.. code-block:: python

   @pulse.kernel
   def sweep(qubits):
       lines, tones = [], []
       for q in qubits:
           line, tone = get_drive_line(q)
           lines.append(line)
           tones.append(tone)
       for amplitude in [0.1, 0.2, 0.3]:
           for line, tone in zip(lines, tones):
               drive(line, square(cal["x_duration"], amplitude), tone)

**If/else** with compile-time conditions:

.. code-block:: python

   @pulse.kernel
   def conditional_pulse(qubit, use_drag):
       drive_line, tone = get_drive_line(qubit)
       if use_drag:
           waveform = drag(cal["x_duration"], 0.3, cal["x_sigma"], cal["x_beta"])
       else:
           waveform = gaussian(cal["x_duration"], 0.3, cal["x_sigma"])
       drive(drive_line, waveform, tone)

**Measurement feedback**: an ``if`` on a measurement result (``if result:`` or
``if not result:``) is kept as a conditional region:

.. code-block:: python

   @pulse.kernel
   def active_reset(qubit):
       drive_line, drive_tone = get_drive_line(qubit)
       readout_line, readout_tone = get_readout_line(qubit)
       result = readout(readout_line, square(cal["readout_duration"], 0.05), readout_tone)
       sync(drive_line, readout_line)
       if result:
           x = drag(cal["x_duration"], 0.5, cal["x_sigma"], cal["x_beta"])
           drive(drive_line, x, drive_tone)

Both branches are traced, and lines and tones they rebind become the
results of the region. Both branches must continue after the ``if``. A
number the branches set differently holds, after the ``if``, the value of
the branch that ran:

.. code-block:: python

   @pulse.kernel
   def flip_on_one(qubit):
       drive_line, drive_tone = get_drive_line(qubit)
       readout_line, readout_tone = get_readout_line(qubit)
       result = readout(readout_line, square(1000, 0.05), readout_tone)
       sync(drive_line, readout_line)
       if result:
           amp = 0.5
       else:
           amp = 0.0
       drive(drive_line, square(40, amp), drive_tone)

The number is one value the measurement selects. On a target that takes it
at run time (an amplitude, a phase or a frequency), every op given it, or a
waveform made from it, reads that value. Otherwise, and always for a
duration or a waveform's shape, each such op runs in each branch with that
branch's value, as if it were written in both. No target takes a selected
value at run time yet. Either way the op waits for the measurement, and
the program schedules and simulates the same. A name assigned in only one branch, with no value
before the ``if``, has none after it: reading it raises a
``CompilationError`` that names it and the line of the ``if``. Other Python
values, qudits, and lines of different kinds must not differ between the
branches. Compiling keeps the
branch as an ``scf.if`` on ``pulse.is_set`` (whether the readout
observed a nonzero level), whose arms yield the lines and tones they
advance; both arms start once the measurement completes, and the code after
the ``if`` waits for the longer arm.

Python Inside Kernels
---------------------

Besides the control flow above, a kernel body may use:

- list, dictionary and set comprehensions, nested and filtered ones too, and
  set displays such as ``{8, 16}``. A comprehension is unrolled while
  tracing like the ``for`` loop it stands for, so
  ``[get_drive_line(q) for q in qubits]`` builds a list of ``(line, tone)``
  pairs and ``[readout(r, wf, t) for _ in range(12)]`` a list of results.
- generator expressions, such as ``sum(a for a in amps)`` or
  ``sync(*(line for line, _ in pairs))``. A generator expression runs to
  the end where it is written, as a list comprehension would, so pulses
  it emits are emitted there, in order, even if what consumes it stops
  early, as ``any()`` does.
- starred assignment, such as ``first, *rest = lines``, and slice
  assignment, such as ``xs[1:3] = ys``.
- ``assert`` and ``raise`` for checks while tracing: a failing
  ``assert n > 0, "msg"`` raises ``CompilationError("assertion failed in
  kernel: msg")``.
- ``sync()`` over lines, lists or tuples of lines, nested too:
  ``sync(lines)`` and ``sync([d0, d1], d2)`` synchronize every line they
  hold.
- calls with ``*args`` and ``**kwargs``, unpacked while tracing, so
  ``sync(*lines)`` synchronizes a list of lines built in a loop and
  ``helper(d, t, **opts)`` passes a dictionary of options.
- lines and tones kept in any container: a list of ``(line, tone)`` pairs, a
  dictionary or nested lists. A qudit has one drive and one readout channel,
  so an operation always uses the channel's current line and tone, whichever
  copy of it the kernel passes. ``append``, ``insert``, ``extend``, ``pop``,
  ``remove`` and ``del xs[i]`` work on lists of lines.

A ``float`` argument stays a runtime parameter, compiled once for every
value, through ``+``, ``-``, ``*``, ``/`` and negation, including as the length
of a ``wait`` (see :ref:`wait-lengths`). Any other use that needs its value
while tracing, such as ``abs``, ``min``, ``math.sin``, ``//``, a comparison,
a ``range`` bound or a waveform duration, makes it static, so each new value
compiles the kernel again. The first such use warns once per kernel and
argument with a ``cudaq_pulse.PulseWarning`` (also a ``RuntimeWarning``)
that names the argument, the operation and the kernel line, and suggests
computing the value on the host, annotating the argument as ``int``, or
giving it a fixed value;
``warnings.filterwarnings("error", category=cudaq_pulse.PulseWarning)``
makes it an error.

.. _wait-lengths:

Wait lengths
~~~~~~~~~~~~

A ``wait`` whose length is computed from ``float`` arguments keeps them
runtime parameters, so a sweep over a delay compiles once:

.. code-block:: python

   import numpy as np

   @pulse.kernel
   def echo(qubit, tau: float):
       d, t = get_drive_line(qubit)
       r, rt = get_readout_line(qubit)
       x90 = drag(cal["x_duration"], 0.25, cal["x_sigma"], cal["x_beta"])
       x180 = drag(cal["x_duration"], 0.5, cal["x_sigma"], cal["x_beta"])
       drive(d, x90, t)
       wait(d, tau / 2 * pulse.ns)
       drive(d, x180, t)
       wait(d, tau / 2 * pulse.ns)
       drive(d, x90, t)
       sync(d, r)
       readout(r, square(cal["readout_duration"], 0.05), rt)

   q = dev.qudit_handles[0]
   compiled_echo = pulse.compile(echo)
   for tau in np.linspace(0, 10_000, 1000):  # nanoseconds
       compiled_echo.lower(q, tau)  # one compilation for the whole sweep

- Each call computes the length from its values, in clock cycles (here
  nanoseconds times ``dev.clock_ghz``), and rounds it to the nearest cycle,
  ties up (``floor(x + 0.5)``): 99.5 cycles is 100, 99.49 is 99.
- The call's schedule is exactly that of the kernel compiled with that
  length: ``sync``, ALAP and ASAP placement, loops, branches and called
  kernels see the rounded length, and other lines are not held back by the
  wait. ``lower()`` only binds the values; the schedule, ``mlir()`` and the
  simulation of a call come from the module with them bound and scheduled
  again, which takes tens of microseconds, not a compile.
- ``wait(line, length, bounds=(lowest, highest))`` bounds the length, in
  cycles or as times; ``highest`` may be ``None``, and without ``bounds`` the length
  only has to be non-negative. A call whose length is not finite, is
  negative or falls outside the bounds raises ``ValueError`` naming the
  kernel, the expression and the value, before anything runs.
- Without values (``pulse.compile(k, device=dev).schedule`` of a kernel that
  only has float parameters, or ``lowered.generic``) the module is scheduled
  with every such wait at its lowest length, 0 unless ``bounds`` says
  otherwise.
- A length that needs more than arithmetic (``int(tau)``, ``tau // 2``,
  ``round(tau)``) still makes the parameter static, with the warning.
  A literal or static length is not rounded: it must be an ``int`` or a
  time that is a whole number of cycles (see :ref:`durations`), and
  ``bounds`` checks it while tracing.

Operations check their arguments when they are traced. Passing a tone
where a line belongs, a number where a waveform belongs, or a readout line
to ``drive`` raises a ``CompilationError`` that names the operation and
the argument, such as ``drive(): argument 1 must be a drive line, got a
tone``, before any MLIR is built.

An error raised while a kernel is traced carries a note with the kernel
source lines that led to it, as a traceback shows them: the line in the
kernel and, for an error inside a helper, the helper's line. A misspelled
operation or attribute gets a suggestion, as in ``unknown pulse op:
drvie; did you mean drive()?``. The kernel language is in
``cudaq_pulse.ops``, so ``pulse.drive(...)`` fails with a message that says
to write ``drive(...)`` or ``pulse.ops.drive(...)``.

Unsupported Patterns
--------------------

The following Python constructs are **not** supported inside kernels
and will raise ``CompilationError``:

- ``while`` loops
- Lambdas and function definitions inside the kernel body; define the
  function at module level, where it is inlined
- Generator functions (``yield``)
- ``try`` / ``except`` and ``with`` blocks
- ``match`` with sequence, mapping or class patterns (literal and ``|``
  patterns work)
- Assignments to globals and to attributes (``obj.x = ...``), class
  definitions, ``async`` code and star imports
- ``if`` conditions on anything that is only known at runtime other than a
  measurement result

The error names the construct and, where there is one, what to write
instead.

Kernels may read globals and variables captured from an enclosing scope; a
change to one is seen on the next call. Calls to ``cudaq_pulse`` ops are
emitted as IR. Other kernels, and helper functions that take IR values, call
``cudaq_pulse`` ops or call a kernel, are inlined, inside loops too. So are
methods of helper objects and ``functools.partial`` objects of such
functions, and helpers wrapped by a decorator: the wrapper is inlined, and it
calls the original. Helpers bind their arguments as Python does, including
default, keyword-only, ``*args`` and ``**kwargs`` parameters. Any other
Python call, such as ``math.sin`` or ``len``, is evaluated at trace time on
concrete values; if one of them calls a ``cudaq_pulse`` op, say through
``map``, the error names the op and the call.
