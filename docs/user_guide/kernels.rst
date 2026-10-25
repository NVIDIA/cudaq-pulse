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

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

.. :spellcheck-enable:

Inside ``@pulse.kernel`` functions, DSL operations (``drive``,
``gaussian``, ``get_drive_line``, etc.) are used as bare names.
Infrastructure stays behind the ``pulse.`` prefix:

.. :spellcheck-disable:

.. code-block:: python

   @pulse.kernel
   def rabi_oscillation(qubit):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(64, 0.5, 16.0), tone)

   dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
   rabi_oscillation(dev.qudit_handles[0])

.. :spellcheck-enable:

Defining a Kernel
-----------------

.. :spellcheck-disable:

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def my_kernel(qubit):
       drive_line, tone = get_drive_line(qubit)
       waveform = gaussian(40, 0.3, 10.0)
       drive(drive_line, waveform, tone)

.. :spellcheck-enable:

Qudit arguments are **qudit handles** of a device: ``dev.qudit_handles[i]``
for one qudit, and ``dev.qudit_handles[0:4]`` or ``dev.qudit_handles[[0, 2]]``
for a register, which a kernel indexes and iterates like a list. The kernel
runs on the device of its handles; see :doc:`devices`. How arguments bind,
which of them the kernel is specialized on, and what a call returns are
described in :doc:`compilation`.

.. :spellcheck-disable:

.. code-block:: python

   @pulse.kernel
   def two(q0, q1, amplitude: float):
       ...

   reg = dev.qudit_handles[0:2]
   two(reg[0], reg[1], 0.3)

.. :spellcheck-enable:

``pulse.qudit_ref()`` and ``pulse.qvec_ref(n)`` are qudit references not
tied to a device: positional (``qudit_ref()``) or at a physical index
(``qudit_ref(4)``). A call with them runs on the process target, and a
kernel may allocate them in its body.

.. :spellcheck-enable:

Control Flow
------------

Kernels support a subset of Python control flow that can be captured
at trace time:

**For loops** with compile-time integer bounds:

.. :spellcheck-disable:

.. code-block:: python

   @pulse.kernel
   def echo_sequence(qubit):
       drive_line, tone = get_drive_line(qubit)
       for i in range(5):
           drive(drive_line, gaussian(40, 0.3, 10.0), tone)
           wait(drive_line, 20)

.. :spellcheck-enable:

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

**The unroll threshold** is ``pulse.compile(k, unroll_threshold=n)``,
``@pulse.kernel(unroll_threshold=n)`` or the ``CUDAQ_PULSE_UNROLL_THRESHOLD``
environment variable, in that order, and 8 by default; 0 keeps every such
loop rolled. Both forms are the same program: without passes their
schedules are identical, and with the default passes fusion may merge
back-to-back pulses of the unrolled form, with the same evolution. The
threshold is part of the cache key, so changing the variable recompiles.

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

.. :spellcheck-disable:

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
               drive(line, square(40, amplitude), tone)

.. :spellcheck-enable:

**If/else** with compile-time conditions:

.. :spellcheck-disable:

.. code-block:: python

   @pulse.kernel
   def conditional_pulse(qubit, use_drag):
       drive_line, tone = get_drive_line(qubit)
       if use_drag:
           waveform = drag(40, 0.3, 10.0, 0.5)
       else:
           waveform = gaussian(40, 0.3, 10.0)
       drive(drive_line, waveform, tone)

.. :spellcheck-enable:

**Measurement feedback**: an ``if`` on a measurement result (``if result:`` or
``if not result:``) is kept as a conditional region:

.. :spellcheck-disable:

.. code-block:: python

   @pulse.kernel
   def active_reset(qubit):
       drive_line, drive_tone = get_drive_line(qubit)
       readout_line, readout_tone = get_readout_line(qubit)
       result = readout(readout_line, square(1000, 0.05), readout_tone)
       sync(drive_line, readout_line)
       if result:
           drive(drive_line, drag(40, 0.5, 10.0, 0.5), drive_tone)

.. :spellcheck-enable:

Both branches are traced, and lines and tones they rebind become the
results of the region. Both branches must continue after the ``if``, and
other Python values must not differ between them. Compiling keeps the
branch as an ``scf.if`` on ``pulse.is_set`` (whether the readout
observed a nonzero level), whose arms yield the lines and tones they
advance; both arms start once the measurement completes, and the code after
the ``if`` waits for the longer arm.

Unsupported Patterns
--------------------

The following Python constructs are **not** supported inside kernels
and will raise ``CompilationError``:

- ``while`` loops
- Lambdas and function definitions inside the kernel body
- List comprehensions or generator expressions
- ``try`` / ``except`` and ``with`` blocks
- ``if`` conditions on anything that is only known at runtime other than a
  measurement result

Kernels may read globals and variables captured from an enclosing scope; a
change to one is seen on the next call. Calls to ``cudaq_pulse`` ops are
emitted as IR. Other kernels, and helper functions that take IR values, call
``cudaq_pulse`` ops or call a kernel, are inlined, inside loops too. Any other
Python call, such as ``math.sin`` or ``len``, is evaluated at trace time on
concrete values.
