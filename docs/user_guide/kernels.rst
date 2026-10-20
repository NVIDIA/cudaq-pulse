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

   compiled_kernel = pulse.compile(rabi_oscillation, [pulse.qudit_ref()],
                                   qubit_freq_hz={0: 5.0e9})

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

Kernel arguments are **qudit references** -- opaque handles representing
quantum degrees of freedom. They are created outside the kernel using
``pulse.qudit_ref()`` or ``pulse.qvec_ref(n)`` and passed in when compiling.

Qudit Allocation
----------------

**Single qudit:**

.. :spellcheck-disable:

.. code-block:: python

   qubit = pulse.qudit_ref()

.. :spellcheck-enable:

Bind an argument to a specific physical target index when needed:

.. :spellcheck-disable:

.. code-block:: python

   target_qubit_4 = pulse.qudit_ref(4)

.. :spellcheck-enable:

**Vector of qudits:**

.. :spellcheck-disable:

.. code-block:: python

   qubits = pulse.qvec_ref(4)
   qubit_0 = qubits[0]
   qubit_1 = qubits[1]

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
does not depend on the trip count. Loops that read the induction variable,
or that carry Python state from one iteration to the next, are unrolled
exactly at trace time. ``cudaq_pulse.compile()`` writes the packed program
directly and always unrolls. Symbolic or runtime-dependent loop bounds, and
``while`` loops, are rejected.

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
other Python values must not differ between them. ``cudaq_pulse.compile()``
cannot lower conditional regions yet and raises an error at the ``if``.

Unsupported Patterns
--------------------

The following Python constructs are **not** supported inside kernels
and will raise ``CompilationError``:

- ``while`` loops
- Lambdas and function definitions inside the kernel body
- List comprehensions or generator expressions
- ``try`` / ``except`` and ``with`` blocks
- Dynamic loop bounds (bounds must be known at trace time)
- ``if`` conditions on anything that is only known at runtime other than a
  measurement result, including comparisons of symbolic parameters

Kernels may read variables captured from an enclosing scope; captured values
are read again on every call. Calls to ``cudaq_pulse`` ops are emitted as IR.
Helper functions that take IR values or call ``cudaq_pulse`` ops are inlined.
Any other Python call, such as ``math.sin`` or ``len``, is evaluated at trace
time on concrete values.
