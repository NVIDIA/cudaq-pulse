..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.

Calling and Compiling Kernels
=============================

A ``@pulse.kernel`` is called like a Python function. The first call with
new arguments compiles the kernel; later calls reuse the compilation and
only run it.

.. code-block:: python

   import cudaq_pulse as pulse

   @pulse.kernel
   def rabi(qubit, amplitude: float):
       drive_line, tone = get_drive_line(qubit)
       drive(drive_line, gaussian(40, amplitude, 10.0), tone)

   dev = pulse.get_device(pulse.Target.TRANSMON_KRINNER_17Q)
   q = dev.qudit_handles[0]

   rabi(q, 0.3)                    # compiles, then runs; returns None
   rabi(q, 0.4)                    # same compilation, new amplitude
   lowered = rabi.lower(q, 0.3)    # compiles (or reuses), does not run
   print(lowered.mlir())

Arguments
---------

Arguments bind exactly as for a Python function: positional and keyword
arguments, defaults, keyword-only parameters, and ``TypeError`` for a call
that does not match the signature. Kernels cannot take ``*args`` or
``**kwargs``. A kernel, ``kernel.lower`` and ``pulse.compile(kernel)`` keep
the decorated function's signature for type checkers, so editors check each
call against it.

Annotations decide what a compilation depends on:

- A parameter annotated ``float`` is a *runtime* value. It stays a function
  argument of the compiled MLIR module, wherever it flows (an amplitude, a
  phase, a tone frequency, the length of a ``wait``), so calling with a new
  value never recompiles. A call whose values set a wait length schedules
  the module again with them, without tracing it; see
  :ref:`wait-lengths`.
- Every other parameter (qudit handles, ``int``, anything unannotated) is
  *static*: the kernel is specialized on its value.
- A ``float`` the program needs while it compiles -- one that sets a
  waveform duration, goes through ``int()``, decides an ``if`` or bounds a
  loop -- is specialized on as well. ``kernel.static_parameters`` and
  ``kernel.runtime_parameters`` show the split.

Return values
-------------

A call returns what the kernel returns: ``None`` without a ``return``, a
measurement's outcome (0, 1, ...) for a measurement made at the top level of
the kernel, values known at compile time, or tuples and lists of these.
Returning a value computed from a ``float`` parameter is an error.

Devices
-------

A kernel runs on the device of the qudit handles it is called with; see
:doc:`devices`. One kernel serves every device it is called with, each with
its own specializations. A call without handles raises ``NoDeviceError``,
unless the kernel is compiled for one device (``device=`` below); the
process target is never used for it. Qudit frequencies and the pulse clock
come from the device.

Caching and Python semantics
----------------------------

Each kernel keeps up to 32 specializations, least recently used first out;
set ``CUDAQ_PULSE_CACHE_SIZE`` or ``@pulse.kernel(cache_size=N)`` to change
that. ``kernel.cache_info()`` reports the specializations held, compiled
and evicted.

Globals and closure variables behave as in Python: the values a kernel read
while it compiled (module globals, closure cells, attribute chains such as
``cfg.amplitude``, the contents of lists and dicts it read) are checked on
every call, and a change compiles the kernel again. A kernel that calls a
host function whose result can change between calls (``random.random()``)
compiles on every call.

``pulse.compile``
-----------------

``pulse.compile(kernel)`` compiles ahead of the call and takes no kernel
arguments. It returns a ``CompiledKernel``: calling it runs the kernel and
returns what the kernel returns, and it has the same ``lower`` as the
kernel. Its specializations are kept for as long as the object lives and are
never evicted.

.. code-block:: python

   sweep = pulse.compile(rabi)
   for amplitude in (0.1, 0.2, 0.3):
       sweep(q, amplitude)          # one compilation for the whole sweep

   pinned = pulse.compile(rabi, device=dev, schedule="asap")

Options, all keyword-only:

``schedule`` *(str, optional)*
   Scheduling policy of the compiled kernel itself: ``"asap"``, ``"alap"``,
   ``"rcp"`` or ``"alap_rcp"``; by default the kernel's own ``schedule=``,
   else ``"alap"``. ASAP and ALAP preserve data and physical-line
   dependencies, including the order of pulses that play on the same tone;
   the RCP variants also enforce ``MachineModel`` resource limits. Kernels
   it calls keep their own ``schedule=`` (see `Per-kernel schedules`_).

``flatten_schedules`` *(bool, default False)*
   Ignore the ``schedule=`` of every called kernel and schedule the whole
   program with ``schedule``, for comparison.

``passes`` *(sequence of Pass or str, optional)*
   Optimization passes, run in the order given: members of
   :class:`cudaq_pulse.Pass` or their names. ``DEFAULT_PASSES`` by default
   (``VERIFY``, ``FOLD_CONSTANTS``, ``REMOVE_UNUSED``, ``FOLD_PHASES``,
   ``FUSE_PULSES``); ``()`` runs none. The members are ``VERIFY`` (check the
   IR), ``FOLD_CONSTANTS`` (compute constant arithmetic at compile time),
   ``REMOVE_UNUSED`` (remove operations whose results nothing uses),
   ``MERGE_WAITS`` (join constant waits in a row on one line),
   ``SIMPLIFY_CONTROL_FLOW`` (remove syncs that order nothing, inline blocks
   that take no time, remove loops that run once or never), ``FOLD_PHASES``
   (fold frame changes into drives), ``FUSE_PULSES`` (merge square pulses in
   a row) and ``HOIST_LOOP_INVARIANTS`` (move loop-invariant operations out
   of loops). An unknown name is a ``ValueError`` that names the closest. See
   :doc:`../architecture/mlir_passes`.

``machine`` *(MachineModel, optional)*
   Resource limits for the RCP schedules.

``device`` *(Device, optional)*
   Pin the compiled kernel to one device. A call with handles of another
   device raises ``DeviceMismatchError``; a call without handles runs on it.
   A pinned kernel whose parameters are all ``float`` is built when it is
   compiled, so ``.mlir()``, ``.schedule`` and ``.metrics`` work on the
   compiled kernel itself.

Calling the kernel directly always uses the defaults: ``k(...)`` takes only
the kernel's own arguments.

Per-kernel schedules
--------------------

A kernel declared with ``@pulse.kernel(schedule="asap")`` or
``schedule="alap"`` and called from another kernel is scheduled on its own
with that policy, then placed whole by its caller's policy, so its internal
timing is the same wherever it is called -- also inside a rolled loop. A
called kernel without ``schedule=`` is scheduled together with its caller.
``rcp`` and ``alap_rcp`` apply to a whole program: they are accepted only by
``pulse.compile``, and not yet for programs that call kernels with their own
``schedule=`` (compile with ``flatten_schedules=True`` instead). A kernel's
policy is part of what it is, so a compiled kernel's specializations depend
on its own policy, ``flatten_schedules`` and the policies of the kernels it
calls. The shape holds when the program is scheduled again, with float
values bound or, in simulation, with the answers of classical calls bound.

Inspecting a compilation
------------------------

``kernel.lower(*args)`` and ``compiled.lower(*args)`` return a ``Lowered``,
the module of one call:

``.mlir(stage="pulse")``
   MLIR text: ``"pulse"`` for the scheduled pulse dialect (with the call's
   float values substituted), ``"llvm"`` after the full Pulse -> QOp ->
   CuDensityMat -> LLVM lowering.

``.schedule``
   The drives, readouts and waits with their start and duration in clock
   cycles (``ScheduledEvent``). ``event.start`` and ``event.end`` are the
   scheduled times as ``int``; they raise ``ValueError`` on an event that is
   not scheduled yet, whose ``start_cycles`` and ``duration_cycles`` are ``None``.

``.timeline(**kwargs)``
   A matplotlib figure of the schedule, one row per line.

``.metrics``
   A ``CompileMetrics`` with per-stage timing in milliseconds (``trace_ms``,
   ``ffi_ms``, ``passes_ms``, ``schedule_ms``, ``total_ms``), ``op_count``
   and the device name.

``.arguments`` and ``.generic``
   The call's float values, and the module without them, in which the float
   parameters are function arguments.

Threads
-------

Calls and cache hits take no lock, and the first call for new static
arguments compiles once even when many threads make it at the same time.
The native stages (module construction, passes, scheduling, lowering and JIT
compilation) release the GIL. Simulations of one specialization from several
threads are safe: each sets the module attributes it lowers with and reads
the lowering in one step.

Long Sessions and Memory
------------------------

Compiled kernels share an MLIR context, which keeps every attribute it has
created (constants, qubit frequencies, timing) until it is freed. A context
is retired, and freed once its last kernel is gone, after it has built a
number of modules or a number of packed program words, whichever comes
first. The limits are read from the environment each time a kernel is
compiled:

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - Variable
     - Default
     - Meaning
   * - ``CUDAQ_PULSE_CONTEXT_MAX_MODULES``
     - 256
     - Modules built in one context before it is retired.
   * - ``CUDAQ_PULSE_CONTEXT_MAX_WORDS``
     - 1048576
     - Packed program words (8 bytes each) built in one context before it
       is retired.

Each must be a whole number from 1 to 2\ :sup:`40`; any other value makes
compiling raise ``ValueError`` naming the variable. Lower values bound the
memory a long session keeps (a loop that compiles a large program for many
devices, for example) at the cost of creating contexts more often, about
0.4 ms each. ``python -m cudaq_pulse.doctor`` prints the limits in effect,
the process target and the kernel cache size.
