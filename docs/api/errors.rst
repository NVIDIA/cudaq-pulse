..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0


Errors
======

Every error cudaq-pulse raises on purpose derives from ``PulseError``, so
``except cudaq_pulse.PulseError`` catches them all. Each also has the
built-in base of its kind of error, so ``except ValueError`` or ``except
RuntimeError`` catches it as it would the built-in one: ``InvalidValueError``
is a ``ValueError``, ``NotFoundError`` and ``UnknownQuantumSystemError`` are
``KeyError``\ s, ``IndexOutOfRangeError`` is an ``IndexError``, and ``CompilationError`` and ``NoProcessQuantumSystemError`` are
``RuntimeError``\ s.

.. autoexception:: cudaq_pulse.PulseError
   :members: problems, location

An error that finds several problems at once, such as
``QuantumSystemDefinitionError``, lists each as a ``Problem`` in ``problems``:

.. autoclass:: cudaq_pulse.Problem

Arguments and names
-------------------

.. autoexception:: cudaq_pulse.InvalidValueError

.. autoexception:: cudaq_pulse.InvalidTypeError

.. autoexception:: cudaq_pulse.NotFoundError

.. autoexception:: cudaq_pulse.IndexOutOfRangeError

.. autoexception:: cudaq_pulse.UnsupportedError

Compiling
---------

.. autoexception:: cudaq_pulse.CompilationError

.. autoexception:: cudaq_pulse.TimingError

.. autoexception:: cudaq_pulse.compiler.UnsupportedInterpreterError

Running
-------

.. autoexception:: cudaq_pulse.ExecutionError

.. autoexception:: cudaq_pulse.InternalError

Quantum systems and targets
---------------------------

.. autoexception:: cudaq_pulse.QuantumSystemDefinitionError

.. autoexception:: cudaq_pulse.NoQuantumSystemError

.. autoexception:: cudaq_pulse.NoProcessQuantumSystemError

.. autoexception:: cudaq_pulse.QuantumSystemMismatchError

.. autoexception:: cudaq_pulse.quantum_systems.UnknownQuantumSystemError

GPU simulation
--------------

``pulse.simulate(..., backend="gpu")`` raises one of the three subclasses of
``GpuRuntimeError`` (``cudaq_pulse.GpuRuntimeError``), an ``ExecutionError``,
when GPU simulation cannot run; see :ref:`gpu-errors`.

.. autoexception:: cudaq_pulse.runtime.GpuRuntimeError

.. autoexception:: cudaq_pulse.runtime.GpuSupportNotBuiltError

.. autoexception:: cudaq_pulse.runtime.GpuRuntimeLoadError

.. autoexception:: cudaq_pulse.runtime.GpuDeviceUnavailableError

Controller backends
-------------------

Compiling for a controller whose limits a program exceeds raises
``ControllerLimitError``; finding a backend raises a ``BackendError``. See
:doc:`controller`.

.. autoexception:: cudaq_pulse.controller.ControllerLimitError

.. autoexception:: cudaq_pulse.backends.BackendError

.. autoexception:: cudaq_pulse.backends.BackendNotFoundError

.. autoexception:: cudaq_pulse.backends.IncompatibleBackendError

Warnings
--------

.. autoexception:: cudaq_pulse.PulseWarning

.. autoexception:: cudaq_pulse.UpgradeNotice
