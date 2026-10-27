..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.


Errors
======

Every error cudaq-pulse raises on purpose derives from ``PulseError``, so
``except cudaq_pulse.PulseError`` catches them all. Each also keeps its
built-in base: ``NoTargetError`` is still a ``RuntimeError`` and
``UnknownTargetError`` a ``LookupError``. A malformed argument raises a
built-in ``ValueError`` or ``TypeError``.

.. autoexception:: cudaq_pulse.PulseError
   :members: problems, location

An error that finds several problems at once, such as
``DeviceDefinitionError``, lists each as a ``Problem`` in ``problems``:

.. autoclass:: cudaq_pulse.Problem

Compiling
---------

.. autoexception:: cudaq_pulse.CompilationError

.. autoexception:: cudaq_pulse.compiler.UnsupportedInterpreterError

Devices and targets
-------------------

.. autoexception:: cudaq_pulse.DeviceDefinitionError

.. autoexception:: cudaq_pulse.NoDeviceError

.. autoexception:: cudaq_pulse.NoTargetError

.. autoexception:: cudaq_pulse.DeviceMismatchError

.. autoexception:: cudaq_pulse.targets.UnknownTargetError

GPU simulation
--------------

``pulse.simulate(..., backend="gpu")`` raises one of the three subclasses of
``GpuRuntimeError`` (``cudaq_pulse.GpuRuntimeError``) when GPU simulation
cannot run; see :ref:`gpu-errors`.

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
