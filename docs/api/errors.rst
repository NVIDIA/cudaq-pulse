..
   Copyright (c) 2026 NVIDIA Corporation & Affiliates.
   All rights reserved.

   This source code and the accompanying materials are made available under
   the terms of the Apache License 2.0 which accompanies this distribution.


Errors
======

Every error cudaq-pulse raises on purpose is one of these, or a built-in
``ValueError`` or ``TypeError`` for a malformed argument.

Compiling
---------

.. autoexception:: cudaq_pulse.CompilationError

.. autoexception:: cudaq_pulse.compiler.UnsupportedInterpreterError

Devices and targets
-------------------

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
