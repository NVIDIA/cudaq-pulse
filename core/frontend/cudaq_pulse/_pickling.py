# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""How cudaq-pulse objects pickle, for process pools and caches on disk.

- A kernel pickles as a function does. One its module and qualified name
  find, outside ``__main__``, pickles as that name. Any other pickles as a
  copy of its function named ``<kernel>.__pulse_pickled_fn__``, its options
  and a token: pickle sends that function by reference, through the kernel,
  and cloudpickle by value, as each sends a function of ``__main__`` or one
  defined inside a function. A process keeps the last ``_KEPT_KERNELS``
  kernels pickled or unpickled this way by token, so a kernel sent again
  (a task's argument, say) is the same kernel, with its specializations.
  A compiled kernel pickles as its kernel, options and device: its
  specializations are compiled again where it is called.
- A target pickles as the inputs it was built from, and a device as its
  target and the inputs it was built from; both are built again, equal to
  the originals. A process keeps the last ``_KEPT_DEVICES`` devices it
  unpickled by fingerprint (which covers the calibration), name, source
  and control system, so that programs of one device unpickled one by one
  share it without building it again. A device makes its description
  once. A control system, its controllers and their processors pickle as
  their inputs and are checked again. A calibration the package ships
  pickles as its name, cooldown and the digest of its file, and unpickles
  as the one shipped here if that has the same digest; any other pickles
  as its values. Qudit handles and registers pickle as their device and
  indices.
- A program (``Lowered``, and a call's view of one) pickles as MLIR
  bytecode and its metadata, stamped with the build that compiled it, and
  runs without compiling again. Only that build unpickles it: another
  raises ``UnsupportedError`` saying to compile it again. The stamp hashes
  the native and the Python sources when the package is built, so any
  change to either refuses older programs.
- An error pickles with its attributes and unpickles without calling its
  ``__init__`` again, so errors raised in a process pool reach the parent
  as they were raised.

Nothing here is imported until something is pickled or unpickled.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import threading
import types
from types import MappingProxyType
from typing import Any

_stamp: tuple[str, str, str, str] | None = None


def build_stamp() -> tuple[str, str, str, str]:
    """What a pickled program records of the build that compiled it: its
    version, the hashes of its native and of its Python sources, both taken
    when it was built, and the Python, whose marshal format its wait-length
    check is pickled in."""
    global _stamp
    if _stamp is None:
        import sys

        from ._build_config import PACKAGE_VERSION
        from ._native import _cudaq_pulse_native as native
        from ._source_stamp import SOURCE_STAMP

        python = str(sys.implementation.cache_tag)
        _stamp = (PACKAGE_VERSION, native.build_stamp(), SOURCE_STAMP, python)
    return _stamp


# ── Frozen descriptions ─────────────────────────────────────────────────────


def _plain(value: Any) -> Any:
    return dict(value) if type(value) is MappingProxyType else value


def frozen_reduce(description: Any) -> tuple[Any, ...]:
    """A frozen dataclass as its class and constructor inputs."""
    inputs = {
        f.name: _plain(getattr(description, f.name))
        for f in dataclasses.fields(description)
        if f.init
    }
    return (rebuilt, (type(description), inputs))


def rebuilt(cls: Any, inputs: dict[str, Any]) -> Any:
    return cls(**inputs)


def rebuilt_links(cls: Any, links: dict[int, Any], names: dict[str, int]) -> Any:
    found = cls()
    dict.update(found, links)
    found.names = MappingProxyType(names)
    return found


# ── Devices ─────────────────────────────────────────────────────────────────

_KEPT_DEVICES = 16
_devices: dict[tuple[Any, ...], Any] = {}
_devices_lock = threading.Lock()


def _device_key(device: Any) -> tuple[Any, ...]:
    # The fingerprint, and the labels and control system it leaves out, so
    # that an unpickled device is the one pickled in every attribute.
    system = device.control_system
    if system is None:
        return (device.fingerprint, device.name, device.source)
    return (device.fingerprint, device.name, device.source, system)


def _keep(key: tuple[Any, ...], device: Any) -> None:
    with _devices_lock:
        _devices.pop(key, None)
        _devices[key] = device
        while len(_devices) > _KEPT_DEVICES:
            del _devices[next(iter(_devices))]


def device_reduce(
    device: Any, inputs: dict[str, Any], state: dict[str, Any] | None = None
) -> tuple[Any, ...]:
    """*device* as its fingerprint, labels and control system and, pickled
    apart, its class, target, the other *inputs* of its constructor and the
    attributes they do not set (*state*): a process that has the device of
    that key does not unpickle the rest. The description is made once per
    device: a device is read-only."""
    key = _device_key(device)
    _keep(key, device)
    description: bytes | None = device.__dict__.get("_pickled")
    if description is None:
        parts: tuple[Any, ...] = (type(device), device.target, inputs)
        if state is not None:
            parts += (state,)
        description = pickle.dumps(parts, pickle.HIGHEST_PROTOCOL)
        device._pickled = description
    return (rebuilt_device, (key, description))


def rebuilt_shipped_calibration(name: str, cooldown: str, digest: str, version: str) -> Any:
    """The calibration *name* (*cooldown*) that this package ships, if it is
    the file of that *digest* that cudaq-pulse *version* shipped."""
    from .targets.calibration import _shipped_unread

    found = _shipped_unread(name, cooldown, digest)
    if found is None:
        from ._build_config import PACKAGE_VERSION
        from ._errors import UnsupportedError
        from ._source_stamp import CALIBRATIONS

        here = CALIBRATIONS.get(name)
        ships = "does not ship it" if here is None else f"ships another (file {here})"
        raise UnsupportedError(
            f"calibration {name!r} was pickled as the one cudaq-pulse {version} ships "
            f"(file {digest}), and cudaq-pulse {PACKAGE_VERSION} {ships}: load the "
            "calibration here, or pickle a Calibration made from its values"
        )
    return found


def rebuilt_device(key: tuple[Any, ...], description: bytes) -> Any:
    found = _devices.get(key)
    if found is not None:
        return found
    cls, target, inputs, *state = pickle.loads(description)
    if len(key) > 3:
        inputs["control_system"] = key[3]
    device = cls(target, **inputs)
    for attributes in state:
        vars(device).update(attributes)
    _keep(key, device)
    return device


# ── Kernels and classical functions ─────────────────────────────────────────


def named(module: str, qualname: str) -> Any:
    import importlib

    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
    return found


_KEPT_KERNELS = 64
_kernels: dict[str, Any] = {}
_kernels_lock = threading.Lock()


def kernel_reduce(kernel: Any) -> str | tuple[Any, ...]:
    """*kernel* by its name where that finds it outside ``__main__``, else
    as its function (see ``Kernel.__pulse_pickled_fn__``), options and a
    token that the processes unpickling it keep it by."""
    module, qualname = kernel.__module__, kernel.__qualname__
    if module != "__main__" and "<locals>" not in qualname:
        try:
            found = named(module, qualname)
        except (ImportError, AttributeError):
            found = None
        if found is kernel:
            return str(qualname)
    token = kernel.__dict__.get("_pickle_token")
    if token is None:
        token = kernel._pickle_token = os.urandom(16).hex()
    _keep_kernel(token, kernel)
    options = {
        "cache_size": kernel._dispatch._capacity,
        "schedule": kernel.schedule,
        "unroll_threshold": kernel.unroll_threshold,
    }
    return (rebuilt_kernel, (kernel.__pulse_pickled_fn__, qualname, token, options))


def function_copy(kernel: Any) -> types.FunctionType:
    """*kernel*'s function, named ``<kernel>.__pulse_pickled_fn__``: made once."""
    copied: types.FunctionType | None = kernel.__dict__.get("_pickled_fn")
    if copied is None:
        fn = kernel.__wrapped__
        copied = types.FunctionType(
            fn.__code__, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__
        )
        copied.__kwdefaults__ = fn.__kwdefaults__
        copied.__annotations__ = fn.__annotations__
        copied.__dict__.update(fn.__dict__)
        copied.__doc__ = fn.__doc__
        copied.__module__ = fn.__module__
        copied.__qualname__ = f"{kernel.__qualname__}.__pulse_pickled_fn__"
        kernel._pickled_fn = copied
    return copied


def _keep_kernel(token: str, kernel: Any) -> None:
    with _kernels_lock:
        _kernels.pop(token, None)
        _kernels[token] = kernel
        while len(_kernels) > _KEPT_KERNELS:
            del _kernels[next(iter(_kernels))]


def rebuilt_kernel(function: Any, qualname: str, token: str, options: dict[str, Any]) -> Any:
    """The kernel kept under *token*; else the kernel *qualname* names, when
    *function* came by reference from it; else a kernel of *function*, which
    came by value."""
    found: Any = _kernels.get(token)
    if found is not None:
        return found
    from .kernel._kernel import Kernel

    try:
        found = named(function.__module__, qualname)
    except (ImportError, AttributeError):
        found = None
    if not (isinstance(found, Kernel) and found.__pulse_pickled_fn__ is function):
        function.__qualname__ = qualname
        found = Kernel(function, **options)
    found._pickle_token = token
    _keep_kernel(token, found)
    return found


def classical_reduce(function: Any, protocol: int) -> Any:
    """A classical function by the name of its implementation when that
    name finds it (it was declared with ``@pulse.classical``), else by value."""
    implementation = function.implementation
    if implementation is not None:
        module = getattr(implementation, "__module__", None)
        qualname = getattr(implementation, "__qualname__", "")
        if module and "<locals>" not in qualname:
            try:
                found = named(module, qualname)
            except (ImportError, AttributeError):
                found = None
            if found is function:
                return (named, (module, qualname))
    return object.__reduce_ex__(function, protocol)


def rebuilt_compiled(kernel: Any, options: Any, device: Any, general: Any) -> Any:
    """A compiled kernel without its specializations, which compile on first
    use; its pinned module, if built, comes with it."""
    from .kernel._kernel import CompiledKernel

    compiled = CompiledKernel.__new__(CompiledKernel)
    compiled.kernel = kernel
    compiled.device = device
    compiled._dispatch = kernel._dispatcher_type(kernel, options, None)
    compiled._dispatch.pin = device
    compiled._dispatch.general = general
    return compiled


# ── Programs ────────────────────────────────────────────────────────────────

# What a program holds of the native module, which pickles as bytecode or is
# made again.
_NATIVE = frozenset({"_module", "_text", "_lock", "_concrete_module"})
_slots: dict[type, tuple[str, ...]] = {}


def _fields(cls: type) -> tuple[str, ...]:
    names = _slots.get(cls)
    if names is None:
        names = tuple(
            name
            for klass in reversed(cls.__mro__)
            for name in getattr(klass, "__slots__", ())
            if name not in _NATIVE
        )
        _slots[cls] = names
    return names


def program_reduce(lowered: Any) -> tuple[Any, ...]:
    """*lowered* as its module's bytecode (none for a call's view, which
    shares its generic module's) and its other fields, with the build stamp."""
    fields: dict[str, Any] = {}
    for name in _fields(type(lowered)):
        try:
            fields[name] = getattr(lowered, name)
        except AttributeError:
            pass
    proxies = tuple(name for name, value in fields.items() if type(value) is MappingProxyType)
    for name in proxies:
        fields[name] = dict(fields[name])
    module = None if lowered._generic is not None else lowered._module.serialize()
    return (rebuilt_program, (build_stamp(), type(lowered), module, fields, proxies))


def rebuilt_program(
    stamp: tuple[Any, ...],
    cls: Any,
    module: tuple[bytes, bytes, bool, list[str]] | None,
    fields: dict[str, Any],
    proxies: tuple[str, ...],
) -> Any:
    here = build_stamp()
    if tuple(stamp) != here:
        from ._errors import UnsupportedError

        raise UnsupportedError(
            f"this program was compiled by cudaq-pulse {_build_name(stamp)}, and this is "
            f"cudaq-pulse {_build_name(here)}; a pickled program runs only on the build "
            "that compiled it: compile the kernel again here"
        )
    from ._native import _cudaq_pulse_native as native

    lowered: Any = cls.__new__(cls)
    for name, value in fields.items():
        setattr(lowered, name, MappingProxyType(value) if name in proxies else value)
    generic: Any = fields.get("_generic")
    if module is None:
        lowered._module = generic._module
        lowered._lock = generic._lock
    else:
        lowered._module = native.PulseModuleBuilder().load(*module)
        lowered._lock = threading.Lock()
    lowered._text = None
    lowered._concrete_module = None
    return lowered


def _build_name(stamp: tuple[Any, ...]) -> str:
    if len(stamp) == 4:
        return f"{stamp[0]} (build {stamp[1]}, sources {stamp[2]}, {stamp[3]})"
    return repr(stamp)


# ── Errors ──────────────────────────────────────────────────────────────────

# Where tracing failed, as decoded bytecode: the error's notes already say it.
_TRACING_ONLY = frozenset({"pulse_source", "pulse_callers"})


def error_reduce(error: BaseException) -> tuple[Any, ...]:
    state = {name: v for name, v in vars(error).items() if name not in _TRACING_ONLY}
    return (rebuilt_error, (type(error), error.args), state or None)


def rebuilt_error(cls: type[BaseException], args: tuple[Any, ...]) -> BaseException:
    return cls.__new__(cls, *args)
