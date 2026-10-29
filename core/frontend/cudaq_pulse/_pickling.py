# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""How cudaq-pulse objects pickle, for process pools and caches on disk.

- A kernel pickles by reference, as a function does: by its module and
  qualified name. In the process that defined it, it unpickles as the same
  kernel, with its specializations; elsewhere, as the kernel of that name.
  A compiled kernel pickles as its kernel, options and device: its
  specializations are compiled again where it is called.
- A target pickles as the inputs it was built from, and a device as its
  target and the inputs it was built from; both are built again, equal to
  the originals. A process keeps the last ``_KEPT_DEVICES`` devices it
  unpickled by fingerprint, so that programs of one device unpickled one
  by one share it without building it again. Qudit handles and registers
  pickle as their device and indices.
- A program (``Lowered``, and a call's view of one) pickles as MLIR
  bytecode and its metadata, stamped with the build that compiled it, and
  runs without compiling again. Only that build unpickles it: another
  raises ``UnsupportedError`` saying to compile it again.
- An error pickles with its attributes and unpickles without calling its
  ``__init__`` again, so errors raised in a process pool reach the parent
  as they were raised.

Nothing here is imported until something is pickled or unpickled.
"""

from __future__ import annotations

import dataclasses
import pickle
import threading
from types import MappingProxyType
from typing import Any

# Changed whenever what a program pickles changes.
_FORMAT = 1
_stamp: tuple[int, str, str] | None = None


def build_stamp() -> tuple[int, str, str]:
    """What a pickled program records of the build that compiled it."""
    global _stamp
    if _stamp is None:
        from ._build_config import PACKAGE_VERSION
        from ._native import _cudaq_pulse_native as native

        _stamp = (_FORMAT, PACKAGE_VERSION, native.build_stamp())
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
_devices: dict[str, Any] = {}
_devices_lock = threading.Lock()


def _keep(device: Any) -> None:
    with _devices_lock:
        _devices.pop(device.fingerprint, None)
        _devices[device.fingerprint] = device
        while len(_devices) > _KEPT_DEVICES:
            del _devices[next(iter(_devices))]


def device_reduce(device: Any, inputs: dict[str, Any]) -> tuple[Any, ...]:
    """*device* as its fingerprint and, pickled apart, its class, target
    and the other *inputs* of its constructor: a process that has the device
    of that fingerprint does not unpickle the rest."""
    _keep(device)
    description = pickle.dumps((type(device), device.target, inputs), pickle.HIGHEST_PROTOCOL)
    return (rebuilt_device, (device.fingerprint, description))


def rebuilt_device(fingerprint: str, description: bytes) -> Any:
    found = _devices.get(fingerprint)
    if found is not None:
        return found
    cls, target, inputs = pickle.loads(description)
    device = cls(target, **inputs)
    _keep(device)
    return device


# ── Kernels and classical functions ─────────────────────────────────────────


def named(module: str, qualname: str) -> Any:
    import importlib

    found: Any = importlib.import_module(module)
    for part in qualname.split("."):
        found = getattr(found, part)
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
    from .kernel._kernel import CompiledKernel, _Dispatcher

    compiled = CompiledKernel.__new__(CompiledKernel)
    compiled.kernel = kernel
    compiled.device = device
    compiled._dispatch = _Dispatcher(kernel, options, None)
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
    module = None if lowered.generic is not None else lowered._module.serialize()
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
    generic: Any = fields.get("generic")
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
    if len(stamp) == 3:
        return f"{stamp[1]} (build {stamp[2]}, format {stamp[0]})"
    return repr(stamp)


# ── Errors ──────────────────────────────────────────────────────────────────

# Where tracing failed, as decoded bytecode: the error's notes already say it.
_TRACING_ONLY = frozenset({"pulse_source", "pulse_callers"})


def error_reduce(error: BaseException) -> tuple[Any, ...]:
    state = {name: v for name, v in vars(error).items() if name not in _TRACING_ONLY}
    return (rebuilt_error, (type(error), error.args), state or None)


def rebuilt_error(cls: type[BaseException], args: tuple[Any, ...]) -> BaseException:
    return cls.__new__(cls, *args)
