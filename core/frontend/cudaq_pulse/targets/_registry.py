# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The process target, chosen outside user code.

``device()`` with no argument returns the device of the target named by
the ``CUDAQ_PULSE_TARGET`` environment variable, which
``python -m cudaq_pulse --target NAME script.py`` sets for the script it
runs. A name is a registered target, optionally with an argument after a
colon (``transmon_generic:4``). There is no default: ``device()`` without
one raises ``NoTargetError``. Kernel calls never read the process target;
they run on the device of their qudit handles.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._errors import InvalidTypeError, InvalidValueError, NotFoundError, PulseError, _closest
from .base import TargetDescription

if TYPE_CHECKING:
    from .._device import Device

ENV_VAR = "CUDAQ_PULSE_TARGET"

TargetFactory = Callable[[str | None], Any]


class NoTargetError(PulseError, RuntimeError):
    """``device()`` was called with no argument and ``CUDAQ_PULSE_TARGET``
    names no process target."""


class UnknownTargetError(NotFoundError):
    """A target name, given to ``device()``, in a system file or in
    ``CUDAQ_PULSE_TARGET``, is not registered; the message lists the
    registered ones."""


def _no_argument(name: str, factory: Callable[[], Any]) -> TargetFactory:
    def make(arg: str | None) -> Any:
        if arg is not None:
            raise InvalidValueError(f"target {name!r} takes no argument, got {arg!r}")
        return factory()

    return make


def _count(name: str, factory: Callable[[int], Any]) -> TargetFactory:
    def make(arg: str | None) -> Any:
        if arg is None or not arg.isdigit():
            raise InvalidValueError(f"target {name!r} needs a qudit count, e.g. {name}:4")
        return factory(int(arg))

    return make


def _builtin_factories() -> dict[str, TargetFactory]:
    from .transmon import transmon_17q, transmon_generic

    return {
        "transmon_17q": _no_argument("transmon_17q", transmon_17q),
        "transmon_generic": _count("transmon_generic", transmon_generic),
    }


_lock = threading.Lock()
_factories: dict[str, TargetFactory] = {}
_resolved: dict[str, Device] = {}


def _registry() -> dict[str, TargetFactory]:
    if not _factories:
        with _lock:
            if not _factories:
                _factories.update(_builtin_factories())
    return _factories


def register_target(name: str, factory: TargetFactory) -> None:
    """Make *factory* available as ``device("name[:arg]")`` and
    ``CUDAQ_PULSE_TARGET=name[:arg]``.

    *factory* receives the text after the colon, or ``None``, and returns a
    ``TargetDescription`` (or an object with ``to_target()``). Registering only adds a
    name: which target a process uses is still chosen by the environment.
    """
    if not name or ":" in name or name != name.strip():
        raise InvalidValueError(f"invalid target name {name!r}")
    registry = _registry()
    with _lock:
        registry[name] = factory
        for spec in [s for s in _resolved if s.partition(":")[0] == name]:
            del _resolved[spec]


def available_targets() -> list[str]:
    """Names that ``device()`` and ``CUDAQ_PULSE_TARGET`` accept."""
    registry = _registry()
    with _lock:
        return sorted(registry)


def make_target(spec: str) -> TargetDescription:
    """A fresh target from a registered name with its optional argument."""
    name, colon, arg = spec.partition(":")
    factory = _registry().get(name)
    if factory is None:
        names = available_targets()
        close = _closest(name, names)
        hint = f"did you mean {close!r}? " if close else ""
        raise UnknownTargetError(
            f"no registered target {spec!r}; {hint}the registered targets are {', '.join(names)}"
        )
    target = factory(arg if colon else None)
    if not isinstance(target, TargetDescription) and hasattr(target, "to_target"):
        target = target.to_target()
    if not isinstance(target, TargetDescription):
        raise InvalidTypeError(
            f"target factory {name!r} returned {type(target).__name__}, not a TargetDescription"
        )
    return target


def current_target() -> Device:
    """The device of the process target; ``NoTargetError`` if none is configured.

    Resolved once per distinct ``CUDAQ_PULSE_TARGET`` value.
    """
    spec = os.environ.get(ENV_VAR)
    if not spec:
        raise NoTargetError(_no_target_message())
    resolved = _resolved.get(spec)
    if resolved is None:
        from .._device import _from_spec

        fresh = _from_spec(spec, f"env {ENV_VAR}={spec}")
        with _lock:
            resolved = _resolved.setdefault(spec, fresh)
    return resolved


def _no_target_message() -> str:
    return (
        "device() with no argument returns the process target, and none is "
        'configured: name a target, as device("transmon_17q"), '
        "or set one of:\n"
        f"    {ENV_VAR}=transmon_17q python script.py\n"
        "    python -m cudaq_pulse --target transmon_17q script.py\n"
        f"Available targets: {', '.join(available_targets())}"
    )
