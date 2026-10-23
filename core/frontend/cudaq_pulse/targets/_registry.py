# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The process target, chosen outside user code.

A kernel runs on the target named by the ``CUDAQ_PULSE_TARGET`` environment
variable, which ``python -m cudaq_pulse --target NAME script.py`` sets for the
script it runs. A name is a registered target, optionally with an argument
after a colon (``transmon_generic:4``). There is no default: a kernel call
without a target raises ``NoTargetError``.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .base import Target

if TYPE_CHECKING:
    from .._device import Device

ENV_VAR = "CUDAQ_PULSE_TARGET"

TargetFactory = Callable[[str | None], Any]


class NoTargetError(RuntimeError):
    """A kernel needs a target and none is configured."""


class UnknownTargetError(LookupError):
    """``CUDAQ_PULSE_TARGET`` names a target that is not registered."""


def _no_argument(name: str, factory: Callable[[], Any]) -> TargetFactory:
    def make(arg: str | None) -> Any:
        if arg is not None:
            raise ValueError(f"target {name!r} takes no argument, got {arg!r}")
        return factory()

    return make


def _count(name: str, factory: Callable[[int], Any]) -> TargetFactory:
    def make(arg: str | None) -> Any:
        if arg is None or not arg.isdigit():
            raise ValueError(f"target {name!r} needs a qudit count, e.g. {name}:4")
        return factory(int(arg))

    return make


def _grid(name: str, factory: Callable[[int, int], Any]) -> TargetFactory:
    def make(arg: str | None) -> Any:
        rows, _, cols = (arg or "").partition("x")
        if not (rows.isdigit() and cols.isdigit()):
            raise ValueError(f"target {name!r} needs a grid size, e.g. {name}:3x3")
        return factory(int(rows), int(cols))

    return make


def _builtin_factories() -> dict[str, TargetFactory]:
    from .rydberg import rydberg_chain, rydberg_square
    from .transmon import transmon_generic, transmon_krinner_17q

    return {
        "transmon_krinner_17q": _no_argument("transmon_krinner_17q", transmon_krinner_17q),
        "transmon_generic": _count("transmon_generic", transmon_generic),
        "rydberg_chain": _count("rydberg_chain", rydberg_chain),
        "rydberg_square": _grid("rydberg_square", rydberg_square),
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
    """Make *factory* available as ``CUDAQ_PULSE_TARGET=name[:arg]``.

    *factory* receives the text after the colon, or ``None``, and returns a
    ``Target`` (or an object with ``to_target()``). Registering only adds a
    name: which target a process uses is still chosen by the environment.
    """
    if not name or ":" in name or name != name.strip():
        raise ValueError(f"invalid target name {name!r}")
    registry = _registry()
    with _lock:
        registry[name] = factory
        for spec in [s for s in _resolved if s.partition(":")[0] == name]:
            del _resolved[spec]


def available_targets() -> list[str]:
    """Names that ``CUDAQ_PULSE_TARGET`` accepts."""
    return sorted(_registry())


def make_target(spec: str) -> Target:
    """A fresh target from a registered name with its optional argument."""
    name, colon, arg = spec.partition(":")
    factory = _registry().get(name)
    if factory is None:
        raise UnknownTargetError(
            f"no registered target {spec!r}. Available: {', '.join(available_targets())}"
        )
    target = factory(arg if colon else None)
    if not isinstance(target, Target) and hasattr(target, "to_target"):
        target = target.to_target()
    if not isinstance(target, Target):
        raise TypeError(f"target factory {name!r} returned {type(target).__name__}, not a Target")
    return target


def current_target(kernel: str | None = None) -> Device:
    """The device of the process target; ``NoTargetError`` if none is configured.

    Resolved once per distinct ``CUDAQ_PULSE_TARGET`` value.
    """
    spec = os.environ.get(ENV_VAR)
    if not spec:
        raise NoTargetError(_no_target_message(kernel))
    resolved = _resolved.get(spec)
    if resolved is None:
        from .._device import Device

        fresh = Device(make_target(spec), name=spec, source=f"env {ENV_VAR}={spec}")
        with _lock:
            resolved = _resolved.setdefault(spec, fresh)
    return resolved


def _no_target_message(kernel: str | None) -> str:
    who = f"{kernel}() needs" if kernel else "running a kernel needs"
    return (
        f"{who} a device: pass qudit handles of one (dev.qudit_handles[i]), or "
        "configure a process target. Set one of:\n"
        f"    {ENV_VAR}=transmon_krinner_17q python script.py\n"
        "    python -m cudaq_pulse --target transmon_krinner_17q script.py\n"
        f"Available targets: {', '.join(available_targets())}"
    )
