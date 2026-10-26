# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Controller backends installed beside cudaq-pulse.

A controller backend turns a compiled kernel into a program for one
controller and runs it there. It ships as its own distribution and
registers an entry point in the group ``cudaq_pulse.backends``, under the
backend's name::

    [project.entry-points."cudaq_pulse.backends"]
    toy = "toy_controller:ToyBackend"

The entry point names a ``Backend`` or a callable without arguments that
returns one. ``available()`` lists the installed backends without importing
them, and ``get(name)`` loads one. A backend follows the ``Backend``
protocol: ``machine_model()`` is the machine model to compile for, so that
programs the controller cannot play fail at compile time;
``emit(program)`` takes ``Lowered.export_controller()`` and returns the
controller's program; ``run(artifact, shots=...)`` runs it::

    from cudaq_pulse import backends

    backend = backends.get("toy")
    lowered = pulse.compile(kernel, device=dev, machine=backend.machine_model()).lower(q)
    artifact = backend.emit(lowered.export_controller())
    result = backend.run(artifact, shots=100)

``register(name, factory)`` adds a backend in this process, without a
distribution.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

    from .compiler import MachineModel

__all__ = [
    "GROUP",
    "Backend",
    "BackendNotFoundError",
    "available",
    "get",
    "register",
    "unregister",
]

GROUP = "cudaq_pulse.backends"


@runtime_checkable
class Backend(Protocol):
    """A controller backend."""

    name: str

    def machine_model(self) -> MachineModel:
        """The machine model to compile for this controller."""
        ...

    def emit(self, program: Mapping[str, Any]) -> Any:
        """The controller's program for ``Lowered.export_controller()``."""
        ...

    def run(self, artifact: Any, *, shots: int = 1) -> Any:
        """Run an emitted program on the controller."""
        ...


class BackendNotFoundError(LookupError):
    """No backend of that name is installed or registered."""


_lock = threading.Lock()
_registered: dict[str, Callable[[], Backend]] = {}
_loaded: dict[tuple[str, str], Backend] = {}


def _entry_points() -> dict[str, EntryPoint]:
    from importlib.metadata import entry_points

    return {ep.name: ep for ep in entry_points(group=GROUP)}


def available() -> list[str]:
    """The names of the backends installed or registered, sorted; none is
    imported."""
    with _lock:
        names = set(_registered)
    return sorted(names | set(_entry_points()))


def register(name: str, factory: Callable[[], Backend]) -> None:
    """Make *factory* the backend *name* in this process; it takes precedence
    over an installed backend of the same name."""
    if not name or not name.isidentifier():
        raise ValueError(f"a backend name is an identifier, got {name!r}")
    with _lock:
        _registered[name] = factory
        _loaded.pop((name, "registered"), None)


def unregister(name: str) -> None:
    """Remove a backend ``register`` added."""
    with _lock:
        _registered.pop(name, None)
        _loaded.pop((name, "registered"), None)


def _instance(name: str, loaded: Any) -> Backend:
    backend = loaded
    if isinstance(loaded, type) or (not isinstance(loaded, Backend) and callable(loaded)):
        backend = loaded()
    if not isinstance(backend, Backend):
        raise TypeError(
            f"backend {name!r} is not a cudaq_pulse.backends.Backend: it needs a name, "
            "machine_model(), emit() and run()"
        )
    return backend


def get(name: str) -> Backend:
    """The backend *name*, loaded on first use and kept."""
    with _lock:
        factory = _registered.get(name)
        key = (name, "registered")
        if factory is not None:
            if key not in _loaded:
                _loaded[key] = _instance(name, factory)
            return _loaded[key]
    found = _entry_points().get(name)
    if found is None:
        installed = ", ".join(available()) or "none"
        raise BackendNotFoundError(
            f"no controller backend {name!r} is installed (installed: {installed}); "
            f"a backend registers an entry point in the {GROUP!r} group"
        )
    key = (name, found.value)
    with _lock:
        backend = _loaded.get(key)
    if backend is None:
        backend = _instance(name, found.load())
        with _lock:
            backend = _loaded.setdefault(key, backend)
    return backend
