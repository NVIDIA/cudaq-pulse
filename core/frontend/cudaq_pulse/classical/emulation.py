# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Emulating a controller that calls classical services, in virtual time.

``export(lowered)`` writes what a controller needs to run a compiled kernel:
its static schedule with each event's epoch, the epoch boundaries, its calls,
and the **manifest** of the classical functions it calls (``manifest``): per
function its dispatch key, link, handler schema and latency budget, which is
what a CUDA-Q Realtime dispatcher's function table holds.

``emulate(lowered, ...)`` runs the program shot by shot in virtual time,
without physics: measurement outcomes come from a source, each call is sent
as a real request (``wire``) over a channel to a service, and the answer
comes back after the modelled transport delay plus the service's measured
time. An answer later than its budget takes the timeout arm of a wait with a
timeout, and is recorded as late otherwise. Answers are bound to the shot as
in simulation, so variable waits get their real lengths and later epochs
shift accordingly.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from .._classical_report import report
from .._errors import (
    InvalidValueError,
    NotFoundError,
    UnsupportedError,
    _CallTimeoutError,
    _positive_int,
)
from ..compiler import _dispatch_keys
from ..runtime._calls import IQModel, _answer_values, _call_inputs, takes_iq
from . import ClassicalFunction, Link, _handlers_for, wire

if TYPE_CHECKING:
    # Exports import this module, and must not import numpy or socket.
    import numpy as np

    from .. import _clocks

__all__ = [
    "CallTrace",
    "RealtimeChannel",
    "device_call_text",
    "EmulationResult",
    "InProcess",
    "Transport",
    "UdpChannel",
    "emulate",
    "export",
    "manifest",
    "serve_udp",
]

FORMAT = "cudaq-pulse.emulation/1"
MANIFEST_FORMAT = "cudaq-pulse.call-manifest/1"


# ── Export ───────────────────────────────────────────────────────────────────


def manifest(lowered: Any) -> dict[str, Any]:
    """The classical functions a compiled kernel calls, for a dispatcher."""
    machine = lowered.options.machine
    functions = []
    for fn in lowered.classical:
        link = machine.link(fn.link)
        link_id = machine.link_id(fn.link)
        key, fixed_by = _dispatch_keys(lowered)[fn.name]
        entry = {"name": fn.name, "dispatch_key": key}
        if fixed_by is not None:
            entry["dispatch_key_fixed_by"] = fixed_by
        functions.append(
            {
                **entry,
                "random": fn.uniform,
                "link": link_id,
                "link_name": machine.link_name(link_id),
                "schema": wire.schema(fn),
                "argument_types": [a.descriptor for a in fn.args],
                "result_types": [r.descriptor for r in fn.results],
                "latency_ns": {
                    "min": fn.min_latency_ns,
                    "nominal": fn.nominal_latency_ns,
                    "max": fn.max_latency_ns,
                },
                "latency_mode": fn.latency_mode,
                "timeout_ns": fn.timeout_ns,
                "deadline_ns": fn.deadline_ns,
                "submit": fn.submit_policy,
                "request_bytes": wire.HEADER_BYTES + sum(a.wire_bytes for a in fn.args),
                "answer_bytes": wire.HEADER_BYTES + sum(r.wire_bytes for r in fn.results),
                "slot_bytes": link.slot_bytes,
                "max_in_flight": link.max_in_flight,
            }
        )
    return {"format": MANIFEST_FORMAT, "functions": functions}


def export(lowered: Any, controller: int | None = None) -> dict[str, Any]:
    """A compiled kernel for a controller emulator: schedule, epochs, calls,
    links and manifest, as JSON-serializable data. Times are in clock cycles
    of the controller, at ``clock_ghz`` (with ``program_clock_ghz`` when that
    is not the program clock, as in ``cudaq_pulse.controller``); each event's
    ``epoch_start_cycles`` is the static start of its epoch, which moves at
    run time by the lengths of the run-time waits before it on its lines. An
    epoch's shift adds to that of the epoch it follows, the one before it
    unless it names another in ``after``. On a control system of several
    controllers, the part of the one *controller* indexes, with only its
    events, calls, links and functions, and ``controller`` its index."""
    machine = lowered.options.machine
    if machine._system is not None and machine.controller is None:
        return export_part(lowered, controller)
    if controller is not None:
        from ..controller import _controller_index

        _controller_index(controller, 1)
    return _export(lowered, True)


_Part = tuple[int, "_clocks.ControllerClock | None", frozenset[str], Sequence[bool]]


def export_part(lowered: Any, controller: int | None) -> dict[str, Any]:
    """``export`` on a control system of several controllers: the part of
    one (as ``cudaq_pulse.controller.export`` chooses it), in its cycles."""
    from ..controller import _part

    _, events, _, parts, index = _part(lowered, controller)
    functions = frozenset(
        event["function"]
        for event, part in zip(events, parts, strict=True)
        if index in part and event["kind"] in ("submit", "answer")
    )
    # The schedule holds the controller events but their syncs, in order.
    takes_part = [
        index in part for event, part in zip(events, parts, strict=True) if event["kind"] != "sync"
    ]
    clocks = (lowered._generic or lowered).clocks
    own = None if clocks is None else clocks.controllers[index]
    return _export(lowered, True, (index, own, functions, takes_part))


def _export(lowered: Any, with_events: bool, part: _Part | None = None) -> dict[str, Any]:
    """``export``; without *with_events*, its ``events`` are left empty. With
    *part* (a controller's index and clock, the functions it calls or takes
    answers of, and whether it takes part in each scheduled event), only
    that controller's part, in its cycles."""
    epochs, after = lowered._epochs()
    starts = [e.start_cycles for e in epochs]
    epoch_of = lowered._event_epochs(syncs=False) if with_events else None
    takes = None
    if part is not None:
        takes = part[3]
        assert not with_events or len(takes) == len(lowered.schedule)
    events = (
        [
            {
                "kind": event.kind,
                "line": event.line,
                "link": event.link,
                "qubit": event.qubit,
                "start_cycles": event.start_cycles,
                "duration_cycles": event.duration_cycles,
                "epoch": (
                    index := epoch_of[position]
                    if epoch_of and event.start_cycles is not None
                    else 0
                ),
                "epoch_start_cycles": starts[index],
            }
            for position, event in enumerate(lowered.schedule)
            if takes is None or takes[position]
        ]
        if with_events
        else []
    )
    machine = lowered.options.machine
    data = {
        "format": FORMAT,
        "clock_ghz": lowered.target.clock_ghz,
        "links": {
            str(machine.link_id(fn.link)): {
                **machine.link(fn.link).cost(),
                "name": machine.link_name(machine.link_id(fn.link)),
            }
            for fn in lowered.classical
            if part is None or fn.name in part[2]
        },
        "epochs": [
            {**asdict(e), "after": a} if k and a != k - 1 else asdict(e)
            for k, (e, a) in enumerate(zip(epochs, after))
        ],
        "events": events,
        "calls": [
            _call_data(c)
            for c in (report(lowered) if lowered.classical else ())
            if part is None or (c.compiled_name in part[2] and c.controller in (None, part[0]))
        ],
        "manifest": manifest(lowered),
    }
    if machine._plain:
        return data
    if part is not None:
        data["manifest"]["functions"] = [
            fn for fn in data["manifest"]["functions"] if fn["name"] in part[2]
        ]
        data["controller"] = part[0]
    clocks = (lowered._generic or lowered).clocks
    if clocks is None:
        return data
    own = clocks.controllers[0] if part is None or part[1] is None else part[1]
    return _in_own_clock(data, events, clocks, own)


def _call_data(call: Any) -> dict[str, Any]:
    """A ``ClassicalCall`` as export data: its fields, in order (the
    ``asdict`` of one, made directly); ``controller`` only for a call over a
    per-controller link. An emulator serves the compiled name:
    ``"function"`` names it, as the manifest's ``"name"`` does, and the
    manifest gives its dispatch key."""
    data = {
        "function": call.compiled_name,
        "link": call.link,
        "iteration": list(call.iteration),
        "submit_cycles": call.submit_cycles,
        "submit_duration_cycles": call.submit_duration_cycles,
        "due_cycles": call.due_cycles,
        "taken_cycles": call.taken_cycles,
        "first_use_cycles": call.first_use_cycles,
        "deadline_cycles": call.deadline_cycles,
        "epoch": call.epoch,
        "link_name": call.link_name,
        "exposed_cycles": call.exposed_cycles,
    }
    if call.controller is not None:
        data["controller"] = call.controller
    return data


def _in_own_clock(data: dict[str, Any], events: list[Any], clocks: Any, own: Any) -> dict[str, Any]:
    """*data*, an export of a program on *clocks*, in the clock *own* of its
    controller."""
    from .. import _clocks

    if own.ratio != 1:
        _clocks.to_own(events, _clocks.EMULATION_EVENT_TIMES, own)
        _clocks.to_own(data["epochs"], _clocks.EPOCH_TIMES, own, _clocks.EPOCH_LOWER_BOUNDS)
        _clocks.to_own(data["calls"], _clocks.CALL_TIMES, own)
        data["clock_ghz"] = own.clock_ghz
    return {
        "format": FORMAT,
        "clock_ghz": data["clock_ghz"],
        **_clocks.clock_keys(clocks, own),
        **data,
    }


# ── Transport and channels ───────────────────────────────────────────────────


@dataclass(frozen=True)
class Transport:
    """The delay of one direction of a link: ``fixed_ns + bytes /
    bytes_per_ns`` plus a jitter drawn from an exponential with mean
    ``jitter_ns`` (0 for none)."""

    fixed_ns: float = 0.0
    bytes_per_ns: float = 0.0
    jitter_ns: float = 0.0

    def delay_ns(self, n_bytes: int, rng: np.random.Generator) -> float:
        delay = self.fixed_ns
        if self.bytes_per_ns > 0:
            delay += n_bytes / self.bytes_per_ns
        if self.jitter_ns > 0:
            delay += float(rng.exponential(self.jitter_ns))
        return delay


class InProcess:
    """A channel to services in this process: handlers by function name,
    called on the decoded request. A handler named for a function whose
    list lengths come from its calls serves each of its versions.
    *dispatch_keys* gives the program's key per compiled name (``emulate``
    passes them); a function not in it is at the hash of its name."""

    def __init__(
        self,
        functions: Sequence[ClassicalFunction[..., Any]],
        handlers: Mapping[str, Callable[..., Any]],
        dispatch_keys: Mapping[str, int] | None = None,
    ):
        self._handlers = dict(handlers)
        self._keys = dict(dispatch_keys or {})

    def call(self, function: ClassicalFunction[..., Any], request: bytes) -> tuple[bytes, float]:
        """Deliver *request*; returns the answer bytes and the service's time in ns."""
        handler = self._handlers.get(function.name) or self._handlers.get(function.base_name)
        if handler is None:
            raise NotFoundError(f"no service for {function.name!r}")
        key = self._keys.get(function.name)
        request_id, timestamp, args = wire.decode_request(function, request, key)
        t0 = time.perf_counter_ns()
        answer = handler(*args)
        service_ns = float(time.perf_counter_ns() - t0)
        return wire.encode_answer(function, answer, request_id, timestamp), service_ns

    def close(self) -> None:
        pass


def serve_udp(
    functions: Sequence[ClassicalFunction[..., Any]],
    handlers: Mapping[str, Callable[..., Any]],
    host: str = "127.0.0.1",
    port: int = 0,
    slot_bytes: int = 65536,
    dispatch_keys: Mapping[str, int] | None = None,
) -> tuple[threading.Thread, tuple[str, int], Callable[[], None]]:
    """Serve *handlers* over UDP: one datagram per request and per answer, in
    the wire format. Returns the server thread, its address and a stop
    function. The service's own time is appended to each answer as 8 bytes.
    A request goes to the function at its dispatch key (*dispatch_keys* by
    compiled name, else the hash of the name). Built-in functions that share
    a key, such as QEC decoders, are told apart by their payload: its length,
    then its first argument (the decoder id)."""
    import socket

    by_key = _by_dispatch_key(functions, dispatch_keys or {})
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    sock.settimeout(0.05)
    running = threading.Event()
    running.set()

    def loop() -> None:
        while running.is_set():
            try:
                data, peer = sock.recvfrom(slot_bytes)
            except TimeoutError:
                continue
            except OSError:
                return
            key, length = struct.unpack_from("<II", data, 4)
            fn = _served(by_key, key, length, data)
            request_id, timestamp, args = wire.decode_request(fn, data, key)
            t0 = time.perf_counter_ns()
            handler = handlers.get(fn.name) or handlers[fn.base_name]
            answer = handler(*args)
            service_ns = time.perf_counter_ns() - t0
            reply = wire.encode_answer(fn, answer, request_id, timestamp)
            sock.sendto(reply + struct.pack("<Q", service_ns), peer)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()

    def stop() -> None:
        running.clear()
        thread.join(timeout=1.0)
        sock.close()

    return thread, sock.getsockname(), stop


def _by_dispatch_key(
    functions: Sequence[ClassicalFunction[..., Any]], keys: Mapping[str, int]
) -> dict[int, dict[int, list[ClassicalFunction[..., Any]]]]:
    """Per dispatch key, per request payload length, the functions there."""
    by_key: dict[int, dict[int, list[ClassicalFunction[..., Any]]]] = {}
    for fn in functions:
        key = keys.get(fn.name, fn._dispatch_key)
        length = sum(a.wire_bytes for a in fn.args)
        there = by_key.setdefault(key, {}).setdefault(length, [])
        if there and (fn._key_route is None or there[0]._key_route is None):
            raise InvalidValueError(
                f"{there[0].name} and {fn.name} are both at dispatch key {key:#x}; a server"
                " tells them apart only by key"
            )
        there.append(fn)
    return by_key


def _served(
    by_key: dict[int, dict[int, list[ClassicalFunction[..., Any]]]],
    key: int,
    length: int,
    data: bytes,
) -> ClassicalFunction[..., Any]:
    """The function a request at *key* with a payload of *length* bytes is for."""
    there = by_key.get(key, {}).get(length)
    if not there:
        raise NotFoundError(f"no function served at dispatch key {key:#x} takes {length} bytes")
    if len(there) == 1:
        return there[0]
    route = wire.decode_request(there[0], data, key)[2][0]
    for fn in there:
        if fn._key_route == route:
            return fn
    raise NotFoundError(f"no function served at dispatch key {key:#x} for decoder {route}")


class RealtimeChannel:
    """A channel through the CUDA-Q Realtime host library, which dispatches
    requests to registered GPU or host services. Not available yet: use
    ``InProcess`` (with ``wire.native`` handlers for compiled code) or
    ``UdpChannel``. The requests and answers are already in its wire
    format, so a channel built on it drops in here."""

    def __init__(self, library: str, *args: Any, **kwargs: Any) -> None:
        raise UnsupportedError(
            "a channel through the CUDA-Q Realtime host library is not available yet; "
            "emulate with InProcess (wire.native handlers run compiled code) or UdpChannel"
        )


def device_call_text(lowered: Any) -> str:
    """CUDA-Q kernel text that makes a compiled program's calls as
    ``cc.device_call`` operations. Not available yet: export the program
    with ``export`` (or ``Lowered.export_emulation()``), whose manifest holds
    each function's dispatch key, payload types and link."""
    raise UnsupportedError(
        "exporting calls as CUDA-Q cc.device_call operations is not available yet; "
        "use Lowered.export_emulation(), whose manifest lists every call"
    )


class UdpChannel:
    """A channel to a service reached over UDP (``serve_udp``)."""

    def __init__(self, address: tuple[str, int], slot_bytes: int = 65536, timeout_s: float = 2.0):
        self._address = address
        import socket

        self._slot_bytes = slot_bytes
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(timeout_s)

    def call(self, function: ClassicalFunction[..., Any], request: bytes) -> tuple[bytes, float]:
        self._sock.sendto(request, self._address)
        try:
            data, _ = self._sock.recvfrom(self._slot_bytes)
        except TimeoutError as error:
            host, port = self._address
            raise _CallTimeoutError(
                f"classical function {function.name!r} got no answer from {host}:{port} within "
                f"{self._sock.gettimeout()} s; check that serve_udp is running there, or raise "
                "UdpChannel(timeout_s=...)"
            ) from error
        service_ns = struct.unpack_from("<Q", data, len(data) - 8)[0]
        return data[:-8], float(service_ns)

    def close(self) -> None:
        self._sock.close()


# ── The virtual-time driver ──────────────────────────────────────────────────


@dataclass(frozen=True)
class CallTrace:
    """One call in an emulated shot. Times in ns of the shot's timeline."""

    function: str
    submit_ns: float
    latency_ns: float
    budget_ns: float
    in_time: bool
    inputs: tuple[Any, ...]
    answer: Any


@dataclass
class EmulationResult:
    """Per shot, the measurement outcomes (in time order) and the calls."""

    outcomes: list[list[int]] = field(default_factory=list)
    calls: list[list[CallTrace]] = field(default_factory=list)
    ends_ns: list[float] = field(default_factory=list)
    wall_s: float = 0.0

    @property
    def late_fraction(self) -> float:
        traces = [c for shot in self.calls for c in shot]
        return sum(not c.in_time for c in traces) / len(traces) if traces else 0.0

    @property
    def calls_per_second(self) -> float:
        n = sum(len(shot) for shot in self.calls)
        return n / self.wall_s if self.wall_s > 0 else math.inf


def _budget_ns(function: ClassicalFunction[..., Any], timeout_ns: float, link: Link) -> float:
    limit = timeout_ns if timeout_ns >= 0 else function.timeout_ns
    base = function.max_latency_ns if limit is None else limit
    return base + link.feedback_ns


def emulate(
    lowered: Any,
    *,
    shots: int = 1,
    channel: Any = None,
    handlers: Mapping[str, Callable[..., Any]] | None = None,
    transport: Transport | Mapping[int | str, Transport] | None = None,
    outcomes: Callable[[int, int, np.random.Generator], int] | None = None,
    service_time: str = "measured",
    seed: int | None = None,
) -> EmulationResult:
    """Run *lowered* for *shots* in virtual time against classical services.

    *channel* reaches the services (``InProcess`` of *handlers* by default,
    the functions' bodies unless given). *transport* models each link's
    delay in each direction, per link (name or id) or for all. *outcomes* gives the
    level a readout reports, from its qudit and the shot's generator
    (uniform 0/1 by default); a call that takes IQ points gets them drawn
    at those levels from the target's dispersive readout model
    (``IQModel``). The answer's latency is the request's
    transport delay, plus the service's time (``"measured"`` wall-clock, or
    ``"none"``), plus the answer's transport delay. *shots* is a positive int.
    """
    if shots.__class__ is not int or shots < 1:
        shots = _positive_int("shots", shots)
    functions = tuple(lowered.classical)
    by_name = {fn.name: fn for fn in functions}
    chosen = _handlers_for(functions, handlers)
    keys = {name: key for name, (key, _) in _dispatch_keys(lowered).items()}
    channel = channel or InProcess(functions, chosen, keys)
    import numpy as np

    rng = np.random.default_rng(seed)
    machine = lowered.options.machine
    schedule_args = lowered.options.schedule_args()
    base = lowered._concrete().simulation_module(*schedule_args)
    clock = lowered.target.clock_ghz
    iq_model = (
        IQModel(lowered.target.target, lowered.qudits, decay=False)
        if any(takes_iq(fn) for fn in functions)
        else None
    )

    def transport_of(link: int) -> Transport:
        if transport is None:
            return Transport()
        if isinstance(transport, Transport):
            return transport
        found = transport.get(link)
        if found is None:
            name = machine.link_name(link)
            found = transport.get(name) if name is not None else None
        return found or Transport()

    def outcome_of(qudit: int) -> int:
        if outcomes is None:
            return int(rng.integers(0, 2))
        return int(outcomes(qudit, 0, rng))

    result = EmulationResult()
    t_wall = time.perf_counter()
    for _ in range(shots):
        module = base
        path: list[tuple[tuple[int, tuple[int, ...]], int]] = []
        shot_outcomes: list[int] = []
        traces: list[CallTrace] = []
        seen: dict[tuple[int, tuple[int, ...]], tuple[int, float]] = {}
        request_id = 0
        while True:
            entries = [[index, *iteration, k] for (index, iteration), k in path]
            found = module.next_measurement(entries)
            call = module.next_call(entries)
            measure_at = math.inf if found is None else found[4]
            call_at = math.inf if call is None else call[1]
            if call is not None and call_at < measure_at - 1e-9:
                index, start, callee, arguments, answered, timeout = call
                fn = by_name[callee]
                known = {key: k for key, k in path}
                point = None
                if iq_model is not None and takes_iq(fn):
                    model = iq_model
                    drives = {
                        (i, tuple(it)): (f, a) for i, it, f, a in module.readout_drives(entries)
                    }

                    def point(key: tuple[int, tuple[int, ...]]) -> tuple[float, float]:
                        qudit, duration = seen[key]
                        frequency, amplitude = drives.get(key, (math.nan, math.nan))
                        return model.sample(rng, qudit, known[key], frequency, amplitude, duration)

                inputs = _call_inputs(fn, arguments, known, point)
                if fn.uniform is not None:
                    # An on-device source: the controller answers, not a link.
                    value = int(rng.integers(fn.uniform))
                    budget = _budget_ns(fn, timeout, machine.link(fn.link))
                    traces.append(
                        CallTrace(callee, start, fn.nominal_latency_ns, budget, True, (), value)
                    )
                    module = module.bind_call(index, [float(value)], *schedule_args)
                    continue
                request = wire.encode_request(fn, inputs, request_id, dispatch_key=keys[callee])
                request_id += 1
                answer_bytes, service_ns = channel.call(fn, request)
                status, answered_id, answer = wire.decode_answer(fn, answer_bytes)
                if status != 0:
                    raise wire.call_failed(callee, answered_id, status)
                link = machine.link(fn.link)
                path_transport = transport_of(machine.link_id(fn.link))
                latency = path_transport.delay_ns(len(request), rng) + path_transport.delay_ns(
                    len(answer_bytes), rng
                )
                if service_time == "measured":
                    latency += service_ns
                budget = _budget_ns(fn, timeout, link)
                in_time = latency <= budget
                traces.append(CallTrace(callee, start, latency, budget, in_time, inputs, answer))
                if not answered:
                    module = module.bind_call(index, [], *schedule_args)
                    continue
                values = _answer_values(fn, answer)
                has_timeout = timeout >= 0 or fn.timeout_ns is not None
                if has_timeout:
                    values = [1.0, *values] if in_time else [0.0] * (1 + len(values))
                module = module.bind_call(index, values, *schedule_args)
                continue
            if found is None:
                break
            ridx, iteration, qudit, begin, finish = found
            level = outcome_of(qudit)
            seen[(ridx, tuple(iteration))] = (qudit, finish - begin)
            path.append(((ridx, tuple(iteration)), level))
            shot_outcomes.append(level)
        ends = [s + d for _, _, _, s, d in module.events() if s is not None and d is not None]
        result.outcomes.append(shot_outcomes)
        result.calls.append(traces)
        result.ends_ns.append(max(ends, default=0) / clock)
    result.wall_s = time.perf_counter() - t_wall
    return result
