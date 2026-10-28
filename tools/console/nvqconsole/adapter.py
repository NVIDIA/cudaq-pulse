# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Every call the console makes into cudaq-pulse.

The server and the web app only see the plain dicts this module returns. A
rename in the library (``Lowered``, ``export_controller``,
``export_emulation``, the pass names, ``controller.resources``) is handled
here and nowhere else.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import itertools
import json
import math
import re
import sys
import tempfile
import threading
import time
import traceback
import types
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse import compiler as _compiler
from cudaq_pulse import controller as _controller

# One compile or simulation at a time: the native module builder is shared.
LOCK = threading.Lock()

# A bundle ships the examples beside the console; a checkout has them at its root.
_HERE = Path(__file__).resolve().parent
_EXAMPLES_DIR = next(
    (d for d in (_HERE.parent / "examples", _HERE.parents[2] / "examples") if d.is_dir()),
    _HERE.parent / "examples",
)
_SCRATCH = Path(tempfile.mkdtemp(prefix="nvqconsole-"))
_MAX_EVENTS = 200_000
_MAX_PROGRAM_EVENTS = 2000
_CACHE_SIZE = 256


# ── Library surface ─────────────────────────────────────────────────────────


def version() -> str:
    return str(getattr(pulse, "__version__", "unknown"))


def pass_names() -> list[str]:
    """Every pass ``passes=`` accepts: the members of ``pulse.Pass``."""
    return [p.value for p in pulse.Pass]


def default_passes() -> list[str]:
    return [str(getattr(p, "value", p)) for p in _compiler.DEFAULT_PASSES]


SCHEDULES = ["alap", "asap", "rcp", "alap_rcp"]


def _controller_program(lowered: Any) -> dict[str, Any]:
    method = getattr(lowered, "controller_program", None) or lowered.export_controller
    return dict(method())


def _controller_usage(lowered: Any, limits: Any = None) -> Any:
    usage = getattr(_controller, "controller_usage", None) or _controller.resources
    return usage(lowered, limits)


# ── Targets and devices ─────────────────────────────────────────────────────

_DEVICES: dict[str, Any] = {}


def targets() -> list[dict[str, Any]]:
    return [{"id": t.value, "name": t.name.lower()} for t in pulse.Target]


def device(target_id: str) -> Any:
    found = _DEVICES.get(target_id)
    if found is None:
        found = pulse.get_device(pulse.Target(target_id))
        _DEVICES[target_id] = found
    return found


def _num(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def device_info(target_id: str) -> dict[str, Any]:
    dev = device(target_id)
    target = dev.target
    qudits = []
    for logical, carrier in enumerate(dev.layout):
        q = target.qubits[carrier]
        qudits.append(
            {
                "index": logical,
                "carrier": carrier,
                "label": getattr(q, "label", "") or "",
                "frequency_ghz": _num(q.frequency_hz / 1e9),
                "anharmonicity_mhz": _num(getattr(q, "anharmonicity_hz", 0.0) / 1e6),
                "t1_us": _num(getattr(q, "t1_us", None)),
                "t2_star_us": _num(getattr(q, "t2_star_us", None)),
                "t2_echo_us": _num(getattr(q, "t2_echo_us", None)),
                "thermal_population": _num(getattr(q, "thermal_population", None)),
                "readout_ghz": _num((q.readout_params or {}).get("ro_freq", 0.0) / 1e9),
            }
        )
    carrier_to_logical = {c: i for i, c in enumerate(dev.layout)}
    couplings = [
        [carrier_to_logical[a], carrier_to_logical[b]]
        for a, b in target.coupling_map
        if a in carrier_to_logical and b in carrier_to_logical
    ]
    try:
        from cudaq_pulse import backends

        available = list(backends.available())
    except Exception:  # noqa: BLE001 - discovery is best effort
        available = []
    return {
        "id": target_id,
        "name": dev.name,
        "architecture": target.architecture,
        "clock_ghz": dev.clock_ghz,
        "qudits": qudits,
        "couplings": couplings,
        "controller_backends": available,
        "calibrated": dev.calibration is not None,
    }


# ── Kernel sources ──────────────────────────────────────────────────────────

# Example file -> the kernel to show first and the arguments its own
# __main__ lowers it with (``q`` is ``dev.qudit_handles``).
EXAMPLE_PRESETS: dict[str, dict[str, str]] = {
    "01_single_qubit_rabi.py": {"kernel": "rabi", "args": "q[0], 100, 0.5"},
    "02_two_qubit_cross_resonance.py": {"kernel": "echoed_cr", "args": "q[0], q[1]"},
    "04_hahn_echo.py": {"kernel": "hahn_echo", "args": "q[0], 10, 0.25, 0.5, 200"},
    "05_waveform_gallery.py": {"kernel": "waveform_showcase", "args": "q[0]"},
    "06_phase_frequency_control.py": {"kernel": "chirped_drive", "args": "q[0], 10, 50e6"},
    "07_multi_qubit_sync.py": {"kernel": "staggered_readout", "args": "q[0], q[1]"},
    "08_readout_and_branching.py": {"kernel": "active_reset", "args": "q[0]"},
    "09_compilation_pipeline.py": {"kernel": "pipeline_demo", "args": "q[0], q[1]"},
    "10_scheduling_comparison.py": {"kernel": "four_qubit", "args": "q[:4]"},
    "13_dynamical_decoupling.py": {"kernel": "xy4", "args": "q[0], 10, 0.5, 100, 4"},
    "16_visualization.py": {"kernel": "vis_demo", "args": "q[0], q[1], q[2]"},
    "17_ghz_state_prep.py": {"kernel": "ghz_3", "args": "q[0], q[1], q[2]"},
    "18_per_kernel_schedules.py": {"kernel": "stabilizer_cycle", "args": "q[:5]"},
    "20_adaptive_calibration.py": {"kernel": "binary_search", "args": "q[0], 2"},
    "21_calibration_protocol.py": {
        "kernel": "calibration_chain",
        "args": "q[0], 20_000.0 * CLOCK_GHZ",
    },
}


def examples() -> list[dict[str, Any]]:
    out = []
    for name, preset in EXAMPLE_PRESETS.items():
        path = _EXAMPLES_DIR / name
        if path.exists():
            doc = path.read_text().split('"""')
            title = doc[1].strip().splitlines()[0] if len(doc) > 1 else name
            out.append({"file": name, "title": title, **preset})
    return out


def example_source(name: str) -> str:
    path = (_EXAMPLES_DIR / name).resolve()
    if path.parent != _EXAMPLES_DIR.resolve() or not path.exists():
        raise FileNotFoundError(name)
    return path.read_text()


class SourceError(Exception):
    """A failure in the user's source or arguments, with where it happened."""

    def __init__(self, stage: str, error: BaseException, path: Path | None) -> None:
        super().__init__(str(error))
        self.stage = stage
        self.kind = type(error).__name__
        notes = list(getattr(error, "__notes__", ()))
        self.message = "\n".join([str(error) or self.kind, *notes])
        self.line: int | None = getattr(error, "lineno", None) if path else None
        frames = traceback.extract_tb(error.__traceback__)
        if path is not None:
            for frame in frames:
                if Path(frame.filename) == path:
                    self.line = frame.lineno
            # A kernel's trace error names its kernel lines in a note, the
            # kernel's own line first.
            found = re.findall(rf'File "{re.escape(str(path))}", line (\d+)', "\n".join(notes))
            if found:
                self.line = int(found[-1])
            self.message = self.message.replace(str(path), "kernel.py").replace(
                path.name, "kernel.py"
            )
        self.trace = [
            f"{Path(f.filename).name}:{f.lineno} in {f.name}"
            for f in frames
            if "cudaq_pulse" not in f.filename or Path(f.filename) == path
        ][-6:]

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "kind": self.kind,
            "message": self.message,
            "line": self.line,
            "trace": self.trace,
        }


_MODULES: dict[str, Any] = {}
_MODULES_LOCK = threading.Lock()
_ATTEMPTS = itertools.count()


def load_source(source: str) -> tuple[Any, Path]:
    """Import *source* as a module from a file of its own, as Python would
    import an example, so that kernels can read their source."""
    key = str(hash(source))
    with _MODULES_LOCK:
        if key in _MODULES:
            return _MODULES[key]
        # A new file for every attempt: requests for one source can arrive
        # together, and a file must not change once a module was read from it.
        n = next(_ATTEMPTS)
        path = _SCRATCH / f"console_{n}.py"
        path.write_text(source)
        name = f"nvqconsole_user_{n}"
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        try:
            exec(compile(source, str(path), "exec"), module.__dict__)
        except BaseException as error:  # noqa: BLE001 - shown to the user
            sys.modules.pop(name, None)
            raise SourceError("import", error, path) from None
        _MODULES[key] = (module, path)
        return module, path


def _kernels(module: Any) -> dict[str, Any]:
    return {name: value for name, value in vars(module).items() if isinstance(value, pulse.Kernel)}


def kernels(source: str) -> list[dict[str, Any]]:
    module, _ = load_source(source)
    out = []
    for name, k in _kernels(module).items():
        params = []
        for p in inspect.signature(k).parameters.values():
            ann = p.annotation
            params.append(
                {
                    "name": p.name,
                    "annotation": ""
                    if ann is inspect.Parameter.empty
                    else getattr(ann, "__name__", str(ann)),
                }
            )
        out.append({"name": name, "params": params, "schedule": k.schedule})
    return out


# ── Compiling ───────────────────────────────────────────────────────────────


def _machine(module: Any) -> Any:
    found = getattr(module, "MACHINE", None)
    return found if isinstance(found, _compiler.MachineModel) else None


def _lower(request: dict[str, Any], passes: Sequence[str]) -> tuple[Any, Any, Path]:
    module, path = load_source(request["source"])
    found = _kernels(module).get(request["kernel"])
    if found is None:
        raise SourceError("kernel", KeyError(f"no @pulse.kernel named {request['kernel']!r}"), None)
    dev = device(request["target"])
    scope = {**vars(module), "q": dev.qudit_handles, "dev": dev, "math": math, "pulse": pulse}
    try:
        args = eval(f"({request.get('args') or ''},)", scope)  # noqa: S307 - local tool
    except BaseException as error:  # noqa: BLE001
        raise SourceError("arguments", error, None) from None
    machine = _machine(module) if request.get("use_machine", True) else None
    try:
        compiled = pulse.compile(
            found,
            device=dev,
            passes=tuple(passes),
            schedule=request.get("schedule") or None,
            machine=machine,
        )
        lowered = compiled.lower(*args)
    except BaseException as error:  # noqa: BLE001
        raise SourceError("compile", error, path) from None
    return lowered, machine, path


_KINDS = ("drive", "readout", "wait", "submit", "answer")


def _events(lowered: Any) -> tuple[dict[str, Any], bool]:
    """The schedule as columns (the web app draws them from typed arrays);
    unscheduled ops are left out."""
    events = [e for e in lowered.schedule if e.start_cycles is not None][:_MAX_EVENTS]
    kinds = list(_KINDS) + sorted({e.kind for e in events} - set(_KINDS))
    index = {k: i for i, k in enumerate(kinds)}
    return {
        "kinds": kinds,
        "kind": [index[e.kind] for e in events],
        "line": [e.line for e in events],
        "qudit": [e.qubit for e in events],
        "start": [e.start_cycles for e in events],
        "duration": [e.duration_cycles or 0 for e in events],
    }, len(events) >= _MAX_EVENTS


def _metrics(lowered: Any) -> dict[str, Any]:
    m = lowered.metrics
    return {f.name: _num(getattr(m, f.name)) for f in dataclasses.fields(m)}


_COMPILED: OrderedDict[str, dict[str, Any]] = OrderedDict()


def compile_stages(request: dict[str, Any]) -> dict[str, Any]:
    """``_compile_stages``, kept for the last few hundred requests: an
    unchanged source returns at once."""
    key = json.dumps(request, sort_keys=True)
    found = _COMPILED.get(key)
    if found is not None:
        _COMPILED.move_to_end(key)
        return {**found, "cached": True}
    started = time.perf_counter()
    result = _compile_stages(request)
    result["server_ms"] = (time.perf_counter() - started) * 1e3
    _COMPILED[key] = result
    if len(_COMPILED) > _CACHE_SIZE:
        _COMPILED.popitem(last=False)
    return {**result, "cached": False}


def _compile_stages(request: dict[str, Any]) -> dict[str, Any]:
    """The traced module (no passes), the module after the chosen passes,
    its schedule and the controller program; or the first error."""
    passes = list(request.get("passes") or [])
    with LOCK:
        try:
            traced, _, _ = _lower(request, ())
            lowered, machine, _ = _lower(request, passes)
        except SourceError as error:
            return {"ok": False, "error": error.as_dict()}
        events, truncated = _events(lowered)
        result: dict[str, Any] = {
            "ok": True,
            "kernel": request["kernel"],
            "target": request["target"],
            "passes": passes,
            "schedule_policy": lowered.options.schedule,
            "clock_ghz": lowered.target.clock_ghz,
            "qudits": list(lowered.qudits),
            "parameters": list(lowered.parameters),
            "machine": None if machine is None else repr(machine),
            "stages": {
                "traced": {"mlir": traced.mlir(), "metrics": _metrics(traced)},
                "optimized": {"mlir": lowered.mlir(), "metrics": _metrics(lowered)},
            },
            "events": events,
            "events_truncated": truncated,
            "epochs": [dataclasses.asdict(e) for e in lowered.epochs()],
            "calls": [
                {k: _num(v) for k, v in dataclasses.asdict(c).items() if k != "iteration"}
                for c in lowered.classical_report()[:200]
            ],
        }
        try:
            program = _controller_program(lowered)
            total = len(program.get("events", []))
            program["event_count"] = total
            if total > _MAX_PROGRAM_EVENTS:
                program["events"] = program["events"][:_MAX_PROGRAM_EVENTS]
            result["controller"] = program
        except BaseException as error:  # noqa: BLE001
            result["controller_error"] = SourceError("export", error, None).as_dict()
        if request.get("llvm"):
            try:
                result["stages"]["llvm"] = {"mlir": lowered.mlir("llvm")}
            except BaseException as error:  # noqa: BLE001
                result["stages"]["llvm"] = {"error": f"{type(error).__name__}: {error}"}
        return result


# ── Simulating ──────────────────────────────────────────────────────────────


def _marginals(rho: Any, dims: Sequence[int]) -> list[list[float]]:
    """Level populations of each qudit (``dims[0]`` least significant)."""
    p = np.real(np.diag(rho)).reshape(tuple(reversed(dims)))
    n = len(dims)
    out = []
    for i in range(n):
        axis = n - 1 - i
        others = tuple(a for a in range(n) if a != axis)
        out.append([float(x) for x in p.sum(axis=others)])
    return out


_SIMS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_SIMS_LOCK = threading.Lock()


def simulate_populations(request: dict[str, Any]) -> dict[str, Any]:
    """Each qudit's level populations at evenly spaced times; the same
    request returns the cached result."""
    points = max(2, min(int(request.get("points", 41)), 121))
    levels = max(2, min(int(request.get("levels", 2)), 4))
    key = json.dumps({**request, "points": points, "levels": levels}, sort_keys=True)
    with _SIMS_LOCK:
        found = _SIMS.get(key)
    if found is not None:
        return {**found, "cached": True}
    found = _simulate_populations(request, points, levels)
    if found["ok"]:
        with _SIMS_LOCK:
            _SIMS[key] = found
            if len(_SIMS) > 64:
                _SIMS.popitem(last=False)
    return {**found, "cached": False}


def _simulate_populations(request: dict[str, Any], points: int, levels: int) -> dict[str, Any]:
    with LOCK:
        try:
            lowered, _, _ = _lower(request, list(request.get("passes") or default_passes()))
        except SourceError as error:
            return {"ok": False, "error": error.as_dict()}
    if lowered.parameters:
        return {
            "ok": False,
            "error": {
                "stage": "simulate",
                "kind": "ValueError",
                "message": "give values for the float parameters: " + ", ".join(lowered.parameters),
                "line": None,
                "trace": [],
            },
        }
    events, _ = _events(lowered)
    clock = lowered.target.clock_ghz
    end_cycles = max((a + b for a, b in zip(events["start"], events["duration"])), default=0)
    end_ns = max(end_cycles / clock, 1.0)
    times = np.linspace(0.0, end_ns, points)
    series = []
    started = time.perf_counter()
    try:
        final = None
        for t in times:
            # The initial state (thermal populations included) at a step's
            # length past the start. The lock is held per point, so a
            # compile waits for one point, not the whole run.
            at = max(float(t), 1e-3)
            with LOCK:
                final = pulse.simulate(lowered, levels=levels, t_end=at)
            populations = _marginals(final.density_matrix, final.dims)
            series.append({"t_ns": float(t), "populations": populations})
    except BaseException as error:  # noqa: BLE001
        return {"ok": False, "error": SourceError("simulate", error, None).as_dict()}
    assert final is not None
    return {
        "ok": True,
        "qudits": list(final.qudits),
        "dims": list(final.dims),
        "series": series,
        "clock_ghz": clock,
        "branches": [{"outcomes": list(o), "p": float(p)} for o, p in final.branches[:32]],
        "wall_s": time.perf_counter() - started,
    }


# ── Experiments: run once, then replayed by the web app ─────────────────────

_EXAMPLE_MODULES: dict[str, Any] = {}
_MODULES_LOCK = threading.Lock()
_ATTEMPTS = itertools.count()


def _example(name: str) -> Any:
    found = _EXAMPLE_MODULES.get(name)
    if found is None:
        path = _EXAMPLES_DIR / name
        spec = importlib.util.spec_from_file_location(f"nvqconsole_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        found = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(found)
        _EXAMPLE_MODULES[name] = found
    return found


def _krinner() -> tuple[Any, Any]:
    from cudaq_pulse.targets import transmon_krinner_17q

    target = transmon_krinner_17q()
    return target, pulse.device({"target": target})


def _param(spec: dict[str, Any], params: dict[str, Any]) -> float:
    value = float(params.get(spec["name"], spec["default"]))
    return min(max(value, spec["min"]), spec["max"])


def _rabi(p: dict[str, float]) -> dict[str, Any]:
    ex = _example("01_single_qubit_rabi.py")
    _, dev = _krinner()
    q = dev.qudit_handles[0]
    length = int(p["length"])
    amps = np.linspace(0.0, p["max_amp"], int(p["points"]))
    with LOCK:
        sweep = pulse.compile(ex.rabi, device=dev)
        p1 = [
            float(pulse.simulate(sweep.lower(q, length, float(a))).density_matrix[1, 1].real)
            for a in amps
        ]
    # Least squares of p1 = c + b * (1 - cos(pi a / a_pi)) / 2 over a grid of a_pi.
    best = (math.inf, 0.0, 0.0, 0.0)
    y = np.asarray(p1)
    for a_pi in np.linspace(amps[1], 2 * amps[-1], 800):
        basis = np.stack([np.ones_like(amps), (1 - np.cos(np.pi * amps / a_pi)) / 2], axis=1)
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        err = float(np.sum((basis @ coef - y) ** 2))
        if err < best[0]:
            best = (err, float(a_pi), float(coef[0]), float(coef[1]))
    _, a_pi, c, b = best
    fine = np.linspace(0.0, amps[-1], 200)
    return {
        "x": [float(a) for a in amps],
        "p1": p1,
        "fit_x": [float(a) for a in fine],
        "fit": [float(c + b * (1 - math.cos(math.pi * a / a_pi)) / 2) for a in fine],
        "pi_amp": a_pi,
        "contrast": b,
    }


def _t1(p: dict[str, float]) -> dict[str, Any]:
    ex = _example("03_t1_t2_dissipator.py")
    target, dev = _krinner()
    q = dev.qudit_handles[0]
    clock = target.clock_ghz
    waits = np.linspace(0, p["max_wait_us"] * 1e3 * clock, int(p["points"])).astype(int)
    with LOCK:
        sweep = pulse.compile(ex.t1_experiment, device=dev)
        rows = []
        for w in waits:
            found = pulse.simulate(sweep.lower(q, int(w)), dt_ns=1 / clock)
            rows.append((found.measurements[0].start_ns, float(found.density_matrix[1, 1].real)))
    t0 = rows[0][0]
    x = np.asarray([(t - t0) / 1e3 for t, _ in rows])
    y = np.asarray([v for _, v in rows])
    floor = target.qubits[0].thermal_population
    keep = y - floor > 1e-3
    slope, _ = np.polyfit(x[keep], np.log(y[keep] - floor), 1)
    t1 = -1 / slope
    fine = np.linspace(0, x[-1], 200)
    return {
        "x": [float(v) for v in x],
        "p1": [float(v) for v in y],
        "fit_x": [float(v) for v in fine],
        "fit": [float(floor + (y[0] - floor) * math.exp(-v / t1)) for v in fine],
        "t1_us": float(t1),
        "t1_true_us": float(target.qubits[0].t1_us),
    }


def _drift(p: dict[str, float]) -> dict[str, Any]:
    from cudaq_pulse.classical import calibration as cal
    from cudaq_pulse.classical.loop import EstimateLog, ParameterStore
    from cudaq_pulse.runtime.drift import DriftingPlant, OrnsteinUhlenbeck, TelegraphT1

    _example("20_adaptive_calibration.py")
    target, _ = _krinner()
    qubit = target.qubits[0]
    iterations, block = 6000, 150
    telegraph = TelegraphT1(qubit.t1_us / 4, qubit.t1_us, rate_hz=p["switch_hz"])
    plant = DriftingPlant(
        target,
        t1=telegraph,
        frequency=OrnsteinUhlenbeck(p["drift_khz"] * 1e3, correlation_s=5.0),
        seed=int(p["seed"]),
    )
    store = ParameterStore(alpha=1.0, beta=10_000.0, mean_hz=0.0, sigma_hz=1e6)
    log = EstimateLog()
    cols: dict[str, list[float]] = {
        k: []
        for k in (
            "time_s",
            "t1_est_us",
            "t1_true_us",
            "offset_est_khz",
            "offset_sigma_khz",
            "offset_true_khz",
        )
    }
    for i in range(iterations):
        alpha, beta, mean, sigma = store.values("alpha", "beta", "mean_hz", "sigma_hz")
        wait_ns = cal.t1_wait(alpha, beta)
        alpha, beta = cal.t1_update(
            alpha, beta, bool(plant.sample(plant.p_survive(wait_ns))), wait_ns
        )
        t, phase = cal.ramsey_setting(mean, sigma, max_wait_ns=2000.0)
        mean, sigma = cal.frequency_update(
            mean, sigma, plant.sample(plant.p_ramsey(t, phase)), t, phase
        )
        if (i + 1) % block == 0:
            t1_est = beta / alpha / 1e3
            log.record(i, "t1_us", t1_est, truth=plant.state.t1_us, time_s=plant.state.time_s)
            log.record(
                i, "offset_hz", mean, uncertainty=sigma, truth=plant.state.frequency_offset_hz
            )
            for key, value in (
                ("time_s", plant.state.time_s),
                ("t1_est_us", t1_est),
                ("t1_true_us", plant.state.t1_us),
                ("offset_est_khz", mean / 1e3),
                ("offset_sigma_khz", sigma / 1e3),
                ("offset_true_khz", plant.state.frequency_offset_hz / 1e3),
            ):
                cols[key].append(float(value))
            alpha, beta = 2.0, 2.0 * beta / alpha
            sigma = max(sigma, 200e3)
        store.update(
            {"alpha": alpha, "beta": beta, "mean_hz": mean, "sigma_hz": sigma}, iteration=i
        )
        plant.advance(1e-3)
    split = (telegraph.low_us + telegraph.high_us) / 2
    blocks = [(e.value, e.truth) for e in log.series("t1_us") if e.truth is not None]
    right = sum((v > split) == (t > split) for v, t in blocks)
    return {
        **cols,
        "shots": iterations,
        "block": block,
        "t1_low_us": telegraph.low_us,
        "t1_high_us": telegraph.high_us,
        "t1_state_right": right,
        "blocks": len(blocks),
        "tracking_rms_khz": log.tracking_error("offset_hz") / 1e3,
    }


def _chain(_: dict[str, float]) -> dict[str, Any]:
    ex = _example("21_calibration_protocol.py")
    target, dev = _krinner()
    q = dev.qudit_handles[0]
    started = time.perf_counter()
    with LOCK:
        compiled = pulse.compile(ex.calibration_chain, device=dev, machine=ex.MACHINE)
        lowered = compiled.lower(q, 20_000.0 * ex.CLOCK_GHZ)
    compile_s = time.perf_counter() - started
    n_readouts = sum(1 for e in lowered.schedule if e.kind == "readout")
    started = time.perf_counter()
    with LOCK:
        result = pulse.simulate(
            lowered,
            plant=ex.chain_plant(target),
            shots=1,
            seed=7,
            paths="sampled",
            readout="model",
        )
    simulate_s = time.perf_counter() - started
    report = _controller_usage(lowered)
    clock = ex.CLOCK_GHZ
    calls = result.calls[0]
    steps = []
    for call in calls:
        answer = call.answer if isinstance(call.answer, tuple) else (call.answer,)
        steps.append(
            {
                "function": call.function,
                "t_us": call.start_ns / 1e3,
                "answer": [_num(a) for a in answer],
            }
        )
    tracker = []
    for k, call in enumerate(c for c in calls if c.function == "adaptive_t1"):
        shape, rate, wait = call.answer
        tracker.append(
            {
                "shot": k + 1,
                "t_us": call.start_ns / 1e3,
                "t1_us": rate / shape / clock / 1e3,
                "t1_sigma_us": rate / shape / math.sqrt(max(shape, 1e-9)) / clock / 1e3,
                "wait_us": wait / clock / 1e3,
                "bit": int(call.inputs[0][0]),
            }
        )
    truth = {
        "readout errors": (0.03, 0.02),
        "detuning (kHz)": 60.0,
        "pi amplitude": ex.PI_AMP / 1.015,
        "pi/2 amplitude": ex.PI_AMP / 2 / 1.015,
        "T1, decay estimator (us)": 18.3,
        "T1, adaptive (us)": 18.3,
    }

    def plain(v: Any) -> Any:
        return [_num(x) for x in v] if isinstance(v, tuple) else _num(v)

    return {
        "compile_s": compile_s,
        "simulate_s": simulate_s,
        "readouts": n_readouts,
        "calls": len(lowered.classical_report()),
        "epochs": len(lowered.epochs()),
        "end_us": max((s["t_us"] for s in steps), default=0.0),
        "results": [
            {"name": name, "value": plain(value), "truth": plain(truth[name])}
            for name, value in ex.chain_results(calls).items()
        ],
        "steps": steps,
        "tracker": tracker,
        "functions": [
            {
                "function": f.function,
                "ops": f.cost.total,
                "state_bytes": f.state_bytes,
                "compute_ns": _num(f.compute_ns),
                "budget_ns": _num(f.max_latency_ns),
                "calls": f.calls_per_shot,
                "on_controller": f.on_controller,
            }
            for f in report.functions
        ],
        "violations": list(report.violations),
    }


def _spec(
    name: str, label: str, default: float, lo: float, hi: float, step: float, unit: str = ""
) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "default": default,
        "min": lo,
        "max": hi,
        "step": step,
        "unit": unit,
    }


EXPERIMENTS: dict[str, dict[str, Any]] = {
    "rabi": {
        "title": "Rabi",
        "plain": "Find how hard to drive the qubit to flip it from 0 to 1.",
        "detail": "Square drive of fixed length, amplitude swept; a cosine fit gives the pi "
        "amplitude.",
        "example": "01_single_qubit_rabi.py",
        "params": [
            _spec("max_amp", "Max amplitude", 0.4, 0.05, 1.0, 0.05),
            _spec("length", "Pulse length", 40, 8, 400, 8, "cycles"),
            _spec("points", "Points", 41, 5, 121, 4),
        ],
        "run": _rabi,
    },
    "t1": {
        "title": "T1",
        "plain": "Flip the qubit to 1 and watch how fast it relaxes back to 0.",
        "detail": "X pulse, wait swept, readout; exponential fit over the thermal floor.",
        "example": "03_t1_t2_dissipator.py",
        "params": [
            _spec("max_wait_us", "Longest wait", 80, 5, 200, 5, "µs"),
            _spec("points", "Points", 21, 5, 61, 4),
        ],
        "run": _t1,
    },
    "chain": {
        "title": "Calibration chain",
        "plain": "Calibrate a qubit end to end in one program: readout, frequency, "
        "pulse amplitudes and lifetime.",
        "detail": "Example 21's calibration_chain lowered for the controller; one simulated "
        "shot; adaptive T1 posterior per shot.",
        "example": "21_calibration_protocol.py",
        "params": [],
        "run": _chain,
    },
    "drift": {
        "title": "Drift tracking",
        "plain": "Keep tracking a qubit whose lifetime and frequency wander over time.",
        "detail": "Telegraph T1 and Ornstein-Uhlenbeck frequency drift; T1 and Ramsey "
        "estimators updated between shots.",
        "example": "20_adaptive_calibration.py",
        "params": [
            _spec("switch_hz", "T1 switch rate", 0.5, 0.05, 3.0, 0.05, "Hz"),
            _spec("drift_khz", "Frequency drift", 100, 10, 400, 10, "kHz"),
            _spec("seed", "Seed", 21, 0, 99, 1),
        ],
        "run": _drift,
    },
}

_RUNS: OrderedDict[str, dict[str, Any]] = OrderedDict()
_RUNS_LOCK = threading.Lock()


def experiments() -> list[dict[str, Any]]:
    return [
        {"id": key, **{k: v for k, v in e.items() if k != "run"}} for key, e in EXPERIMENTS.items()
    ]


def run_experiment(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """The whole run at once; the same parameters return the cached run."""
    spec = EXPERIMENTS.get(name)
    if spec is None:
        return {"ok": False, "error": f"unknown experiment {name!r}"}
    values = {s["name"]: _param(s, params) for s in spec["params"]}
    key = json.dumps([name, values], sort_keys=True)
    with _RUNS_LOCK:
        found = _RUNS.get(key)
        if found is not None:
            return {**found, "cached": True}
        started = time.perf_counter()
        try:
            data = spec["run"](values)
        except BaseException as error:  # noqa: BLE001
            return {"ok": False, "error": SourceError("run", error, None).as_dict()}
        found = {
            "ok": True,
            "experiment": name,
            "params": values,
            "data": data,
            "server_ms": (time.perf_counter() - started) * 1e3,
        }
        _RUNS[key] = found
        if len(_RUNS) > 64:
            _RUNS.popitem(last=False)
    return {**found, "cached": False}


def warm_up(on_ready: Callable[[str, float], None] | None = None) -> None:
    """Run every experiment at its defaults, fastest first, so the first Go
    replays at once."""
    for name in ("rabi", "t1", "drift"):
        started = time.perf_counter()
        try:
            run_experiment(name, {})
        except Exception:  # noqa: BLE001 - shown when the run is requested
            traceback.print_exc()
        if on_ready is not None:
            on_ready(name, time.perf_counter() - started)
    started = time.perf_counter()
    for _ in device_sweeps("transmon_krinner_17q"):
        pass
    if on_ready is not None:
        on_ready("device sweeps", time.perf_counter() - started)
    started = time.perf_counter()
    run_experiment("chain", {})
    if on_ready is not None:
        on_ready("chain", time.perf_counter() - started)


def warm_simulations(
    names: Sequence[str], request: Callable[[str], dict[str, Any]], points: int, levels: int
) -> None:
    """Simulate these examples at their presets, as the web app asks for them,
    so their first Simulate replays at once."""
    for name in names:
        try:
            simulate_populations({**request(name), "points": points, "levels": levels})
        except Exception:  # noqa: BLE001 - shown when the simulation is requested
            traceback.print_exc()


# ── Device-wide sweeps: every qudit, streamed as each one finishes ──────────

_SWEEPS: dict[str, list[dict[str, Any]]] = {}
_SWEEPS_LOCK = threading.Lock()
SWEEP_DETUNINGS_HZ = [float(v) for v in np.linspace(-4e6, 4e6, 21)]
SWEEP_DURATIONS = [int(v) for v in np.linspace(0, 2000, 16)]
_SWEEP_POINTS = 15


def _sweep_plant(target: Any, seed: int) -> tuple[Any, dict[Any, float]]:
    """The simulated device: each qudit up to 1.5 MHz off its nominal
    frequency, which the sweeps have to find."""
    rng = np.random.default_rng(seed)
    offsets = {k: float(rng.uniform(-1.5e6, 1.5e6)) for k in target.qubits}
    qubits = {
        k: dataclasses.replace(q, frequency_hz=q.frequency_hz + offsets[k])
        for k, q in target.qubits.items()
    }
    return dataclasses.replace(target, qubits=qubits), offsets


def _decay_fit(x: np.ndarray, y: np.ndarray, floor: float, rising: bool) -> float | None:
    """The time constant of an exponential approach to *floor*."""
    gap = (floor - y) if rising else (y - floor)
    keep = gap > 1e-3
    if keep.sum() < 3:
        return None
    slope, _ = np.polyfit(x[keep], np.log(gap[keep]), 1)
    return float(-1 / slope) if slope < 0 else None


def _sweep_qudit(sweeps: dict[str, Any], dev: Any, plant: Any, k: int, key: Any) -> dict[str, Any]:
    q = dev.qudit_handles[k]
    spec = dev.target.qubits[key]
    clock = dev.target.clock_ghz
    dt = 1 / clock
    amp = 0.157 * 40 / 1000
    with LOCK:
        chevron = [
            [
                float(
                    pulse.simulate(
                        sweeps["rabi"].lower(q, max(d, 1), amp, df), plant=plant, dt_ns=1.0
                    )
                    .density_matrix[1, 1]
                    .real
                )
                for df in SWEEP_DETUNINGS_HZ
            ]
            for d in SWEEP_DURATIONS
        ]
        t1_waits = np.linspace(0, 3 * spec.t1_us * 1e3 * clock, _SWEEP_POINTS).astype(int)
        t1_rows = [
            pulse.simulate(sweeps["t1"].lower(q, int(w)), plant=plant, dt_ns=dt) for w in t1_waits
        ]
        t2_taus = (
            np.linspace(0, 3 * (spec.t2_echo_us or spec.t1_us) * 1e3 * clock, _SWEEP_POINTS)
            // 2
            * 2
        ).astype(int)
        t2_rows = [
            pulse.simulate(sweeps["echo"].lower(q, int(w)), plant=plant, dt_ns=dt) for w in t2_taus
        ]
    grid = np.asarray(chevron)
    dets = np.asarray(SWEEP_DETUNINGS_HZ)
    peak = grid.max(axis=0)
    j = int(peak.argmax())
    offset = float(dets[j])
    if 0 < j < len(dets) - 1:
        a, b, c = peak[j - 1], peak[j], peak[j + 1]
        denom = a - 2 * b + c
        if denom < 0:
            offset += float(0.5 * (a - c) / denom * (dets[1] - dets[0]))
    # The Rabi rate on resonance: a cosine fit of the column nearest the peak.
    t_ns = np.asarray(SWEEP_DURATIONS) / clock
    column = grid[:, j]
    best = (math.inf, 0.0)
    for f in np.linspace(0.1e6, 5e6, 500):
        basis = np.stack([np.ones_like(t_ns), (1 - np.cos(2 * np.pi * f * t_ns * 1e-9)) / 2], 1)
        coef, *_ = np.linalg.lstsq(basis, column, rcond=None)
        err = float(np.sum((basis @ coef - column) ** 2))
        if err < best[0]:
            best = (err, float(f))
    rabi_hz = best[1]
    t1x = np.asarray(
        [(r.measurements[0].start_ns - t1_rows[0].measurements[0].start_ns) / 1e3 for r in t1_rows]
    )
    t1y = np.asarray([float(r.density_matrix[1, 1].real) for r in t1_rows])
    t2x = t2_taus / clock / 1e3
    t2y = np.asarray([float(r.density_matrix[1, 1].real) for r in t2_rows])
    floor = spec.thermal_population or 0.0
    return {
        "qudit": k,
        "label": spec.label,
        "chevron": chevron,
        "t1": {"x": [float(v) for v in t1x], "y": [float(v) for v in t1y]},
        "t2": {"x": [float(v) for v in t2x], "y": [float(v) for v in t2y]},
        "fit": {
            "offset_khz": offset / 1e3,
            "pi_ns": 1e9 / (2 * rabi_hz),
            "t1_us": _decay_fit(t1x, t1y, floor, rising=False),
            "t2_us": _decay_fit(t2x, t2y, 0.5, rising=True),
        },
    }


def device_sweeps(target_id: str, seed: int = 3) -> Iterator[dict[str, Any]]:
    """Spectroscopy, T1 and Hahn-echo sweeps of every qudit of a device,
    one event per qudit as it finishes; a repeated run streams the cached
    events."""
    from . import sweep_kernels

    cache_key = f"{target_id}:{seed}"
    with _SWEEPS_LOCK:
        cached = _SWEEPS.get(cache_key)
    if cached is not None:
        yield from ({**e, "cached": True} for e in cached)
        return
    dev = device(target_id)
    target = dev.target
    plant, offsets = _sweep_plant(target, seed)
    keys = list(target.qubits)
    events: list[dict[str, Any]] = []

    def emit(event: dict[str, Any]) -> dict[str, Any]:
        events.append(event)
        return event

    yield emit(
        {
            "type": "start",
            "target": target_id,
            "qudits": [{"index": i, "label": target.qubits[k].label} for i, k in enumerate(keys)],
            "detunings_khz": [v / 1e3 for v in SWEEP_DETUNINGS_HZ],
            "durations_ns": [d / target.clock_ghz for d in SWEEP_DURATIONS],
        }
    )
    ex3 = _example("03_t1_t2_dissipator.py")
    started = time.perf_counter()
    try:
        with LOCK:
            sweeps = {
                "rabi": pulse.compile(sweep_kernels.detuned_rabi, device=dev),
                "t1": pulse.compile(ex3.t1_experiment, device=dev),
                "echo": pulse.compile(ex3.t2_echo, device=dev),
            }
        for i, k in enumerate(keys):
            spec = target.qubits[k]
            result = _sweep_qudit(sweeps, dev, plant, i, k)
            result["truth"] = {
                "offset_khz": offsets[k] / 1e3,
                "t1_us": spec.t1_us,
                "t2_us": spec.t2_echo_us,
            }
            yield emit({"type": "qudit", "elapsed_s": time.perf_counter() - started, **result})
    except Exception as error:  # noqa: BLE001 - shown in the tab
        yield {"type": "error", **SourceError("sweep", error, None).as_dict()}
        return
    yield emit({"type": "done", "elapsed_s": time.perf_counter() - started})
    with _SWEEPS_LOCK:
        _SWEEPS[cache_key] = events


# ── Precomputed runs: saved once, loaded at start ───────────────────────────


def save_runs(path: Path) -> None:
    """Every finished run, compile and simulation, for ``load_runs``."""
    with _RUNS_LOCK, _SIMS_LOCK, _SWEEPS_LOCK:
        data = {
            "version": version(),
            "runs": list(_RUNS.items()),
            "sims": list(_SIMS.items()),
            "sweeps": list(_SWEEPS.items()),
            "compiled": [(k, v) for k, v in _COMPILED.items() if v.get("ok")],
        }
    path.write_text(json.dumps(data, separators=(",", ":")))


def load_runs(path: Path) -> bool:
    """Loads ``save_runs``'s file when it was made with this cudaq-pulse."""
    data = json.loads(path.read_text())
    if data.get("version") != version():
        return False
    # Marked so the web app says "precomputed" instead of the original run time.
    for _, value in data["runs"] + data["sims"] + data["compiled"]:
        value["precomputed"] = True
    for _, events in data["sweeps"]:
        for event in events:
            event["precomputed"] = True
    with _RUNS_LOCK, _SIMS_LOCK, _SWEEPS_LOCK:
        _RUNS.update(data["runs"])
        _SIMS.update(data["sims"])
        _SWEEPS.update(data["sweeps"])
        _COMPILED.update(data["compiled"])
    return True


__all__ = [
    "EXPERIMENTS",
    "SCHEDULES",
    "SourceError",
    "compile_stages",
    "default_passes",
    "device_info",
    "device_sweeps",
    "example_source",
    "examples",
    "experiments",
    "kernels",
    "load_runs",
    "pass_names",
    "run_experiment",
    "save_runs",
    "simulate_populations",
    "targets",
    "version",
    "warm_simulations",
    "warm_up",
]
