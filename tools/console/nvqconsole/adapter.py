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
import math
import re
import sys
import tempfile
import threading
import time
import traceback
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import cudaq_pulse as pulse
import numpy as np
from cudaq_pulse import compiler as _compiler
from cudaq_pulse import controller as _controller

# One compile or simulation at a time: the native module builder is shared.
LOCK = threading.Lock()

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples"
_SCRATCH = Path(tempfile.mkdtemp(prefix="nvqconsole-"))
_MAX_EVENTS = 6000


# ── Library surface ─────────────────────────────────────────────────────────


def version() -> str:
    return str(getattr(pulse, "__version__", "unknown"))


def pass_names() -> list[str]:
    """Every pass ``passes=`` accepts, and the default pipeline."""
    enum = getattr(pulse, "Pass", None)
    if enum is not None:
        return [str(getattr(p, "value", p)) for p in enum]
    return list(_compiler._PASS_MAP)


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
    "21_calibration_protocol.py": {"kernel": "binary_search", "args": "q[0], 4"},
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


def load_source(source: str) -> tuple[Any, Path]:
    """Import *source* as a module from a file of its own, as Python would
    import an example, so that kernels can read their source."""
    key = str(hash(source))
    if key in _MODULES:
        return _MODULES[key]
    path = _SCRATCH / f"console_{len(_MODULES)}.py"
    path.write_text(source)
    name = f"nvqconsole_user_{len(_MODULES)}"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
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


def _events(lowered: Any) -> tuple[list[dict[str, Any]], bool]:
    events = lowered.schedule
    truncated = len(events) > _MAX_EVENTS
    return [
        {
            "kind": e.kind,
            "line": e.line,
            "qudit": e.qubit,
            "start": e.start_cycles,
            "duration": e.duration_cycles,
        }
        for e in events[:_MAX_EVENTS]
    ], truncated


def _metrics(lowered: Any) -> dict[str, Any]:
    m = lowered.metrics
    return {f.name: _num(getattr(m, f.name)) for f in dataclasses.fields(m)}


def compile_stages(request: dict[str, Any]) -> dict[str, Any]:
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
            if len(program.get("events", [])) > _MAX_EVENTS:
                program["events"] = program["events"][:_MAX_EVENTS]
                program["events_truncated"] = True
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


def simulate_populations(request: dict[str, Any]) -> dict[str, Any]:
    """Each qudit's level populations at evenly spaced times."""
    points = max(2, min(int(request.get("points", 41)), 121))
    levels = max(2, min(int(request.get("levels", 2)), 4))
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
                    "message": "give values for the float parameters: "
                    + ", ".join(lowered.parameters),
                    "line": None,
                    "trace": [],
                },
            }
        events, _ = _events(lowered)
        clock = lowered.target.clock_ghz
        end_cycles = max(
            (e["start"] + e["duration"] for e in events if e["start"] is not None), default=0
        )
        end_ns = max(end_cycles / clock, 1.0)
        times = np.linspace(0.0, end_ns, points)
        series = []
        started = time.perf_counter()
        try:
            final = None
            for t in times:
                # The initial state (thermal populations included) at a step's
                # length past the start.
                at = max(float(t), 1e-3)
                final = pulse.simulate(lowered, levels=levels, t_end=at)
                series.append(
                    {"t_ns": float(t), "populations": _marginals(final.density_matrix, final.dims)}
                )
        except BaseException as error:  # noqa: BLE001
            return {"ok": False, "error": SourceError("simulate", error, None).as_dict()}
        assert final is not None
        return {
            "ok": True,
            "qudits": list(final.qudits),
            "dims": list(final.dims),
            "series": series,
            "events": events,
            "clock_ghz": clock,
            "branches": [{"outcomes": list(o), "p": float(p)} for o, p in final.branches[:32]],
            "wall_s": time.perf_counter() - started,
        }


# ── Calibration runs (examples 20 and 21) ───────────────────────────────────

_EXAMPLE_MODULES: dict[str, Any] = {}


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


def drift_tracking(
    seed: int = 21, iterations: int = 6000, block: int = 150
) -> Iterator[dict[str, Any]]:
    """Example 20, part 2: T1 and frequency tracked between shots on a
    drifting device; one record per block of shots."""
    ex = _example("20_adaptive_calibration.py")
    from cudaq_pulse.classical import calibration as cal
    from cudaq_pulse.classical.loop import EstimateLog, ParameterStore
    from cudaq_pulse.runtime.drift import DriftingPlant, OrnsteinUhlenbeck, TelegraphT1

    target, _ = _krinner()
    qubit = target.qubits[0]
    telegraph = TelegraphT1(qubit.t1_us / 4, qubit.t1_us, rate_hz=0.5)
    plant = DriftingPlant(
        target, t1=telegraph, frequency=OrnsteinUhlenbeck(100e3, correlation_s=5.0), seed=seed
    )
    store = ParameterStore(alpha=1.0, beta=10_000.0, mean_hz=0.0, sigma_hz=1e6)
    log = EstimateLog()
    yield {
        "type": "start",
        "iterations": iterations,
        "block": block,
        "t1_low_us": telegraph.low_us,
        "t1_high_us": telegraph.high_us,
        "source": ex.__name__,
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
            yield {
                "type": "point",
                "i": i + 1,
                "time_s": plant.state.time_s,
                "t1_est_us": t1_est,
                "t1_true_us": plant.state.t1_us,
                "offset_est_khz": mean / 1e3,
                "offset_sigma_khz": sigma / 1e3,
                "offset_true_khz": plant.state.frequency_offset_hz / 1e3,
                "store_version": store.version,
            }
            alpha, beta = 2.0, 2.0 * beta / alpha
            sigma = max(sigma, 200e3)
        store.update(
            {"alpha": alpha, "beta": beta, "mean_hz": mean, "sigma_hz": sigma}, iteration=i
        )
        plant.advance(1e-3)
    split = (telegraph.low_us + telegraph.high_us) / 2
    blocks = [(e.value, e.truth) for e in log.series("t1_us") if e.truth is not None]
    right = sum((v > split) == (t > split) for v, t in blocks)
    yield {
        "type": "done",
        "t1_state_right": right,
        "blocks": len(blocks),
        "tracking_rms_khz": log.tracking_error("offset_hz") / 1e3,
        "store_version": store.version,
    }


def binary_search(rounds: int = 15, seeds: int = 4) -> Iterator[dict[str, Any]]:
    """Example 21, part 2: the frequency binary search, one simulated shot
    per detuning; one record per round of each."""
    ex = _example("21_calibration_protocol.py")
    target, dev = _krinner()
    q = dev.qudit_handles[0]
    with LOCK:
        lowered = pulse.compile(ex.binary_search, device=dev, machine=ex.MACHINE).lower(q, rounds)
    yield {"type": "start", "rounds": rounds, "sigma0_khz": ex.SIGMA0_HZ / 1e3}
    for seed in range(seeds):
        offset = (seed - 1.5) * 300e3
        with LOCK:
            found = pulse.simulate(
                lowered,
                plant=ex.search_plant(target, offset),
                shots=1,
                seed=seed,
                paths="sampled",
                readout="model",
            )
        yield {"type": "series", "seed": seed, "offset_khz": offset / 1e3}
        for k, call in enumerate(found.calls[0]):
            mean, sigma, wait_cycles, _ = call.answer
            yield {
                "type": "point",
                "seed": seed,
                "round": k + 1,
                "t_us": call.start_ns / 1e3,
                "mean_khz": mean / 1e3,
                "sigma_khz": sigma / 1e3,
                "wait_ns": wait_cycles / ex.CLOCK_GHZ,
            }
    yield {"type": "done"}


_CHAIN: dict[str, Any] = {}
_CHAIN_LOCK = threading.Lock()


def _chain_result() -> dict[str, Any]:
    """Example 21, part 1 and 3: the calibration chain compiled for the
    controller, one shot of it simulated, and its controller usage."""
    with _CHAIN_LOCK:
        if _CHAIN:
            return _CHAIN
        ex = _example("21_calibration_protocol.py")
        target, dev = _krinner()
        q = dev.qudit_handles[0]
        started = time.perf_counter()
        with LOCK:
            compiled = pulse.compile(ex.calibration_chain, device=dev, machine=ex.MACHINE)
            lowered = compiled.lower(q, 20_000.0 * ex.CLOCK_GHZ)
            compile_s = time.perf_counter() - started
            n_readouts = sum(1 for e in lowered.schedule if e.kind == "readout")
            n_calls = len(lowered.classical_report())
            n_epochs = len(lowered.epochs())
            started = time.perf_counter()
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
                    "t1_us": rate / shape / clock / 1e3,
                    "t1_sigma_us": rate / shape / math.sqrt(max(shape, 1e-9)) / clock / 1e3,
                    "wait_us": wait / clock / 1e3,
                    "bit": int(call.inputs[0][0]),
                }
            )
        results = ex.chain_results(calls)
        truth = {
            "readout errors": (0.03, 0.02),
            "detuning (kHz)": 60.0,
            "pi amplitude": ex.PI_AMP / 1.015,
            "pi/2 amplitude": ex.PI_AMP / 2 / 1.015,
            "T1, decay estimator (us)": 18.3,
            "T1, adaptive (us)": 18.3,
        }
        _CHAIN.update(
            {
                "compile_s": compile_s,
                "simulate_s": simulate_s,
                "readouts": n_readouts,
                "calls": n_calls,
                "epochs": n_epochs,
                "results": [
                    {
                        "name": name,
                        "value": [_num(v) for v in value]
                        if isinstance(value, tuple)
                        else _num(value),
                        "truth": list(truth[name])
                        if isinstance(truth[name], tuple)
                        else truth[name],
                    }
                    for name, value in results.items()
                ],
                "steps": steps,
                "tracker": tracker,
                "functions": [
                    {
                        "function": f.function,
                        "link": str(f.link),
                        "ops": f.cost.total,
                        "op_classes": dict(f.cost.ops),
                        "state_bytes": f.state_bytes,
                        "compute_ns": _num(f.compute_ns),
                        "budget_ns": _num(f.max_latency_ns),
                        "calls": f.calls_per_shot,
                        "on_controller": f.on_controller,
                    }
                    for f in report.functions
                ],
                "links": [
                    {
                        k: _num(v) if not isinstance(v, str) else v
                        for k, v in dataclasses.asdict(u).items()
                    }
                    for u in report.links
                ],
                "limits": None if report.limits is None else report.limits.as_dict(),
                "violations": list(report.violations),
                "report": str(report),
            }
        )
        return _CHAIN


def calibration_chain() -> Iterator[dict[str, Any]]:
    cached = bool(_CHAIN)
    yield {"type": "start", "cached": cached}
    chain = _chain_result()
    yield {"type": "summary", **{k: v for k, v in chain.items() if k not in ("steps", "tracker")}}
    for step in chain["steps"]:
        yield {"type": "step", **step}
    yield {"type": "done"}


def warm_up() -> None:
    """Build the slow calibration chain ahead of the first request."""
    try:
        _chain_result()
    except Exception:  # noqa: BLE001 - shown when the run is requested
        traceback.print_exc()


__all__ = [
    "SCHEDULES",
    "SourceError",
    "binary_search",
    "calibration_chain",
    "compile_stages",
    "default_passes",
    "device_info",
    "drift_tracking",
    "example_source",
    "examples",
    "kernels",
    "pass_names",
    "simulate_populations",
    "targets",
    "version",
    "warm_up",
]
