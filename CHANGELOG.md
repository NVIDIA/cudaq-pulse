# Changelog

All notable changes to cudaq-pulse are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the [versioning policy](CONTRIBUTING.md#versioning-and-stability): while the
major version is 0, a minor release may break the API.

Each GitHub release takes its notes from the section for its version, so
every change worth telling users about goes in the top section, the next
release's `## [X.Y.Z] - Unreleased`, in the pull request that makes it.

## [0.1.0] - Unreleased

The first public release, a research preview: APIs, dialects and numerical
behavior may change in any 0.x release.

### Added

- **Calibration on the controller.** `cudaq_pulse.classical.calibration`
  adds the estimators of millisecond on-controller calibration in controller
  form: adaptive T1 tracking with assignment errors (`adaptive_t1_update`,
  `adaptive_t1_wait`), the frequency binary search
  (`binary_search_update`, `binary_search_wait`, `binary_search_phase`),
  the analytical decay estimator (`decay_time`) and the sparse phase
  estimator (`sparse_phase`, `pi_train_amplitude`, `ramsey_detuning`).
  `examples/21_calibration_protocol.py` runs the millisecond calibration
  chain of M. A. Marciniak, R. T. Birke, J. B. Severin, F. Berritta,
  D. Kjær, F. Nilsson, S. N. Themadath, S. Kallatt, J. L. Webb, K. Bentsen,
  T. Madsen, Z. Sun, S. Krøjer, C. W. Warren, J. Hastrup and M. Kjaergaard,
  "Millisecond-Scale Calibration and Benchmarking of Superconducting
  Qubits", arXiv:2602.11912 (2026), as one program with its estimators on
  the controller link, its adaptive T1 step after F. Berritta et al.,
  "Real-time adaptive tracking of fluctuating relaxation rates in
  superconducting qubits", arXiv:2506.09576 (2025);
  `benchmarks/classical/calibration_protocol.py` checks it end to end over
  many seeds against the published numbers.
- **What classical functions cost a controller.** `controller.resources()`
  reports each classical function's operations by class (read from its
  body), state in bytes, compute time against `max_latency_ns`, and calls,
  slots and buffer bytes per shot, and whether its body is in controller
  form (fixed-size state, bounded loops, no allocation or exceptions).
  `ControllerLimits` gains `op_cycles`, `compute_clock_ghz`, `max_ops`,
  `state_bytes`, `max_calls_per_shot`, `buffer_bytes`, `controller_form`
  and `compute_links`; compiling fails with `ControllerLimitError` when a
  program exceeds them.
- **Faster simulation of long programs.** Idle waits and square pulses in
  their frame are propagated exactly, at a cost independent of their
  length, and driven stretches on registers of up to four levels by RK4
  maps built at once (a 100 us T1 shot: 104 s to under 1 ms).
  `simulate(paths="sampled")` follows only the paths the shots take, so a
  program that measures many times in one shot costs time linear in its
  measurements. Finding a program's next measurement and next call is now
  linear in its length.

- **Kernels.** `@pulse.kernel` traces a Python function into Pulse-dialect
  MLIR. Calling a kernel with qudit handles compiles it for their device on
  first use, caches the specialization, and runs it; `kernel.lower(...)`
  stops before running, and `pulse.compile(kernel)` compiles ahead of the
  call. `float` parameters stay symbolic, so a parameter sweep compiles once.
  Kernels call other kernels, loop with `range`, and branch on measurement
  results. Kernels take their qudits as parameters; a call without qudit
  handles raises `NoDeviceError` unless the kernel was compiled for a device
  (`pulse.compile(k, device=dev)`).
- **Loop unrolling.** A `range` loop whose variable is never read is kept
  as one rolled loop, unless its estimated unrolled size (trip count times
  the calls in its body) is at most the unroll threshold: 8 by default, the
  measured break-even point, set with `pulse.compile(k, unroll_threshold=)`,
  `@pulse.kernel(unroll_threshold=)` or `CUDAQ_PULSE_UNROLL_THRESHOLD`.
- **Runtime wait lengths.** A `wait` whose length is computed from `float`
  parameters with `+`, `-`, `*` and `/` (`wait(d, tau)`, `wait(d, tau / 2)`)
  keeps them runtime parameters: a T1, Ramsey or echo sweep compiles once,
  and a call with a new length costs about as much as a cached call. Each
  call rounds the length to the nearest clock cycle (ties up) and schedules
  exactly as the kernel compiled with that length would. The optional
  `wait(line, length, bounds=(lowest, highest))` bounds it in cycles; a
  length that is not finite, is negative or is out of bounds raises
  `ValueError` at the call. A literal length must be a whole number of
  cycles. A length computed from the answer of a classical call makes the
  wait a variable region: it is rounded the same way, clamped to its
  bounds, and opens a new epoch (below). A length from both, such as
  `wait(d, answer + tau)`, compiles once too: each call checks that its
  float part (the length with its answers at 0) is finite and at most the
  upper bound, and the whole length is clamped to the bounds at run time.
- **Python in kernels.** List, dict and set comprehensions, set displays
  and generator expressions are unrolled while tracing, so
  `[get_drive_line(q) for q in qubits]` builds a list of lines and
  `sum(a for a in amps)` works; a lambda or nested `def` raises a
  `CompilationError` that says what to write instead. Calls with
  `*args` and `**kwargs` are unpacked
  while tracing, so `sync(*lines)` synchronizes a list of lines built in a
  loop. Helpers called from a kernel may take keyword-only, `*args` and
  `**kwargs` parameters; methods of helper objects, `functools.partial`
  objects and decorated helpers that use pulse ops are inlined too. Lines
  and tones may be kept in any container, such as a list of `(line, tone)`
  pairs or a dict, and lists of them support `pop`, `insert`, `extend`,
  `remove` and `del xs[i]`. `sync()` takes lists and tuples of lines, as
  in `sync(lines)`. A pulse op given the wrong kind of argument, such as a
  tone where its line belongs, raises a `CompilationError` that names the
  op and the argument when it is traced, not an MLIR verifier error.
  Errors raised while tracing carry a note with the file, line and source
  of the kernel line, and of each helper line, that raised them; a
  misspelled op or attribute gets a "did you mean" suggestion, and
  `pulse.drive(...)` says that the kernel language is in `cudaq_pulse.ops`.
  A failing `assert cond, "msg"` reports "assertion failed in kernel: msg".
  Starred assignment, `first, *rest = xs`, works, and so does slice
  assignment on every supported Python. A construct the tracer does not
  support, such as a `match` class pattern or an assignment to a global,
  is named in the error, with what to write instead, rather than reported
  as an unsupported bytecode instruction. A `float` argument that tracing
  makes static, because `abs`, `min`, `math.*`, a comparison or a waveform
  duration needs its value, warns once with a `RuntimeWarning` naming the
  argument and the operation, since each new value then compiles the kernel
  again.
- **Operations.** The kernel language lives in `cudaq_pulse.ops` and is
  used by bare name inside kernels. Drive and readout channels per qudit; `drive`, `readout`,
  `wait` and `sync`; phase and frequency control (`shift_phase` follows the
  McKay et al. 2017 virtual-Z convention); Gaussian, square, DRAG (beta in
  clock cycles), cosine, tanh-ramp, Gaussian-square and sampled waveforms;
  waveform sums, differences, scaling and negation (exact) and products
  (sampled once per clock cycle).
- **Editor support with no configuration.** The wheel installs
  `__builtins__.pyi` at the root of site-packages, generated at build time
  from `cudaq_pulse.ops.__all__`, so Pylance, Cursor, pyright and
  basedpyright know the operations used by bare name inside kernels: no
  undefined-name warnings, with completion, hover documentation and checked
  arguments. The names are builtins in the whole environment, outside
  kernels too; see the new Editor Support page. The getting-started advice
  to add a `pyrightconfig.json` is gone.
- `python -m cudaq_pulse.doctor --setup-editor` configures mypy and ruff,
  which take no names from an installed package: it shows the change to the
  project's mypy and ruff configuration and writes it when confirmed or
  with `--yes`. The mypy change enables `cudaq_pulse.mypy_plugin`, which
  types the operations in files that import cudaq_pulse; the ruff change
  lists them as `builtins`.
- `cudaq_pulse.ops.__all__` lists the kernel language.
- `ScheduledEvent.start` and `ScheduledEvent.end`: an event's scheduled
  times in clock cycles as `int`, raising `ValueError` on an event that is
  not scheduled yet.
- `Kernel` and `CompiledKernel` are generic in the decorated function's
  parameters, so type checkers check `k(...)`, `k.lower(...)` and
  `pulse.compile(k)(...)` against the kernel's signature.
- `square()` is typed to take the complex and `[re, im]` amplitudes it
  accepts.
- `custom(duration, f)` samples the Python function `f` once per clock cycle
  while the kernel is traced, so custom envelopes compile, schedule and
  simulate like the built-in ones; it takes a callable, not an envelope
  name, and up to 252 clock cycles, like `custom_samples`.
- **Compilation.** Verification, virtual-Z folding and drive fusion by
  default; ASAP, ALAP and resource-constrained schedules
  (`cudaq_pulse.compiler.MachineModel`); per-kernel schedule policies and
  schedule blocks. `pulse.Lowered.mlir()` returns the scheduled Pulse MLIR
  or the LLVM-dialect module, `Lowered.schedule` the timed events
  (`cudaq_pulse.compiler.ScheduledEvent`), `Lowered.metrics` the compile
  times (`cudaq_pulse.compiler.CompileMetrics`), and `Lowered.timeline()` a
  plot (with the `[viz]` extra).
- **Devices and targets.** `pulse.get_device` and `pulse.device` describe a
  machine as a target plus a qudit layout; devices compare by content and
  share compiled kernels. Target descriptions are
  `cudaq_pulse.targets.TargetDescription`; `pulse.Target` names the shipped
  targets: a 17-qubit transmon device built from Krinner et al., Nature 605,
  669 (2022), and Rydberg chain and square arrays (compile only).
- **Calibrations and the transmon model.** `cudaq_pulse.targets.Calibration`
  loads fitted values, each with its kind, source, uncertainty and cooldown,
  apart from the published target, and a device config takes `"calibration"`
  and `"cooldown"`. `cudaq_pulse.targets.transmon_model` computes
  flux-tunable transmon levels and charge-coupled pairs from circuit values.
  The shipped `krinner17` calibration has one device-wide charge coupling,
  fitted to the measured `|11>`-`|20>` coupling of X2-D4, so the Krinner
  target's exchange (5.67-6.08 MHz) and idle ZZ (-4.5 to -7.3 kHz) depend on
  each pair's frequencies; calibrated qubits convert detunings to and from flux quanta.
  It also holds in-model sudden-net-zero CZ knobs for all 24 pairs (mean
  duration 65.0 +- 4.8 ns against the published 68(7) ns), each with its
  predicted phase and leakage, flags and alternatives.
- **Simulation.** `pulse.simulate` evolves a compiled call with T1, T2 or
  Hahn-echo dephasing, thermal populations, exchange couplings, residual ZZ
  and any number of transmon levels, either with a NumPy reference
  integrator (`backend="cpu"`) or on NVIDIA GPUs through cuDensityMat
  (`backend="gpu"`, Linux, experimental). Programs that measure and branch
  are evolved as an exact ensemble over outcomes. The simulated space holds
  only the qudits the program gets a line on, whatever their device indices
  (a pair at qudits 13 and 3 is a two-mode problem); `SimulationResult.qudits`
  and `Lowered.qudits` give the device qudit of each mode, and the other
  qudits are classical spectators.
- **Classical calls at QPU runtime.** `classical`, a classical real-time
  dialect for co-processing at QPU runtime: pulse programs call GPU, host or
  FPGA services (decoders, estimators, calibration updates) while the QPU
  runs and use their answers in the same shot. `@pulse.classical` declares a
  function with typed arguments and results, a latency model and a link;
  `fn.submit(...).result()` calls it in a kernel, with `timeout_ns=` for a
  fallback. Type checkers and editors see the decorator with its keyword
  options, and `.submit()` returns a `Pending`. Links are named in
  `MachineModel(links={...})`. Every schedule budgets the worst-case
  latency, holds link slots and reports each call's slack and its hidden and
  exposed latency (`Lowered.classical_report()`); `pulse-classical-check`
  verifies slots, deadlines and ordering. A wait whose length is an answer,
  and a blocking wait for an answer, are barriers: the timeline splits into
  epochs (`Lowered.epochs()`) whose start shifts with the run-time lengths
  before them.
- **Simulating and emulating calls.** `pulse.simulate` binds answers per
  measurement path, with timeouts from the latency model, assignment errors
  and relaxation during readout, IQ points from a dispersive readout model,
  and `plant=` to simulate a drifted device without compiling again.
  `Lowered.export_emulation()` writes the schedule, epochs and call manifest
  for a controller; `classical.emulation.emulate` runs a program in virtual
  time against real services, in process or over UDP. Requests and answers
  use the CUDA-Q real-time wire format (`classical.wire`);
  `classical.qec.Decoder` talks to a CUDA-QX real-time decoder. The
  cuDensityMat runtime keeps a classical handler table per run.
- **Calibrating while the QPU runs.** `classical.calibration` (adaptive T1,
  a frequency binary search, three-point decay and phase estimators,
  golden-section and Nelder-Mead steps), `classical.random_index` (an
  on-device random source), `classical.loop` (a parameter store whose values
  programs read at run time, and an estimate log) and `runtime.drift` (a
  drifting device with its truth trace and surrogates). Examples
  `19_classical_calls.py` and `20_adaptive_calibration.py`, and the
  "Classical Calls at QPU Runtime" guide.
- **Controller backends.** `Lowered.export_controller()` exports a
  scheduled kernel for an out-of-tree controller backend, format
  `cudaq-pulse.controller/1` (`cudaq_pulse.controller.FORMAT`): the lines
  with their roles, and every drive, readout, wait, sync and classical call
  with its exact start and end, waveform and parameters (and, with
  `sample_rate_ghz=`, its samples), tone frequency and frame phase (including
  the phase a detuned tone gains between drives), epoch and the
  measurement- or answer-conditioned branches it lies in. Within a version,
  keys and event kinds are only added. `MachineModel(controller=
  ControllerLimits(...))` states what a controller can play (clock, issue
  spacing, waveform memory, tones, readout acquirers, feedback latency,
  answer timeouts, program size) and makes compilation fail with
  `ControllerLimitError` when a program exceeds it, parametric kernels
  included. `cudaq_pulse.backends` finds backends installed under the
  `cudaq_pulse.backends` entry-point group; a backend that fails to load, is
  not a `Backend` or does not read this export format raises
  `IncompatibleBackendError`, and an unknown name `BackendNotFoundError`,
  both `BackendError`s. `examples/controller_backend/` is a toy backend, and
  the "Writing a Controller Backend" guide walks through one.
- **Public API.** The top level is exactly `kernel`, `Kernel`, `compile`,
  `CompiledKernel`, `Lowered`, `simulate`, `SimulationResult`, `device`,
  `get_device`, `Device`, `Qudit`, `Register`, `Target`, `PulseError`,
  `CompilationError`, `NoTargetError`, `NoDeviceError`,
  `DeviceMismatchError`, `GpuRuntimeError`, `viz`, `classical`,
  `controller` and `backends`, plus `__version__`.
  Every error cudaq-pulse raises on purpose derives from `PulseError` and
  keeps its built-in base (`NoTargetError` is still a `RuntimeError`).
  `cudaq_pulse.compiler`, `cudaq_pulse.runtime.simulate` and
  `cudaq_pulse.targets.transmon_model` list their public names in `__all__`,
  as the other documented modules do. `cudaq_pulse.targets.UnknownTargetError`
  (an unregistered target name) and
  `cudaq_pulse.compiler.UnsupportedInterpreterError` (a Python whose bytecode
  the kernel tracer does not know) are public, and every public name is in
  the API reference, errors on their own page.
- Target registration, the shipped targets and calibrations, and the kernel
  tracer's caches are safe to use from several threads on free-threaded
  CPython: each target and calibration is built once.
- **Dialects and tools.** The Pulse, Classical, QOp and CuDensityMat MLIR
  dialects with their lowerings to LLVM; `cudaq-pulse-opt`, the MLIR driver, which
  also loads out-of-tree pass plugins; `python -m cudaq_pulse.doctor`, which
  reports what an installation contains and whether GPU execution works.
- **Packaging.** One abi3 wheel per platform for CPython 3.12 and newer:
  Linux x86_64 and aarch64 (manylinux_2_28) with the CUDA 12 and CUDA 13
  GPU runtimes, selected by the `[cu12]` or `[cu13]` extra, and macOS arm64
  (compiler only). The wheels link their own MLIR, so they need neither
  CUDA-Q nor an LLVM installation.

[0.1.0]: https://github.com/NVIDIA/cudaq-pulse/releases/tag/v0.1.0
