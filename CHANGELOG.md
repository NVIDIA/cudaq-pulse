# Changelog

All notable changes to cudaq-pulse are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the [versioning policy](CONTRIBUTING.md#versioning-and-stability): while the
major version is 0, a minor release may break the API.

Each GitHub release takes its notes from the section for its version, so
every change worth telling users about goes under **Unreleased** in the pull
request that makes it.

## [Unreleased]

## [0.1.0] - Unreleased

The first public release, a research preview: APIs, dialects and numerical
behavior may change in any 0.x release.

### Added

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
  `ValueError` at the call.
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
  duration needs its value, warns once with a `RuntimeWarning` naming the argument
  and the operation, since each new value then compiles the kernel again. List, dict and set comprehensions and set
  displays trace on every supported Python; generator expressions, lambdas
  and nested functions are rejected with what to write instead.
- **Operations.** The kernel language lives in `cudaq_pulse.ops` and is
  used by bare name inside kernels. Drive and readout channels per qudit; `drive`, `readout`,
  `wait` and `sync`; phase and frequency control (`shift_phase` follows the
  McKay et al. 2017 virtual-Z convention); Gaussian, square, DRAG (beta in
  clock cycles), cosine, tanh-ramp, Gaussian-square and sampled waveforms;
  waveform sums, differences, scaling and negation (exact) and products
  (sampled once per clock cycle).
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
- **Public API.** The top level is exactly `kernel`, `Kernel`, `compile`,
  `CompiledKernel`, `Lowered`, `simulate`, `SimulationResult`, `device`,
  `get_device`, `Device`, `Qudit`, `Register`, `Target`,
  `CompilationError`, `NoTargetError`, `NoDeviceError`,
  `DeviceMismatchError`, `GpuRuntimeError` and `viz`, plus `__version__`.
- **Dialects and tools.** The Pulse, QOp and CuDensityMat MLIR dialects
  with their lowerings to LLVM; `cudaq-pulse-opt`, the MLIR driver, which
  also loads out-of-tree pass plugins; `python -m cudaq_pulse.doctor`, which
  reports what an installation contains and whether GPU execution works.
- **Packaging.** One abi3 wheel per platform for CPython 3.12 and newer:
  Linux x86_64 and aarch64 (manylinux_2_28) with the CUDA 12 and CUDA 13
  GPU runtimes, selected by the `[cu12]` or `[cu13]` extra, and macOS arm64
  (compiler only). The wheels link their own MLIR, so they need neither
  CUDA-Q nor an LLVM installation.

[Unreleased]: https://github.com/NVIDIA/cudaq-pulse/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/NVIDIA/cudaq-pulse/releases/tag/v0.1.0
