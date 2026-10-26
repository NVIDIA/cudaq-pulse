# Changelog

All notable changes to cudaq-pulse are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
the [versioning policy](CONTRIBUTING.md#versioning-and-stability): while the
major version is 0, a minor release may break the API.

Each GitHub release takes its notes from the section for its version, so
every change worth telling users about goes under **Unreleased** in the pull
request that makes it.

## [Unreleased]

### Added

- **Classical calls at QPU runtime.** `classical`, a classical real-time
  dialect for co-processing at QPU runtime: pulse programs call GPU, host or
  FPGA services (decoders, estimators, calibration updates) while the QPU
  runs and use their answers in the same shot. `@pulse.classical` declares a
  function with typed arguments and results, a latency model and a link;
  `fn.submit(...).result()` calls it in a kernel, with `timeout_ns=` for a
  fallback. Links are named in `MachineModel(links={...})`. Every schedule
  budgets the worst-case latency, holds link slots and reports each call's
  slack and its hidden and exposed latency (`Lowered.classical_report()`);
  `pulse-classical-check` verifies slots, deadlines and ordering.
- **Variable waits.** `wait(line, answer, bounds=(lo, hi))` takes its length
  from an answer; the timeline splits into epochs (`Lowered.epochs()`).
- **Simulating and emulating calls.** `pulse.simulate` binds answers per
  measurement path, with timeouts from the latency model, assignment errors
  and relaxation during readout, IQ points from a dispersive readout model,
  and `plant=` to simulate a drifted device without compiling again.
  `Lowered.export_emulation()` writes the schedule, epochs and call manifest
  for a controller; `classical.emulation.emulate` runs a program in virtual
  time against real services, in process or over UDP. Requests and answers
  use the CUDA-Q real-time wire format; `classical.qec.Decoder` talks to a
  CUDA-QX real-time decoder. The cuDensityMat runtime keeps a classical
  handler table per run.
- **Calibrating while the QPU runs.** `classical.calibration` (adaptive T1,
  a frequency binary search, three-point decay and phase estimators,
  golden-section and Nelder-Mead steps), `classical.random_index` (an
  on-device random source), `classical.loop` (a parameter store whose values
  programs read at run time, and an estimate log) and `runtime.drift` (a
  drifting device with its truth trace and surrogates). Example
  `19_classical_calls.py` and the "Classical Calls at QPU Runtime" guide.

### Fixed

- A kernel that called a method of a global object (`fn.submit(...)`,
  `cfg.scaled(...)`) was compiled again on every call; it is now cached.

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
