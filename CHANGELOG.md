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
  Kernels call other kernels, loop with `range` (kept rolled while the count
  is symbolic), and branch on measurement results.
- **Operations.** Drive and readout channels per qudit; `drive`, `readout`,
  `wait` and `sync`; phase and frequency control (`shift_phase` follows the
  McKay et al. 2017 virtual-Z convention); Gaussian, square, DRAG (beta in
  clock cycles), cosine, tanh-ramp, Gaussian-square and sampled waveforms;
  waveform sums, differences, scaling and negation (exact) and products
  (sampled once per clock cycle).
- **Compilation.** Verification, virtual-Z folding and drive fusion by
  default; ASAP, ALAP and resource-constrained schedules
  (`MachineModel`); per-kernel schedule policies and schedule blocks.
  `Lowered.mlir()` returns the scheduled Pulse MLIR or the LLVM-dialect
  module, `Lowered.schedule` the timed events, and `Lowered.timeline()` a
  plot (with the `[viz]` extra).
- **Devices and targets.** `pulse.get_device` and `pulse.device` describe a
  machine as a target plus a qudit layout; devices compare by content and
  share compiled kernels. Shipped targets: a 17-qubit transmon device built
  from Krinner et al., Nature 605, 669 (2022), and Rydberg chain and square
  arrays (compile only).
- **Simulation.** `pulse.simulate` evolves a compiled call with T1, T2 or
  Hahn-echo dephasing, thermal populations, exchange couplings, residual ZZ
  and any number of transmon levels, either with a NumPy reference
  integrator (`backend="cpu"`) or on NVIDIA GPUs through cuDensityMat
  (`backend="gpu"`, Linux, experimental). Programs that measure and branch
  are evolved as an exact ensemble over outcomes.
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
