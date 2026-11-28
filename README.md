<p align="center">
  <img src="https://raw.githubusercontent.com/NVIDIA/cudaq-pulse/main/docs/_static/cudaq-pulse-banner.svg"
       alt="CUDA-Q Pulse: a Python pulse kernel beside a timeline of drive, readout and GPU lanes. A readout result goes to a GPU decoder over NVQLink within a latency budget, and the answer conditions the next drive pulse." width="100%">
</p>

<p align="center">
  <a href="https://github.com/NVIDIA/cudaq-pulse/actions/workflows/ci.yml"><img src="https://github.com/NVIDIA/cudaq-pulse/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://nvidia.github.io/cudaq-pulse/"><img src="https://github.com/NVIDIA/cudaq-pulse/actions/workflows/docs.yml/badge.svg?branch=main" alt="Docs"></a>
  <a href="https://github.com/NVIDIA/cudaq-pulse/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"></a>
</p>

# CUDA-Q Pulse

## About

`cudaq-pulse` is a Python library and compiler for programming quantum
processors at the pulse level: the drive and readout pulses, their timing,
and the real-time classical decisions between them. It is for people who
build and run QPUs and their control systems: hardware and control
engineers, calibration and quantum error correction researchers, and
controller vendors who want one open format to compile to. It belongs to the
CUDA-Q family and shares its goal of one programming model across QPUs, but
works at the level below CUDA-Q's gates and kernels and installs on its own,
without CUDA-Q. The [documentation](https://nvidia.github.io/cudaq-pulse/)
covers the tutorials, user guide and API reference.

## Highlights and Key Features

`cudaq-pulse` lets developers write pulses, measurements and the real-time
classical logic between them as one Python program. That program compiles to
an open MLIR pulse dialect that any controller can target.

- **Real-time control:** measure-and-react, feedback and calls to classical
  services (such as decoders) are part of the program. The compiler checks
  each answer arrives before the pulse that depends on it, given the
  controller's latency.
- **QEC at scale:** syndrome-extraction cycles compile in milliseconds; a
  distance-31 surface code cycle compiles in 263 ms.
- **Composable QPUs:** a QPU is described as data: its qudits, lines,
  controllers and the links between them. One program can span several
  controllers, and it is checked against that description when it compiles.
- **Hardware portability:** the same compiled program exports to controller
  backends, so Python front ends are decoupled from each underlying control
  interface.

Underneath are a Python kernel DSL, the Pulse and Classical MLIR dialects, and
their compiler passes. `cudaq-pulse` does not depend on CUDA-Q. The wheel
contains its own statically linked MLIR, so `pip` is the only installer you
need.

`cudaq-pulse` is an experimental, community-driven research project under
active development. We're releasing it now because our partners want a
stable, universal pulse dialect that separates Python DSLs from quantum
control interfaces.

Documentation: <https://nvidia.github.io/cudaq-pulse/>

## Project Status

> [!NOTE]
> The pulse layer is where much of the open work in quantum control happens,
> and we want to build it in the open with the people who use it. Expect
> bugs, incomplete features, and incompatible changes to the Python APIs,
> MLIR dialects and file formats between releases. It is not a supported CUDA-Q product feature. We invite
> you to try it with your own quantum systems and help shape its direction.

## Install

Install from PyPI; `pip` picks the wheel for your platform:

```bash
python -m venv .venv && source .venv/bin/activate

# Compiler and CPU simulation: compile kernels, inspect the MLIR, simulate
# with the NumPy reference, run cudaq-pulse-opt.
pip install cudaq-pulse
```

For GPU simulation on Linux, also install one CUDA extra, never both:

```bash
# CUDA 12 libraries: NVIDIA driver 525.60.13 or newer.
pip install "cudaq-pulse[cu12]"

# CUDA 13 libraries instead: NVIDIA driver 580 or newer.
# pip install "cudaq-pulse[cu13]"
```

Then check what was installed and whether it works:

```bash
python -m cudaq_pulse.doctor
```

`cudaq-pulse-opt`, the MLIR driver, comes along automatically: it is in
`cudaq-pulse-tools`, a wheel of its own, one per platform for every Python,
which `cudaq-pulse` requires at the same version. Each release on the
[GitHub releases page](https://github.com/NVIDIA/cudaq-pulse/releases) also
carries the wheels and the source distribution.

`[gpu]` is an alias for `[cu12]`. The `cu12` and `cu13` extras install into
the same directories, so install only one of them per environment.

| Platform | Python | What you get |
| --- | --- | --- |
| Linux x86_64 and aarch64, glibc 2.28+ | 3.12+ | Compiler and CPU simulation; GPU simulation with `[cu12]` or `[cu13]` |
| macOS 11+ arm64 | 3.12+ | Compiler and CPU simulation |

GPU execution needs an NVIDIA GPU (compute capability 7.0 or newer) and a
driver for the CUDA major you installed:

| Extra | Minimum Linux driver |
| --- | --- |
| `[cu12]` (or `[gpu]`) | 525.60.13 |
| `[cu13]` | 580 |

The driver is the only system requirement. The extras install cuDensityMat and
the CUDA runtime libraries from PyPI, so no CUDA toolkit, cuQuantum SDK, or
container is required.

### Troubleshooting

Run `python -m cudaq_pulse.doctor` first. It prints the version, the wheel
platform, whether CPU simulation works, which CUDA extra is installed, each
GPU library with its path or load error, and the driver version and device
count. Without a CUDA extra it passes on CPU simulation alone and says how
to enable GPU simulation. With one installed, it exits non-zero if GPU
simulation does not work: a driver too old for the extra, both extras
installed, or a library that does not load.

Without a usable GPU, everything except GPU simulation still works, including
CPU simulation, the default. GPU simulation (`backend="gpu"`) fails with a
specific error rather than degrading:

| Situation | Error raised by GPU simulation |
| --- | --- |
| Wheel without GPU support (macOS, or a compiler-only source build) | `GpuSupportNotBuiltError` |
| GPU support built in, but no CUDA extra, both extras, or a library fails to load | `GpuRuntimeLoadError` |
| Libraries load, but no device, or the driver is too old | `GpuDeviceUnavailableError` |

All three derive from `cudaq_pulse.runtime.GpuRuntimeError`. The runtime
for CUDA 12 or 13 is chosen from the installed extra; set
`CUDAQ_PULSE_CUDA=12` or `CUDAQ_PULSE_CUDA=13` to choose it explicitly.

## Quick example

```python
import cudaq_pulse as pulse


@pulse.kernel
def rabi_oscillation(qubit, amplitude: float, duration, sigma):
    drive_line = get_drive_line(qubit)
    readout_line = get_readout_line(qubit)
    drive(drive_line, gaussian(duration, amplitude, sigma))
    sync(drive_line, readout_line)
    return readout(readout_line, readout_line.get_calibration(pulse.gates.READOUT))


qs = pulse.quantum_system("transmon_surface17")
qubit = qs.qudits[0]
duration = 32 * pulse.ns
sigma = 8 * pulse.ns

# A call compiles the kernel and simulates one shot on the CPU, no GPU needed.
outcome = rabi_oscillation(qubit, 0.5, duration, sigma)
print(outcome)  # the readout's outcome: 1 in about 96 shots of 100
print(rabi_oscillation.lower(qubit, 0.5, duration, sigma).mlir())  # the scheduled Pulse MLIR
```

Calling the kernel compiles it for the quantum system of its qudit handles,
with frequencies from the quantum system, simulates one shot and returns what
the kernel returns: here the outcome of its readout. This pulse leaves the
qubit almost fully excited, so most shots read 1. `kernel.lower(*args)`
compiles the call without running it, and `pulse.compile(kernel)` compiles
ahead of the call. The pulse's values are arguments of the kernel: the
kernel is compiled for its duration and sigma, and a `float` parameter stays
symbolic, so a sweep over the amplitude compiles once. The compiler
traces the Python kernel, builds Pulse dialect IR, applies the selected
transformations, and emits scheduled MLIR. The wheel also installs
`cudaq-pulse-opt`, the MLIR driver for the pulse dialects:

```bash
cudaq-pulse-opt --pulse-to-qop kernel.mlir
```

The [user guide](https://nvidia.github.io/cudaq-pulse/) covers the kernel
model, operations, compilation, custom passes, and GPU execution.
[`examples/`](https://github.com/NVIDIA/cudaq-pulse/tree/main/examples) has runnable programs, and
[`examples/custom_passes/`](https://github.com/NVIDIA/cudaq-pulse/tree/main/examples/custom_passes) shows how to write
passes in Python and as C++ plugins.

[`challenges/`](https://github.com/NVIDIA/cudaq-pulse/tree/main/challenges) has
hands-on problems to solve and score locally.

Kernels can also call classical functions while the QPU runs
(`@pulse.classical`, `fn.submit(...).result()`): decoders, estimators or
calibration updates whose answers the same shot uses, scheduled with their
latency budget and simulated or emulated in virtual time. A compiled kernel
exports to an out-of-tree controller backend with
`ScheduledKernel.export_to_controller()`, and `cudaq_pulse.backends` finds the backends
installed beside cudaq-pulse; see the "Classical Calls at QPU Runtime" and
"Writing a Controller Backend" guides.

## Run the test suite against your installation

The tests live in this repository, not in the wheel. They import whichever
`cudaq_pulse` is installed, so they check exactly what you installed:

```bash
git clone https://github.com/NVIDIA/cudaq-pulse.git
cd cudaq-pulse
pip install --group test        # pytest, hypothesis, lit (pip 25.1+)

pytest tests -m "not gpu"       # any installation
pytest tests                    # GPU installations: also runs the GPU suite
```

Selecting the GPU tests on an installation that cannot run them stops pytest
immediately with the reason, instead of skipping them.

## Build from source

See [CONTRIBUTING.md](https://github.com/NVIDIA/cudaq-pulse/blob/main/CONTRIBUTING.md). In short,
`pip install ./packaging/cudaq-pulse-tools .` from a clone builds the two
wheels, `cudaq-pulse-tools` and `cudaq-pulse`, locally against the pinned, checksum-verified MLIR
toolchain, which CMake downloads on supported platforms (or set `MLIR_DIR` to
your own MLIR 22 build). Nothing else is built from source. Each GitHub
release carries the wheels and the source distribution.

## Notice and disclaimer

NOTICE AND DISCLAIMER: This software automatically retrieves, accesses or interacts with external materials. Those retrieved materials are not distributed with this software and are governed solely by separate terms, conditions and licenses. You are solely responsible for finding, reviewing and complying with all applicable terms, conditions, and licenses, and for verifying the security, integrity and suitability of any retrieved materials for your specific use case. This software is provided "AS IS", without warranty of any kind. The author makes no representations or warranties regarding any retrieved materials, and assumes no liability for any losses, damages, liabilities or legal consequences from your use or inability to use this software or any retrieved materials. Use this software and the retrieved materials at your own risk.

What installing and building retrieve:

- The optional GPU extras install NVIDIA packages from PyPI under NVIDIA's
  own licenses. `[cu12]` (and `[gpu]`, the same set) installs
  [cudensitymat-cu12](https://pypi.org/project/cudensitymat-cu12/),
  [nvidia-cuda-runtime-cu12](https://pypi.org/project/nvidia-cuda-runtime-cu12/),
  [nvidia-cublas-cu12](https://pypi.org/project/nvidia-cublas-cu12/),
  [nvidia-cusolver-cu12](https://pypi.org/project/nvidia-cusolver-cu12/),
  [nvidia-cusparse-cu12](https://pypi.org/project/nvidia-cusparse-cu12/),
  [nvidia-curand-cu12](https://pypi.org/project/nvidia-curand-cu12/) and
  [nvidia-nvjitlink-cu12](https://pypi.org/project/nvidia-nvjitlink-cu12/);
  `[cu13]` installs [cudensitymat-cu13](https://pypi.org/project/cudensitymat-cu13/)
  and the CUDA runtime and libraries through
  [cuda-toolkit](https://pypi.org/project/cuda-toolkit/).
- Building from source downloads the pinned LLVM/MLIR 22.1.8 toolchain
  archive (Apache-2.0 WITH LLVM-exception) from this repository's
  [`mlir-toolchain-22.1.8` release](https://github.com/NVIDIA/cudaq-pulse/releases/tag/mlir-toolchain-22.1.8).
- pip and uv install the Python dependencies listed in `pyproject.toml`.

## Contributing and support

[CONTRIBUTING.md](https://github.com/NVIDIA/cudaq-pulse/blob/main/CONTRIBUTING.md) covers building, testing and the
versioning policy, [SUPPORT.md](https://github.com/NVIDIA/cudaq-pulse/blob/main/SUPPORT.md) where to ask for help, and
[CHANGELOG.md](https://github.com/NVIDIA/cudaq-pulse/blob/main/CHANGELOG.md) what changed in each release. Everyone taking
part follows the [Code of Conduct](https://github.com/NVIDIA/cudaq-pulse/blob/main/CODE_OF_CONDUCT.md).

## License

cudaq-pulse is licensed under the [Apache License 2.0](https://github.com/NVIDIA/cudaq-pulse/blob/main/LICENSE). The
wheels also contain third-party software (LLVM and MLIR, nanobind,
robin-map) under the licenses listed, with their texts, in
[THIRD_PARTY_NOTICES](https://github.com/NVIDIA/cudaq-pulse/blob/main/THIRD_PARTY_NOTICES); the texts are also in
[LICENSES/](https://github.com/NVIDIA/cudaq-pulse/tree/main/LICENSES).

## Security

Report security vulnerabilities to NVIDIA as described in
[SECURITY.md](https://github.com/NVIDIA/cudaq-pulse/blob/main/SECURITY.md), not through GitHub issues.
