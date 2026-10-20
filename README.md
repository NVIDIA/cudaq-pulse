# CUDA-Q pulse

CUDA-Q pulse (`cudaq-pulse`) is a pulse-level quantum programming research
package built on MLIR. It provides a Python kernel DSL, pulse and operator
dialects, compiler passes, and an experimental GPU execution path on NVIDIA
cuDensityMat.

> [!WARNING]
> CUDA-Q pulse is **research-preview software**. It is not production software
> and is not a product-supported CUDA-Q feature. Its Python APIs, MLIR
> dialects, runtime interfaces, build options, numerical behavior, and file
> layout may change incompatibly or be removed without notice. No stability,
> compatibility, performance, or production-readiness guarantee is provided.

`cudaq-pulse` does not depend on CUDA-Q. The wheel contains its own statically
linked MLIR, so `pip` is the only installer you need.

## Install

```bash
python -m venv .venv && source .venv/bin/activate

# Compiler only: compile kernels, inspect the MLIR, run cudaq-pulse-opt.
pip install cudaq-pulse

# Compiler plus GPU simulation (Linux), with the CUDA 12 or CUDA 13 libraries.
pip install "cudaq-pulse[cu12]"
pip install "cudaq-pulse[cu13]"

# Check what was installed and whether it works.
python -m cudaq_pulse.doctor
```

`[gpu]` is an alias for `[cu12]`. The `cu12` and `cu13` extras install into
the same directories, so install only one of them per environment.

| Platform | Python | What you get |
| --- | --- | --- |
| Linux x86_64 and aarch64, glibc 2.28+ | 3.12+ | Compiler; GPU execution with `[cu12]` or `[cu13]` |
| macOS 11+ arm64 | 3.12+ | Compiler only |

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
platform, which CUDA extra is installed, each GPU library with its path or
load error, and the driver version and device count. It exits non-zero if
GPU execution does not work (`--allow-no-gpu` accepts a working compiler-only
installation).

Without a usable GPU, everything except execution still works. Execution fails
with a specific error rather than degrading:

| Situation | Error raised by `pulse.evolve` / `CompiledKernel.run` |
| --- | --- |
| Wheel without GPU support (macOS, or a compiler-only source build) | `GpuSupportNotBuiltError` |
| GPU support built in, but no CUDA extra, both extras, or a library fails to load | `GpuRuntimeLoadError` |
| Libraries load, but no device, or the driver is too old | `GpuDeviceUnavailableError` |

All three derive from `cudaq_pulse.runtime.gpu.GpuRuntimeError`. The runtime
for CUDA 12 or 13 is chosen from the installed extra; set
`CUDAQ_PULSE_CUDA=12` or `CUDAQ_PULSE_CUDA=13` to choose it explicitly.

## Quick example

```python
import cudaq_pulse as pulse


@pulse.kernel
def rabi_oscillation(qubit):
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, gaussian(64, 0.5, 16.0), tone)


compiled_kernel = pulse.compile(
    rabi_oscillation,
    [pulse.qudit_ref()],
    qubit_freq_hz={0: 5.0e9},
)
print(compiled_kernel.mlir)
```

The compiler traces the Python kernel, builds Pulse dialect IR, applies the
selected transformations, and emits scheduled MLIR. The wheel also installs
`cudaq-pulse-opt`, the MLIR driver for the pulse dialects:

```bash
cudaq-pulse-opt --pulse-to-qop kernel.mlir
```

The [user guide](docs/index.rst) covers the kernel model, operations,
compilation, custom passes, and GPU execution. [`examples/`](examples) has
runnable programs, and [`examples/custom_passes/`](examples/custom_passes)
shows how to write passes in Python and as C++ plugins.

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

See [CONTRIBUTING.md](CONTRIBUTING.md). In short, `pip install .` from a
clone builds a wheel locally against the pinned, checksum-verified MLIR
toolchain, which CMake downloads on supported platforms (or set `MLIR_DIR` to
your own MLIR 22 build). Nothing else is built from source. PyPI carries
wheels only; the source distribution is attached to each GitHub release.

## Current scope and limitations

- The compiler, dialects, lowering experiments, and CPU unit tests are the
  primary research surface.
- GPU evolution models transmons in per-qubit rotating frames, as qubits or
  with any number of levels (including their anharmonicity). Target T1/T2
  data, XX couplings, residual ZZ terms, and calibrated drive-amplitude
  scaling are supported. Programs that measure and branch on the result are
  evolved between measurements as an exact ensemble over outcomes, on the GPU
  or with a NumPy reference (`pulse.simulate`).
- GPU evolution supports the `rk1`, `rk2`, `rk4`, `magnus`, and
  `crank_nicolson` integrators. Observable evaluation, neutral-atom models,
  unspecialized parameters, arbitrary Python waveform callbacks, and waveform
  algebra are rejected explicitly by the execution lowering.
- GPU numerical results are research regression coverage, with no production
  compatibility or accuracy commitment.

## Repository contents

- `core/frontend/` — Python DSL, compiler pipeline, targets, and
  runtime
- `core/mlir/` — Pulse, `QOp`, and CuDensityMat dialects, transformations,
  bindings, and `cudaq-pulse-opt`
- `core/runtime/` — cuDensityMat runtime library loaded by JIT-compiled code
- `cmake/` — dependency, Python staging, testing, and documentation modules
- `tools/` — MLIR toolchain build/fetch scripts and CI helpers
- `test/` — lit and `FileCheck` compiler regression tests
- `tests/` — pytest unit, workload, and GPU tests
- `examples/` — pulse programming examples
- `docs/` — user, API, and architecture documentation
- `benchmarks/` — compiler and simulation benchmarks

## License

[Apache License 2.0](LICENSE).
