# Contributing to cudaq-pulse

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it provides Python, CMake, Ninja, and
  every Python tool below)
- A C++17 compiler: Clang or GCC on Linux, Xcode command-line tools on macOS
- For GPU work only: Linux, an NVIDIA GPU and driver, and the CUDA 12.9
  toolkit (or no toolkit at all; see [GPU builds](#gpu-builds))

You do not need CUDA-Q, an LLVM build, or Docker.

## Set up

```bash
git clone https://github.com/NVIDIA/cudaq-pulse.git
cd cudaq-pulse

uv sync                                  # .venv with build, test, lint, docs tools
python3 tools/fetch-mlir-toolchain.py    # pinned MLIR 22.1.8 into .cache/mlir/
```

`uv sync` installs tooling only (`[tool.uv] package = false`); the package is
built by CMake, below. The fetch script downloads the toolchain archive for
your platform, verifies its SHA-256 against `tools/mlir-toolchain.lock.json`,
and unpacks it. It refuses archives it cannot verify. Configuring runs it
for you if you skip this step; on a platform without a pinned archive, pass
`-DMLIR_DIR=<prefix>/lib/cmake/mlir` for your own MLIR 22.1.8 build.

## Build and test

```bash
source .venv/bin/activate

cmake -S . -B build/dev -G Ninja \
  -DCUDAQ_PULSE_BUILD_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/dev --target check-pulse
```

`check-pulse` builds everything and runs the lit/FileCheck tests in `test/`
and the non-GPU pytest suite in `tests/`. The staged package lives in
`build/dev/python`; to use it directly:

```bash
export PYTHONPATH="$PWD/build/dev/python"
export PATH="$PWD/build/dev/bin:$PATH"     # cudaq-pulse-opt
python examples/01_single_qubit_rabi.py
```

Python sources are staged as symlinks, so edits to `core/frontend/` take
effect without rebuilding; C++ changes need `cmake --build build/dev`.

The property tests use hypothesis profiles (`tests/property_settings.py`),
chosen with `HYPOTHESIS_PROFILE`: `ci`, the default, is derandomized so every
run checks the same examples; `dev` draws new examples; `nightly` draws ten
times as many. A failure prints a `@reproduce_failure` line; a nightly
failure also gives its seed in the job summary, replayed with
`HYPOTHESIS_PROFILE=nightly pytest <test> --hypothesis-seed=<seed>`.

## Writing passes

Compiler passes are C++ MLIR passes in `core/mlir/transforms/`. To try one
without changing the tree, build it as a plugin and load it with
`cudaq-pulse-opt --load-pass-plugin`; checks over a compiled kernel's
schedule can be plain Python. [`examples/custom_passes/`](examples/custom_passes)
has a minimal example of each, and
[docs/user_guide/custom_passes.rst](docs/user_guide/custom_passes.rst)
walks through them. `check-pulse` builds the example plugin and runs it in
`test/Plugins/`.

## Lint and type-check

```bash
pre-commit install                       # once; runs the hooks on each commit
pre-commit run --all-files

MYPYPATH=build/dev/python mypy -p cudaq_pulse
```

The hooks run the ruff linter and formatter, clang-format, whitespace and
YAML/TOML checks, and a license-header check
(`tools/check-license-headers.py`). New source files need the NVIDIA
Apache-2.0 header used by their neighbors.

mypy runs in strict mode against the staged package, because it needs the
type stub that the build generates for the native extension.

## Benchmarks

```bash
PYTHONPATH=build/dev/python python benchmarks/compiler/compile_time.py \
  --baseline benchmarks/baselines/compile_time.json
```

CI fails if any workload's op count differs from the baseline. If a change
alters op counts on purpose, regenerate it with `--write-baseline` and
explain the change in the pull request. Timing is only gated when you pass
`--gate FACTOR`; it is too noisy on shared runners to gate in CI.

`benchmarks/compiler/mlir_pipeline_bench.py` breaks compile time down by
stage and by pass and reports IR sizes; `benchmarks/simulation/qec_scaling.py`
shows how compile time grows with surface-code distance. CI runs both at a
small size to keep them working.

## GPU builds

GPU support is off by default and, when requested, mandatory: configuration
fails rather than silently producing a compiler-only build.

```bash
uv pip install -r pyproject.toml --extra cu12
SITE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

cmake -S . -B build/gpu -G Ninja \
  -DCUDAQ_PULSE_BUILD_TESTS=ON \
  -DCUDAQ_PULSE_ENABLE_GPU=ON \
  -DCUDAQ_PULSE_CUDA12_ROOT="/usr/local/cuda-12.9;$SITE" \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build/gpu --target check-pulse
cmake --build build/gpu --target check-pulse-gpu
```

The first command installs the `[cu12]` extra's dependencies (not the package
itself) into `.venv`: the same libraries `pip install "cudaq-pulse[cu12]"`
gives users, including the cuDensityMat headers the build compiles against.
`check-pulse-gpu` runs the cuDensityMat linkage smoke test and the numerical
GPU suite, and fails if no device is usable.

The build produces one runtime library per CUDA major,
`libcudm-runtime-cu12.so` and `libcudm-runtime-cu13.so`, from
`CUDAQ_PULSE_CUDA12_ROOT` and `CUDAQ_PULSE_CUDA13_ROOT`. Set either or both.
Each is a list of directories that together provide that major's CUDA runtime
(`cuda_runtime_api.h`, `crt/host_defines.h`, `libcudart.so.NN`) and
cuDensityMat (`cudensitymat.h`, `libcudensitymat.so.0`), as toolkit-style
`include/` and `lib/` prefixes or as site-packages trees. Above, the toolkit
supplies the CUDA headers and the `[cu12]` wheels supply cuDensityMat.

Without a toolkit, `crt/host_defines.h` comes from one more wheel
(`nvidia-cuda-nvcc-cu12` for CUDA 12, `nvidia-cuda-crt` for CUDA 13), and the
site-packages directory alone is the root; `tools/ci/manylinux-before-all.sh`
does this for the release wheels. For CUDA 13, use `--extra cu13` in a
separate environment (the two extras cannot share one) and pass
`-DCUDAQ_PULSE_CUDA13_ROOT`. At run time the loader picks the runtime matching
the installed extra; `CUDAQ_PULSE_CUDA=12|13` forces one.

## Build a wheel

```bash
pip install .            # or: uv build --wheel
```

This uses the same fetched toolchain. Wheels are compiler-only unless
`CUDAQ_PULSE_ENABLE_GPU=ON` is set in the environment (as the release build
does on Linux; see `[tool.cibuildwheel]` in `pyproject.toml`).

## The MLIR toolchain

`tools/build-mlir-toolchain.sh` builds the slim, static LLVM/MLIR 22.1.8
distribution that cudaq-pulse links against. The `mlir-toolchain` workflow
runs it for every platform. Pushing a tag named after `release_tag` in
`tools/mlir-toolchain.lock.json` (for example `mlir-toolchain-22.1.8`) builds
the archives, uploads them to that release, and commits their checksums to
the lock file on `staging`. A manual run with `publish` enabled uploads too,
but leaves committing the printed lock file to you.

## Continuous integration

Workflows use third-party actions pinned by full commit SHA, with the release
in a trailing comment, and container images pinned by digest;
`tools/check-workflow-pins.py` (a pre-commit hook) rejects tags, and
Dependabot proposes weekly updates for actions and `uv.lock`.

- `ci.yml`: ruff and pre-commit; build, lit, pytest, mypy, and the benchmark
  baseline on Linux x86_64, Linux aarch64, and macOS. A manually dispatched
  run with `gpu` enabled also builds with GPU support on the runner named by
  the `GPU_RUNNER` repository variable.
- `wheels.yml`: cibuildwheel for `cp312-abi3` (3.12 and newer) on
  manylinux_2_28 (x86_64, aarch64, with the CUDA 12 and CUDA 13 runtimes) and
  macOS arm64 (compiler-only), plus an sdist. It then installs
  the wheels in clean containers without a CUDA toolkit. A manually
  dispatched run with `gpu` enabled also runs the GPU tests from the wheels on
  the `GPU_RUNNER` runner.
- `docs.yml`: Sphinx with warnings as errors; deploys `main` to GitHub Pages.
- `property-nightly.yml`: the property tests with random examples on
  staging every night (and on manual dispatch), with the seed in the job
  summary and the example database kept in the Actions cache. GitHub runs
  scheduled workflows from the default branch, so the nightly schedule takes
  effect once this file is on `main`; until then, dispatch it manually.

## Releases

Work lands on `staging`, where every workflow runs but nothing is published.

1. Tag a release candidate on `staging`, for example `v0.1.0rc1`. Tags
   containing `rc` publish the wheels to TestPyPI; check them on a clean
   machine with
   `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "cudaq-pulse[cu12]"`.
2. Squash `staging` into one commit on `main`, with the `staging` workflow
   triggers removed and the toolchain checksums in the lock file.
3. Tag `v0.1.0` on `main`. Other `v*` tags publish the wheels to PyPI.

Both publish jobs use trusted publishing (environments `testpypi` and `pypi`)
and upload wheels only. Every `v*` tag also gets a GitHub release with the
wheels and the sdist attached; the sdist is not uploaded to PyPI, so
unsupported platforms get "no matching distribution" instead of a long
source build.
