# Contributing to cudaq-pulse

Everyone taking part in the project follows the
[Code of Conduct](CODE_OF_CONDUCT.md). [SUPPORT.md](SUPPORT.md) says where to
ask questions and report bugs; report security vulnerabilities as described
in [SECURITY.md](SECURITY.md), not in an issue.

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
Dependabot proposes weekly updates for actions and `uv.lock`, each a week
after its release. [zizmor](https://docs.zizmor.sh/) audits the workflows
(a pre-commit hook, and online in CI), and every job sets
`timeout-minutes` (`tests/test_release_tools.py` checks it).

- `ci.yml`: ruff and pre-commit; build, lit, pytest, mypy, and the benchmark
  baseline on Linux x86_64 (Python 3.12 to 3.15), Linux aarch64 (3.12 and
  3.15), and macOS (3.12), and the lit and pytest suites on free-threaded
  CPython 3.14t with the GIL off (configured with
  `-DCUDAQ_PULSE_FREE_THREADED=ON`, which declares the extension safe
  without the GIL; release wheels are not free-threaded yet). The `gpu` job
  builds with GPU support and runs the GPU tests on the runner named by the
  `GPU_RUNNER` repository variable: nightly once the variable is set, and on
  a manual run with `gpu` enabled.
- `wheels.yml`: cibuildwheel for `cp312-abi3` (3.12 and newer) on
  manylinux_2_28 (x86_64, aarch64, with the CUDA 12 and CUDA 13 runtimes) and
  macOS arm64 (compiler-only), plus an sdist. It then installs
  the wheels in clean containers without a CUDA toolkit. `gpu-install-test`
  runs the GPU tests from the wheels on the `GPU_RUNNER` runner, on every
  release tag and on a manual run with `gpu` enabled, and gates the release
  (see [Releases](#releases)).

`GPU_RUNNER` is a runner label (`linux-gpu`) or a JSON list of labels
(`["self-hosted", "linux", "x64", "gpu"]`). Only tag pushes, the nightly
schedule and manual runs reach it, never pull requests, so a self-hosted
runner does not run code from forks; it should still be ephemeral and
isolated. While it is unset, every GPU job is skipped and the CPU jobs are
unaffected.
- `docs.yml`: Sphinx with warnings as errors; deploys `main` to GitHub Pages.
- `property-nightly.yml`: the property tests with random examples on
  staging every night (and on manual dispatch), with the seed in the job
  summary and the example database kept in the Actions cache. GitHub runs
  scheduled workflows from the default branch, so the nightly schedule takes
  effect once this file is on `main`; until then, dispatch it manually.

## Versioning and stability

Versions follow [PEP 440](https://peps.python.org/pep-0440/) in the shape of
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`, with
pre-releases `X.Y.Z{a,b,rc}N` and post releases `X.Y.Z.postN`.

- **What is covered.** The public API is the names in the `__all__` of
  `cudaq_pulse` and of its documented submodules, their documented behavior,
  the documented environment variables, and the `cudaq-pulse-opt` and
  `python -m cudaq_pulse.doctor` command lines. Modules and names that start
  with an underscore, the native extension, the C++ headers, and the textual
  form of the MLIR dialects are internal and may change in any release.
  Numerical results are covered by the tests' tolerances, not bit for bit.
- **While the major version is 0** (the research preview), a minor release
  may break the public API; a patch release fixes bugs and does not break it
  on purpose. From 1.0, only a major release breaks it.
- **Deprecation.** Where it is practical, a name or behavior that is going
  away warns with `DeprecationWarning` for at least one minor release before
  it is removed. Every break and deprecation is listed under **Changed**,
  **Deprecated** or **Removed** in `CHANGELOG.md`.
- **Python versions.** CPython 3.12 and newer. A new CPython is supported
  from its first release, and dropping one happens only in a minor release.
- **Post releases** (`X.Y.Z.postN`) fix packaging or metadata only, never
  code; they need their own dated section in `CHANGELOG.md`.

### Withdrawing a broken release

Release assets are write-once and tags are never moved or deleted, so a
broken release is withdrawn, not replaced:

1. Open an issue that describes the problem and who is affected.
2. Put a warning at the top of the GitHub release's notes that links the
   issue and names the release to use instead. Keep the assets: deleting
   them breaks pinned installs and reproducibility.
3. Release the fix as a new version: `X.Y.(Z+1)` for a code fix, or
   `X.Y.Z.postN` for a packaging-only fix, with a `CHANGELOG.md` entry that
   names the withdrawn release.
4. Once releases are on PyPI, also yank the broken version there
   ([PEP 592](https://peps.python.org/pep-0592/)) with the issue link as
   the reason. Yanking hides it from new resolutions but keeps exact pins
   working; never delete files from PyPI.

A vulnerability is handled through [SECURITY.md](SECURITY.md) first, and the
withdrawal follows the coordinated disclosure.

## Releases

Work lands on `staging`, where every workflow runs but nothing is published.
Releases are GitHub releases only for now: nothing is uploaded to PyPI or
TestPyPI.

1. Tag a release candidate, for example `v0.1.0rc1`. `wheels.yml` builds and
   tests the wheels and the sdist and publishes them as a GitHub pre-release.
2. Squash `staging` into one commit on `main`, with the `staging` workflow
   triggers removed and the toolchain checksums in the lock file.
3. Date the version's section in `CHANGELOG.md` (`## [0.1.0] - YYYY-MM-DD`)
   and leave an empty `## [Unreleased]` section above it.
4. Tag `v0.1.0` on `main`.

Release notes come from `CHANGELOG.md`, not from the commit history:
`tools/release-notes.py` prints the section for the version, and a
pre-release uses the section of the release it precedes. Add user-visible
changes under `## [Unreleased]` in the pull request that makes them.

Before anything is released, the `release-meta` job checks the tag with
`tools/release-meta.py`: it must name a version `X.Y.Z`, `X.Y.Z.postN` or
`X.Y.Z{a,b,rc}N` that the setuptools-scm `tag_regex` accepts, a final
version must be on `main`, `CHANGELOG.md` must have a dated, non-empty
section for it, and the release must not exist yet. The release is created as a
draft, every built file's version is checked against the tag and gets a
signed build-provenance attestation (check one with
`gh attestation verify <file> --repo NVIDIA/cudaq-pulse`), the assets are
uploaded once (never replaced), and only then is the release published. The
MLIR toolchain assets are write-once too. The end of `wheels.yml` records how
PyPI and TestPyPI uploads will be re-enabled with trusted publishing.

### GPU verification

No release is published without GPU verification of the tagged commit; the
`gpu-gate` job in `wheels.yml` enforces it, and it fails closed:

- **With a GPU runner** (`GPU_RUNNER` set), the tag's run executes
  `gpu-install-test` and publishes only if it passes. A failed GPU run
  cannot be overridden: fix the problem and tag a new version.
- **Without a GPU runner**, the tag's run builds and tests everything else
  and then stops at `gpu-gate`. To release it:
  1. Run the GPU test suite on the tagged commit on a machine with an
     NVIDIA GPU: build it as in [GPU builds](#gpu-builds) and run
     `check-pulse` and `check-pulse-gpu`, or download the tag run's
     `wheels-linux-x86_64` artifact into `wheelhouse/` in a checkout of the
     tag and run `bash tools/ci/install-test.sh gpu`. Every test must pass,
     with no GPU test skipped.
  2. Run the `wheels` workflow manually on the tag (*Run workflow*, *Use
     workflow from*: the tag) with `gpu-verified-sha` set to the full commit
     SHA the tag points to (`git rev-parse vX.Y.Z^{commit}`). This is the
     maintainer's attestation that step 1 passed on that commit; the run's
     summary records who made it.

  The attestation counts only on a manual run, only when it names the exact
  commit being released, and only when `gpu-install-test` did not run.
