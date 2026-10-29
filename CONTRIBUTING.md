# Contributing to cudaq-pulse

Everyone taking part in the project follows the
[Code of Conduct](CODE_OF_CONDUCT.md). [SUPPORT.md](SUPPORT.md) says where to
ask questions and report bugs; report security vulnerabilities as described
in [SECURITY.md](SECURITY.md), not in an issue.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (it provides Python, CMake, Ninja, and
  every Python tool below)
- A C++20 compiler: GCC or Clang on Linux, the Xcode command-line tools on
  macOS. CI builds with GCC 13 (Ubuntu 24.04), the manylinux_2_28 toolchain
  and the Apple Clang of macOS 15
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

cmake --preset dev                         # build/dev, with the tests
cmake --build --preset check               # build, then run check-pulse
```

`CMakePresets.json` has a configure preset for each build in this guide:
`dev`, `docs`, `gpu`, `free-threaded`, `asan` and `tsan`, each in
`build/<preset>` and using `.venv/bin/python`; `cmake --list-presets` lists
them. Add `-D` options after `--preset`, and keep personal presets in an
untracked `CMakeUserPresets.json`.

`check-pulse` builds everything and runs the lit/FileCheck tests in `test/`,
the non-GPU pytest suite in `tests/`, and `check-pulse-runtime`: the
cuDensityMat runtime built against a host fake of cuDensityMat
(`tests/runtime/fake_cudensitymat/`), so its callback table and integrators
are tested without a GPU. The staged package lives in
`build/dev/python`; to use it directly:

```bash
export PYTHONPATH="$PWD/build/dev/python"
export PATH="$PWD/build/dev/bin:$PATH"     # cudaq-pulse-opt
python examples/01_single_qubit_rabi.py
```

Python sources are staged as symlinks, so edits to `core/frontend/` take
effect without rebuilding; C++ changes need `cmake --build build/dev`, and a
new or renamed Python file needs a configure (`cmake build/dev`).

### Run a subset

`test/` holds the lit and FileCheck tests of the dialects and passes;
`tests/` holds the pytest suite. `check-pulse` runs `check-pulse-mlir` (lit),
`check-pulse-python` (pytest without the GPU tests) and
`check-pulse-runtime` (the cuDensityMat host test); each is a target of its
own. For a faster loop, run the tools directly:

```bash
export PYTHONPATH="$PWD/build/dev/python"
pytest tests/test_compile.py -k metrics    # one file, filtered by name
pytest tests -m "not gpu"                  # what check-pulse-python runs
pytest tests/physics -m physics --run-nightly   # the physics suite, slow checks too
pytest tests/test_perf_guard.py --run-benchmarks
lit -sv build/dev/test/Dialect/Pulse        # one lit directory
```

The markers are `gpu` (needs a GPU runtime; selecting one without a GPU
fails instead of skipping), `physics` (everything in `tests/physics`),
`nightly` (slow physics checks, skipped without `--run-nightly`),
`benchmark` (timing guards, skipped without `--run-benchmarks`) and
`scaling` (benchmarks that compare two sizes in one run). `tests/conftest.py`
sets `CUDAQ_PULSE_TARGET=ladder:8192` and `CUDAQ_PULSE_UNROLL_THRESHOLD=0`
for the whole session, so a kernel's loops stay rolled unless a test says
otherwise.

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

## Documentation

```bash
cmake --preset docs
cmake --build --preset docs                # pulse-docs; HTML in build/docs/docs/html
```

The dialect reference pages under `architecture/dialects/` are generated by
`mlir-tblgen -gen-dialect-doc` from the ODS files, so `pulse-docs` builds the
site from a copy of `docs/` with those pages added; running `sphinx-build` on
`docs/` directly leaves them out. Describe a new op with `description` in its
`.td` definition.

## Sanitizers

`-DCUDAQ_PULSE_SANITIZERS="address;undefined"` (or `thread`) instruments
cudaq-pulse's own C++; the prebuilt MLIR toolchain stays uninstrumented. The
`sanitizers` workflow builds both variants on Linux. The address and
undefined-behavior build runs `check-pulse-mlir`, `check-pulse-runtime` and
the Python suite; the thread build runs `check-pulse-runtime`, whose GPU
runtime host test evolves runs concurrently. Under address and undefined
behavior, `check-pulse-mlir` and `check-pulse-runtime` block a merge. The
Python suite there and the thread build are informational until the MLIR
toolchain is built with the sanitizers too. For the Python suite, preload the
runtimes into the uninstrumented interpreter:

```bash
LD_PRELOAD="$(gcc -print-file-name=libasan.so):$(gcc -print-file-name=libubsan.so)" \
ASAN_OPTIONS=detect_leaks=0 PYTHONPATH=build/san/python python -m pytest tests -m "not gpu"
```

## Benchmarks

### The perf gate

The `perf-gate` CI job blocks a pull request on measurements that repeat
exactly from run to run, not on wall-clock time. It runs the cases in
`benchmarks/perf_gate/cases.py`: compiles of small and large kernels,
including kernels over lists of qudits and kernels with classical calls,
plus the first compile in a process, binding and specializing new values,
`submit()`, the controller and emulator exports, and the setup of a
simulation. It compares each case with `benchmarks/baselines/perf_gate.json`
on two measures:

- **Instructions** executed per repetition, counted by Valgrind's Callgrind
  on Linux x86-64. A case fails at more than 0.4% above its baseline. It
  also fails at more than 1% below, so that a speedup is recorded in the
  baseline instead of hiding a later regression.
- **Program sizes**: op count, packed program bytes and export bytes. Any
  change fails. These are the same on every platform, and
  `tests/test_perf_gate.py` checks them in the regular test suite too.

```bash
PYTHONPATH=build/dev/python python tools/perf-gate.py      # compare
PYTHONPATH=build/dev/python python tools/perf-gate.py --update
```

Without Valgrind (macOS, or Linux on another architecture), the tool
compares and updates only the program sizes. The instruction counts depend
on the toolchain, so they are taken in CI's environment, where the image,
its packages, uv and Python are pinned. If a change is intended, the job
summary of the failed run prints the one command that makes that run's
measurements the baseline:

```bash
gh run download <run-id> -R <owner>/<repo> -n perf-gate-baseline -D benchmarks/baselines
```

Commit the updated `perf_gate.json` and say why in the pull request.

**Bootstrap mode.** A baseline that says `"bootstrap": true` and holds no
instruction counts is in bootstrap mode. The job still measures every case
twice and reports the counts and their spread, but they don't fail the run;
program sizes block as always. The summary says so in a banner, together
with the command above. Taking a run's measurements as the baseline removes
the field, and `"bootstrap": true` next to any instruction count is an
error (in the job and in `tests/test_perf_gate.py`), so the gate can't stay
in bootstrap mode once it has counts. Bootstrap mode also can't reach a
release: the field fails the perf gate on `main`, and `tools/release-meta.py`
refuses a release tag while it is there. Without the field, a case with no
count fails. If the two runs of a bootstrap differ at all for some case, the
summary flags it: give that case its own `"tolerance"` before committing
the baseline. A new case added later needs its count taken the same way,
from the run that fails on it.

**Counts that can't be right fail in every mode**, bootstrap included: a
case without a count, a count of zero, a compile under 100,000 instructions
per repetition, or a heavier case (a large compile, the first compile)
measuring less than twice a lighter one. Such a run's counts are left out of
its `perf-gate-baseline` artifact. The `perf-gate-callgrind-dumps` artifact
holds each run's raw Callgrind dumps and Valgrind log for inspection
(`callgrind_annotate` reads them); locally, `--dumps DIR` keeps them.

The
summary lists every case with its baseline, its count in this run, the
change, and the spread between two identical runs made in the same job. A
case whose own numbers move more than the default allows can carry a
`"tolerance"` (a fraction) in the baseline. Changing a pin in the job (image
digest, `UBUNTU_SNAPSHOT`, uv or Python version, MLIR toolchain) changes the
counts, so take a new baseline in the same pull request.

Wall-clock times are informational only: shared runners are too noisy to
block on. The test job still reports `tests/test_perf_guard.py`'s
baseline ratios as a warning, and fails on its `scaling` tests, which
compare two sizes of one program in one process. Release-time wall-clock
comparisons stay a manual A/B run on a quiet machine.

### Other benchmarks

```bash
PYTHONPATH=build/dev/python python benchmarks/compiler/compile_time.py \
  --baseline benchmarks/baselines/compile_time.json
```

`compile_time.py` times the canonical workloads and checks their op counts
against `compile_time.json`. `tests/classical/test_no_call_identity.py`
checks the same op counts on every platform and in the wheels, and
`tests/test_perf_gate.py` checks that they agree with `perf_gate.json`. If a
change alters them on purpose, regenerate the file with `--write-baseline`
as well. Timing is only gated when you pass `--gate FACTOR`.

`benchmarks/compiler/mlir_pipeline_bench.py` breaks compile time down by
stage and by pass and reports IR sizes; `benchmarks/simulation/qec_scaling.py`
shows how compile time grows with surface-code distance. CI runs both at a
small size to keep them working. `benchmarks/compiler/specialize_bench.py`
compares a call with a new float value against a fresh compile and fails if
the call is not at least ten times cheaper; the nightly workflow runs it.

## GPU builds

GPU support is off by default and, when requested, mandatory: configuration
fails rather than silently producing a compiler-only build.

```bash
uv pip install -r pyproject.toml --extra cu12
SITE="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

cmake --preset gpu -DCUDAQ_PULSE_CUDA12_ROOT="/usr/local/cuda-12.9;$SITE"
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

Configuring needs the pinned MLIR 22.1.8 toolchain:

1. `python3 tools/fetch-mlir-toolchain.py` downloads the archive for your
   platform from the `mlir-toolchain-22.1.8` release, checks its SHA-256
   against `tools/mlir-toolchain.lock.json` and unpacks it into
   `.cache/mlir/`. Configuring runs it when the toolchain is missing.
2. On a platform without a pinned archive, build one with
   `tools/build-mlir-toolchain.sh --out-dir DIR` (it downloads the LLVM
   22.1.8 sources), unpack the archive it writes, and configure with
   `-DMLIR_DIR=<prefix>/lib/cmake/mlir -DLLVM_DIR=<prefix>/lib/cmake/llvm`.
   Another MLIR 22.1.8 build works the same way.

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
  builds with GPU support and runs the GPU tests (see [GPU CI](#gpu-ci)).
- `wheels.yml`: cibuildwheel for `cp312-abi3` (3.12 and newer) on
  manylinux_2_28 (x86_64, aarch64, with the CUDA 12 and CUDA 13 runtimes) and
  macOS arm64 (compiler-only), plus an sdist. It then installs
  the wheels in clean containers without a CUDA toolkit. `gpu-install-test`
  runs the GPU tests from the wheels (see [GPU CI](#gpu-ci)) and gates the
  release (see [Releases](#releases)).
- `gpu-tests.yml`: the GPU tests, called by `ci.yml` and `wheels.yml`.
- `docs.yml`: Sphinx with warnings as errors; deploys `main` to GitHub Pages.
- `property-nightly.yml`: the property tests with random examples, the
  slow physics checks (`nightly`) and the perf guard, every night and on
  manual dispatch, with the seed in the job summary and the example database
  kept in the Actions cache. GitHub runs scheduled workflows from the default
  branch, so the schedule tests `main`; a manual dispatch tests the branch it
  is dispatched on.

### GPU CI

The GPU tests run on NVIDIA's shared GitHub Actions GPU runners, through
`gpu-tests.yml`. Each test runs in a container that gets the machine's GPU
and driver and nothing else. `tools/ci/gpu-matrix.py` lists the rows,
following what the GPU code varies on:

- `wheels.yml` (`gpu-install-test`) installs the run's wheels with one CUDA
  extra per row, in images that have no CUDA toolkit
  (`install-test.sh --driver-only` fails if one is present), and runs the
  doctor and `pytest -m gpu`. On every push to `staging` it runs `[cu12]`
  and `[cu13]` on x86_64 with the newest driver. Release tags and manual
  runs with `gpu` also run both extras on aarch64 and on x86_64 with the
  oldest driver, plus `[cu13]` on the newest Python (the abi3 wheel). On
  the oldest driver, an extra whose minimum driver (CUDA 12.0 for `[cu12]`,
  CUDA 13.0 for `[cu13]`) is newer than the installed one must be refused
  by the doctor with that reason; on any other row an old driver fails the
  test. Free-threaded Python has no release wheels and no row.
- `ci.yml` (`gpu`) builds from source with `CUDAQ_PULSE_ENABLE_GPU=ON` in
  the CUDA 12.9 development image with the `[cu12]` libraries, and runs
  `check-pulse-gpu`, on every push to `main` and `staging`, on vetted pull
  requests (below) and on manual runs with `gpu`.

The `GPU_RUNNERS` repository variable switches GPU CI on, and names a runner
for each role, as a JSON object:

```json
{"x86_64": "<label>", "x86_64-min-driver": "<label>", "aarch64": "<label>"}
```

`x86_64` has the newest driver, `x86_64-min-driver` the oldest. A value may be
a JSON list of labels, and a bare label means `x86_64` alone. A row whose
role is missing is skipped and listed in the job summary. While
`GPU_RUNNERS` is unset, no GPU job runs or waits for a runner, and
`gpu-gate` requires the manual attestation (see
[GPU verification](#gpu-verification)).

**Pull requests.** The `pull_request` event never runs a GPU job, since a
pull request from a fork controls the workflow files it runs with. Instead,
the [copy-pr-bot](https://github.com/apps/copy-pr-bot) GitHub App
(configured in `.github/copy-pr-bot.yaml`) pushes a copy of the pull request
to a `pull-request/<N>` branch of this repository, and `ci.yml` runs the
`gpu` job there alone. Commits the app already trusts are copied on their
own; for any other, a maintainer first reviews the commit, workflow changes
included, and approves it with the app's `/ok to test` comment. Every new
commit needs approval again. The GPU jobs use no secrets and a
read-only token. The pull request shows the result as the `ci` checks of
its head commit.

**Switching it on** (repository administrators):

1. NVIDIA's runner administrators grant this repository the GPU runners.
2. Set `GPU_RUNNERS` to their labels (Settings, *Secrets and variables*,
   *Actions*, *Variables*).
3. Install the copy-pr-bot app on the repository, for pull requests.
4. Run `ci` and `wheels` manually with `gpu` on `staging`, and check that
   every row ran on a GPU and passed.

Deleting `GPU_RUNNERS` switches everything off again. Uninstalling the app
stops the pull-request copies.

To run the wheels' GPU tests on a machine with an NVIDIA GPU, put the wheels
in `wheelhouse/` and run `bash tools/ci/install-test.sh gpu-cu12` (or
`gpu-cu13`, or `gpu` for both).

## Versioning and stability

Versions follow [PEP 440](https://peps.python.org/pep-0440/) in the shape of
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`, with
pre-releases `X.Y.Z{a,b,rc}N` and post releases `X.Y.Z.postN`.

- **What is covered.** The public API is the names in the `__all__` of
  `cudaq_pulse` and of its documented submodules (`cudaq_pulse.ops`,
  `cudaq_pulse.compiler`, `cudaq_pulse.targets`,
  `cudaq_pulse.targets.transmon_model`, `cudaq_pulse.runtime`,
  `cudaq_pulse.runtime.simulate`, `cudaq_pulse.viz`, `cudaq_pulse.controller`,
  `cudaq_pulse.backends`, `cudaq_pulse.classical`,
  `cudaq_pulse.classical.wire`, `cudaq_pulse.classical.qec`,
  `cudaq_pulse.classical.emulation`, `cudaq_pulse.classical.calibration`,
  `cudaq_pulse.classical.loop` and `cudaq_pulse.runtime.drift`), their
  documented behavior, the `cudaq_pulse.backends` entry-point group, the
  controller export format (whose version is `cudaq_pulse.controller.FORMAT`),
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
3. Date the version's section in `CHANGELOG.md` (`## [0.1.0] - YYYY-MM-DD`).
   The next release's section, `## [X.Y.Z] - Unreleased`, starts with the
   first change after the tag.
4. Tag `v0.1.0` on `main`.

Release notes come from `CHANGELOG.md`, not from the commit history:
`tools/release-notes.py` prints the section for the version, and a
pre-release uses the section of the release it precedes. Add user-visible
changes to the top section, the next release's `## [X.Y.Z] - Unreleased`,
in the pull request that makes them. It is the only undated section, so
everything in it reaches the release notes.

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

- **With GPU runners** (`GPU_RUNNERS` set; see [GPU CI](#gpu-ci)), the
  tag's run executes every `gpu-install-test` row and publishes only if they
  all pass. A failed GPU run cannot be overridden: fix the problem and tag a
  new version.
- **Without GPU runners**, the tag's run builds and tests everything else
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
