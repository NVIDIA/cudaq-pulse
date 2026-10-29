#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Install cudaq-pulse from a directory of built wheels into fresh virtual
# environments and test the installed package. The tests run from a copy of
# tests/ outside the checkout, so the source tree is never importable.
#
# Usage: tools/ci/install-test.sh [--python PY] [--wheelhouse DIR]
#                                 [--driver-only] [--allow-old-driver] MODE...
#
# Modes:
#   base      install cudaq-pulse; compile a kernel, run the installed
#             cudaq-pulse-opt on it, the doctor (no GPU required), and
#             pytest -m "not gpu"
#   extras    install [cu12] and [cu13], each in its own environment; the
#             doctor must load every library and report that no device is
#             present
#   gpu-cu12  install [cu12] in a fresh environment; the doctor must pass and
#   gpu-cu13  pytest -m gpu must succeed (likewise for [cu13])
#   gpu       gpu-cu12, then gpu-cu13
#
# The GPU modes read the driver's CUDA version from libcuda.so.1 first. A
# driver older than the extra's minimum (CUDA 12.0 for [cu12], CUDA 13.0 for
# [cu13]) fails the test, unless --allow-old-driver is given: then the doctor
# must refuse that driver by name, and the GPU tests are not run.
# --driver-only fails if a CUDA toolkit is present, so that the GPU modes
# prove the extras alone supply every CUDA library.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="python3"
wheelhouse="${repo_root}/wheelhouse"
driver_only=false
allow_old_driver=false
modes=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) python="$2"; shift 2 ;;
    --wheelhouse) wheelhouse="$(cd "$2" && pwd)"; shift 2 ;;
    --driver-only) driver_only=true; shift ;;
    --allow-old-driver) allow_old_driver=true; shift ;;
    base | extras | gpu) modes+=("$1"); shift ;;
    gpu-cu12 | gpu-cu13) modes+=("$1"); shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ${#modes[@]} -eq 0 ]]; then
  echo "error: no mode given (base, extras, gpu-cu12, gpu-cu13, gpu)" >&2
  exit 2
fi

shopt -s nullglob
wheels=("$wheelhouse"/cudaq_pulse-*.whl)
shopt -u nullglob
if [[ ${#wheels[@]} -eq 0 ]]; then
  echo "error: no cudaq_pulse-*.whl in $wheelhouse" >&2
  exit 1
fi
# Pinning the version keeps pip from preferring a release on the index over
# the wheels under test.
version="$(basename "${wheels[0]}" | cut -d- -f2)"
test_requirements=("pytest>=8" "hypothesis>=6")

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

venv=""

# new_venv NAME REQUIREMENT...: create $work/NAME, install into it, set $venv.
new_venv() {
  venv="${work}/$1"
  shift
  echo "::group::Install $* into a fresh environment"
  "$python" -m venv "$venv"
  "${venv}/bin/python" -m pip install --quiet --no-cache-dir --upgrade pip
  "${venv}/bin/python" -m pip install --no-cache-dir \
    --find-links "$wheelhouse" --only-binary cudaq-pulse "$@"
  "${venv}/bin/python" -m pip list
  echo "::endgroup::"
}

in_venv() {
  (cd "$work" && PATH="${venv}/bin:${PATH}" VIRTUAL_ENV="$venv" "$@")
}

drop_venv() {
  rm -rf "$venv"
  venv=""
}

run_pytest() {
  local markers="$1" dir
  dir="${work}/tests-$(basename "$venv")"
  mkdir -p "$dir"
  cp -R "${repo_root}/tests" "${repo_root}/pyproject.toml" "$dir/"
  # tests/test_frontend_features.py and tests/classical/test_no_call_identity.py
  # check the canonical workloads and their baselined op counts.
  mkdir -p "$dir/benchmarks/compiler" "$dir/benchmarks/baselines"
  cp "${repo_root}/benchmarks/compiler/compile_time.py" "$dir/benchmarks/compiler/"
  cp "${repo_root}/benchmarks/baselines/compile_time.json" "$dir/benchmarks/baselines/"
  (cd "$dir" && PATH="${venv}/bin:${PATH}" VIRTUAL_ENV="$venv" \
    python -m pytest -p no:cacheprovider -m "$markers")
}

cat >"${work}/smoke_compile.py" <<'EOF'
import sys
from pathlib import Path

import cudaq_pulse as pulse

package = Path(pulse.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve()), f"not installed: {package}"


@pulse.kernel
def rabi(qubit, duration: int, amplitude: float):
    drive_line = get_drive_line(qubit)
    drive(drive_line, square(duration, amplitude))


device = pulse.device("transmon_17q")
compiled = pulse.compile(rabi).lower(device.qudit_handles[0], 100, 0.5)
assert compiled.mlir().strip(), "compile produced no MLIR"
Path("kernel.mlir").write_text(compiled.mlir())
print(f"compiled from {package.parent} in {compiled.metrics.total_ms:.3f} ms")
EOF

cat >"${work}/invalid.mlir" <<'EOF'
func.func @main() {
  %d = arith.constant 0 : i64
  %a = arith.constant 0.1 : f64
  %w = pulse.square %d, %a, %a : i64, f64, f64 -> !pulse.waveform
  return
}
EOF

# expect_in FILE TEXT: fail unless FILE contains TEXT.
expect_in() {
  if ! grep -qF -- "$2" "${work}/$1"; then
    echo "error: cudaq-pulse-opt output $1 lacks '$2'" >&2
    exit 1
  fi
}

# The installed cudaq-pulse-opt verifies, schedules and lowers the kernel
# smoke_compile.py wrote, and rejects invalid IR with its diagnostic.
opt_checks() {
  in_venv cudaq-pulse-opt --pulse-verify kernel.mlir -o verified.mlir
  expect_in verified.mlir "pulse.drive"
  in_venv cudaq-pulse-opt --pulse-merge-waits --pulse-fold-phases --pulse-fuse-pulses \
    --pulse-schedule-asap kernel.mlir -o scheduled.mlir
  expect_in scheduled.mlir "start_cycles = 0"
  in_venv cudaq-pulse-opt --pulse-to-qop --qop-to-cudm --cudm-to-llvm --canonicalize \
    --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts \
    kernel.mlir -o lowered.mlir
  expect_in lowered.mlir "llvm.call @cudm_evolve"
  if in_venv cudaq-pulse-opt invalid.mlir -o /dev/null 2>"${work}/invalid.err"; then
    echo "error: cudaq-pulse-opt accepted invalid.mlir" >&2
    exit 1
  fi
  expect_in invalid.err "'pulse.square' op duration must be positive, got 0"
}

# Exits 0 if a CUDA toolkit is visible: nvcc, a toolkit directory, a CUDA
# runtime on the loader path, or CUDA_HOME/CUDA_PATH.
toolkit_present() {
  local dir
  command -v nvcc && return 0
  for dir in /usr/local/cuda /usr/local/cuda-* /opt/cuda; do
    [[ -e "$dir" ]] && { echo "$dir"; return 0; }
  done
  ldconfig -p 2>/dev/null | grep -F 'libcudart.so' && return 0
  [[ -n "${CUDA_HOME:-}${CUDA_PATH:-}" ]] && { echo "CUDA_HOME/CUDA_PATH is set"; return 0; }
  return 1
}

# cuDriverGetVersion from the driver's libcuda.so.1 (12080 is CUDA 12.8), or
# 0 without a driver; read directly, not through cudaq-pulse.
driver_cuda_version() {
  "$python" - <<'EOF'
import ctypes

try:
    libcuda = ctypes.CDLL("libcuda.so.1")
except OSError:
    print(0)
else:
    version = ctypes.c_int(0)
    print(version.value if libcuda.cuDriverGetVersion(ctypes.byref(version)) == 0 else 0)
EOF
}

# gpu_extra cuNN: the GPU tests from a fresh environment with only [cuNN].
gpu_extra() {
  local extra="$1" minimum driver
  case "$extra" in
    cu12) minimum=12000 ;;
    cu13) minimum=13000 ;;
  esac
  if command -v nvidia-smi >/dev/null; then nvidia-smi; fi
  driver="$(driver_cuda_version)"
  if [[ "$driver" -eq 0 ]]; then
    echo "error: no NVIDIA driver (libcuda.so.1 is not loadable)" >&2
    exit 1
  fi
  new_venv "gpu-${extra}" "cudaq-pulse[${extra}]==${version}" "${test_requirements[@]}"
  if [[ "$driver" -ge "$minimum" ]]; then
    in_venv python -m cudaq_pulse.doctor
    run_pytest "gpu"
  elif [[ "$allow_old_driver" == true ]]; then
    # The doctor must fail on the driver's age alone: every library loads,
    # and the message names the extra and its minimum driver.
    if in_venv python -m cudaq_pulse.doctor >"${work}/doctor.out" 2>&1; then
      cat "${work}/doctor.out"
      echo "error: the doctor accepted a driver older than [${extra}] supports" >&2
      exit 1
    fi
    cat "${work}/doctor.out"
    if ! grep -qF "cudaq-pulse[${extra}] needs driver" "${work}/doctor.out"; then
      echo "error: the doctor did not refuse the driver by its version" >&2
      exit 1
    fi
    in_venv python -m cudaq_pulse.doctor --expect-no-device
    echo "::notice::driver supports CUDA $((driver / 1000)).$((driver % 1000 / 10)), older than" \
      "[${extra}] needs: the doctor refused it as expected; GPU tests not run"
  else
    echo "error: the driver supports CUDA $((driver / 1000)).$((driver % 1000 / 10));" \
      "[${extra}] needs CUDA $((minimum / 1000)).0 or newer" >&2
    exit 1
  fi
  drop_venv
}

if [[ "$driver_only" == true ]] && found="$(toolkit_present)"; then
  echo "error: a CUDA toolkit is present (${found}); --driver-only tests machines" \
    "that have only the NVIDIA driver" >&2
  exit 1
fi

for mode in "${modes[@]}"; do
  case "$mode" in
    base)
      new_venv base "cudaq-pulse==${version}" "${test_requirements[@]}"
      in_venv python smoke_compile.py
      in_venv cudaq-pulse-opt --help >/dev/null
      opt_checks
      in_venv python -m cudaq_pulse.doctor --allow-no-gpu
      run_pytest "not gpu"
      drop_venv
      ;;
    extras)
      for extra in cu12 cu13; do
        new_venv "$extra" "cudaq-pulse[${extra}]==${version}"
        in_venv python -m cudaq_pulse.doctor --expect-no-device
        drop_venv
      done
      ;;
    gpu-cu12 | gpu-cu13) gpu_extra "${mode#gpu-}" ;;
    gpu)
      gpu_extra cu12
      gpu_extra cu13
      ;;
  esac
done
