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
# Usage: tools/ci/install-test.sh [--python PY] [--wheelhouse DIR] MODE...
#
# Modes:
#   base    install cudaq-pulse; compile a kernel, run cudaq-pulse-opt --help,
#           the doctor (no GPU required), and pytest -m "not gpu"
#   extras  install [cu12] and [cu13], each in its own environment; the doctor
#           must load every library and report that no device is present
#   gpu     install [cu12] and [cu13], each in its own environment; the doctor
#           must pass and pytest -m gpu must succeed

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="python3"
wheelhouse="${repo_root}/wheelhouse"
modes=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) python="$2"; shift 2 ;;
    --wheelhouse) wheelhouse="$(cd "$2" && pwd)"; shift 2 ;;
    base | extras | gpu) modes+=("$1"); shift ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ ${#modes[@]} -eq 0 ]]; then
  echo "error: no mode given (base, extras, gpu)" >&2
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
  # tests/test_frontend_features.py checks the canonical workload op counts.
  mkdir -p "$dir/benchmarks/compiler"
  cp "${repo_root}/benchmarks/compiler/compile_time.py" "$dir/benchmarks/compiler/"
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
def rabi(duration, amplitude):
    qubit = pulse.qudit_ref()
    drive_line, tone = get_drive_line(qubit)
    drive(drive_line, square(duration, amplitude), tone)


compiled = pulse.compile(rabi, [100, 0.5], qubit_freq_hz={0: 5e9})
assert compiled.mlir.strip(), "compile produced no MLIR"
print(f"compiled from {package.parent} in {compiled.metrics.total_ms:.3f} ms")
EOF

for mode in "${modes[@]}"; do
  case "$mode" in
    base)
      new_venv base "cudaq-pulse==${version}" "${test_requirements[@]}"
      in_venv python smoke_compile.py
      in_venv cudaq-pulse-opt --help >/dev/null
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
    gpu)
      for extra in cu12 cu13; do
        new_venv "gpu-${extra}" "cudaq-pulse[${extra}]==${version}" "${test_requirements[@]}"
        in_venv python -m cudaq_pulse.doctor
        run_pytest "gpu"
        drop_venv
      done
      ;;
  esac
done
