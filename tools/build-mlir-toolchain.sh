#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build the MLIR toolchain that cudaq-pulse compiles against and package it as
# mlir-toolchain-<version>-<os>-<arch>.tar.xz plus a .sha256 file.
#
# Release Linux toolchains are built inside the manylinux_2_28 image (see
# .github/workflows/mlir-toolchain.yml) so the static archives are ABI
# compatible with the wheels. The same script builds a native macOS toolchain
# for local development.
#
# Usage: tools/build-mlir-toolchain.sh [--work-dir DIR] [--out-dir DIR] [--jobs N]

set -euo pipefail

LLVM_VERSION="22.1.8"
LLVM_SOURCE_SHA256="922f1817a0df7b1489272d18134ee0087a8b068828f87ac63b9861b1a9965888"
LLVM_SOURCE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/llvm-project-${LLVM_VERSION}.src.tar.xz"

work_dir="${PWD}/.cache/mlir-build"
out_dir="${PWD}/dist/mlir-toolchain"
jobs=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --work-dir) work_dir="$2"; shift 2 ;;
    --out-dir) out_dir="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    *) echo "error: unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$(uname -s)" in
  Linux) os="linux" ;;
  Darwin) os="macos" ;;
  *) echo "error: unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac
case "$(uname -m)" in
  x86_64 | amd64) arch="x86_64" ;;
  aarch64 | arm64) arch="aarch64" ;;
  *) echo "error: unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

for tool in cmake ninja tar xz; do
  command -v "$tool" >/dev/null || {
    echo "error: required tool '$tool' is not on PATH" >&2
    exit 1
  }
done

if [[ -z "$jobs" ]]; then
  jobs="$(getconf _NPROCESSORS_ONLN)"
fi

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

name="mlir-toolchain-${LLVM_VERSION}-${os}-${arch}"
src_archive="${work_dir}/llvm-project-${LLVM_VERSION}.src.tar.xz"
src_dir="${work_dir}/llvm-project-${LLVM_VERSION}.src"
build_dir="${work_dir}/build-${os}-${arch}"
prefix="${work_dir}/${name}"

mkdir -p "$work_dir" "$out_dir"

if [[ ! -f "$src_archive" ]]; then
  echo "Downloading ${LLVM_SOURCE_URL}"
  curl -sSfL -o "${src_archive}.part" "$LLVM_SOURCE_URL"
  mv "${src_archive}.part" "$src_archive"
fi
actual="$(sha256 "$src_archive")"
if [[ "$actual" != "$LLVM_SOURCE_SHA256" ]]; then
  echo "error: checksum mismatch for $src_archive" >&2
  echo "  expected $LLVM_SOURCE_SHA256" >&2
  echo "  actual   $actual" >&2
  exit 1
fi

if [[ ! -f "${src_dir}/.extracted" ]]; then
  echo "Extracting LLVM ${LLVM_VERSION} sources"
  rm -rf "$src_dir"
  src_root="$(basename "$src_dir")"
  tar -C "$work_dir" --exclude='*/.vscode' -xJf "$src_archive" \
    "${src_root}/llvm" "${src_root}/mlir" "${src_root}/cmake" "${src_root}/third-party"
  touch "${src_dir}/.extracted"
fi

# Only these components are installed and exported. FileCheck, not, and count
# drive the lit suite and the tablegen tools generate the dialect sources.
# Every exported executable must be present in the archive or
# find_package(MLIR) fails, so the export set and the install set are
# controlled by the same list.
distribution_components=(
  llvm-headers llvm-libraries cmake-exports llvm-tblgen
  FileCheck not count
  mlir-headers mlir-libraries mlir-cmake-exports mlir-tblgen
)
distribution_components_list="$(IFS=';'; echo "${distribution_components[*]}")"

cmake_args=(
  -G Ninja
  -S "${src_dir}/llvm"
  -B "$build_dir"
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$prefix"
  -DLLVM_ENABLE_PROJECTS=mlir
  -DLLVM_TARGETS_TO_BUILD=host
  -DLLVM_ENABLE_RTTI=OFF
  -DLLVM_ENABLE_ASSERTIONS=OFF
  -DLLVM_ENABLE_PIC=ON
  -DLLVM_BUILD_LLVM_DYLIB=OFF
  -DLLVM_LINK_LLVM_DYLIB=OFF
  -DLLVM_INSTALL_UTILS=ON
  -DLLVM_INCLUDE_TESTS=OFF
  -DLLVM_INCLUDE_EXAMPLES=OFF
  -DLLVM_INCLUDE_BENCHMARKS=OFF
  -DLLVM_INCLUDE_DOCS=OFF
  -DLLVM_ENABLE_BINDINGS=OFF
  -DLLVM_ENABLE_ZLIB=OFF
  -DLLVM_ENABLE_ZSTD=OFF
  -DLLVM_ENABLE_LIBXML2=OFF
  -DLLVM_ENABLE_LIBEDIT=OFF
  -DLLVM_ENABLE_LIBPFM=OFF
  -DMLIR_INCLUDE_TESTS=OFF
  -DMLIR_INCLUDE_INTEGRATION_TESTS=OFF
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
  -DLLVM_DISTRIBUTION_COMPONENTS="$distribution_components_list"
)
if [[ "$os" == "macos" ]]; then
  cmake_args+=(-DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}")
fi

cmake "${cmake_args[@]}"
rm -rf "$prefix"
cmake --build "$build_dir" --parallel "$jobs" --target install-distribution

rm -rf "${prefix}/share"
cp "${src_dir}/llvm/LICENSE.TXT" "${prefix}/LICENSE.TXT"
if [[ -f "${src_dir}/mlir/LICENSE.TXT" ]]; then
  cp "${src_dir}/mlir/LICENSE.TXT" "${prefix}/LICENSE-mlir.TXT"
fi
if [[ "$os" == "linux" ]]; then
  strip_args=(--strip-unneeded)
else
  strip_args=(-S)
fi
for binary in "${prefix}"/bin/*; do
  if [[ -f "$binary" && ! -L "$binary" && -x "$binary" ]]; then
    strip "${strip_args[@]}" "$binary"
  fi
done

# The archive redistributes LLVM and MLIR, so it must carry their license.
[[ -s "${prefix}/LICENSE.TXT" ]] || {
  echo "error: ${prefix}/LICENSE.TXT is missing; the archive must include it" >&2
  exit 1
}

archive="${out_dir}/${name}.tar.xz"
echo "Packaging ${archive}"
tar -C "$work_dir" -cf - "$name" | xz -T0 -6 >"${archive}.part"
mv "${archive}.part" "$archive"
printf '%s  %s\n' "$(sha256 "$archive")" "$(basename "$archive")" >"${archive}.sha256"
cat "${archive}.sha256"
