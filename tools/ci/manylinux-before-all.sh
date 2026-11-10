#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs once per manylinux_2_28 container before cibuildwheel builds any wheel.
# Installs the build-time CUDA 12 and CUDA 13 headers and libraries from the
# same pip wheels the [cu12] and [cu13] extras install at run time, into
# /opt/cuda12 and /opt/cuda13, and unpacks the checksum-verified MLIR
# toolchain. Any failure aborts the wheel build.

set -euo pipefail

# Each runtime is compiled against the oldest CUDA minor its extra allows, so
# it runs with any newer runtime of that major. crt/host_defines.h ships with
# nvcc for CUDA 12 and in its own wheel for CUDA 13.
CUDENSITYMAT_VERSION="0.7.0"
cuda12_wheels=(
  "nvidia-cuda-runtime-cu12==12.9.79"
  "nvidia-cuda-nvcc-cu12==12.9.86"
  "cudensitymat-cu12==${CUDENSITYMAT_VERSION}"
)
cuda13_wheels=(
  "nvidia-cuda-runtime==13.0.48"
  "nvidia-cuda-crt==13.0.48"
  "cudensitymat-cu13==${CUDENSITYMAT_VERSION}"
)

python3.12 -m pip install --no-deps --target /opt/cuda12 "${cuda12_wheels[@]}"
test -f /opt/cuda12/nvidia/cuda_runtime/include/cuda_runtime_api.h
test -f /opt/cuda12/nvidia/cuda_nvcc/include/crt/host_defines.h
test -f /opt/cuda12/nvidia/cuda_runtime/lib/libcudart.so.12
test -f /opt/cuda12/cuquantum/include/cudensitymat.h
test -f /opt/cuda12/cuquantum/lib/libcudensitymat.so.0

python3.12 -m pip install --no-deps --target /opt/cuda13 "${cuda13_wheels[@]}"
test -f /opt/cuda13/nvidia/cu13/include/cuda_runtime_api.h
test -f /opt/cuda13/nvidia/cu13/include/crt/host_defines.h
test -f /opt/cuda13/nvidia/cu13/lib/libcudart.so.13
test -f /opt/cuda13/cuquantum/include/cudensitymat.h
test -f /opt/cuda13/cuquantum/lib/libcudensitymat.so.0

# The workflow downloads the toolchain archive into .cache/mlir/downloads
# before cibuildwheel starts; this verifies its checksum and unpacks it.
python3.12 tools/fetch-mlir-toolchain.py
