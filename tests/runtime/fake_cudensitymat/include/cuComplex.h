// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Host-only stand-in for the CUDA complex types the cuDensityMat runtime uses,
// so cudm_runtime.cpp builds and runs without CUDA (see fake_cudensitymat.cpp).

#pragma once

struct cuDoubleComplex {
  double x;
  double y;
};

inline cuDoubleComplex make_cuDoubleComplex(double real, double imaginary) {
  return {real, imaginary};
}
