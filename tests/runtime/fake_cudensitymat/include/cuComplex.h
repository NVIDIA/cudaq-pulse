/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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
