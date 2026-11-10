# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_conversion_library(CudaqPulseQOpToCuDensityMat
  QOpToCuDensityMat.cpp

  DEPENDS
  CudaqPulseQOpIncGen
  CudaqPulseCuDensityMatIncGen

  LINK_LIBS PUBLIC
  CudaqPulseQOp
  CudaqPulseCuDensityMat
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransforms
)
