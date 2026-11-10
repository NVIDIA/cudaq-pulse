# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_conversion_library(CudaqPulsePulseToQOp
  PulseToQOp.cpp

  DEPENDS
  CudaqPulsePulseIncGen
  CudaqPulseQOpIncGen

  LINK_LIBS PUBLIC
  CudaqPulsePulse
  CudaqPulseClassical
  CudaqPulseQOp
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRTransforms
)
