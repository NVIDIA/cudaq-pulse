# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(CudaqPulsePulseTransforms
  PulseVerify.cpp
  PulseVerifyCoupled.cpp
  Canonicalize.cpp
  VirtualZ.cpp
  Fusion.cpp
  Rewrite.cpp
  ScheduleAlap.cpp
  ScheduleAlapCoupled.cpp
  ClassicalCheck.cpp
  FrameChanges.cpp
  FrameChangesCoupled.cpp
  LowerSelects.cpp

  DEPENDS
  CudaqPulsePulseIncGen
  CudaqPulseClassicalIncGen

  LINK_LIBS PUBLIC
  CudaqPulsePulse
  CudaqPulseClassical
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransformUtils
)
