# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_mlir_dialect_library(CudaqPulseClassical
  ClassicalDialect.cpp
  ClassicalOps.cpp

  DEPENDS
  CudaqPulseClassicalIncGen
  CudaqPulsePulseIncGen

  LINK_LIBS PUBLIC
  CudaqPulsePulse
  MLIRIR
  MLIRSupport
  MLIRBytecodeOpInterface
)
