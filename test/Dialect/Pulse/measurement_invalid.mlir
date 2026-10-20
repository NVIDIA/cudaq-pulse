// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: not cudaq-pulse-opt %s 2>&1 | FileCheck %s

// CHECK: 'pulse.is_set' op measurement must be produced by pulse.readout, got builtin.unrealized_conversion_cast
func.func @not_from_readout(%b: i1) -> i1 {
  %m = builtin.unrealized_conversion_cast %b : i1 to !pulse.measurement
  %c = pulse.is_set %m : !pulse.measurement -> i1
  return %c : i1
}
