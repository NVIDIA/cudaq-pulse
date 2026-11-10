// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The runtime computes in complex128 only, so f64 is the only compute type.

// RUN: not cudaq-pulse-opt %s 2>&1 | FileCheck %s

// CHECK: error: expected ::cudm::ComputeType to be one of: f64
func.func @single_precision() {
  %h = cudm.init_handle : !cudm.handle
  %s = cudm.create_state %h {purity = #cudm<purity pure>, data_type = #cudm<compute_type f32>, mode_extents = array<i64: 2>} : (!cudm.handle) -> !cudm.state
  return
}
