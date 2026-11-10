// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A three-level qudit lowers to 3x3 matrix leaves, with its anharmonicity,
// and on through cuDensityMat to LLVM.
// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-to-qop --qop-to-cudm --cudm-to-llvm %s | FileCheck %s --check-prefix=LLVM

// CHECK: qop.mode_extents = array<i64: 3>
// CHECK: qop.matrix_leaf({{.*}}) {dimensions = array<i64: 3>, targets = array<i64: 0>} : tensor<3x3x2xf64>
// CHECK: qop.callback_scalar @drive_envelope_0_x
// CHECK: qop.callback_scalar @drive_envelope_0_y
// LLVM: llvm.call @cudm_elementary_op_create
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.levels = array<i64: 3>, pulse.anharmonicity_hz = array<f64: -2.000000e+08>, pulse.t1_times = [1.000000e+04 : f64]} {
  func.func @main() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d1, %t1 = pulse.drive %d0, %x, %t0 <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
