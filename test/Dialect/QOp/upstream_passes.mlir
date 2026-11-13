// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --cse %s | FileCheck %s
// RUN: cudaq-pulse-opt --canonicalize %s | FileCheck %s
// RUN: cudaq-pulse-opt --symbol-dce %s | FileCheck %s
// RUN: cudaq-pulse-opt --sccp %s | FileCheck %s
// RUN: cudaq-pulse-opt --inline %s | FileCheck %s
// RUN: cudaq-pulse-opt --qop-to-cudm --canonicalize %s > %t.cudm
// RUN: cudaq-pulse-opt --cse --qop-to-cudm --canonicalize %s | diff %t.cudm -
// RUN: cudaq-pulse-opt --canonicalize --qop-to-cudm --canonicalize %s | diff %t.cudm -
// RUN: cudaq-pulse-opt --symbol-dce --qop-to-cudm --canonicalize %s | diff %t.cudm -
// RUN: cudaq-pulse-opt --sccp --qop-to-cudm --canonicalize %s | diff %t.cudm -
// RUN: cudaq-pulse-opt --inline --qop-to-cudm --canonicalize %s | diff %t.cudm -
// RUN: cudaq-pulse-opt --cse --canonicalize --symbol-dce --sccp --inline --qop-to-cudm --canonicalize %s | diff %t.cudm -

// Upstream passes on a QOp program: its operators are pure, and qop.evolve,
// which writes the state, keeps every one it uses. Each pass leaves the
// program, and the cuDensityMat program it lowers to, as they were.

// CHECK-LABEL: func.func @main
// CHECK-DAG: %[[Z:.*]] = qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_z>}
// CHECK-DAG: %[[X:.*]] = qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_x>}
// CHECK-DAG: %[[LOWERING:.*]] = qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_lowering>}
// CHECK-DAG: %[[ENVELOPE:.*]] = qop.callback_scalar @drive_envelope_0_x
// CHECK-DAG: %[[STATIC:.*]] = qop.make_product(%{{.*}}, %[[Z]])
// CHECK-DAG: %[[DRIVE:.*]] = qop.make_product(%[[ENVELOPE]], %[[X]])
// CHECK-DAG: %[[DAMPING:.*]] = qop.make_product(%{{.*}}, %[[LOWERING]])
// CHECK-DAG: %[[H:.*]] = qop.make_sum(%[[STATIC]], %[[DRIVE]]) : !qop.op
// CHECK-DAG: %[[C:.*]] = qop.make_sum(%[[DAMPING]]) : !qop.op
// CHECK: %[[L:.*]] = qop.lindblad(%[[H]], %[[C]]) : !qop.super_op
// CHECK-NEXT: qop.evolve %[[L]] : !qop.super_op
// CHECK-NEXT: return
module attributes {qop.qudit_count = 1 : i64, qop.t_start = 0.0 : f64, qop.t_end = 100.0 : f64, qop.step_count = 100 : i64} {
func.func @main() {
  %c0 = arith.constant 0 : i64
  %sz = qop.spin(%c0) {kind = #qop<handler_kind spin_z>} : !qop.handler
  %coeff_z = qop.const_scalar {real = 15.707963 : f64, imag = 0.0 : f64} : !qop.scalar
  %term_z = qop.make_product(%coeff_z, %sz) : !qop.product
  %sx = qop.spin(%c0) {kind = #qop<handler_kind spin_x>} : !qop.handler
  %cb_x = qop.callback_scalar @drive_envelope_0_x {cudm.callback_kind = 0 : i32, cudm.callback_params = array<f64: 0.0, 40.0, 0.3, 10.0>} : !qop.scalar
  %term_x = qop.make_product(%cb_x, %sx) : !qop.product
  %H = qop.make_sum(%term_z, %term_x) : !qop.op
  %lowering = qop.spin(%c0) {kind = #qop<handler_kind spin_lowering>} : !qop.handler
  %collapse_coeff = qop.const_scalar {real = 0.1 : f64, imag = 0.0 : f64} : !qop.scalar
  %collapse_product = qop.make_product(%collapse_coeff, %lowering) : !qop.product
  %collapse = qop.make_sum(%collapse_product) : !qop.op
  %L = qop.lindblad(%H, %collapse) : !qop.super_op
  qop.evolve %L : !qop.super_op
  return
}
}
