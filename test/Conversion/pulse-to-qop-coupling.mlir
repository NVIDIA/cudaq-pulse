// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// Each qudit has its own rotating frame, so an exchange coupling between
// qudits at 5.0 and 5.1 GHz turns at their difference: four products whose
// callbacks are cos and sin of (-2 pi 0.1 GHz) t with amplitude g. Between
// resonant qudits it is static. A crosstalk ZZ rate is a cross-Kerr n n term.
// RUN: cudaq-pulse-opt --split-input-file --pulse-to-qop %s | FileCheck %s
// RUN: cudaq-pulse-opt --split-input-file --pulse-to-qop --qop-to-cudm %s | FileCheck %s --check-prefix=CUDM

// CHECK-LABEL: module
// CHECK: qop.callback_scalar @coupling_0_0 {cudm.callback_kind = 1 : i32, cudm.callback_params = array<f64: 0.000000e+00, 1.7976931348623157E+308, 0.031415926535897934, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, {{-0.6283185307179[0-9]*}}>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_x>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_x>}
// CHECK: qop.callback_scalar @coupling_0_1 {{.*}} 0.031415926535897934, {{.*}}, 0.000000e+00, {{-0.6283185307179[0-9]*}}>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_y>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_y>}
// CHECK: qop.callback_scalar @coupling_0_2 {{.*}} 0.031415926535897934, {{.*}}, 1.000000e+00, {{-0.6283185307179[0-9]*}}>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_y>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_x>}
// CHECK: qop.callback_scalar @coupling_0_3 {{.*}} -0.031415926535897934, {{.*}}, 1.000000e+00, {{-0.6283185307179[0-9]*}}>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_x>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_y>}
// CHECK: qop.matrix_leaf({{.*}}) {dimensions = array<i64: 2>, targets = array<i64: 0>}
// CHECK: qop.matrix_leaf({{.*}}) {dimensions = array<i64: 2>, targets = array<i64: 1>}
// CHECK: qop.const_scalar {imag = 0.000000e+00 : f64, real = 6.283185307179587E-5 : f64}
// CUDM-LABEL: module
// CUDM-COUNT-4: cudm.append_elementary_product {{.*}} cudm.callback_kind = 1 : i32{{.*}} modes_acted_on = array<i32: 0, 1>
// CUDM: cudm.append_elementary_product {{.*}} coeff_real = 6.283185307179587E-5 : f64, duality = array<i32: 0, 0>, modes_acted_on = array<i32: 0, 1>
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.coupling_pairs = array<i64: 0, 1>, pulse.coupling_strength_hz = array<f64: 5.000000e+06>, pulse.crosstalk_pairs = array<i64: 0, 1>, pulse.crosstalk_strength_hz = array<f64: 1.000000e+04>} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
    %q1 = pulse.qudit_alloc {qubit = 1 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q0 {qubit = 0 : i64, frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %d1, %t1 = pulse.get_drive_line %q1 {qubit = 1 : i64, frequency_hz = 5.100000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %zero, %zero : i64, f64, f64 -> !pulse.waveform
    %d2, %t2 = pulse.drive %d0, %x, %t0 {start_vtu = 0 : i64, duration_vtu = 40 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}

// -----

// CHECK-LABEL: module
// CHECK-NOT: @coupling_
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_x>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_x>}
// CHECK: qop.const_scalar {imag = 0.000000e+00 : f64, real = 0.015707963267948967 : f64}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_y>}
// CHECK: qop.spin(%{{.*}}) {kind = #qop<handler_kind spin_y>}
// CHECK: qop.const_scalar {imag = 0.000000e+00 : f64, real = 0.015707963267948967 : f64}
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.coupling_pairs = array<i64: 0, 1>, pulse.coupling_strength_hz = array<f64: 5.000000e+06>} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
    %q1 = pulse.qudit_alloc {qubit = 1 : i64} : !pulse.qref
    %d0, %t0 = pulse.get_drive_line %q0 {qubit = 0 : i64, frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %d1, %t1 = pulse.get_drive_line %q1 {qubit = 1 : i64, frequency_hz = 5.000000e+09 : f64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %zero, %zero : i64, f64, f64 -> !pulse.waveform
    %d2, %t2 = pulse.drive %d0, %x, %t0 {start_vtu = 0 : i64, duration_vtu = 40 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
