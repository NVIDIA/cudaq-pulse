// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// pulse.modes lists the simulated qudits, one mode each in order: qutrits 3
// and 13 are modes 0 and 1 of a 3 x 3 space, not modes of a 3^14 one. Qudit 5
// is not simulated, so its anharmonicity is left out.
// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-to-qop --qop-to-cudm %s | FileCheck %s --check-prefix=CUDM

// CHECK: qop.mode_extents = array<i64: 3, 3>
// CHECK-NOT: targets = array<i64: {{[2-9]|1[0-9]}}>
// The coupling of qudits 13 and 3 acts on modes 1 and 0.
// CHECK: qop.callback_scalar @coupling_0_0
// CHECK: qop.matrix_leaf({{.*}}) {dimensions = array<i64: 3>, targets = array<i64: 1>}
// CHECK: qop.matrix_leaf({{.*}}) {dimensions = array<i64: 3>, targets = array<i64: 0>}
// CHECK: qop.callback_scalar @coupling_0_3
// The anharmonicities of qudits 3 and 13, but not 5 (-0.5969... rad/ns).
// CHECK: qop.const_scalar {imag = 0.000000e+00 : f64, real = -0.62831853071795862 : f64}
// CHECK-NOT: real = -0.5969
// CHECK: qop.const_scalar {imag = 0.000000e+00 : f64, real = -0.56548667764616278 : f64}
// The drive on qudit 13 acts on mode 1.
// CHECK: qop.callback_scalar @drive_envelope_0_x
// CHECK: qop.matrix_leaf({{.*}}) {dimensions = array<i64: 3>, targets = array<i64: 1>}
// CHECK-NOT: targets = array<i64: {{[2-9]|1[0-9]}}>
// CUDM: mode_extents = array<i64: 3, 3>
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.modes = array<i64: 3, 13>, pulse.quantum_system = #pulse.quantum_system<level_counts = [2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3], anharmonicity_hz = [0.0, 0.0, 0.0, -2.000000e+08, 0.0, -1.900000e+08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.800000e+08]>, pulse.coupling_pairs = array<i64: 13, 3>, pulse.coupling_strength_hz = array<f64: 5.000000e+06>} {
  func.func @main() {
    %q13 = pulse.qudit_alloc {qudit = 13 : i64} : !pulse.qudit
    %q3 = pulse.qudit_alloc {qudit = 3 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q13 {frequency_hz = 5.100000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %d1, %t1 = pulse.get_drive_line %q3 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d2, %t2 = pulse.drive %d0, %x, %t0 <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
