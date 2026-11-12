// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// pulse.true_qudit_freq_hz puts a qudit at its true frequency, here 1 MHz
// above the 5 GHz tone the program was compiled with: each drive envelope
// turns at the detuning, 2 pi x 1e-3 rad/ns. NaN leaves a qudit as compiled.
// RUN: cudaq-pulse-opt --pulse-to-qop %s | FileCheck %s

// CHECK: qop.callback_scalar @drive_envelope_0_x {{.*}}cudm.callback_params = array<f64: {{.*}}, 0.0062831853071795866>
// CHECK: qop.callback_scalar @drive_envelope_0_y {{.*}}cudm.callback_params = array<f64: {{.*}}, 0.0062831853071795866>
// CHECK: qop.callback_scalar @drive_envelope_1_x {{.*}}cudm.callback_params = array<f64: {{.*}}, 0.000000e+00>
module attributes {pulse.clock_ghz = 1.000000e+00 : f64, pulse.levels = array<i64: 2, 2>, pulse.t1_times = [1.000000e+04 : f64, 1.000000e+04 : f64], pulse.true_qudit_freq_hz = array<f64: 5.001000e+09, 0x7FF8000000000000>} {
  func.func @main() {
    %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
    %d0, %t0 = pulse.get_drive_line %q0 {frequency_hz = 5.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %d1, %t1 = pulse.get_drive_line %q1 {frequency_hz = 5.100000e+09 : f64} : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %d2, %t2 = pulse.drive %d0, %x, %t0 <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %d3, %t3 = pulse.drive %d1, %x, %t1 <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    return
  }
}
