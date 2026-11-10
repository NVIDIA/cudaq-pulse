// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --pulse-verify --pulse-schedule-asap %s | FileCheck %s

// A readout gives a measurement and an iq_acquire an IQ point; both return
// their line and tone, so the frame change before each stays on the tone it
// threads. The readout is scheduled; the iq_acquire is given no start or
// duration.

// CHECK-LABEL: func.func @main
// CHECK: %[[T1:.*]] = pulse.shift_phase
// CHECK: %{{.*}}, %[[T2:.*]], %[[M:.*]] = pulse.readout %{{.*}}, %{{.*}}, %[[T1]], iq <{duration_cycles = 400 : i64, start_cycles = 0 : i64}>
// CHECK: %[[T3:.*]] = pulse.shift_phase %[[T2]]
// CHECK: pulse.iq_acquire %{{.*}}, %{{.*}}, %[[T3]] : !pulse.readout_line
// CHECK: pulse.is_set %[[M]]
module attributes {pulse.clock_ghz = 1.000000e+00 : f64} {
  func.func @main() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %r0, %t0 = pulse.get_readout_line %q {frequency_hz = 7.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %c400 = arith.constant 400 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %c400, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    %half_pi = arith.constant 1.5707963267948966 : f64
    %t1 = pulse.shift_phase %t0, %half_pi : !pulse.tone, f64 -> !pulse.tone
    %r1, %t2, %m = pulse.readout %r0, %w, %t1, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %t3 = pulse.shift_phase %t2, %half_pi : !pulse.tone, f64 -> !pulse.tone
    %r2, %t4, %iq = pulse.iq_acquire %r1, %w, %t3 : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
    %i, %qv = pulse.extract_iq %iq : !pulse.iq_data -> f64, f64
    %b = pulse.is_set %m : !pulse.measurement -> i1
    return
  }
}
