// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --pulse-verify > /dev/null

// Ops no other round-trip test prints: each prints as it parses, and the
// functions return what they build.

// An IQ acquisition returns its line and tone, so a tone is acquired with
// again.
// CHECK-LABEL: func.func @iq_acquire_twice
// CHECK: %[[R1:.*]], %[[T1:.*]], %[[IQ1:.*]] = pulse.iq_acquire %{{.*}}, %{{.*}}, %{{.*}} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
// CHECK: %[[R2:.*]], %[[T2:.*]], %[[IQ2:.*]] = pulse.iq_acquire %[[R1]], %{{.*}}, %[[T1]]
// CHECK: %[[I:.*]], %[[Q:.*]] = pulse.extract_iq %[[IQ1]] : !pulse.iq_data -> f64, f64
// CHECK: pulse.create_iq %[[I]], %[[Q]] : f64, f64 -> !pulse.iq_data
// CHECK: pulse.iq_magnitude %[[IQ2]] : !pulse.iq_data -> f64
// CHECK: pulse.iq_phase %[[IQ2]] : !pulse.iq_data -> f64
func.func @iq_acquire_twice(%q: !pulse.qref, %w: !pulse.waveform) -> (!pulse.readout_line, !pulse.tone, !pulse.iq_data, f64, f64) {
  %r, %t = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %r1, %t1, %iq1 = pulse.iq_acquire %r, %w, %t : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  %r2, %t2, %iq2 = pulse.iq_acquire %r1, %w, %t1 : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  %i, %qv = pulse.extract_iq %iq1 : !pulse.iq_data -> f64, f64
  %iq3 = pulse.create_iq %i, %qv : f64, f64 -> !pulse.iq_data
  %mag = pulse.iq_magnitude %iq2 : !pulse.iq_data -> f64
  %ph = pulse.iq_phase %iq2 : !pulse.iq_data -> f64
  return %r2, %t2, %iq3, %mag, %ph : !pulse.readout_line, !pulse.tone, !pulse.iq_data, f64, f64
}

// A tone of its own, and the time ops.
// CHECK-LABEL: func.func @tone_and_time
// CHECK: %[[TONE:.*]] = pulse.tone %{{.*}}, %{{.*}} : f64, f64 -> !pulse.tone
// CHECK: %[[T0:.*]] = pulse.get_current_time %[[D:.*]] : !pulse.drive_line -> !pulse.timepoint
// CHECK: %[[D1:.*]], %{{.*}} = pulse.drive %[[D]], %{{.*}}, %[[TONE]]
// CHECK: %[[T1:.*]] = pulse.get_current_time %[[D1]] : !pulse.drive_line -> !pulse.timepoint
// CHECK: %[[DUR:.*]] = pulse.timepoint_subtract %[[T1]], %[[T0]] : (!pulse.timepoint, !pulse.timepoint) -> !pulse.duration
// CHECK: pulse.duration_to_int %[[DUR]] : (!pulse.duration) -> i64
func.func @tone_and_time(%d: !pulse.drive_line, %w: !pulse.waveform) -> (!pulse.drive_line, !pulse.tone, i64) {
  %f = arith.constant 5.0e9 : f64
  %p = arith.constant 0.0 : f64
  %tone = pulse.tone %f, %p : f64, f64 -> !pulse.tone
  %t0 = pulse.get_current_time %d : !pulse.drive_line -> !pulse.timepoint
  %d1, %tone1 = pulse.drive %d, %w, %tone : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %t1 = pulse.get_current_time %d1 : !pulse.drive_line -> !pulse.timepoint
  %dur = pulse.timepoint_subtract %t1, %t0 : (!pulse.timepoint, !pulse.timepoint) -> !pulse.duration
  %n = pulse.duration_to_int %dur : (!pulse.duration) -> i64
  return %d1, %tone1, %n : !pulse.drive_line, !pulse.tone, i64
}
