// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt %s --mlir-print-op-generic | cudaq-pulse-opt | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-verify %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=VERIFY --allow-empty

// A coupling line reaches the control of its qudits, one or several, and
// plays a waveform with no tone: pulse.drive on it takes the line and the
// waveform and returns the line. It waits and syncs as any line, and a
// drive on a drive line prints as it always has.

// VERIFY-NOT: error
// VERIFY: warning: coupling line allocated but never used (qudits 0, 1)
// VERIFY-NOT: error

// CHECK-LABEL: func @coupling_lines
// CHECK: %[[Q0:.*]] = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
// CHECK: %[[Q1:.*]] = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
// CHECK: %[[C:.*]] = pulse.get_coupling_line %[[Q0]] : (!pulse.qudit) -> !pulse.coupling_line
// CHECK: %[[P:.*]] = pulse.get_coupling_line %[[Q0]], %[[Q1]] : (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line
// CHECK: %[[D:.*]], %[[T:.*]] = pulse.get_drive_line %[[Q0]]
// CHECK: %[[C1:.*]] = pulse.drive %[[C]], %[[W:.*]] : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
// CHECK: %[[C2:.*]] = pulse.wait %[[C1]], %{{.*}} : (!pulse.coupling_line, !pulse.duration) -> !pulse.coupling_line
// CHECK: %[[D1:.*]], %[[T1:.*]] = pulse.drive %[[D]], %[[W]], %[[T]] : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
// CHECK: %[[S:.*]]:2 = pulse.sync %[[C2]], %[[D1]] : !pulse.coupling_line, !pulse.drive_line -> !pulse.coupling_line, !pulse.drive_line
// CHECK: %[[C3:.*]] = pulse.drive %[[S]]#0, %[[W]] : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
// CHECK: pulse.get_current_time %[[C3]] : !pulse.coupling_line -> !pulse.timepoint
func.func @coupling_lines() -> !pulse.timepoint {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q0 : (!pulse.qudit) -> !pulse.coupling_line
  %p = pulse.get_coupling_line %q0, %q1 : (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line
  %d, %t = pulse.get_drive_line %q0 {frequency_hz = 5.0e9 : f64}
      : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %duration = arith.constant 40 : i64
  %amplitude = arith.constant 0.31 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %duration, %amplitude, %zero : i64, f64, f64 -> !pulse.waveform
  %c1 = pulse.drive %c, %w : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %cycles = arith.constant 20 : i64
  %length = pulse.duration_from_int %cycles : (i64) -> !pulse.duration
  %c2 = pulse.wait %c1, %length : (!pulse.coupling_line, !pulse.duration) -> !pulse.coupling_line
  %d1, %t1 = pulse.drive %d, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %c3, %d2 = pulse.sync %c2, %d1 : !pulse.coupling_line, !pulse.drive_line -> !pulse.coupling_line, !pulse.drive_line
  %c4 = pulse.drive %c3, %w : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %now = pulse.get_current_time %c4 : !pulse.coupling_line -> !pulse.timepoint
  return %now : !pulse.timepoint
}

// A coupling line crosses an scf.if and an scf.for as any line does, and a
// measurement-conditioned pulse on it is an arm's drive.

// CHECK-LABEL: func @carried
// CHECK: %[[IF:.*]] = scf.if %{{.*}} -> (!pulse.coupling_line) {
// CHECK: pulse.drive %{{.*}}, %{{.*}} : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
// CHECK: scf.yield %{{.*}} : !pulse.coupling_line
// CHECK: %{{.*}} = scf.for %{{.*}} iter_args(%[[IN:.*]] = %[[IF]]) -> (!pulse.coupling_line) {
// CHECK: %[[OUT:.*]] = pulse.drive %[[IN]], %{{.*}} : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
// CHECK: scf.yield %[[OUT]] : !pulse.coupling_line
func.func @carried(%bit : i1) {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q0 : (!pulse.qudit) -> !pulse.coupling_line
  %samples = pulse.custom_samples [0.1, 0.2, -0.3, 0.0] : !pulse.waveform
  %c1 = scf.if %bit -> (!pulse.coupling_line) {
    %driven = pulse.drive %c, %samples : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
    scf.yield %driven : !pulse.coupling_line
  } else {
    scf.yield %c : !pulse.coupling_line
  }
  %lower = arith.constant 0 : index
  %upper = arith.constant 3 : index
  %step = arith.constant 1 : index
  %c2 = scf.for %i = %lower to %upper step %step iter_args(%line = %c1) -> (!pulse.coupling_line) {
    %next = pulse.drive %line, %samples : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
    scf.yield %next : !pulse.coupling_line
  }
  return
}
