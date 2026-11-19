// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A condition on measurement results a kernel kept in a value: selects of
// true and false on each measurement, combined, and an if on the result. The
// if becomes ifs on the measurements, each tested where the condition still
// depends on it, as `if s0 and not s1:` written in the if gives.
// RUN: cudaq-pulse-opt --pulse-lower-selects %s | FileCheck %s
// RUN: cudaq-pulse-opt --pulse-lower-selects --pulse-schedule-alap --pulse-verify %s | FileCheck %s --check-prefix=TIME

// s0 and not s1: an if on s0 whose then arm is an if on s1; the drive is in
// the else arm of the inner if only.
// CHECK-LABEL: func.func @and_not
// CHECK-NOT: pulse.selected_values
// CHECK-NOT: arith.select
// CHECK: %[[S0:.*]] = pulse.is_set %[[M0:.*]] :
// CHECK-NEXT: scf.if %[[S0]]
// CHECK-NEXT: %[[S1:.*]] = pulse.is_set
// CHECK-NEXT: scf.if %[[S1]]
// CHECK-NOT: pulse.drive %
// CHECK: } else {
// CHECK: pulse.drive %
// CHECK: } else {
// CHECK-NOT: pulse.drive %
// CHECK: return
// TIME-LABEL: func.func @and_not
// TIME: pulse.readout {{.*}}start_cycles = 0
// TIME: pulse.readout {{.*}}start_cycles = 0
// TIME: pulse.drive {{.*}}start_cycles = 100
func.func @and_not() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %r1, %rt1 = pulse.get_readout_line %q1 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c100 = arith.constant 100 : i64
  %zero = arith.constant 0.0 : f64
  %amp = arith.constant 0.1 : f64
  %c0 = arith.constant 0 : i64
  %true = arith.cmpi eq, %c0, %c0 : i64
  %false = arith.cmpi ne, %c0, %c0 : i64
  %probe = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %r0a, %rt0a, %m0 = pulse.readout %r0, %probe, %rt0, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %r1a, %rt1a, %m1 = pulse.readout %r1, %probe, %rt1, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %b0 = pulse.is_set %m0 : !pulse.measurement -> i1
  %s0 = arith.select %b0, %true, %false : i1
  %b1 = pulse.is_set %m1 : !pulse.measurement -> i1
  %s1 = arith.select %b1, %true, %false : i1
  %n1 = arith.cmpi eq, %s1, %false : i1
  %x = arith.andi %s0, %n1 : i1
  %wf = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = scf.if %x -> (!pulse.drive_line, !pulse.tone) {
    %d2, %t2 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d2, %t2 : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d, %t : !pulse.drive_line, !pulse.tone
  }
  return
}

// s0 or s1: the drive in the then arm of the if on s0, and in the inner if on
// s1 of its else arm; s1 is not tested where s0 decides.
// CHECK-LABEL: func.func @or
// CHECK: %[[S0:.*]] = pulse.is_set
// CHECK-NEXT: scf.if %[[S0]]
// CHECK-NOT: pulse.is_set
// CHECK: pulse.drive %
// CHECK: } else {
// CHECK-NEXT: %[[S1:.*]] = pulse.is_set
// CHECK-NEXT: scf.if %[[S1]]
// CHECK: pulse.drive %
// CHECK-NOT: pulse.is_set
// CHECK: return
func.func @or() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q0 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %r1, %rt1 = pulse.get_readout_line %q1 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c100 = arith.constant 100 : i64
  %zero = arith.constant 0.0 : f64
  %amp = arith.constant 0.1 : f64
  %c0 = arith.constant 0 : i64
  %true = arith.cmpi eq, %c0, %c0 : i64
  %false = arith.cmpi ne, %c0, %c0 : i64
  %probe = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %r0a, %rt0a, %m0 = pulse.readout %r0, %probe, %rt0, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %r1a, %rt1a, %m1 = pulse.readout %r1, %probe, %rt1, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %b0 = pulse.is_set %m0 : !pulse.measurement -> i1
  %s0 = arith.select %b0, %true, %false : i1
  %b1 = pulse.is_set %m1 : !pulse.measurement -> i1
  %s1 = arith.select %b1, %true, %false : i1
  %x = arith.ori %s0, %s1 : i1
  %wf = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = scf.if %x -> (!pulse.drive_line, !pulse.tone) {
    %d2, %t2 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d2, %t2 : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d, %t : !pulse.drive_line, !pulse.tone
  }
  return
}
