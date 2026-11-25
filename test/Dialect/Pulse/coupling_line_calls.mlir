// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --pulse-verify | FileCheck %s --check-prefix=KEEP
// RUN: cudaq-pulse-opt %s --pulse-verify --pulse-expand-calls --pulse-verify | FileCheck %s

// A coupling line beside a placed call: the caller fetches the line and
// plays on it, and calls a function that plays on a drive line. pulse-verify
// accepts the module as it is, and with the call written out it is the flat
// program, which the checks of a module that holds a coupling line accept.

// The schedule passes, in the variant of a module that holds a coupling
// line, refuse the call as they refuse any.
// RUN: not cudaq-pulse-opt %s --pulse-schedule-asap 2>&1 | FileCheck %s --check-prefix=PLACE

// KEEP: module attributes {pulse.coupling_lines}
// KEEP: pulse.call @x90

// PLACE: the schedule passes do not place a pulse.call yet; write the ops of @x90 in its place before scheduling

// CHECK: module attributes {pulse.coupling_lines}
// CHECK-NOT: func.func private @x90
// CHECK-LABEL: func.func @shot
// CHECK-NOT: pulse.call
// CHECK: pulse.drive_baseband {{.*}}<{duration_cycles = 60 : i64, start_cycles = 0 : i64}>
// CHECK-NOT: pulse.call
// CHECK: pulse.drive {{.*}}<{duration_cycles = 40 : i64, start_cycles = 20 : i64}>
// CHECK-NOT: pulse.call
// CHECK: pulse.drive_baseband {{.*}}<{duration_cycles = 60 : i64, start_cycles = 60 : i64}>
// CHECK-NOT: pulse.call
module attributes {pulse.coupling_lines} {
func.func private @x90(%d: !pulse.drive_line, %t: !pulse.tone) -> (!pulse.drive_line, !pulse.tone)
    attributes {pulse.length_cycles = 40 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %n = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %w = pulse.square %n, %a, %z : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %w, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %d1, %t1 : !pulse.drive_line, !pulse.tone
}
func.func @shot() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c0 = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %n = arith.constant 60 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %w = pulse.square %n, %a, %z : i64, f64, f64 -> !pulse.waveform
  %c1 = pulse.drive_baseband %c0, %w <{duration_cycles = 60 : i64, start_cycles = 0 : i64}> : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %e:2 = pulse.call @x90(%d0, %t0) <{duration_cycles = 40 : i64, start_cycles = 20 : i64}> : (!pulse.drive_line, !pulse.tone) -> (!pulse.drive_line, !pulse.tone)
  %c2 = pulse.drive_baseband %c1, %w <{duration_cycles = 60 : i64, start_cycles = 60 : i64}> : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
}
