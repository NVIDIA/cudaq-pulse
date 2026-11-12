// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// pulse.schedule_state: scheduling marks a function timed, or placed when it
// moves an op by a line delay; a function without the attribute is
// unscheduled, and pulse-merge-waits takes only that.
// RUN: cudaq-pulse-opt --split-input-file --pulse-schedule-asap %s | FileCheck %s
// RUN: cudaq-pulse-opt --split-input-file --pulse-schedule-asap --pulse-verify %s | FileCheck %s
// RUN: cudaq-pulse-opt --split-input-file --pulse-schedule-asap %s | cudaq-pulse-opt --split-input-file | FileCheck %s
// RUN: not cudaq-pulse-opt --split-input-file --pulse-schedule-asap --pulse-merge-waits %s 2>&1 | FileCheck %s --check-prefix=MERGE

// CHECK-LABEL: func.func @no_delays
// CHECK-SAME: attributes {pulse.schedule_state = #pulse.schedule_state<timed>}
// MERGE: pulse-merge-waits takes an unscheduled function, and this one is timed; run it before pulse-schedule
func.func @no_delays() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

module attributes {pulse.line_delays = {drive = array<i64: 3>, readout = array<i64: 0>}} {
// CHECK-LABEL: func.func @delayed
// CHECK-SAME: attributes {pulse.schedule_state = #pulse.schedule_state<placed>}
// CHECK: pulse.drive {{.*}}<{duration_cycles = 8 : i64, start_cycles = 3 : i64}>
// MERGE: pulse-merge-waits takes an unscheduled function, and this one is placed
func.func @delayed() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
}

// -----

// A delay of zero moves nothing: the function is timed.
module attributes {pulse.line_delays = {drive = array<i64: 0>, readout = array<i64: 0>}} {
// CHECK-LABEL: func.func @zero_delay
// CHECK-SAME: attributes {pulse.schedule_state = #pulse.schedule_state<timed>}
func.func @zero_delay() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %w = pulse.square %c8, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
}
