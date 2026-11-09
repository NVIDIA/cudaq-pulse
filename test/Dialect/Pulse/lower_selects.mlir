// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Values a measurement selects: kept where the target takes the parameter at
// run time, else the op that reads one runs in both arms of an if.
// RUN: cudaq-pulse-opt --pulse-lower-selects %s | FileCheck %s --check-prefixes=CHECK,ARMS
// RUN: cudaq-pulse-opt --pulse-lower-selects=runtime=amplitude,phase,frequency %s | FileCheck %s --check-prefixes=CHECK,KEEP
// RUN: cudaq-pulse-opt --pulse-lower-selects --pulse-schedule-asap %s | FileCheck %s --check-prefix=TIME
// RUN: cudaq-pulse-opt --pulse-lower-selects=runtime=amplitude,phase,frequency --pulse-schedule-asap %s | FileCheck %s --check-prefix=TIME
// RUN: cudaq-pulse-opt --pulse-lower-selects --pulse-schedule-alap %s | FileCheck %s --check-prefix=TIME
// RUN: cudaq-pulse-opt --pulse-lower-selects=runtime=amplitude,phase,frequency --pulse-schedule-alap %s | FileCheck %s --check-prefix=TIME
// RUN: cudaq-pulse-opt --pulse-lower-selects=runtime=amplitude --pulse-schedule-alap --pulse-verify %s > /dev/null
// RUN: not cudaq-pulse-opt --pulse-lower-selects=runtime=duration %s 2>&1 | FileCheck %s --check-prefix=BAD
// BAD: unknown run-time parameter 'duration'; choose amplitude, phase or frequency

// An amplitude: one select read by the waveform, or the waveform and the
// drive in each arm.
// CHECK-LABEL: func.func @amplitude
// CHECK-SAME: pulse.selected_values
// KEEP: %[[SEL:.*]] = arith.select
// KEEP: pulse.square %{{.*}}, %[[SEL]]
// KEEP-NOT: scf.if
// ARMS-NOT: arith.select
// ARMS: scf.if
// ARMS: pulse.square
// ARMS: pulse.drive %
// ARMS: } else {
// ARMS: pulse.square
// ARMS: pulse.drive %
// TIME-LABEL: func.func @amplitude
// TIME: pulse.readout {{.*}}start_cycles = 0
// TIME: pulse.drive {{.*}}start_cycles = 100
func.func @amplitude() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c100 = arith.constant 100 : i64
  %zero = arith.constant 0.0 : f64
  %low = arith.constant 0.1 : f64
  %high = arith.constant 0.3 : f64
  %probe = pulse.square %c100, %low, %zero : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r, %probe, %rt, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %bit = pulse.is_set %m : !pulse.measurement -> i1
  %amp = arith.select %bit, %high, %low : f64
  %wf = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// A phase on a frame read by a drive on its own line: the drive waits for
// the measurement whether the shift is kept or in the arms.
// CHECK-LABEL: func.func @phase
// KEEP: %[[SEL:.*]] = arith.select
// KEEP: pulse.shift_phase %{{.*}}, %[[SEL]]
// ARMS: scf.if
// ARMS: pulse.shift_phase
// ARMS: } else {
// ARMS: pulse.shift_phase
// TIME-LABEL: func.func @phase
// TIME: pulse.readout {{.*}}start_cycles = 0
// TIME: pulse.drive {{.*}}start_cycles = 100
func.func @phase() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c100 = arith.constant 100 : i64
  %zero = arith.constant 0.0 : f64
  %amp = arith.constant 0.1 : f64
  %pi = arith.constant 3.14159 : f64
  %half = arith.constant 1.5 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %probe = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r, %probe, %rt, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %bit = pulse.is_set %m : !pulse.measurement -> i1
  %ph = arith.select %bit, %pi, %half : f64
  %t2 = pulse.shift_phase %t1, %ph : !pulse.tone, f64 -> !pulse.tone
  %d2, %t3 = pulse.drive %d1, %x, %t2 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// A duration changes timing: the op runs in both arms at every setting,
// with the arithmetic between the select and the op.
// CHECK-LABEL: func.func @duration
// CHECK-NOT: arith.select
// CHECK: scf.if
// CHECK: arith.addi
// CHECK: pulse.square
// CHECK: pulse.drive %
// CHECK: } else {
// CHECK: arith.addi
// CHECK: pulse.square
// CHECK: pulse.drive %
// Each arm's duration is arithmetic on constants until it is folded; the
// schedule plays each pulse for that many cycles.
// TIME-LABEL: func.func @duration
// TIME: pulse.readout {{.*}}start_cycles = 0
// TIME: pulse.drive {{.*}}duration_cycles = 88 : i64, start_cycles = 100
// TIME: pulse.drive {{.*}}duration_cycles = 48 : i64
func.func @duration() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c8 = arith.constant 8 : i64
  %c40 = arith.constant 40 : i64
  %c80 = arith.constant 80 : i64
  %c100 = arith.constant 100 : i64
  %zero = arith.constant 0.0 : f64
  %amp = arith.constant 0.1 : f64
  %probe = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r, %probe, %rt, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %bit = pulse.is_set %m : !pulse.measurement -> i1
  %n = arith.select %bit, %c80, %c40 : i64
  %len = arith.addi %n, %c8 : i64
  %wf = pulse.square %len, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// An elif: the outer select's op is copied first, the inner one only into
// the arm that reads it, three drives in all.
// CHECK-LABEL: func.func @elif
// ARMS: scf.if %[[A:.*]] ->
// ARMS: pulse.drive %
// ARMS: } else {
// ARMS: scf.if
// ARMS: pulse.drive %
// ARMS: } else {
// ARMS: pulse.drive %
// ARMS-NOT: pulse.drive %
// KEEP-COUNT-2: arith.select
// KEEP-NOT: scf.if
func.func @elif() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c100 = arith.constant 100 : i64
  %zero = arith.constant 0.0 : f64
  %one = arith.constant 0.1 : f64
  %two = arith.constant 0.2 : f64
  %three = arith.constant 0.3 : f64
  %probe = pulse.square %c100, %one, %zero : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r, %probe, %rt, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %r2, %rt2, %n = pulse.readout %r1, %probe, %rt1, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %a = pulse.is_set %m : !pulse.measurement -> i1
  %b = pulse.is_set %n : !pulse.measurement -> i1
  %inner = arith.select %b, %two, %one : f64
  %amp = arith.select %a, %three, %inner : f64
  %wf = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %d1, %t1 = pulse.drive %d, %wf, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// Without a select the function is left as it is.
// CHECK-LABEL: func.func @none
// CHECK-NOT: pulse.selected_values
// CHECK: pulse.drive %
func.func @none() {
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %d1, %t1 = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// A tone made from a selected frequency only computes from it: each arm
// makes its tone and plays it.
// CHECK-LABEL: func.func @selected_tone_frequency
// ARMS: scf.if
// ARMS-NEXT: = pulse.tone %
// ARMS-NEXT: = pulse.drive %
// ARMS: } else {
// ARMS-NEXT: = pulse.tone %
// ARMS-NEXT: = pulse.drive %
func.func @selected_tone_frequency() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %rt = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %c100 = arith.constant 100 : i64
  %zero = arith.constant 0.0 : f64
  %amp = arith.constant 0.1 : f64
  %f0 = arith.constant 5.0e9 : f64
  %f1 = arith.constant 5.1e9 : f64
  %probe = pulse.square %c100, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %wf = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %r1, %rt1, %m = pulse.readout %r, %probe, %rt, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %bit = pulse.is_set %m : !pulse.measurement -> i1
  %f = arith.select %bit, %f1, %f0 : f64
  %tone = pulse.tone %f, %zero : f64, f64 -> !pulse.tone
  %d1, %t1 = pulse.drive %d, %wf, %tone : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
