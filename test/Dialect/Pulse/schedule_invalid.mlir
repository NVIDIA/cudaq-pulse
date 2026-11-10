// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-asap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-alap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-rcp --verify-diagnostics

// One case per diagnostic of the schedule passes that fails a program before
// it is placed. A pulse or wait whose length is not known cannot be placed:
// it is an error, not a placement that assumes some length.

// A waveform that is an argument has no known duration.
func.func @argument_waveform(%w: !pulse.waveform) {
  %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{cannot determine the duration of this pulse; specialize the waveform to a constant duration first}}
  %d1, %t1 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d1, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// Nor does a built-in waveform whose duration is an argument.
func.func @argument_duration(%dur: i64) {
  %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %amp = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %c40 = arith.constant 40 : i64
  %w = pulse.square %dur, %amp, %z : i64, f64, f64 -> !pulse.waveform
  %x = pulse.square %c40, %amp, %z : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{cannot determine the duration of this pulse}}
  %d1, %t1 = pulse.drive %d0, %w, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d1, %x, %t1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// Waveform arithmetic has the duration of its operands.
func.func @scaled_argument_waveform(%w: !pulse.waveform, %d0: !pulse.drive_line, %t0: !pulse.tone) {
  %s = arith.constant 0.5 : f64
  %half = pulse.scale %w, %s : !pulse.waveform, f64 -> !pulse.waveform
  // expected-error @+1 {{cannot determine the duration of this pulse}}
  %d1, %t1 = pulse.drive %d0, %half, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// A wait whose length is an argument is only placed when it is variable,
// with its bounds.
func.func @argument_wait(%n: i64, %d0: !pulse.drive_line) {
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{cannot determine the length of this wait; give it a constant length, or make it a pulse.variable_wait with its bounds}}
  %d1 = pulse.wait %d0, %len : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @negative_wait(%d0: !pulse.drive_line) {
  %n = arith.constant -5 : i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{a wait cannot last -5 cycles}}
  %d1 = pulse.wait %d0, %len : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @loop_of_argument_trips(%n: index, %d0: !pulse.drive_line, %t0: !pulse.tone) {
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %x = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  // expected-error @+1 {{cannot schedule a loop with non-constant bounds}}
  %dn, %tn = scf.for %i = %lb to %n step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %d1, %t1 = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// A qudit has one drive line however often it is acquired, so the two
// versions below would give one line two timelines.
func.func @drive_line_acquired_twice() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
  // expected-note @+1 {{acquired first here}}
  %a, %ta = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{this qudit's drive line is already acquired; use the line acquired first}}
  %b, %tb = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %a1, %ta1 = pulse.drive %a, %x, %ta : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b1, %tb1 = pulse.drive %b, %x, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}
