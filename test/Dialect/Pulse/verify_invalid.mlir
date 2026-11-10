// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// One case per diagnostic of pulse-verify, and the programs it must accept.

// A qudit has one drive line: acquiring it twice on one path would give two
// versions of the same line.
func.func @drive_line_acquired_twice() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-note @+1 {{acquired first here}}
  %a, %ta = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{this qudit's drive line is already acquired; use the line acquired first}}
  %b, %tb = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a1, %ta1 = pulse.drive %a, %x, %ta : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %b1, %tb1 = pulse.drive %b, %x, %tb : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

func.func @readout_line_acquired_twice(%q: !pulse.qudit) {
  // expected-note @+1 {{acquired first here}}
  %a, %ta = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  // expected-error @+1 {{this qudit's readout line is already acquired}}
  %b, %tb = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %s:2 = pulse.sync %a, %b : !pulse.readout_line, !pulse.readout_line -> !pulse.readout_line, !pulse.readout_line
  return
}

// -----

// Its drive and readout lines are two lines, and so are two qudits' drive
// lines.
func.func @one_line_of_each_kind(%q: !pulse.qudit, %p: !pulse.qudit) {
  %d, %td = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r, %tr = pulse.get_readout_line %q : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %e, %te = pulse.get_drive_line %p : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %s:3 = pulse.sync %d, %r, %e : !pulse.drive_line, !pulse.readout_line, !pulse.drive_line -> !pulse.drive_line, !pulse.readout_line, !pulse.drive_line
  return
}

// -----

// The timing check keys a line on its qudit and kind, so pulses on two
// versions of one line overlap.
func.func @overlap_across_acquisitions() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-note @+1 {{acquired first here}}
  %a, %ta = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{this qudit's drive line is already acquired}}
  %b, %tb = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %a1, %ta1 = pulse.drive %a, %x, %ta <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  // expected-error @+1 {{operation overlaps or precedes its physical-line predecessor: start 0 < prior end 40}}
  %b1, %tb1 = pulse.drive %b, %x, %tb <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// An op in a loop is reported once, not once per iteration checked.
func.func @loop_op_without_its_period() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %dn, %tn = scf.for %i = %lb to %ub step %step iter_args(%di = %d, %ti = %t) -> (!pulse.drive_line, !pulse.tone) {
    // expected-error @+1 {{scheduled operation needs one period_cycles entry per enclosing scf.for}}
    %d1, %t1 = pulse.drive %di, %x, %ti <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

func.func @loop_op_before_time_zero() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %dn, %tn = scf.for %i = %lb to %ub step %step iter_args(%di = %d, %ti = %t) -> (!pulse.drive_line, !pulse.tone) {
    // expected-error @+1 {{operation is scheduled before time zero: start -1000}}
    %d1, %t1 = pulse.drive %di, %x, %ti <{duration_cycles = 40 : i64, period_cycles = array<i64: 40>, start_cycles = -1000 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

func.func @loop_op_with_a_negative_period() {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %lb = arith.constant 0 : index
  %ub = arith.constant 8 : index
  %step = arith.constant 1 : index
  %dn, %tn = scf.for %i = %lb to %ub step %step iter_args(%di = %d, %ti = %t) -> (!pulse.drive_line, !pulse.tone) {
    // expected-error @+1 {{period_cycles must not be negative}}
    %d1, %t1 = pulse.drive %di, %x, %ti <{duration_cycles = 40 : i64, period_cycles = array<i64: -40>, start_cycles = 1000 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

func.func @scheduled_loop_of_argument_trips(%n: index) {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d, %t = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %dn, %tn = scf.for %i = %lb to %n step %step iter_args(%di = %d, %ti = %t) -> (!pulse.drive_line, !pulse.tone) {
    // expected-error @+1 {{a scheduled loop needs constant bounds}}
    %d1, %t1 = pulse.drive %di, %x, %ti <{duration_cycles = 40 : i64, period_cycles = array<i64: 40>, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

func.func @scheduled_drive_on_an_argument_line(%d: !pulse.drive_line, %t: !pulse.tone) {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{cannot trace timed operation to a physical line}}
  %d1, %t1 = pulse.drive %d, %x, %t <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// A function is scheduled as a whole or not at all.
func.func @partly_scheduled() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %line, %tone = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 4.700000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %c100 = arith.constant 100 : i64
  %w100 = pulse.duration_from_int %c100 : (i64) -> !pulse.duration
  %c30 = arith.constant 30 : i64
  %w30 = pulse.duration_from_int %c30 : (i64) -> !pulse.duration
  // expected-note @+1 {{scheduled here}}
  %l1 = pulse.wait %line, %w100 <{duration_cycles = 100 : i64, start_cycles = 0 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  // expected-error @+1 {{operation is not scheduled, but other operations of its function are}}
  %l2 = pulse.wait %l1, %w30 : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  %l3, %t3 = pulse.drive %l2, %x, %tone <{duration_cycles = 40 : i64, start_cycles = 130 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

func.func @duration_that_is_not_the_pulse() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %line, %tone = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 4.700000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{duration_cycles is 999, but the operation takes 40 cycles}}
  %l1, %t1 = pulse.drive %line, %x, %tone <{duration_cycles = 999 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

func.func @start_without_a_duration() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %line, %tone = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c100 = arith.constant 100 : i64
  %w100 = pulse.duration_from_int %c100 : (i64) -> !pulse.duration
  // expected-error @+1 {{a scheduled operation needs duration_cycles}}
  %l1 = pulse.wait %line, %w100 <{start_cycles = 0 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @duration_without_a_start() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %line, %tone = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c100 = arith.constant 100 : i64
  %w100 = pulse.duration_from_int %c100 : (i64) -> !pulse.duration
  // expected-error @+1 {{a scheduled operation needs start_cycles}}
  %l1 = pulse.wait %line, %w100 <{duration_cycles = 100 : i64}> : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

func.func @scheduled_pulse_of_unknown_length(%x: !pulse.waveform) {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %line, %tone = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  // expected-error @+1 {{a scheduled operation needs a known length}}
  %l1, %t1 = pulse.drive %line, %x, %tone <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

// -----

// An op in a loop that never runs has a duration but no start.
func.func @zero_trip_loop_in_a_scheduled_function() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %line, %tone = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %a = arith.constant 4.700000e-02 : f64
  %z = arith.constant 0.000000e+00 : f64
  %x = pulse.square %c40, %a, %z : i64, f64, f64 -> !pulse.waveform
  %l1, %t1 = pulse.drive %line, %x, %tone <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %ln, %tn = scf.for %i = %lb to %lb step %step iter_args(%l = %l1, %t = %t1) -> (!pulse.drive_line, !pulse.tone) {
    %l2, %t2 = pulse.drive %l, %x, %t <{duration_cycles = 40 : i64}> : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %l2, %t2 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// A time read does not consume its line: it may be read, read again and
// then played on, and read in a loop that does not carry it.
func.func @time_reads_then_a_drive(%d: !pulse.drive_line, %w: !pulse.waveform, %t: !pulse.tone) -> (i64, i64, !pulse.drive_line, !pulse.tone) {
  %tp0 = pulse.get_current_time %d : !pulse.drive_line -> !pulse.timepoint
  %n0 = pulse.timepoint_to_int %tp0 : !pulse.timepoint -> i64
  %tp1 = pulse.get_current_time %d : !pulse.drive_line -> !pulse.timepoint
  %n1 = pulse.timepoint_to_int %tp1 : !pulse.timepoint -> i64
  %d1, %t1 = pulse.drive %d, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %n0, %n1, %d1, %t1 : i64, i64, !pulse.drive_line, !pulse.tone
}

// -----

func.func @time_read_in_a_loop(%d: !pulse.drive_line) -> !pulse.drive_line {
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %tp = pulse.get_current_time %d : !pulse.drive_line -> !pulse.timepoint
  }
  return %d : !pulse.drive_line
}

// -----

// A drive still consumes it: two drives of one version are two uses (of an
// argument, reported at its function).
// expected-error @+1 {{linear pulse value is consumed more than once on one path}}
func.func @time_read_and_two_drives(%d: !pulse.drive_line, %w: !pulse.waveform, %t: !pulse.tone, %u: !pulse.tone) -> (!pulse.timepoint, !pulse.drive_line, !pulse.drive_line) {
  %tp = pulse.get_current_time %d : !pulse.drive_line -> !pulse.timepoint
  %d1, %t1 = pulse.drive %d, %w, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  %d2, %t2 = pulse.drive %d, %w, %u : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return %tp, %d1, %d2 : !pulse.timepoint, !pulse.drive_line, !pulse.drive_line
}
