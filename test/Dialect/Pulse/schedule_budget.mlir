// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-asap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-alap --verify-diagnostics

// max_duration_cycles on a schedule block, and pulse.max_duration_cycles on
// the module, bound every execution of the block and the whole program.

!d = !pulse.drive_line
!t = !pulse.tone
!w = !pulse.waveform

// Two 40-cycle drives on one line fit 80 cycles, called 100 times.
func.func @fits(%d0: !d, %t0: !t) {
  %c = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 10.0 : f64
  %x = pulse.gaussian %c, %a, %s : i64, f64, f64 -> !w
  %lb = arith.constant 0 : index
  %ub = arith.constant 100 : index
  %step = arith.constant 1 : index
  %dn, %tn = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!d, !t) {
    %d1, %t1 = pulse.schedule_block "asap" (%d, %t) {callee = "cycle", max_duration_cycles = 80 : i64} : (!d, !t) -> (!d, !t) {
    ^bb0(%db: !d, %tb: !t):
      %d2, %t2 = pulse.drive %db, %x, %tb : !d, !w, !t -> !d, !t
      %d3, %t3 = pulse.drive %d2, %x, %t2 : !d, !w, !t -> !d, !t
      pulse.yield %d3, %t3 : !d, !t
    }
    scf.yield %d1, %t1 : !d, !t
  }
  return
}

// -----

!d = !pulse.drive_line
!t = !pulse.tone

func.func @overruns(%d0: !d, %t0: !t) {
  %c = arith.constant 40 : i64
  %a = arith.constant 0.1 : f64
  %s = arith.constant 10.0 : f64
  %x = pulse.gaussian %c, %a, %s : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{each call of kernel 'cycle' takes 80 cycles, more than max_duration_cycles=79; let its operations overlap, shorten them or raise max_duration_cycles}}
  %d1, %t1 = pulse.schedule_block "asap" (%d0, %t0) {callee = "cycle", max_duration_cycles = 79 : i64} : (!d, !t) -> (!d, !t) {
  ^bb0(%db: !d, %tb: !t):
    %d2, %t2 = pulse.drive %db, %x, %tb : !d, !pulse.waveform, !t -> !d, !t
    %d3, %t3 = pulse.drive %d2, %x, %t2 : !d, !pulse.waveform, !t -> !d, !t
    pulse.yield %d3, %t3 : !d, !t
  }
  return
}

// -----

// A wait whose length a parameter sets counts at its highest length.
!d = !pulse.drive_line

func.func @variable_wait(%tau: f64, %d0: !d) {
  %n = arith.fptosi %tau : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{each call of kernel 'idle' takes up to 400 cycles (8 as scheduled, its variable waits add up to 392), more than max_duration_cycles=100}}
  %d1 = pulse.schedule_block "asap" (%d0) {callee = "idle", max_duration_cycles = 100 : i64} : (!d) -> (!d) {
  ^bb0(%db: !d):
    %d2 = pulse.wait %db, %len {pulse.variable_wait, pulse.min_cycles = 8 : i64, pulse.max_cycles = 400 : i64}
        : (!d, !pulse.duration) -> !d
    pulse.yield %d2 : !d
  }
  return
}

// -----

!d = !pulse.drive_line

func.func @unbounded_wait(%tau: f64, %d0: !d) {
  %n = arith.fptosi %tau : f64 to i64
  %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d1 = pulse.schedule_block "asap" (%d0) {callee = "idle", max_duration_cycles = 100 : i64} : (!d) -> (!d) {
  ^bb0(%db: !d):
    // expected-error @+1 {{each call of kernel 'idle' has max_duration_cycles=100, but this wait in it has no highest length; bound it with wait(..., bounds=(lowest, highest)), or drop max_duration_cycles}}
    %d2 = pulse.wait %db, %len {pulse.variable_wait, pulse.min_cycles = 8 : i64}
        : (!d, !pulse.duration) -> !d
    pulse.yield %d2 : !d
  }
  return
}

// -----

// The program: 100 iterations of a 40-cycle drive end at 4000.
module attributes {pulse.max_duration_cycles = 3999 : i64} {
  // expected-error @+1 {{kernel 'main' takes 4000 cycles, more than max_duration_cycles=3999}}
  func.func @program(%d0: !pulse.drive_line, %t0: !pulse.tone) {
    %c = arith.constant 40 : i64
    %a = arith.constant 0.1 : f64
    %s = arith.constant 10.0 : f64
    %x = pulse.gaussian %c, %a, %s : i64, f64, f64 -> !pulse.waveform
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index
    %dn, %tn = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
      %d1, %t1 = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
    }
    return
  }
}

// -----

module attributes {pulse.max_duration_cycles = 4000 : i64} {
  func.func @program_fits(%d0: !pulse.drive_line, %t0: !pulse.tone) {
    %c = arith.constant 40 : i64
    %a = arith.constant 0.1 : f64
    %s = arith.constant 10.0 : f64
    %x = pulse.gaussian %c, %a, %s : i64, f64, f64 -> !pulse.waveform
    %lb = arith.constant 0 : index
    %ub = arith.constant 100 : index
    %step = arith.constant 1 : index
    %dn, %tn = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
      %d1, %t1 = pulse.drive %d, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
      scf.yield %d1, %t1 : !pulse.drive_line, !pulse.tone
    }
    return
  }
}
