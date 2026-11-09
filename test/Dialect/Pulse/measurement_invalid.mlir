// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// A measurement comes from a readout on every path it may take: through
// either arm of an scf.if, a loop's initial and yielded value (only the
// yielded one after a loop that runs), or a schedule block. A function
// argument is the caller's.

func.func @not_from_readout(%b: i1) -> i1 {
  %m = builtin.unrealized_conversion_cast %b : i1 to !pulse.measurement
  // expected-error @+1 {{'pulse.is_set' op measurement must be produced by pulse.readout, got builtin.unrealized_conversion_cast}}
  %c = pulse.is_set %m : !pulse.measurement -> i1
  return %c : i1
}

// -----

func.func @outcome_not_from_a_readout(%b: i1) -> i64 {
  %m = builtin.unrealized_conversion_cast %b : i1 to !pulse.measurement
  // expected-error @+1 {{'pulse.outcome' op measurement must be produced by pulse.readout, got builtin.unrealized_conversion_cast}}
  %k = pulse.outcome %m : !pulse.measurement -> i64
  return %k : i64
}

// -----

// Each arm reads out.
func.func @from_a_readout_in_each_arm(%b: i1) -> i1 {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
  %r, %to = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %wf = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %m = scf.if %b -> (!pulse.measurement) {
    %r1, %t1, %m1 = pulse.readout %r, %wf, %to, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %m1 : !pulse.measurement
  } else {
    %r1, %t1, %m1 = pulse.readout %r, %wf, %to, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %m1 : !pulse.measurement
  }
  %res = pulse.is_set %m : !pulse.measurement -> i1
  return %res : i1
}

// -----

func.func @from_a_cast_in_one_arm(%b: i1) -> i1 {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qref
  %r, %to = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %c40 = arith.constant 40 : i64
  %amp = arith.constant 0.1 : f64
  %zero = arith.constant 0.0 : f64
  %wf = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
  %m = scf.if %b -> (!pulse.measurement) {
    %r1, %t1, %m1 = pulse.readout %r, %wf, %to, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %m1 : !pulse.measurement
  } else {
    %m2 = builtin.unrealized_conversion_cast %b : i1 to !pulse.measurement
    scf.yield %m2 : !pulse.measurement
  }
  // expected-error @+1 {{'pulse.is_set' op measurement must be produced by pulse.readout, got builtin.unrealized_conversion_cast}}
  %res = pulse.is_set %m : !pulse.measurement -> i1
  return %res : i1
}

// -----

// A loop carries the last iteration's readout; its initial value is one too.
func.func @carried_by_a_loop(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i1 {
  %r1, %t1, %m0 = pulse.readout %r0, %wf, %t0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %rn, %tn, %mn = scf.for %i = %lb to %ub step %step iter_args(%r = %r1, %t = %t1, %m = %m0) -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %r2, %t2, %m2 = pulse.readout %r, %wf, %t, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %r2, %t2, %m2 : !pulse.readout_line, !pulse.tone, !pulse.measurement
  }
  %res = pulse.is_set %mn : !pulse.measurement -> i1
  return %res : i1
}

// -----

// A schedule block passes a measurement through, in and out.
func.func @through_a_schedule_block(%m: !pulse.measurement, %d: !pulse.drive_line) -> i1 {
  %d1, %m1 = pulse.schedule_block "asap" (%d, %m) : (!pulse.drive_line, !pulse.measurement) -> (!pulse.drive_line, !pulse.measurement) {
  ^bb0(%db: !pulse.drive_line, %mb: !pulse.measurement):
    %bit = pulse.is_set %mb : !pulse.measurement -> i1
    pulse.yield %db, %mb : !pulse.drive_line, !pulse.measurement
  }
  %res = pulse.is_set %m1 : !pulse.measurement -> i1
  return %res : i1
}

// -----

// After a loop that runs, its result is the last iteration's readout: the
// initial value, a placeholder, is never read.
func.func @after_a_loop_that_runs(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i1 {
  %none = ub.poison : !pulse.measurement
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %rn, %tn, %mn = scf.for %i = %lb to %ub step %step iter_args(%r = %r0, %t = %t0, %m = %none) -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
    %r2, %t2, %m2 = pulse.readout %r, %wf, %t, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %r2, %t2, %m2 : !pulse.readout_line, !pulse.tone, !pulse.measurement
  }
  %res = pulse.is_set %mn : !pulse.measurement -> i1
  return %res : i1
}

// -----

// In its first iteration a loop body reads the initial value.
func.func @placeholder_read_in_the_first_iteration(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) {
  %none = ub.poison : !pulse.measurement
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %rn, %tn, %mn = scf.for %i = %lb to %ub step %step iter_args(%r = %r0, %t = %t0, %m = %none) -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
    // expected-error @+1 {{'pulse.is_set' op measurement must be produced by pulse.readout, got ub.poison}}
    %bit = pulse.is_set %m : !pulse.measurement -> i1
    %r2, %t2, %m2 = pulse.readout %r, %wf, %t, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %r2, %t2, %m2 : !pulse.readout_line, !pulse.tone, !pulse.measurement
  }
  return
}

// -----

// A loop that may not run returns its initial value.
func.func @after_a_loop_of_argument_trips(%n: index, %r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i1 {
  %none = ub.poison : !pulse.measurement
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  %rn, %tn, %mn = scf.for %i = %lb to %n step %step iter_args(%r = %r0, %t = %t0, %m = %none) -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
    %r2, %t2, %m2 = pulse.readout %r, %wf, %t, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    scf.yield %r2, %t2, %m2 : !pulse.readout_line, !pulse.tone, !pulse.measurement
  }
  // expected-error @+1 {{'pulse.is_set' op measurement must be produced by pulse.readout, got ub.poison}}
  %res = pulse.is_set %mn : !pulse.measurement -> i1
  return %res : i1
}
