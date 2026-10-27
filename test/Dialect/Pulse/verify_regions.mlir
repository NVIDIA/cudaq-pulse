// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: not cudaq-pulse-opt --pulse-verify --split-input-file %s 2>&1 | FileCheck %s

// Each arm may consume the line once: only one arm runs.
// CHECK-NOT: error
func.func @one_use_per_arm(%x: !pulse.waveform) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %r1, %rt1, %m = pulse.readout %r0, %x, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %bit = pulse.is_set %m : !pulse.measurement -> i1
  %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %d, %t = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d, %t : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// CHECK: error: linear pulse value is consumed more than once on one path
func.func @two_uses_in_one_arm(%x: !pulse.waveform, %bit: i1) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %d, %t = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %e, %u = pulse.drive %d0, %x, %t : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %e, %u : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// CHECK: error: linear pulse value from outside an scf.for is used in its body
func.func @line_not_carried(%x: !pulse.waveform) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %d, %t = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// CHECK: error: operation in a measurement-conditioned branch starts at 100, before its measurement completes at 300
func.func @branch_before_measurement(%x: !pulse.waveform) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %r1, %rt1, %m = pulse.readout %r0, %x, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 300 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %bit = pulse.is_set %m : !pulse.measurement -> i1
  %d1, %t1 = scf.if %bit -> (!pulse.drive_line, !pulse.tone) {
    %d, %t = pulse.drive %d0, %x, %t0 {start_cycles = 100 : i64, duration_cycles = 40 : i64} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    scf.yield %d, %t : !pulse.drive_line, !pulse.tone
  } else {
    scf.yield %d0, %t0 : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// The second iteration's drive starts before the first one's wait is over.
// CHECK: error: operations on one line in a loop must share its period
func.func @loop_periods_disagree(%x: !pulse.waveform, %w: !pulse.duration) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %dd, %tt = pulse.drive %d, %x, %t {start_cycles = 0 : i64, duration_cycles = 40 : i64, period_cycles = array<i64: 80>} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %dw = pulse.wait %dd, %w {start_cycles = 40 : i64, duration_cycles = 60 : i64, period_cycles = array<i64: 100>} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %dw, %tt : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// CHECK: error: operation overlaps or precedes its physical-line predecessor: start 80 < prior end 100
func.func @loop_iterations_overlap(%x: !pulse.waveform, %w: !pulse.duration) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %lb = arith.constant 0 : index
  %ub = arith.constant 4 : index
  %step = arith.constant 1 : index
  %d2, %t2 = scf.for %i = %lb to %ub step %step iter_args(%d = %d0, %t = %t0) -> (!pulse.drive_line, !pulse.tone) {
    %dd, %tt = pulse.drive %d, %x, %t {start_cycles = 0 : i64, duration_cycles = 40 : i64, period_cycles = array<i64: 80>} : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
    %dw = pulse.wait %dd, %w {start_cycles = 40 : i64, duration_cycles = 60 : i64, period_cycles = array<i64: 80>} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %dw, %tt : !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// A line acquired in a loop body is one allocation for every iteration.
// CHECK: error: a line cannot be acquired inside an scf.for; acquire it before the loop and carry it through iter_args
func.func @line_acquired_in_loop(%x: !pulse.waveform) {
  %q = pulse.qudit_alloc : !pulse.qref
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
    %d, %t = pulse.drive %d0, %x, %t0 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  }
  return
}

// -----

// CHECK: error: a line cannot be acquired inside an scf.for
func.func @readout_line_acquired_in_loop(%x: !pulse.waveform) {
  %q = pulse.qudit_alloc : !pulse.qref
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    %r0, %rt0 = pulse.get_readout_line %q : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
    %r1, %rt1, %m = pulse.readout %r0, %x, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  }
  return
}
