// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// pulse-verify on classical calls: answers cannot shape the timeline (V1),
// blocking waits stay out of blocks and branch arms (V2), and links and
// futures are linear like lines.

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

func.func @duration_from_answer() {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @f : i64
  %two = arith.constant 2 : i64
  %d = arith.muli %n, %two : i64
  %a = arith.constant 0.1 : f64
  // expected-error @+1 {{a duration cannot depend on the answer of a classical call}}
  %w = pulse.square %d, %a, %a : i64, f64, f64 -> !pulse.waveform
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

func.func @wait_from_answer() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @f : i64
  // expected-error @+1 {{a duration cannot depend on the answer of a classical call}}
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  // expected-error @+1 {{a wait whose length is not a constant must be marked pulse.variable_wait}}
  %d1 = pulse.wait %d0, %dur : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

func.func @loop_bound_from_answer() {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @f : i64
  %ub = arith.index_cast %n : i64 to index
  %lb = arith.constant 0 : index
  %step = arith.constant 1 : index
  // expected-error @+1 {{a loop bound cannot depend on the answer of a classical call}}
  scf.for %i = %lb to %ub step %step {
  }
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "blocking",
    submit = "eager"}

func.func @blocking_in_block() {
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l2 = pulse.schedule_block "asap" (%l0) : (!classical.link) -> (!classical.link) {
  ^bb0(%l: !classical.link):
    %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
    // expected-error @+1 {{a blocking wait cannot be inside a pulse.schedule_block}}
    %v = classical.wait %h, @f : f64
    pulse.yield %l1 : !classical.link
  }
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "blocking",
    submit = "eager"}

func.func @blocking_in_branch(%m: !pulse.measurement) {
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l0, @f() : () -> (!classical.link, !classical.future)
  %c = pulse.is_set %m : !pulse.measurement -> i1
  scf.if %c {
    // expected-error @+1 {{a blocking wait cannot be inside one arm of an scf.if}}
    %v = classical.wait %h, @f : f64
  }
  return
}

// -----

func.func @link_in_loop() {
  %lb = arith.constant 0 : index
  %ub = arith.constant 2 : index
  %step = arith.constant 1 : index
  scf.for %i = %lb to %ub step %step {
    // expected-error @+1 {{a link cannot be acquired inside an scf.for}}
    %l = classical.get_link {device = 0 : i64} : !classical.link
  }
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

func.func @link_used_twice() {
  // expected-error @+1 {{linear pulse value is consumed more than once on one path}}
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %l2, %g = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

func.func @future_waited_twice() {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{linear pulse value is consumed more than once on one path}}
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %a = classical.wait %h, @f : f64
  %b = classical.wait %h, @f : f64
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

// A variable wait may take its length from an answer.
func.func @variable_wait_is_allowed() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @f : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d1 = pulse.wait %d0, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 0 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

func.func @variable_wait_in_block() {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @f : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d2 = pulse.schedule_block "asap" (%d0) : (!pulse.drive_line) -> (!pulse.drive_line) {
  ^bb0(%d: !pulse.drive_line):
    // expected-error @+1 {{a variable wait cannot be inside a pulse.schedule_block}}
    %d1 = pulse.wait %d, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 0 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    pulse.yield %d1 : !pulse.drive_line
  }
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

func.func @variable_wait_in_branch(%m: !pulse.measurement) {
  %q = pulse.qudit_alloc : !pulse.qref
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @f : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %c = pulse.is_set %m : !pulse.measurement -> i1
  %d2 = scf.if %c -> (!pulse.drive_line) {
    // expected-error @+1 {{a variable wait cannot be inside one arm of an scf.if}}
    %d1 = pulse.wait %d0, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 0 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    scf.yield %d1 : !pulse.drive_line
  } else {
    scf.yield %d0 : !pulse.drive_line
  }
  return
}
