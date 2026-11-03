// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-classical-check --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-classical-check=summarize-trips=0 --verify-diagnostics

// One case per diagnostic of pulse-classical-check. The schedules are written
// by hand so that each breaks exactly one rule; the scheduler never produces
// them. Long loops are checked from one copy of their body, or with
// summarize-trips=0 iteration by iteration, with the same diagnostics.

module attributes {pulse.clock_ghz = 1.0 : f64,
                   classical.links = {"0" = {slot_bytes = 32 : i64}}} {
  // expected-error @+1 {{a request of 'f' takes 40 bytes, but a slot of link 0 holds 32}}
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = ["f64", "f64"], results = [], latency_min_ns = 0.0,
      latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
      submit = "eager"}
}

// -----

module attributes {pulse.clock_ghz = 1.0 : f64,
                   classical.links = {"0" = {slot_bytes = 30 : i64}}} {
  // expected-error @+1 {{an answer of 'f' takes 32 bytes, but a slot of link 0 holds 30}}
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
      submit = "eager"}
}

// -----

// A deadline shorter than the budget: 1 cycle to submit and 1000 to answer
// is 1001, against a deadline of 800 (slack -201).
module attributes {pulse.clock_ghz = 1.0 : f64} {
  // expected-error @+1 {{'f' cannot meet its deadline: its answer is due 1001 cycles after the submission starts, but the deadline is 800 (slack -201)}}
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 500.0, latency_max_ns = 1000.0, mode = "bounded",
      submit = "eager", deadline_ns = 800.0}
}

// -----

// The link's feedback latency counts against the deadline too.
module attributes {pulse.clock_ghz = 1.0 : f64,
                   classical.links = {"0" = {feedback_ns = 300.0 : f64}}} {
  // expected-error @+1 {{its answer is due 1301 cycles after the submission starts, but the deadline is 1200 (slack -101)}}
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 500.0, latency_max_ns = 1000.0, mode = "bounded",
      submit = "eager", deadline_ns = 1200.0}
}

// -----

// S7: the answer is taken before it is due (10 + 1 + 100 = 111).
module attributes {pulse.clock_ghz = 1.0 : f64} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}
  func.func @early() {
    %l = classical.get_link {device = 0 : i64} : !classical.link
    %l1, %h = classical.submit %l, @f() {start_cycles = 10 : i64, duration_cycles = 1 : i64} : () -> (!classical.link, !classical.future)
    // expected-error @+1 {{the answer of 'f' is taken at 50, before it is due at 111}}
    %v = classical.wait %h, @f {start_cycles = 50 : i64, duration_cycles = 0 : i64} : f64
    return
  }
}

// -----

// S7 in a loop: due 1 + 100 + 10k, taken 101 + 5k, late from k = 1 on.
module attributes {pulse.clock_ghz = 1.0 : f64} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}
  func.func @drifting() {
    %l0 = classical.get_link {device = 0 : i64} : !classical.link
    %lb = arith.constant 0 : index
    %ub = arith.constant 4 : index
    %step = arith.constant 1 : index
    %ln = scf.for %i = %lb to %ub step %step iter_args(%l = %l0) -> (!classical.link) {
      %l1, %h = classical.submit %l, @f() {start_cycles = 0 : i64, duration_cycles = 1 : i64, period_cycles = array<i64: 10>} : () -> (!classical.link, !classical.future)
      // expected-error @+1 {{the answer of 'f' is taken at 116, before it is due at 131}}
      %v = classical.wait %h, @f {start_cycles = 101 : i64, duration_cycles = 0 : i64, period_cycles = array<i64: 5>} : f64
      scf.yield %l1 : !classical.link
    }
    return
  }
}

// -----

// S8: three requests in flight on a link with two slots.
module attributes {pulse.clock_ghz = 1.0 : f64,
                   classical.links = {"0" = {max_in_flight = 2 : i64}}} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = [], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}
  func.func @crowded() {
    %l0 = classical.get_link {device = 0 : i64} : !classical.link
    // expected-error @+1 {{link 0 holds 3 requests at once at 2, more than its 2 slots}}
    %l1, %a = classical.submit %l0, @f() {start_cycles = 0 : i64, duration_cycles = 1 : i64} : () -> (!classical.link, !classical.future)
    %l2, %b = classical.submit %l1, @f() {start_cycles = 1 : i64, duration_cycles = 1 : i64} : () -> (!classical.link, !classical.future)
    %l3, %c = classical.submit %l2, @f() {start_cycles = 2 : i64, duration_cycles = 1 : i64} : () -> (!classical.link, !classical.future)
    return
  }
}

// -----

// S9: the second submission starts before the first is sent.
module attributes {pulse.clock_ghz = 1.0 : f64,
                   classical.links = {"0" = {bytes_per_ns = 1.0 : f64}}} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = [], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}
  func.func @overlapping() {
    %l0 = classical.get_link {device = 0 : i64} : !classical.link
    %l1, %a = classical.submit %l0, @f() {start_cycles = 0 : i64, duration_cycles = 24 : i64} : () -> (!classical.link, !classical.future)
    // expected-error @+1 {{submission on link 0 starts at 10, before the one before it in program order ends at 24}}
    %l2, %b = classical.submit %l1, @f() {start_cycles = 10 : i64, duration_cycles = 24 : i64} : () -> (!classical.link, !classical.future)
    return
  }
}

// -----

// A future carried out of the loop that submitted it.
module attributes {pulse.clock_ghz = 1.0 : f64} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = ["f64"], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}
  func.func @escaping(%h0: !classical.future) {
    %l0 = classical.get_link {device = 0 : i64} : !classical.link
    %lb = arith.constant 0 : index
    %ub = arith.constant 1 : index
    %step = arith.constant 1 : index
    %ln, %hn = scf.for %i = %lb to %ub step %step iter_args(%l = %l0, %h = %h0) -> (!classical.link, !classical.future) {
      %l1, %g = classical.submit %l, @f() {start_cycles = 0 : i64, duration_cycles = 1 : i64} : () -> (!classical.link, !classical.future)
      scf.yield %l1, %g : !classical.link, !classical.future
    }
    // expected-error @+1 {{the answer of 'f' is taken in another loop iteration than its submission}}
    %v = classical.wait %hn, @f {start_cycles = 500 : i64, duration_cycles = 0 : i64} : f64
    return
  }
}

// -----

// reaction_deadline_ns: the answer is due 100 + 1 + 800 = 901, 801 cycles after the
// readout it carries ends at 100, against 600.
module attributes {pulse.clock_ghz = 1.0 : f64} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
      latency_nominal_ns = 800.0, latency_max_ns = 800.0, mode = "bounded",
      submit = "eager", reaction_deadline_ns = 600.0}
  func.func @slow(%r0: !pulse.readout_line, %rt0: !pulse.tone, %ro: !pulse.waveform) {
    %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 100 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %l = classical.get_link {device = 0 : i64} : !classical.link
    // expected-error @+1 {{'f' cannot react in time: its answer is due 801 cycles after the readouts it carries end, but reaction_deadline_ns allows 600 (slack -201); submit it sooner, lower max_latency_ns, use a faster link or raise reaction_deadline_ns}}
    %l1, %h = classical.submit %l, @f(%m) {start_cycles = 100 : i64, duration_cycles = 1 : i64} : (!pulse.measurement) -> (!classical.link, !classical.future)
    return
  }
}

// -----

// reaction_deadline_ns in a loop: the readout ends at 100 before the loop, and the
// submission starts at 100 + 50k, so the last of 8 iterations is the worst.
module attributes {pulse.clock_ghz = 1.0 : f64} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
      latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager", reaction_deadline_ns = 400.0}
  func.func @drifting(%r0: !pulse.readout_line, %rt0: !pulse.tone, %ro: !pulse.waveform) {
    %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 100 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %l0 = classical.get_link {device = 0 : i64} : !classical.link
    %lb = arith.constant 0 : index
    %ub = arith.constant 8 : index
    %step = arith.constant 1 : index
    %ln = scf.for %i = %lb to %ub step %step iter_args(%l = %l0) -> (!classical.link) {
      // expected-error @+1 {{'f' cannot react in time: in iteration (7) its answer is due 451 cycles after the readouts it carries end, but reaction_deadline_ns allows 400 (slack -51)}}
      %l1, %h = classical.submit %l, @f(%m) {start_cycles = 100 : i64, duration_cycles = 1 : i64, period_cycles = array<i64: 50>} : (!pulse.measurement) -> (!classical.link, !classical.future)
      scf.yield %l1 : !classical.link
    }
    return
  }
}

// -----

// reaction_deadline_ns: a wait whose length is known only when the program runs, between
// the readout and the submission, counts at its highest length: 101 + 300.
module attributes {pulse.clock_ghz = 1.0 : f64} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
      latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager", reaction_deadline_ns = 200.0}
  func.func @stretched(%tau: f64, %d0: !pulse.drive_line, %r0: !pulse.readout_line, %rt0: !pulse.tone, %ro: !pulse.waveform) {
    %r1, %rt1, %m = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 100 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    %n = arith.fptosi %tau : f64 to i64
    %len = pulse.duration_from_int %n : (i64) -> !pulse.duration
    %d1 = pulse.wait %d0, %len {pulse.variable_wait, pulse.min_cycles = 0 : i64, pulse.max_cycles = 300 : i64, start_cycles = 100 : i64, duration_cycles = 0 : i64}
        : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    %l = classical.get_link {device = 0 : i64} : !classical.link
    // expected-error @+1 {{its answer is due 401 cycles after the readouts it carries end, but reaction_deadline_ns allows 200 (slack -201)}}
    %l1, %h = classical.submit %l, @f(%m) {start_cycles = 100 : i64, duration_cycles = 1 : i64} : (!pulse.measurement) -> (!classical.link, !classical.future)
    return
  }
}

// -----

// reaction_deadline_ns: a measurement out of a branch has no readout known statically.
module attributes {pulse.clock_ghz = 1.0 : f64} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = ["bits:1"], results = ["bits:1"], latency_min_ns = 0.0,
      latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager", reaction_deadline_ns = 2000.0}
  func.func @branch(%c: i1, %r0: !pulse.readout_line, %rt0: !pulse.tone, %ro: !pulse.waveform) {
    %r1, %rt1, %m = scf.if %c -> (!pulse.readout_line, !pulse.tone, !pulse.measurement) {
      %r2, %rt2, %m2 = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 100 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
      scf.yield %r2, %rt2, %m2 : !pulse.readout_line, !pulse.tone, !pulse.measurement
    } else {
      %r3, %rt3, %m3 = pulse.readout %r0, %ro, %rt0, "iq" {start_cycles = 0 : i64, duration_cycles = 50 : i64} : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
      scf.yield %r3, %rt3, %m3 : !pulse.readout_line, !pulse.tone, !pulse.measurement
    }
    %l = classical.get_link {device = 0 : i64} : !classical.link
    // expected-error @+1 {{'f' has reaction_deadline_ns, but a measurement it carries comes out of a branch or a loop iteration, so its readout is not known; read it out where the request is made, or drop reaction_deadline_ns}}
    %l1, %h = classical.submit %l, @f(%m) {start_cycles = 100 : i64, duration_cycles = 1 : i64} : (!pulse.measurement) -> (!classical.link, !classical.future)
    return
  }
}

// -----

// S8 in a long loop: two requests start every 40 cycles, each in flight for
// 101, and the one before the loop is still in flight over the first three
// iterations.
module attributes {pulse.clock_ghz = 1.0 : f64,
                   classical.links = {"0" = {max_in_flight = 3 : i64}}} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = [], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}
  func.func @streaming() {
    %l0 = classical.get_link {device = 0 : i64} : !classical.link
    // expected-error @+1 {{link 0 holds 7 requests at once at 82, more than its 3 slots}}
    %l1, %a = classical.submit %l0, @f() {start_cycles = 0 : i64, duration_cycles = 1 : i64} : () -> (!classical.link, !classical.future)
    %lb = arith.constant 0 : index
    %ub = arith.constant 1000 : index
    %step = arith.constant 1 : index
    %ln = scf.for %i = %lb to %ub step %step iter_args(%l = %l1) -> (!classical.link) {
      %l2, %h = classical.submit %l, @f() {start_cycles = 1 : i64, duration_cycles = 1 : i64, period_cycles = array<i64: 40>} : () -> (!classical.link, !classical.future)
      %l3, %g = classical.submit %l2, @f() {start_cycles = 2 : i64, duration_cycles = 1 : i64, period_cycles = array<i64: 40>} : () -> (!classical.link, !classical.future)
      scf.yield %l3 : !classical.link
    }
    return
  }
}

// -----

// S9 in a long loop: the second request of each iteration starts 2 cycles
// after the first is sent in iteration 0 and 1 cycle earlier each
// iteration, so from iteration 3 on before it is sent.
module attributes {pulse.clock_ghz = 1.0 : f64} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = [], latency_min_ns = 0.0,
      latency_nominal_ns = 1.0, latency_max_ns = 1.0, mode = "bounded",
      submit = "eager"}
  func.func @drifting_order() {
    %l0 = classical.get_link {device = 0 : i64} : !classical.link
    %lb = arith.constant 0 : index
    %ub = arith.constant 1000 : index
    %step = arith.constant 1 : index
    %ln = scf.for %i = %lb to %ub step %step iter_args(%l = %l0) -> (!classical.link) {
      %l1, %h = classical.submit %l, @f() {start_cycles = 0 : i64, duration_cycles = 2 : i64, period_cycles = array<i64: 100>} : () -> (!classical.link, !classical.future)
      // expected-error @+1 {{submission on link 0 starts at 301, before the one before it in program order ends at 302}}
      %l2, %g = classical.submit %l1, @f() {start_cycles = 4 : i64, duration_cycles = 2 : i64, period_cycles = array<i64: 99>} : () -> (!classical.link, !classical.future)
      scf.yield %l2 : !classical.link
    }
    return
  }
}

// -----

// S8 in the middle of a long loop: a request in a branch after the loop,
// out of program order but not sent in it, starts while iterations 498 to
// 500 hold the loop's three in flight.
module attributes {pulse.clock_ghz = 1.0 : f64,
                   classical.links = {"0" = {max_in_flight = 3 : i64}}} {
  classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
      args = [], results = [], latency_min_ns = 0.0,
      latency_nominal_ns = 50.0, latency_max_ns = 100.0, mode = "bounded",
      submit = "eager"}
  func.func @interleaved(%c: i1) {
    %l0 = classical.get_link {device = 0 : i64} : !classical.link
    %lb = arith.constant 0 : index
    %ub = arith.constant 1000 : index
    %step = arith.constant 1 : index
    %ln = scf.for %i = %lb to %ub step %step iter_args(%l = %l0) -> (!classical.link) {
      // expected-error @+1 {{link 0 holds 4 requests at once at 20001, more than its 3 slots}}
      %l1, %h = classical.submit %l, @f() {start_cycles = 1 : i64, duration_cycles = 1 : i64, period_cycles = array<i64: 40>} : () -> (!classical.link, !classical.future)
      scf.yield %l1 : !classical.link
    }
    %lm = scf.if %c -> (!classical.link) {
      %l2, %g = classical.submit %ln, @f() {start_cycles = 20001 : i64, duration_cycles = 1 : i64} : () -> (!classical.link, !classical.future)
      scf.yield %l2 : !classical.link
    } else {
      scf.yield %ln : !classical.link
    }
    return
  }
}
