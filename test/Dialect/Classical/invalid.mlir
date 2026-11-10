// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// One case per diagnostic of the classical op verifiers. Schedule-time rules
// (link capacity, order, deadlines) are in pulse-classical-check tests.

// expected-error @+1 {{has 9 arguments; at most 8 are allowed}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = ["u8", "u8", "u8", "u8", "u8", "u8", "u8", "u8", "u8"], results = [],
    latency_min_ns = 0.0, latency_nominal_ns = 0.0, latency_max_ns = 0.0,
    mode = "bounded", submit = "eager"}

// -----

// expected-error @+1 {{has 5 results; at most 4 are allowed}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["u8", "u8", "u8", "u8", "u8"],
    latency_min_ns = 0.0, latency_nominal_ns = 0.0, latency_max_ns = 0.0,
    mode = "bounded", submit = "eager"}

// -----

// expected-error @+1 {{has a malformed argument type 'bits:0'}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = ["bits:0"], results = [], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}

// -----

// expected-error @+1 {{has a malformed result type 'complex'}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["complex"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}

// -----

// expected-error @+1 {{has a malformed argument type 'f64[x]'}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = ["f64[x]"], results = [], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}

// -----

// expected-error @+1 {{latency must satisfy 0 <= min <= nominal <= max, got 1.000000e+01, 5.000000e+00, 2.000000e+01}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = [], latency_min_ns = 10.0, latency_nominal_ns = 5.0,
    latency_max_ns = 20.0, mode = "bounded", submit = "eager"}

// -----

// expected-error @+1 {{mode must be "bounded", "timeout" or "blocking", got "soon"}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "soon", submit = "eager"}

// -----

// expected-error @+1 {{a timeout callable needs a positive timeout_ns}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "timeout",
    submit = "eager"}

// -----

// expected-error @+1 {{timeout_ns is only meaningful with mode "timeout"}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "bounded", submit = "eager", timeout_ns = 5.0}

// -----

// expected-error @+1 {{submit must be "eager" or "lazy", got "late"}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "bounded", submit = "late"}

// -----

// expected-error @+1 {{deadline_ns must be positive}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "bounded", submit = "eager",
    deadline_ns = 0.0}

// -----

// expected-error @+1 {{function_id 1 is not the FNV-1a hash of 'f' (3809224601)}}
classical.callable @f {function_id = 1 : i64, device = 0 : i64, args = [],
    results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "bounded", submit = "eager"}

// -----

// expected-error @+1 {{function_id must be a 32-bit unsigned value}}
classical.callable @f {function_id = 4294967296 : i64, device = 0 : i64,
    args = [], results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "bounded", submit = "eager",
    function_id_override}

// -----

// expected-error @+1 {{device must not be negative}}
classical.callable @f {function_id = 3809224601 : i64, device = -1 : i64,
    args = [], results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "bounded", submit = "eager"}

// -----

func.func @negative_link() {
  // expected-error @+1 {{'classical.get_link' op device must not be negative}}
  %l = classical.get_link {device = -2 : i64} : !classical.link
  return
}

// -----

func.func @unknown_callee() {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{'nothing' does not name a classical.callable}}
  %l1, %h = classical.submit %l, @nothing() : () -> (!classical.link, !classical.future)
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = ["bits:2"], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}

func.func @too_few(%m: !pulse.measurement) {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{passes 1 values; 'f' takes more (argument 0 is bits:2)}}
  %l1, %h = classical.submit %l, @f(%m) : (!pulse.measurement) -> (!classical.link, !classical.future)
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = ["f64"], results = [], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}

func.func @too_many(%x: f64) {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{passes 2 values; 'f' takes 1}}
  %l1, %h = classical.submit %l, @f(%x, %x) : (f64, f64) -> (!classical.link, !classical.future)
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = ["bits:1"], results = [], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}

func.func @not_a_measurement(%x: f64) {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{value 0 has type 'f64', but argument 0 of 'f' is bits:1}}
  %l1, %h = classical.submit %l, @f(%x) : (f64) -> (!classical.link, !classical.future)
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = ["i64"], results = [], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}

func.func @float_for_int(%x: f64) {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{value 0 has type 'f64', but argument 0 of 'f' is i64}}
  %l1, %h = classical.submit %l, @f(%x) : (f64) -> (!classical.link, !classical.future)
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}
classical.callable @g {function_id = 3792446982 : i64, device = 0 : i64,
    args = [], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "bounded",
    submit = "eager"}

func.func @wrong_callee() {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  // expected-error @+1 {{waits for 'g', but the future was submitted to 'f'}}
  %v = classical.wait %h, @g : f64
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["bits:2"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 0.0, mode = "timeout",
    timeout_ns = 10.0, submit = "eager"}

func.func @wrong_results() {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  // expected-error @+1 {{result types do not match 'f': expected ('i1', 'i1', 'i1')}}
  %a, %b = classical.wait %h, @f : i1, i1
  return
}

// -----

classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "bounded", submit = "eager"}

func.func @nothing_to_wait_for() {
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  // expected-error @+1 {{'f' returns nothing; there is nothing to wait for}}
  "classical.wait"(%h) {callee = @f} : (!classical.future) -> ()
  return
}

// -----

// expected-error @+1 {{reaction_deadline_ns must be positive}}
classical.callable @f {function_id = 3809224601 : i64, device = 0 : i64,
    args = [], results = [], latency_min_ns = 0.0, latency_nominal_ns = 0.0,
    latency_max_ns = 0.0, mode = "bounded", submit = "eager",
    reaction_deadline_ns = 0.0}
