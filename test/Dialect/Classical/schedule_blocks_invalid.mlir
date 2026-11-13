// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A block is placed whole, so the scheduler rejects a wait of run-time length
// in one even when the verifier has not run.
// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics --pulse-schedule-asap
// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics --pulse-schedule-alap

classical.callable @f {dispatch_key = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "blocking",
    submit = "eager"}

func.func @blocking_in_block() {
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l2 = pulse.schedule_block asap (%l0) : (!classical.link) -> (!classical.link) {
  ^bb0(%l: !classical.link):
    %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
    // expected-error @+1 {{a blocking wait cannot be inside a pulse.schedule_block, whose length is fixed; return the future from the kernel and take the answer after it returns, or call the kernel without schedule=}}
    %v = classical.wait %h, @f : f64
    pulse.yield %l1 : !classical.link
  }
  return
}

// -----

classical.callable @f {dispatch_key = 3809224601 : i64, device = 0 : i64,
    args = [], results = ["i64"], latency_min_ns = 0.0,
    latency_nominal_ns = 0.0, latency_max_ns = 10.0, mode = "bounded",
    submit = "eager"}

func.func @variable_wait_in_block() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %d0, %t0 = pulse.get_drive_line %q : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %l = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l, @f() : () -> (!classical.link, !classical.future)
  %n = classical.wait %h, @f : i64
  %dur = pulse.duration_from_int %n : (i64) -> !pulse.duration
  %d2 = pulse.schedule_block asap (%d0) : (!pulse.drive_line) -> (!pulse.drive_line) {
  ^bb0(%d: !pulse.drive_line):
    // expected-error @+1 {{a variable wait cannot be inside a pulse.schedule_block, whose length is fixed; call the kernel without schedule= or move the wait out of it}}
    %d1 = pulse.wait %d, %dur {classical.variable, pulse.variable_wait, pulse.min_cycles = 0 : i64} : (!pulse.drive_line, !pulse.duration) -> !pulse.drive_line
    pulse.yield %d1 : !pulse.drive_line
  }
  return
}
