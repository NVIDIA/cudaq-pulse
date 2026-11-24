// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-asap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-alap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-schedule-rcp --verify-diagnostics
// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// The schedule passes refuse an op they cannot time. The ops of a coupling
// line are ops they time: a drive with no tone, a wait and a sync on it
// raise no diagnostic (coupling_line_schedule.mlir checks where they are
// placed).

func.func @a_coupling_line_is_timed() {
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q0 : (!pulse.qudit) -> !pulse.coupling_line
  %p = pulse.get_coupling_line %q0, %q1 : (!pulse.qudit, !pulse.qudit) -> !pulse.coupling_line
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %c1 = pulse.drive %c, %x : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  %c8 = arith.constant 8 : i64
  %held = pulse.duration_from_int %c8 : (i64) -> !pulse.duration
  %p1 = pulse.wait %p, %held : (!pulse.coupling_line, !pulse.duration) -> !pulse.coupling_line
  %c2, %p2 = pulse.sync %c1, %p1 : !pulse.coupling_line, !pulse.coupling_line -> !pulse.coupling_line, !pulse.coupling_line
  %p3 = pulse.drive %p2, %x : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}

// -----

// A call that takes a coupling line is refused as one that takes any other
// line: it would be timed as taking no time while the called function's
// pulse plays on the line.
func.func private @detunes(%line: !pulse.coupling_line) -> !pulse.coupling_line {
  %c40 = arith.constant 40 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c40, %am, %z : i64, f64, f64 -> !pulse.waveform
  %out = pulse.drive %line, %x : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return %out : !pulse.coupling_line
}

func.func @call_that_takes_a_coupling_line() {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %c8 = arith.constant 8 : i64
  %am = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %x = pulse.square %c8, %am, %z : i64, f64, f64 -> !pulse.waveform
  %c1 = pulse.drive %c, %x : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  // expected-error @+1 {{'func.call' takes or gives a qudit, a line, a tone, a measurement or a classical handle, and the schedule passes cannot tell what it plays or how long it takes; write the ops it stands for in its place before scheduling}}
  %c2 = call @detunes(%c1) : (!pulse.coupling_line) -> !pulse.coupling_line
  %c3 = pulse.drive %c2, %x : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
