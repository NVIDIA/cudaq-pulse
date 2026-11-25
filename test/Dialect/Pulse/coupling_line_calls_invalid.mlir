// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics --pulse-verify

// What a coupling line does not do with a placed call yet: cross one, as an
// argument or a result, and be fetched or played on in a function that a
// call runs.

module attributes {pulse.coupling_lines} {
func.func private @tune()
    attributes {pulse.length_cycles = 60 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-error @+1 {{a function that a pulse.call runs cannot fetch a coupling line or play on one yet; write its ops in the caller, beside the call}}
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  %n = arith.constant 60 : i64
  %a = arith.constant 0.1 : f64
  %z = arith.constant 0.0 : f64
  %w = pulse.square %n, %a, %z : i64, f64, f64 -> !pulse.waveform
  // expected-error @+1 {{a function that a pulse.call runs cannot fetch a coupling line or play on one yet}}
  %c1 = pulse.drive_baseband %c, %w <{duration_cycles = 60 : i64, start_cycles = 0 : i64}> : !pulse.coupling_line, !pulse.waveform -> !pulse.coupling_line
  return
}
func.func @fetched_in_a_callee() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  pulse.call @tune() <{duration_cycles = 60 : i64, start_cycles = 0 : i64}> : () -> ()
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func private @hold(%c: !pulse.coupling_line)
    attributes {pulse.length_cycles = 0 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  return
}
func.func @handed_to_a_call() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %c0 = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  // expected-error @+1 {{'pulse.call' op argument 0 is a '!pulse.coupling_line'; a call takes lines, tones, measurements, links, f64 and i64}}
  pulse.call @hold(%c0) <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : (!pulse.coupling_line) -> ()
  return
}
}

// -----

module attributes {pulse.coupling_lines} {
func.func private @give() -> !pulse.coupling_line
    attributes {pulse.length_cycles = 0 : i64, pulse.schedule_state = #pulse.schedule_state<timed>}
func.func @returned_by_a_call() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  // expected-error @+1 {{'pulse.call' op result #0 must be variadic of a pulse line, tone, measurement, link or call handle, but got '!pulse.coupling_line'}}
  %c = pulse.call @give() <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : () -> (!pulse.coupling_line)
  return
}
}

// -----

// A function that only a called function calls is a callee too.

module attributes {pulse.coupling_lines} {
func.func private @inner()
    attributes {pulse.length_cycles = 0 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  // expected-error @+1 {{a function that a pulse.call runs cannot fetch a coupling line or play on one yet}}
  %c = pulse.get_coupling_line %q : (!pulse.qudit) -> !pulse.coupling_line
  return
}
func.func private @outer()
    attributes {pulse.length_cycles = 0 : i64, pulse.schedule_state = #pulse.schedule_state<timed>} {
  pulse.call @inner() <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : () -> ()
  return
}
func.func @fetched_two_calls_down() attributes {pulse.schedule_state = #pulse.schedule_state<timed>} {
  pulse.call @outer() <{duration_cycles = 0 : i64, start_cycles = 0 : i64}> : () -> ()
  return
}
}
