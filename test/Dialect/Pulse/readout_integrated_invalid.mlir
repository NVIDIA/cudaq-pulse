// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --pulse-verify --verify-diagnostics

// An integrated readout gives its IQ point, !pulse.iq_data, and a readout of
// any other mode a level, !pulse.measurement. A point has no level to read,
// and a call takes it only as an IQ argument.

func.func @integrated_gives_a_point(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) {
  // expected-error @+1 {{'pulse.readout' op of mode integrated gives '!pulse.iq_data', got '!pulse.measurement'}}
  %r1, %t1, %m = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  return
}

// -----

func.func @level_gives_a_level(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) {
  // expected-error @+1 {{'pulse.readout' op of mode level gives '!pulse.measurement', got '!pulse.iq_data'}}
  %r1, %t1, %p = pulse.readout %r0, %wf, %t0, level : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  return
}

// -----

func.func @branch_on_a_point(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i1 {
  %r1, %t1, %p = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  // expected-error @+1 {{'pulse.is_set' op operand #0 must be The level a readout observed, but got '!pulse.iq_data'}}
  %c = pulse.is_set %p : !pulse.iq_data -> i1
  return %c : i1
}

// -----

func.func @level_of_a_point(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) -> i64 {
  %r1, %t1, %p = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  // expected-error @+1 {{'pulse.outcome' op operand #0 must be The level a readout observed, but got '!pulse.iq_data'}}
  %k = pulse.outcome %p : !pulse.iq_data -> i64
  return %k : i64
}

// -----

// Sent as bits, a call would read its level; as an IQ argument it reads the
// point, as it may.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>,
    classical.links = {"0" = {bytes_per_ns = 0.0 : f64, feedback_ns = 8.0 : f64,
                              fixed_ns = 4.0 : f64, slots = 64 : i64,
                              slot_bytes = 1024 : i64}}} {

classical.callable @decide {args = ["bits:1"], device = 0 : i64,
    dispatch_key = 3994489633 : i64, latency_max_ns = 100.0 : f64,
    latency_min_ns = 0.0 : f64, latency_nominal_ns = 100.0 : f64,
    mode = "bounded", results = ["f64"], submit = "eager"}
classical.callable @fit {args = ["iq:1"], device = 0 : i64,
    dispatch_key = 2934209580 : i64, latency_max_ns = 100.0 : f64,
    latency_min_ns = 0.0 : f64, latency_nominal_ns = 100.0 : f64,
    mode = "bounded", results = ["f64"], submit = "eager"}

func.func @sent(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) {
  %r1, %t1, %p = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  %l1, %h = classical.submit %l0, @fit(%p) : (!pulse.iq_data) -> (!classical.link, !classical.future)
  // expected-error @+1 {{'classical.submit' op value 0 has type '!pulse.iq_data', but argument 0 of 'decide' is bits:1}}
  %l2, %h2 = classical.submit %l1, @decide(%p) : (!pulse.iq_data) -> (!classical.link, !classical.future)
  return
}

}

// -----

// A call takes an IQ point from a readout, on every path, as pulse.is_set
// takes a level: not one pulse.iq_acquire gives, which the exports have no
// readout event for.
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000>,
    classical.links = {"0" = {bytes_per_ns = 0.0 : f64, feedback_ns = 8.0 : f64,
                              fixed_ns = 4.0 : f64, slots = 64 : i64,
                              slot_bytes = 1024 : i64}}} {

classical.callable @fit {args = ["iq:1"], device = 0 : i64,
    dispatch_key = 2934209580 : i64, latency_max_ns = 100.0 : f64,
    latency_min_ns = 0.0 : f64, latency_nominal_ns = 100.0 : f64,
    mode = "bounded", results = ["f64"], submit = "eager"}

func.func @acquired(%r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) {
  %r1, %t1, %p = pulse.iq_acquire %r0, %wf, %t0 : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{'classical.submit' op sends an IQ point that must be produced by pulse.readout, got pulse.iq_acquire}}
  %l1, %h = classical.submit %l0, @fit(%p) : (!pulse.iq_data) -> (!classical.link, !classical.future)
  return
}

func.func @acquired_on_one_path(%b: i1, %r0: !pulse.readout_line, %t0: !pulse.tone, %wf: !pulse.waveform) {
  %r1, %t1, %p = scf.if %b -> (!pulse.readout_line, !pulse.tone, !pulse.iq_data) {
    %r2, %t2, %p2 = pulse.readout %r0, %wf, %t0, integrated : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
    scf.yield %r2, %t2, %p2 : !pulse.readout_line, !pulse.tone, !pulse.iq_data
  } else {
    %r2, %t2, %p2 = pulse.iq_acquire %r0, %wf, %t0 : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.iq_data
    scf.yield %r2, %t2, %p2 : !pulse.readout_line, !pulse.tone, !pulse.iq_data
  }
  %l0 = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{'classical.submit' op sends an IQ point that must be produced by pulse.readout, got pulse.iq_acquire}}
  %l1, %h = classical.submit %l0, @fit(%p) : (!pulse.iq_data) -> (!classical.link, !classical.future)
  return
}

}
