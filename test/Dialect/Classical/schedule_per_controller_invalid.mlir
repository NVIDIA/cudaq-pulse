// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// A call over a link with one instance per controller runs on one controller.
// RUN: cudaq-pulse-opt %s --pulse-schedule-asap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --pulse-schedule-alap --verify-diagnostics

module attributes {pulse.clock_ghz = 1.0 : f64, pulse.n_qubits = 2 : i64,
    pulse.controllers = {drive = array<i64: 0, 1>, readout = array<i64: 0, 1>,
      outcome_latency = array<i64: 0, 0, 0, 0>, spread = array<i64: 0, 0, 0, 0>,
      uncertainty_ns = array<f64: 0.5, 0.5>},
    classical.links = {"0" = {fixed_ns = 0.0 : f64, bytes_per_ns = 0.0 : f64,
      feedback_ns = 0.0 : f64, max_in_flight = 1 : i64, slot_bytes = 1024 : i64,
      per_controller, name = "decoder"}}} {

classical.callable @decode {function_id = 1, device = 0 : i64,
    args = ["bits:1"], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", function_id_override}

func.func @measured_on_one_used_on_another() {
  %c10 = arith.constant 10 : i64
  %c8 = arith.constant 8 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c10, %a, %z : i64, f64, f64 -> !pulse.waveform
  %q0 = pulse.qudit_alloc {qubit = 0 : i64} : !pulse.qref
  %r0, %rt0 = pulse.get_readout_line %q0 {qubit = 0 : i64} : (!pulse.qref) -> (!pulse.readout_line, !pulse.tone)
  %q1 = pulse.qudit_alloc {qubit = 1 : i64} : !pulse.qref
  %d1, %dt1 = pulse.get_drive_line %q1 {qubit = 1 : i64} : (!pulse.qref) -> (!pulse.drive_line, !pulse.tone)
  %r0a, %rt0a, %m0 = pulse.readout %r0, %ro, %rt0, "iq" : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %link = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{the call of 'decode' over link 'decoder', which has one instance per controller, uses controllers[0] and controllers[1]}}
  %link1, %f0 = classical.submit %link, @decode(%m0) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %x0 = classical.wait %f0, @decode : f64
  %w0 = pulse.square %c8, %x0, %z : i64, f64, f64 -> !pulse.waveform
  %d1a, %dt1a = pulse.drive %d1, %w0, %dt1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

}
