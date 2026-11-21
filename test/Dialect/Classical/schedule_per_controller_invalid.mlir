// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A call over a link with one instance per controller runs on one controller.
// RUN: cudaq-pulse-opt %s --pulse-schedule-asap --verify-diagnostics
// RUN: cudaq-pulse-opt %s --pulse-schedule-alap --verify-diagnostics

module attributes {pulse.qudit_count = 2 : i64,
    pulse.control_system = #pulse.control_system<clock_period_fs = 1000000,
      lines = [#pulse.line<drive, qudits = [0]>, #pulse.line<drive, qudits = [1], controller = 1>,
               #pulse.line<readout, qudits = [0]>, #pulse.line<readout, qudits = [1], controller = 1>],
      controllers = [
        #pulse.controller<clock_ratio = 2, start_uncertainty_cycles = 1>,
        #pulse.controller<clock_ratio = 2, start_uncertainty_cycles = 1>]>,
    classical.links = {"0" = {fixed_ns = 0.0 : f64, bytes_per_ns = 0.0 : f64,
      feedback_ns = 0.0 : f64, slots = 1 : i64, slot_bytes = 1024 : i64,
      per_controller, name = "decoder"}}} {

classical.callable @decode {dispatch_key = 1, device = 0 : i64,
    args = ["bits:1"], results = ["f64"], latency_min_ns = 0.0,
    latency_nominal_ns = 100.0, latency_max_ns = 100.0, mode = "bounded",
    submit = "eager", dispatch_key_override}

func.func @measured_on_one_used_on_another() {
  %c10 = arith.constant 10 : i64
  %c8 = arith.constant 8 : i64
  %a = arith.constant 0.05 : f64
  %z = arith.constant 0.0 : f64
  %ro = pulse.square %c10, %a, %z : i64, f64, f64 -> !pulse.waveform
  %q0 = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
  %r0, %rt0 = pulse.get_readout_line %q0 : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
  %q1 = pulse.qudit_alloc {qudit = 1 : i64} : !pulse.qudit
  %d1, %dt1 = pulse.get_drive_line %q1 : (!pulse.qudit) -> (!pulse.drive_line, !pulse.tone)
  %r0a, %rt0a, %m0 = pulse.readout %r0, %ro, %rt0, iq : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
  %link = classical.get_link {device = 0 : i64} : !classical.link
  // expected-error @+1 {{the call of 'decode' over link 'decoder', which has one instance per controller, uses controllers[0] and controllers[1]}}
  %link1, %f0 = classical.submit %link, @decode(%m0) : (!pulse.measurement) -> (!classical.link, !classical.future)
  %x0 = classical.wait %f0, @decode : f64
  %w0 = pulse.square %c8, %x0, %z : i64, f64, f64 -> !pulse.waveform
  %d1a, %dt1a = pulse.drive %d1, %w0, %dt1 : !pulse.drive_line, !pulse.waveform, !pulse.tone -> !pulse.drive_line, !pulse.tone
  return
}

}
