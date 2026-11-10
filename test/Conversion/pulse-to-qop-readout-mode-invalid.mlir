// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Only iq readouts are simulated; another mode is refused rather than
// simulated as iq.
// RUN: cudaq-pulse-opt --pulse-to-qop --verify-diagnostics %s

module attributes {pulse.clock_ghz = 1.000000e+00 : f64} {
  func.func @main() {
    %q = pulse.qudit_alloc {qudit = 0 : i64} : !pulse.qudit
    %r0, %t0 = pulse.get_readout_line %q {frequency_hz = 7.000000e+09 : f64} : (!pulse.qudit) -> (!pulse.readout_line, !pulse.tone)
    %c40 = arith.constant 40 : i64
    %amp = arith.constant 0.05 : f64
    %zero = arith.constant 0.0 : f64
    %w = pulse.square %c40, %amp, %zero : i64, f64, f64 -> !pulse.waveform
    // expected-error @+1 {{readout mode trace is not supported by the simulator yet; only iq is}}
    %r1, %t1, %m = pulse.readout %r0, %w, %t0, trace <{duration_cycles = 40 : i64, start_cycles = 0 : i64}> : !pulse.readout_line, !pulse.waveform, !pulse.tone -> !pulse.readout_line, !pulse.tone, !pulse.measurement
    return
  }
}
