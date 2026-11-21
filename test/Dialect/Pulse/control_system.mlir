// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --split-input-file %s | cudaq-pulse-opt --split-input-file | FileCheck %s

// A control system prints its fields in one order, whatever order they
// parse in, and leaves out those at their default. The clock period is an
// exact fraction of femtoseconds: 1250000/3 fs is a 2.4 GHz clock.

// CHECK: module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1250000/3, lines = [#pulse.line<drive, qudits = [0], delay_cycles = 3>, #pulse.line<readout, qudits = [0, 1, 2, 3], max_concurrent_tones = 4, controller = 1>], controllers = [#pulse.controller<feedback_latency_cycles = 100, peer_latency_cycles = {1 = 40}, start_uncertainty_cycles = 1, persistent_frames>, #pulse.controller<grid_cycles = 2, clock_ratio = 1/2, feedback_latency_cycles = 50, peer_latency_cycles = 20, start_uncertainty_cycles = 1, blocks_on_calls, max_duration_cycles = 4000>]>}
module attributes {pulse.control_system = #pulse.control_system<
    controllers = [
      #pulse.controller<persistent_frames, start_uncertainty_cycles = 1,
                        peer_latency_cycles = {1 = 40}, feedback_latency_cycles = 100,
                        grid_cycles = 1, clock_ratio = 1>,
      #pulse.controller<max_duration_cycles = 4000, blocks_on_calls, grid_cycles = 2,
                        start_uncertainty_cycles = 1, peer_latency_cycles = 20,
                        clock_ratio = 1/2, feedback_latency_cycles = 50>],
    lines = [#pulse.line<drive, delay_cycles = 3, qudits = [0], controller = 0>,
             #pulse.line<readout, controller = 1, qudits = [0, 1, 2, 3],
                         max_concurrent_tones = 4, delay_cycles = 0>],
    clock_period_fs = 1250000/3>} {
}

// -----

// A whole period prints without its denominator.
// CHECK: module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 500000>}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 500000/1>} {
}

// -----

// CHECK: module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<>]>}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [], controllers = [#pulse.controller<>]>} {
}

// -----

// A controller's facts at their default print nothing: one peer latency for
// every peer of 0, none by index, a clock ratio of 1. A whole ratio prints
// without its denominator, and peers by index by increasing index.
// CHECK: module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 500000, controllers = [#pulse.controller<>, #pulse.controller<>, #pulse.controller<clock_ratio = 2, peer_latency_cycles = {0 = 7, 3 = 0}>, #pulse.controller<clock_ratio = 3/5, feedback_latency_cycles = 6, start_uncertainty_cycles = 2>]>}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 500000, controllers = [
    #pulse.controller<peer_latency_cycles = {}, clock_ratio = 1/1, feedback_latency_cycles = 0,
                      start_uncertainty_cycles = 0>,
    #pulse.controller<peer_latency_cycles = 0>,
    #pulse.controller<peer_latency_cycles = {0 = 7, 3 = 0}, clock_ratio = 2/1>,
    #pulse.controller<start_uncertainty_cycles = 2, clock_ratio = 3/5, feedback_latency_cycles = 6>]>} {
}

// -----

// Any 64-bit facts whose latencies between controllers fit 64 bits: a
// 1.234567 GHz controller of a 10 MHz program, and the largest latency.
// CHECK: module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 100000000, controllers = [#pulse.controller<clock_ratio = 1234567/10000, feedback_latency_cycles = 2199023255552, peer_latency_cycles = {1 = 2199023255552}>, #pulse.controller<peer_latency_cycles = 9223372036854775807>]>}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 100000000, controllers = [
    #pulse.controller<clock_ratio = 1234567/10000, feedback_latency_cycles = 2199023255552,
                      peer_latency_cycles = {1 = 2199023255552}>,
    #pulse.controller<peer_latency_cycles = 9223372036854775807>]>} {
}
