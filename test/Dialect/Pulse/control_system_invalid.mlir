// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// One case per diagnostic of #pulse.control_system, #pulse.line and
// #pulse.controller.

// expected-error@+1 {{#pulse.control_system needs its clock_period_fs}}
module attributes {pulse.control_system = #pulse.control_system<lines = []>} {}

// -----

// expected-error@+1 {{clock_period_fs 1000000/2 is not in lowest terms}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000/2>} {}

// -----

// expected-error@+1 {{clock_period_fs is positive, got 0/1}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 0>} {}

// -----

// expected-error@+1 {{clock_period_fs 1/9007199255 has no exact clock}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1/9007199255>} {}

// -----

// expected-error@+1 {{unknown field 'clock_ghz' of #pulse.control_system}}
module attributes {pulse.control_system = #pulse.control_system<clock_ghz = 2.0>} {}

// -----

// expected-error@+1 {{a line is on controller 1, of 1}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<drive, qudits = [0], controller = 1>]>} {}

// -----

// expected-error@+1 {{two drive lines reach qudit 1}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<drive, qudits = [0, 1]>, #pulse.line<drive, qudits = [1]>]>} {}

// -----

// A controller gives facts, not tables of latencies.
// expected-error@+1 {{unknown field 'outcome_latency_cycles' of #pulse.controller}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<outcome_latency_cycles = [0, 0], spread_cycles = [0, 0]>, #pulse.controller<>]>} {}

// -----

// expected-error@+1 {{controller 1's peer_latency_cycles names itself}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<>, #pulse.controller<peer_latency_cycles = {1 = 4}>]>} {}

// -----

// expected-error@+1 {{controller 0's peer_latency_cycles names controller 2, of 2}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<peer_latency_cycles = {1 = 3, 2 = 4}>, #pulse.controller<>]>} {}

// -----

// expected-error@+1 {{expected a #pulse.line}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.controller<>]>} {}

// -----

// expected-error@+1 {{unknown line kind 'flux'; a line is drive or readout}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<flux, qudits = [0]>]>} {}

// -----

// expected-error@+1 {{a line reaches at least one qudit}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<drive>]>} {}

// -----

// expected-error@+1 {{a line reaches qudit 2 twice}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<readout, qudits = [2, 2]>]>} {}

// -----

// expected-error@+1 {{max_concurrent_tones is 1 or more, got 0}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, lines = [#pulse.line<readout, qudits = [2], max_concurrent_tones = 0>]>} {}

// -----

// expected-error@+1 {{grid_cycles is 1 or more, got 0}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<grid_cycles = 0>]>} {}

// -----

// expected-error@+1 {{start_uncertainty_cycles is 0 or more, got -1}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<start_uncertainty_cycles = -1>]>} {}

// -----

// expected-error@+1 {{feedback_latency_cycles is 0 or more, got -6}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<feedback_latency_cycles = -6>]>} {}

// -----

// expected-error@+1 {{peer_latency_cycles is 0 or more, got -2}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<peer_latency_cycles = {1 = -2}>]>} {}

// -----

// expected-error@+1 {{peer_latency_cycles names controllers by increasing index, got 1 after 3}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<peer_latency_cycles = {3 = 5, 1 = 5}>]>} {}

// -----

// A peer the outcome cannot reach has its syntax kept for later.
// expected-error@+1 {{unreachable peers are not supported yet}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<peer_latency_cycles = {1 = 10, 2 = none}>]>} {}

// -----

// expected-error@+1 {{clock_ratio 6/4 is not in lowest terms}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<clock_ratio = 6/4>]>} {}

// -----

// expected-error@+1 {{clock_ratio is positive, got 0/1}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<clock_ratio = 0>]>} {}

// -----

// expected-error@+1 {{#pulse.controller gives 'blocks_on_calls' twice}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<blocks_on_calls, blocks_on_calls>]>} {}

// -----

// The latencies between controllers, in program cycles, fit 64 bits: here
// 2^62 own cycles at a quarter of the program clock are 2^64 program cycles.
// expected-error@+1 {{the outcome of a readout on controller 0 reaches controller 1 more than 2^63 - 1 program cycles after the readout ends}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<clock_ratio = 1/4, feedback_latency_cycles = 4611686018427387904>, #pulse.controller<>]>} {}

// -----

// expected-error@+1 {{the outcome of a readout on controller 1 reaches controller 0 more than 2^63 - 1 program cycles after the readout ends}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<start_uncertainty_cycles = 1>, #pulse.controller<peer_latency_cycles = {0 = 9223372036854775807}>]>} {}
