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

// expected-error@+1 {{each controller's outcome_latency_cycles and spread_cycles list all 2 controllers}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<outcome_latency_cycles = [0, 0], spread_cycles = [0, 0]>, #pulse.controller<>]>} {}

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

// expected-error@+1 {{uncertainty_ns is finite and 0 or more}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<uncertainty_ns = -1.0>]>} {}

// -----

// expected-error@+1 {{#pulse.controller gives 'blocks_on_calls' twice}}
module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1000000, controllers = [#pulse.controller<blocks_on_calls, blocks_on_calls>]>} {}
