// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --split-input-file %s | cudaq-pulse-opt --split-input-file | FileCheck %s

// A control system prints its fields in one order, whatever order they
// parse in, and leaves out those at their default. The clock period is an
// exact fraction of femtoseconds: 1250000/3 fs is a 2.4 GHz clock.

// CHECK: module attributes {pulse.control_system = #pulse.control_system<clock_period_fs = 1250000/3, lines = [#pulse.line<drive, qudits = [0], delay_cycles = 3>, #pulse.line<readout, qudits = [0, 1, 2, 3], max_concurrent_tones = 4, controller = 1>], controllers = [#pulse.controller<outcome_latency_cycles = [100, 140], spread_cycles = [0, 3], uncertainty_ns = 5.000000e-01, persistent_frames>, #pulse.controller<grid_cycles = 2, outcome_latency_cycles = [140, 100], spread_cycles = [3, 0], uncertainty_ns = 5.000000e-01, blocks_on_calls, max_duration_cycles = 4000>]>}
module attributes {pulse.control_system = #pulse.control_system<
    controllers = [
      #pulse.controller<persistent_frames, spread_cycles = [0, 3],
                        outcome_latency_cycles = [100, 140], uncertainty_ns = 0.5,
                        grid_cycles = 1>,
      #pulse.controller<max_duration_cycles = 4000, blocks_on_calls, grid_cycles = 2,
                        outcome_latency_cycles = [140, 100], spread_cycles = [3, 0],
                        uncertainty_ns = 0.5>],
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
