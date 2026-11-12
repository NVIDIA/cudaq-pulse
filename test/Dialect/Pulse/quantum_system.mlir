// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt --split-input-file %s | cudaq-pulse-opt --split-input-file | FileCheck %s

// A quantum system prints its fields in one order, whatever order they
// parse in, and leaves out those it does not give; an f64 keeps every bit,
// an infinite time too.

// CHECK: module attributes {pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [5.000000e+09, 5.100000e+09], level_counts = [3, 2], anharmonicity_hz = [-2.000000e+08, 0.000000e+00], t1_ns = [5.000000e+04, 0x7FF0000000000000], t2_ns = [3.000000e+04, 0.000000e+00], drive_scale_rad_per_ns = [1.000000e+00, 0.0012345678901234567], static_terms = [#pulse.static_term<zz, qudits = [0, 1], rate_hz = 4.500000e+04>]>}
module attributes {pulse.quantum_system = #pulse.quantum_system<
    static_terms = [#pulse.static_term<zz, qudits = [0, 1], rate_hz = 4.5e+04>],
    t2_ns = [3.0e+04, 0.0], t1_ns = [5.0e+04, 0x7FF0000000000000],
    drive_scale_rad_per_ns = [1.0, 1.2345678901234567e-03],
    anharmonicity_hz = [-2.0e+08, 0.0], level_counts = [3, 2],
    qudit_frequency_hz = [5.0e+09, 5.1e+09]>} {
}

// -----

// CHECK: module attributes {pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [5.000000e+09]>}
module attributes {pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [5.0e+09], t1_ns = []>} {
}

// -----

// CHECK: module attributes {pulse.quantum_system = #pulse.quantum_system<>}
module attributes {pulse.quantum_system = #pulse.quantum_system<>} {
}
