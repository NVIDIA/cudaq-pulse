// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// One case per diagnostic of #pulse.quantum_system.

// expected-error@+1 {{unknown field 'thermal_populations' of #pulse.quantum_system}}
module attributes {pulse.quantum_system = #pulse.quantum_system<thermal_populations = [0.0]>} {}

// -----

// expected-error@+1 {{#pulse.quantum_system gives 't1_ns' twice}}
module attributes {pulse.quantum_system = #pulse.quantum_system<t1_ns = [1.0], t1_ns = [2.0]>} {}

// -----

// expected-error@+1 {{level_counts lists 1 qudits, and the other lists 2}}
module attributes {pulse.quantum_system = #pulse.quantum_system<qudit_frequency_hz = [5.0e+09, 5.1e+09], level_counts = [3]>} {}

// -----

// expected-error@+1 {{a qudit has at least 2 levels, got 1}}
module attributes {pulse.quantum_system = #pulse.quantum_system<level_counts = [3, 1]>} {}

// -----

// expected-error@+1 {{t1_ns entries are 0 or more, got -1}}
module attributes {pulse.quantum_system = #pulse.quantum_system<t1_ns = [-1.0]>} {}

// -----

// expected-error@+1 {{t2_ns entries are 0 or more, got nan}}
module attributes {pulse.quantum_system = #pulse.quantum_system<t2_ns = [0x7FF8000000000000]>} {}

// -----

// expected-error@+1 {{expected a #pulse.static_term}}
module attributes {pulse.quantum_system = #pulse.quantum_system<static_terms = [1 : i64]>} {}

// -----

// expected-error@+1 {{expected '['}}
module attributes {pulse.quantum_system = #pulse.quantum_system<static_terms = 1>} {}
