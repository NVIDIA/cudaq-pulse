// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RUN: cudaq-pulse-opt %s | cudaq-pulse-opt | FileCheck %s

// A module's static Hamiltonian terms print as they parse, and print again
// the same; a rate keeps every bit of its f64.

// CHECK: module attributes {pulse.static_terms = [#pulse.static_term<zz, qudits = [3, 4], rate_hz = 4.500000e+04>, #pulse.static_term<zz, qudits = [7, 0], rate_hz = -0.0012345678901234567>, #pulse.static_term<zz, qudits = [1, 2], rate_hz = 0.000000e+00>]}
module attributes {pulse.static_terms = [
    #pulse.static_term<zz, qudits = [3, 4], rate_hz = 4.5e+04>,
    #pulse.static_term<zz, qudits = [7, 0], rate_hz = -1.2345678901234567e-03>,
    #pulse.static_term<zz, qudits = [1, 2], rate_hz = 0.0>]} {
}
