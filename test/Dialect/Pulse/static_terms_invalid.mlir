// ============================================================================ //
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// RUN: cudaq-pulse-opt %s --split-input-file --verify-diagnostics

// One case per diagnostic of #pulse.static_term.

// expected-error@+1 {{a zz term acts on 2 qudits, got 3}}
module attributes {pulse.static_terms = [#pulse.static_term<zz, qudits = [0, 1, 2], rate_hz = 1.0>]} {}

// -----

// expected-error@+1 {{a zz term acts on 2 qudits, got 1}}
module attributes {pulse.static_terms = [#pulse.static_term<zz, qudits = [0], rate_hz = 1.0>]} {}

// -----

// expected-error@+1 {{names qudit 2 twice}}
module attributes {pulse.static_terms = [#pulse.static_term<zz, qudits = [2, 2], rate_hz = 1.0>]} {}

// -----

// expected-error@+1 {{qudit -1 is not a device index}}
module attributes {pulse.static_terms = [#pulse.static_term<zz, qudits = [-1, 2], rate_hz = 1.0>]} {}

// -----

// expected-error@+1 {{rate_hz must be finite, got inf}}
module attributes {pulse.static_terms = [#pulse.static_term<zz, qudits = [0, 1], rate_hz = 0x7FF0000000000000>]} {}

// -----

// expected-error@+1 {{rate_hz must be finite, got nan}}
module attributes {pulse.static_terms = [#pulse.static_term<zz, qudits = [0, 1], rate_hz = 0x7FF8000000000000>]} {}

// -----

// expected-error@+1 {{rate_hz must be finite, got -inf}}
module attributes {pulse.static_terms = [#pulse.static_term<zz, qudits = [0, 1], rate_hz = 0xFFF0000000000000>]} {}

// -----

// expected-error@+1 {{unknown static term kind 'xx'}}
module attributes {pulse.static_terms = [#pulse.static_term<xx, qudits = [0, 1], rate_hz = 1.0>]} {}
