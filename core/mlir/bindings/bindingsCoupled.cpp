//===- bindingsCoupled.cpp - The bindings' code for coupling lines --------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// bindings.cpp compiled for a program that holds a coupling line: its packed
// decoder and its event walks, behind the functions of BindingsCoupled.h.
// Nothing else of bindings.cpp is reached from them, so nothing else of it
// is in this unit's object, and it defines no Python module.
//
//===----------------------------------------------------------------------===//

// What this unit does not reach of bindings.cpp is unused here by design.
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wunused-member-function"
#pragma clang diagnostic ignored "-Wunneeded-internal-declaration"
#pragma clang diagnostic ignored "-Wunused-const-variable"
#endif

#define PULSE_BINDINGS_COUPLED 1
#include "bindings.cpp" // NOLINT(bugprone-suspicious-include)
