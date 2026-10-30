// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// --version prints the package's version, then the LLVM it was built with.
// RUN: cudaq-pulse-opt --version | FileCheck %s
// CHECK: {{^}}cudaq-pulse-opt {{[0-9]+\.[0-9]+[^ ]*}} (LLVM {{[0-9]+\.[0-9]+\.[0-9]+[^ ]*}}){{$}}
