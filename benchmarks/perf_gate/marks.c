/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Callgrind client requests for benchmarks/perf_gate/cases.py, which loads
// this library with ctypes. Outside Valgrind each request is a no-op.

#include <valgrind/callgrind.h>

void perf_gate_start(void) {
  CALLGRIND_ZERO_STATS;
  CALLGRIND_START_INSTRUMENTATION;
}

void perf_gate_stop(const char *name) {
  CALLGRIND_STOP_INSTRUMENTATION;
  CALLGRIND_DUMP_STATS_AT(name);
}
