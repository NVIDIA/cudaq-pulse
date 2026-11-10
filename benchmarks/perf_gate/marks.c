// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Callgrind client requests for benchmarks/perf_gate/cases.py, which loads
// this library with ctypes. Outside Valgrind each request is a no-op.

// Switching instrumentation on or off zeroes Callgrind's running total but
// not the total at the last dump, from which a dump's "summary:" is taken:
// zero the counters after switching on, and dump before switching off.

#include <valgrind/callgrind.h>

void perf_gate_start(void) {
  CALLGRIND_START_INSTRUMENTATION;
  CALLGRIND_ZERO_STATS;
}

void perf_gate_stop(const char *name) {
  CALLGRIND_DUMP_STATS_AT(name);
  CALLGRIND_STOP_INSTRUMENTATION;
}
