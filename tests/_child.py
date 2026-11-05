# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The limits of a child process that proves a lowering's time and memory
bounded: the code a test's child runs first."""

# Linux's ru_maxrss carries the parent's peak across fork and exec, which a
# long test run takes past any of these caps; VmHWM is this process's own.
_LIMITS = """
import os, resource, signal, sys, threading, time
signal.alarm({wall})
resource.setrlimit(resource.RLIMIT_CPU, ({cpu}, {cpu}))
def _peak():
    if sys.platform == "darwin":
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) << 10
    raise RuntimeError("no VmHWM in /proc/self/status")
def _watch():
    while True:
        if (peak := _peak()) > {mb} << 20:
            os.write(2, f"peak RSS {{peak >> 20}} MB, past {mb} MB\\n".encode())
            os._exit(3)
        time.sleep(0.005)
_peak()
threading.Thread(target=_watch, daemon=True).start()
"""


def limits(cpu_s: int, rss_mb: int) -> str:
    """Code that stops the process it runs in past *cpu_s* seconds of CPU,
    twice that of wall time, or a peak resident size of *rss_mb* MB, the
    last with exit status 3 and the peak on stderr."""
    return _LIMITS.format(wall=2 * cpu_s, cpu=cpu_s, mb=rss_mb)
