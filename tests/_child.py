# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The limits of a child process that proves a lowering's time and memory
bounded: the code a test's child runs first."""

from tests import _tools

# Linux's ru_maxrss carries the parent's peak across fork and exec, which a
# long test run takes past any of these caps; VmHWM is this process's own.
# An exit callback stops the watcher and joins it, after a last look at the
# peak: a daemon thread that wakes while the interpreter finalizes can end
# after it wrote the peak and before it exits with 3. A fork of the child
# drops the callback, as it has no watcher to stop.
_LIMITS = """
import atexit, os, resource, signal, sys, threading
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
_exiting = threading.Event()
def _watch():
    last = False
    while not last:
        last = _exiting.wait(0.005)
        if (peak := _peak()) > {mb} << 20:
            os.write(2, f"peak RSS {{peak >> 20}} MB, past {mb} MB\\n".encode())
            os._exit(3)
_peak()
_watcher = threading.Thread(target=_watch, daemon=True)
_watcher.start()
@atexit.register
def _stop():
    _exiting.set()
    _watcher.join()
# A fork has no watcher, and may hold the Event's lock the watcher held.
os.register_at_fork(after_in_child=lambda: atexit.unregister(_stop))
"""


def limits(cpu_s: int, rss_mb: int) -> str:
    """Code that stops the process it runs in past *cpu_s* seconds of CPU,
    twice that of wall time, or a peak resident size of *rss_mb* MB, the
    last with exit status 3 and the peak on stderr.

    Under a sanitizer runtime the test that asks is skipped: the limits are
    sized for the uninstrumented build."""
    _tools.skip_raw_limit("a child's CPU and memory limit")
    return _LIMITS.format(wall=2 * cpu_s, cpu=cpu_s, mb=rss_mb)
