# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The limits of a test's child process: its memory cap ends it with status 3,
and the thread that watches the cap is gone before the interpreter finalizes."""

import re
import subprocess
import sys

from tests import _child

_PAST_64_MB = re.compile(r"^peak RSS \d+ MB, past 64 MB$", re.MULTILINE)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)


def test_a_child_under_its_limits_exits_0_with_its_output():
    out = _run(_child.limits(30, 512) + 'print("done")\n')
    assert (out.returncode, out.stdout) == (0, "done\n"), out.stderr[-2000:]


def test_a_child_past_its_memory_cap_is_stopped_while_it_runs():
    # os._exit runs no exit callback: status 3 is the watcher's, mid-run.
    code = 'import time\ndata = b"x" * (128 << 20)\ntime.sleep(30)\nos._exit(0)\n'
    out = _run(_child.limits(30, 64) + code)
    assert out.returncode == 3 and _PAST_64_MB.search(out.stderr), out.stderr[-2000:]


def test_a_child_that_passes_its_memory_cap_as_it_ends_is_stopped():
    out = _run(_child.limits(30, 64) + 'data = b"x" * (64 << 20)\n')
    assert out.returncode == 3 and _PAST_64_MB.search(out.stderr), out.stderr[-2000:]


def test_the_watcher_is_gone_once_the_exit_callbacks_have_run():
    # Exit callbacks run last registered first: this one after the limits' own.
    first = "import atexit, threading\natexit.register(lambda: print(threading.active_count()))\n"
    out = _run(first + _child.limits(30, 512))
    assert (out.returncode, out.stdout) == (0, "1\n"), out.stderr[-2000:]


def test_a_fork_of_the_child_exits_without_joining_a_watcher():
    # The fork holds the Event's lock, as if forked while the watcher waited
    # (or it already does).
    code = (
        "import os, sys, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    _exiting._cond.acquire(False)\n"
        "    sys.exit(0)\n"
        "for _ in range(1000):\n"
        "    done, status = os.waitpid(pid, os.WNOHANG)\n"
        "    if done:\n"
        "        print(os.waitstatus_to_exitcode(status))\n"
        "        break\n"
        "    time.sleep(0.01)\n"
        "else:\n"
        "    os.kill(pid, 9)\n"
        "    os.waitpid(pid, 0)\n"
        '    print("hung")\n'
    )
    out = _run(_child.limits(30, 512) + code)
    assert (out.returncode, out.stdout) == (0, "0\n"), out.stderr[-2000:]
