# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a controller can do (``Controller.supports``)."""

from __future__ import annotations

import enum

__all__ = ["Capability"]


class Capability(enum.StrEnum):
    """What a controller can do, for ``Controller.supports``.

    ``play`` a drive; ``readout`` a qudit; ``wait`` for a length computed
    while the QPU runs; ``sync`` lines; ``call`` a classical function and
    take its ``answer``; ``branch`` on an outcome or an answer;
    ``runtime_frame_update`` a tone's frequency or phase from a value
    computed while the QPU runs; ``forward_outcome`` of one of its readouts
    to another controller, which branches on it; ``loop``: run a rolled
    loop itself, holding its body once. A controller without ``loop`` plays
    a rolled loop unrolled, so each iteration counts against its
    ``max_pulses``. Unrolling does not change its waveform memory: a
    waveform is stored once per shape however often it plays, so only the
    pulse count goes up.
    """

    PLAY = "play"
    READOUT = "readout"
    WAIT = "wait"
    SYNC = "sync"
    CALL = "call"
    ANSWER = "answer"
    BRANCH = "branch"
    RUNTIME_FRAME_UPDATE = "runtime_frame_update"
    FORWARD_OUTCOME = "forward_outcome"
    LOOP = "loop"
