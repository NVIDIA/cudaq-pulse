# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for CUDA-Q QEC's real-time decoding functions and decoders.

``qec`` holds objects named and placed as ``cudaq_qec.qec``'s are (native
functions, not Python ones), so ``pulse.classical`` takes them as it takes
the real ones. ``Decoders`` answers as CUDA-Q QEC's simulation library does:
each enqueued round is decoded and its corrections accumulate (XOR) until a
reset or a get with reset.
"""

from __future__ import annotations

import contextlib
import types
from collections.abc import Callable, Iterator, Sequence

from cudaq_pulse.classical import _cudaqx

MODULE = "cudaq_qec._pycudaqx_qec_the_suffix_matters_cudaq_qec.qec"


class RealtimeFunction:
    """What cudaq_qec.qec's functions are to Python: a native function that
    does nothing outside a CUDA-Q kernel."""

    def __init__(self, name: str, qualname: str | None = None, module: str = MODULE) -> None:
        self.__name__ = name
        self.__qualname__ = qualname or name
        self.__module__ = module
        self.__doc__ = f"{name}(...) -> None"

    def __call__(self, *args: object) -> None:
        return None


NAMES = (
    "enqueue_syndromes",
    "enqueue_syndromes_mapped",
    "enqueue_syndromes_test",
    "get_corrections",
    "get_logical_measurements",
    "reset_decoder",
)

#: cudaq_qec.qec as cudaq-qec 0.7 and 0.8 (nanobind) have it.
qec = types.SimpleNamespace(**{n: RealtimeFunction(n) for n in NAMES})
#: As cudaq-qec 0.6.0 (pybind11) has it.
qec_060 = types.SimpleNamespace(**{n: RealtimeFunction(n, f"PyCapsule.{n}") for n in NAMES})


class Decoders:
    """Decoders by id, each a function from one round's syndrome bits to
    its corrections."""

    def __init__(self, **decoders: Callable[[list[int]], Sequence[int]]) -> None:
        self.decode = {int(k.removeprefix("d")): fn for k, fn in decoders.items()}
        self.corrections: dict[int, list[int]] = {}
        self.calls: list[tuple[object, ...]] = []

    def _decoder(self, decoder_id: int) -> Callable[[list[int]], Sequence[int]]:
        if decoder_id not in self.decode:
            raise _cudaqx.ExecutionError(f"CUDA-Q QEC decoder {decoder_id}: Decoder not found")
        return self.decode[decoder_id]

    def reset(self, decoder_id: int) -> None:
        self._decoder(decoder_id)
        self.calls.append(("reset", decoder_id))
        self.corrections.pop(decoder_id, None)

    def enqueue(self, decoder_id: int, syndromes: Sequence[int], tag: int) -> None:
        bits = [1 if s else 0 for s in syndromes]
        self.calls.append(("enqueue", decoder_id, tuple(bits), tag))
        fix = list(self._decoder(decoder_id)(bits))
        held = self.corrections.get(decoder_id, [0] * len(fix))
        self.corrections[decoder_id] = [a ^ b for a, b in zip(held, fix, strict=True)]

    def get(self, decoder_id: int, size: int, reset: bool) -> list[int]:
        self._decoder(decoder_id)
        self.calls.append(("get", decoder_id, size, reset))
        held = self.corrections.get(decoder_id, [0] * size)
        if len(held) != size:
            raise _cudaqx.ExecutionError(
                f"CUDA-Q QEC decoder {decoder_id}: return size {size} is not the observable"
                f" count {len(held)}"
            )
        if reset:
            self.corrections.pop(decoder_id, None)
        return list(held)


@contextlib.contextmanager
def answering(decoders: Decoders) -> Iterator[Decoders]:
    """Answer CUDA-Q QEC calls with *decoders* inside the block."""
    before = _cudaqx._use_backend(decoders)
    try:
        yield decoders
    finally:
        _cudaqx._use_backend(before)


def repetition(bits: list[int]) -> list[int]:
    """The distance-3 repetition code's correction for syndromes (s0, s1)."""
    s0, s1 = bits
    return [int(s0 and not s1), int(s0 and s1), int(s1 and not s0)]
