# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Real-time decoding with the messages of the CUDA-Q QEC decoders.

A ``Decoder`` declares the three calls of a CUDA-QX real-time decoder with
their dispatch keys and payloads, so a program built with it talks to a
CUDA-QX decoding server unchanged. Every decoder sends the same three keys;
the server tells them apart by the decoder id in the payload:

- ``enqueue_syndromes(decoder, measurements, round)``: send one round's
  syndrome bits (``{decoder id, round, mapping id, count}`` and the bits);
  nothing comes back.
- ``get_corrections(decoder, reset=False)``: the corrections decoded from
  everything enqueued since the last reset; the call to wait on.
- ``reset_decoder(decoder)``: clear what was enqueued.

In simulation the decoder is a Python function from the syndrome bits
enqueued on a path to correction bits; each path keeps its own::

    from cudaq_pulse.classical import qec

    decoder = qec.Decoder(0, syndromes=2, corrections=1, max_latency_ns=2000,
                          decode=lambda bits: [bits[0] ^ bits[1]])

    @pulse.kernel
    def memory(...):
        qec.reset_decoder(decoder)
        for r in range(rounds):
            ...
            qec.enqueue_syndromes(decoder, [m0, m1], r)
        fix = qec.get_corrections(decoder).result()
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from .._errors import InvalidTypeError, InvalidValueError
from . import ClassicalFunction, _Bits, _Byte, _Integer, function_id

__all__ = [
    "Decoder",
    "enqueue_syndromes",
    "enqueue_syndromes_mapped",
    "get_corrections",
    "reset_decoder",
]

ENQUEUE_SYNDROMES_ID = function_id("enqueue_syndromes")
GET_CORRECTIONS_ID = function_id("get_corrections")
RESET_DECODER_ID = function_id("reset_decoder")


class Decoder:
    """One real-time decoder: its id and the sizes of its messages.

    ``syndromes`` is the number of syndrome bits a round enqueues and
    ``corrections`` the number of correction bits ``get_corrections``
    returns. ``decode`` maps all syndrome bits enqueued since the last reset,
    in order, to the correction bits; it is the decoder simulation runs.
    """

    def __init__(
        self,
        decoder_id: int,
        *,
        syndromes: int,
        corrections: int,
        max_latency_ns: float,
        enqueue_latency_ns: float = 0.0,
        link: int = 0,
        decode: Callable[[list[int]], Sequence[int]] | None = None,
    ) -> None:
        self.decoder_id = decoder_id
        self.syndromes = syndromes
        self.corrections = corrections
        self.decode = decode
        suffix = f"{decoder_id}_{syndromes}_{corrections}"
        self.enqueue_function = ClassicalFunction._declared(
            None,
            (_Integer(), _Integer(), _Integer(), _Integer(), _Bits(syndromes)),
            (),
            name=f"enqueue_syndromes_{suffix}",
            max_latency_ns=enqueue_latency_ns,
            link=link,
            _dispatch_key=ENQUEUE_SYNDROMES_ID,
            _key_route=decoder_id,
        )
        self.corrections_function = ClassicalFunction._declared(
            None,
            (_Integer(), _Integer(), _Byte()),
            (_Bits(corrections),),
            name=f"get_corrections_{suffix}",
            max_latency_ns=max_latency_ns,
            link=link,
            _dispatch_key=GET_CORRECTIONS_ID,
            _key_route=decoder_id,
        )
        self.reset_function = ClassicalFunction._declared(
            None,
            (_Integer(),),
            (),
            name=f"reset_decoder_{suffix}",
            max_latency_ns=enqueue_latency_ns,
            link=link,
            _dispatch_key=RESET_DECODER_ID,
            _key_route=decoder_id,
        )
        if decode is not None:
            for fn, handler in zip(
                (self.enqueue_function, self.corrections_function, self.reset_function),
                self.handlers().values(),
                strict=False,
            ):
                fn.implementation = handler

    def handlers(self) -> dict[str, Callable[..., Any]]:
        """Simulation handlers for the three calls, for ``simulate(handlers=)``.

        ``get_corrections`` decodes the syndromes the path enqueued since its
        last reset, so every path keeps its own decoder state.
        """
        decode = self.decode
        enqueue_name = self.enqueue_function.name
        reset_name = self.reset_function.name
        corrections = self.corrections

        def enqueue(decoder_id: int, round: int, mapping: int, count: int, bits: list[int]) -> None:
            return None

        def reset(decoder_id: int) -> None:
            return None

        def get(decoder_id: int, return_size: int, clear: int, history: Sequence[Any] = ()) -> Any:
            if decode is None:
                raise InvalidTypeError("this Decoder has no decode function to simulate it with")
            bits: list[int] = []
            for record in history:
                if record.function == reset_name:
                    bits = []
                elif record.function == enqueue_name:
                    bits.extend(record.inputs[4])
            answer = list(decode(bits))
            if len(answer) != corrections:
                raise InvalidValueError(
                    f"decode returned {len(answer)} bits, expected {corrections}"
                )
            return answer

        get.takes_history = True  # type: ignore[attr-defined]
        return {
            enqueue_name: enqueue,
            self.corrections_function.name: get,
            reset_name: reset,
        }


def enqueue_syndromes(
    decoder: Decoder, syndromes: Sequence[Any], round: Any = 0, mapping: int = 0
) -> Any:
    """Send one round of syndrome measurements to *decoder* (in a kernel)."""
    return decoder.enqueue_function.submit(
        decoder.decoder_id, round, mapping, decoder.syndromes, syndromes
    )


def enqueue_syndromes_mapped(
    decoder: Decoder, syndromes: Sequence[Any], round: Any, mapping: int
) -> Any:
    """``enqueue_syndromes`` with the syndrome-mapping row given, under the
    name CUDA-QX uses for it."""
    return enqueue_syndromes(decoder, syndromes, round, mapping)


def get_corrections(decoder: Decoder, reset: bool = False) -> Any:
    """Ask *decoder* for its corrections; take them with ``.result()``."""
    return decoder.corrections_function.submit(decoder.decoder_id, decoder.corrections, int(reset))


def reset_decoder(decoder: Decoder) -> Any:
    """Clear what *decoder* has been sent (in a kernel)."""
    return decoder.reset_function.submit(decoder.decoder_id)
