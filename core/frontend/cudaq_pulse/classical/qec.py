# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-time decoding with the messages of the CUDA-Q QEC decoders.

A ``Decoder`` declares the three calls of a CUDA-QX real-time decoder with
their dispatch keys and payloads, so a program built with it talks to a
CUDA-QX decoding server unchanged. Every decoder sends the same three keys;
the server tells them apart by the decoder id in the payload:

- ``enqueue_syndromes(decoder, measurements, tag)``: send one round's
  syndrome bits (``{decoder id, tag, mapping id, count}`` and the bits);
  nothing comes back.
- ``get_corrections(decoder, reset=False)``: the corrections decoded from
  the syndromes enqueued since the last decode or reset; the call to wait on.
  Asked again, it gives the same corrections until the next enqueue starts a
  new volume; ``reset=True`` drops them once read. With nothing to decode the
  decoder answers NOT_READY, which a wait with a timeout takes as late.
- ``reset_decoder(decoder)``: clear what was enqueued.

The decoder's ``decode`` is a Python function from the syndrome bits of one
volume to correction bits. It runs with the state a CUDA-QX decoding server
keeps, per path in simulation and per shot in emulation, and per decoder in
``emulation.serve_udp``. A ``Decoder`` declares no volume size, so a get
decodes what has arrived; a server waits for the volume its configuration
gives and answers NOT_READY before it::

    from cudaq_pulse.classical import qec

    decoder = qec.Decoder(0, syndrome_count=2, correction_count=1, max_latency_ns=2000,
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
from typing import TYPE_CHECKING, Any

from .._errors import InvalidTypeError, InvalidValueError
from . import ClassicalFunction, _Bits, _Byte, _Integer, _NotReadyError, dispatch_key

if TYPE_CHECKING:
    from .._units import Time

__all__ = [
    "Decoder",
    "enqueue_syndromes",
    "get_corrections",
    "reset_decoder",
]

ENQUEUE_SYNDROMES_ID = dispatch_key("enqueue_syndromes")
GET_CORRECTIONS_ID = dispatch_key("get_corrections")
RESET_DECODER_ID = dispatch_key("reset_decoder")


class Decoder:
    """One real-time decoder: its id and the sizes of its messages.

    ``syndrome_count`` is the number of syndrome bits a round enqueues and
    ``correction_count`` the number of correction bits ``get_corrections``
    returns. ``decode`` maps the syndrome bits of one volume, in the order
    they were enqueued, to the correction bits; it is the decoder simulation
    and emulation run.

    The latencies are those of ``get_corrections``, as for
    ``@pulse.classical``: ``min_latency_ns``, ``nominal_latency_ns`` (the
    maximum unless given) and ``max_latency_ns`` give the latency model
    simulation draws from, and the compiled program waits for the maximum.
    With ``timeout_ns``, every ``get_corrections(decoder).result()`` waits
    that long and gives ``(in_time, corrections)``. ``enqueue_latency_ns``
    is the budget of ``enqueue_syndromes`` and ``reset_decoder``.
    """

    def __init__(
        self,
        decoder_id: int,
        *,
        syndrome_count: int,
        correction_count: int,
        max_latency_ns: float | Time,
        min_latency_ns: float | Time = 0.0,
        nominal_latency_ns: float | Time | None = None,
        timeout_ns: float | Time | None = None,
        enqueue_latency_ns: float | Time = 0.0,
        link: int | str = 0,
        decode: Callable[[list[int]], Sequence[int]] | None = None,
    ) -> None:
        self.decoder_id = decoder_id
        self.syndrome_count = syndrome_count
        self.correction_count = correction_count
        self.decode = decode
        suffix = f"{decoder_id}_{syndrome_count}_{correction_count}"
        self.enqueue_function = ClassicalFunction._declared(
            None,
            (_Integer(), _Integer(), _Integer(), _Integer(), _Bits(syndrome_count)),
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
            (_Bits(correction_count),),
            name=f"get_corrections_{suffix}",
            max_latency_ns=max_latency_ns,
            min_latency_ns=min_latency_ns,
            nominal_latency_ns=nominal_latency_ns,
            latency_mode="bounded" if timeout_ns is None else "timeout",
            timeout_ns=timeout_ns,
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
        """Handlers for the three calls, for ``simulate(handlers=)``,
        ``emulate(handlers=)`` and ``serve_udp``.

        They keep the decoder's state as a CUDA-QX decoding server does. In
        simulation and in-process emulation it is rebuilt from the path's
        own calls, so every path and every shot starts afresh; called
        without them, as ``serve_udp`` does, the handlers keep one state,
        changed by each call as it comes.
        """
        decode = self.decode
        enqueue_name = self.enqueue_function.name
        get_name = self.corrections_function.name
        reset_name = self.reset_function.name
        correction_count = self.correction_count

        def decoded(bits: list[int]) -> list[int]:
            if decode is None:
                raise InvalidTypeError("this Decoder has no decode function to simulate it with")
            answer = list(decode(bits))
            if len(answer) != correction_count:
                raise InvalidValueError(
                    f"decode returned {len(answer)} bits, expected {correction_count}"
                )
            return answer

        served = _Session()

        def enqueue(
            decoder_id: int,
            tag: int,
            mapping: int,
            count: int,
            bits: list[int],
            history: Sequence[Any] | None = None,
        ) -> None:
            if history is None:
                served.enqueue(bits)

        def reset(decoder_id: int, history: Sequence[Any] | None = None) -> None:
            if history is None:
                served.reset()

        def get(
            decoder_id: int, return_size: int, clear: int, history: Sequence[Any] | None = None
        ) -> Any:
            if history is None:
                return served.get(decoded, clear)
            session = _Session()
            for record in history:
                if record.function == enqueue_name:
                    session.enqueue(record.inputs[4])
                elif record.function == get_name:
                    try:
                        session.replay_get(record.inputs[2])
                    except _NotReadyError:
                        pass
                elif record.function == reset_name:
                    session.reset()
            return session.get(decoded, clear)

        enqueue.takes_history = True  # type: ignore[attr-defined]
        get.takes_history = True  # type: ignore[attr-defined]
        reset.takes_history = True  # type: ignore[attr-defined]
        return {enqueue_name: enqueue, get_name: get, reset_name: reset}


class _Session:
    """A decoder's state as a CUDA-QX decoding server keeps it: the bits of
    the volume being collected, and the corrections of the last decode until
    the next enqueue starts a new volume (or a get with reset drops them)."""

    __slots__ = ("bits", "complete", "decoded")

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.bits: list[int] = []
        self.complete = False
        self.decoded: list[int] | None = None

    def enqueue(self, bits: Sequence[int]) -> None:
        if self.complete:
            self.reset()
        self.bits.extend(bits)

    def get(self, decoded: Callable[[list[int]], list[int]], clear: int) -> list[int]:
        self.replay_get(0)
        if self.decoded is None:
            self.decoded = decoded(self.bits)
        answer = self.decoded
        if clear:
            self.reset()
        return answer

    def replay_get(self, clear: int) -> None:
        """A get, as far as the state goes: it completes the volume (its
        decode runs when an answer is needed), or finds nothing to decode."""
        if not self.complete:
            if not self.bits:
                raise _NotReadyError
            self.complete = True
        if clear:
            self.reset()


def enqueue_syndromes(
    decoder: Decoder, syndromes: Sequence[Any], tag: Any = 0, mapping: int = 0
) -> Any:
    """Send one round of syndrome measurements to *decoder* (in a kernel)."""
    return decoder.enqueue_function.submit(
        decoder.decoder_id, tag, mapping, decoder.syndrome_count, syndromes
    )


def get_corrections(decoder: Decoder, reset: bool = False) -> Any:
    """Ask *decoder* for its corrections; take them with ``.result()``."""
    return decoder.corrections_function.submit(
        decoder.decoder_id, decoder.correction_count, int(reset)
    )


def reset_decoder(decoder: Decoder) -> Any:
    """Clear what *decoder* has been sent (in a kernel)."""
    return decoder.reset_function.submit(decoder.decoder_id)
