# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-Q QEC's decoders answering pulse kernels, with the real cudaq_qec.

Skipped where cudaq_qec does not import (it installs on Linux). The decoder
is ``single_error_lut``, which needs no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

cudaq_qec = pytest.importorskip("cudaq_qec", reason="needs CUDA-Q QEC (Linux)")

import cudaq_pulse as pulse  # noqa: E402
from cudaq_pulse.classical import _cudaqx, emulation  # noqa: E402

from tests.classical.test_simulation import _device  # noqa: E402

REPETITION = """
decoders:
  - id: 0
    type: single_error_lut
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1]
"""

TABLE = {(0, 0): [0, 0, 0], (1, 0): [1, 0, 0], (1, 1): [0, 1, 0], (0, 1): [0, 0, 1]}


@pytest.fixture
def decoder():
    before = _cudaqx._use_backend(None)
    assert cudaq_qec.configure_decoders_from_str(REPETITION) == 0
    try:
        yield
    finally:
        cudaq_qec.finalize_decoders()
        _cudaqx._use_backend(before)


enqueue_syndromes = pulse.classical(cudaq_qec.qec.enqueue_syndromes, link="gpu")
get_corrections = pulse.classical(cudaq_qec.qec.get_corrections, link="gpu", max_latency_ns=500)
reset_decoder = pulse.classical(cudaq_qec.qec.reset_decoder, link="gpu")
LINKS = pulse.ControlSystem(links={"gpu": pulse.Link(feedback_ns=50)})


@pulse.kernel
def repetition_round(a0, a1, decoder_id: int):
    s0 = readout(get_readout_line(a0), square(100, 0.05))
    s1 = readout(get_readout_line(a1), square(100, 0.05))
    enqueue_syndromes.submit(decoder_id, [s0, s1], 0)
    get_corrections.submit(decoder_id, 3, False).result()


def _lowered(decoder_id=0):
    dev = _device(n=2)
    compiled = pulse.compile(repetition_round, quantum_system=dev, control_system=LINKS)
    return compiled.lower(*dev.qudits[:2], decoder_id)


@pytest.mark.usefixtures("decoder")
def test_the_configured_decoder_answers_each_path():
    plus = np.array([1, 1]) / np.sqrt(2)
    result = pulse.simulate(_lowered(), initial_state=np.kron(plus, plus), levels=2)
    assert {tuple(b.levels): b.calls[-1].answer for b in result.branches} == TABLE


@pytest.mark.usefixtures("decoder")
def test_the_configured_decoder_answers_each_shot():
    run = emulation.emulate(_lowered(), shots_count=20, seed=4, service_time="none")
    for outcomes, calls in zip(run.outcomes, run.calls, strict=True):
        assert calls[-1].answer == TABLE[tuple(outcomes)]


@pytest.mark.usefixtures("decoder")
def test_outside_a_kernel_the_decoder_answers_now():
    for syndrome, want in TABLE.items():
        reset_decoder(0)
        enqueue_syndromes(0, list(syndrome), 0)
        assert get_corrections(0, 3, False) == want
    # Corrections accumulate until a reset, as CUDA-Q QEC's simulation does.
    reset_decoder(0)
    enqueue_syndromes(0, [1, 0], 0)
    enqueue_syndromes(0, [0, 1], 1)
    assert get_corrections(0, 3, True) == [1, 0, 1]


@pytest.mark.usefixtures("decoder")
def test_a_decoder_that_is_not_configured_is_an_error_not_a_crash():
    with pytest.raises(pulse.ExecutionError, match="CUDA-Q QEC decoder 5"):
        pulse.simulate(_lowered(decoder_id=5), levels=2)
    reset_decoder(0)
    with pytest.raises(pulse.ExecutionError, match="CUDA-Q QEC decoder 0"):
        enqueue_syndromes(0, [1] * 9, 0)
    # The process goes on, and so does the decoder.
    reset_decoder(0)
    enqueue_syndromes(0, [1, 1], 0)
    assert get_corrections(0, 3, False) == [0, 1, 0]
