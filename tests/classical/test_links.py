# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Links named in the control system, and functions that name them."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse import ControlSystem
from cudaq_pulse.classical import Link, emulation
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q

DECODER = Link(fixed_ns=100, feedback_ns=20)
CONTROLLER = Link(fixed_ns=4, slots=4)


def _function(link):
    def decide(bits: pulse.Bits) -> pulse.Bits:
        return [bits[0]]

    return pulse.classical(decide, max_latency_ns=100, link=link)


def _kernel(fn):
    @pulse.kernel
    def feedback(q):
        d = get_drive_line(q)
        r = get_readout_line(q)
        m = readout(r, square(100, 0.05))
        if fn.submit([m]).result()[0]:
            drive(d, square(40, 0.1))

    return feedback


def test_a_link_is_named_or_given_by_its_position():
    machine = ControlSystem(links={"decoder": DECODER, "controller": CONTROLLER})
    named = pulse.compile(_kernel(_function("controller")), control_system=machine).lower(Q[0])
    numbered = pulse.compile(_kernel(_function(1)), control_system=machine).lower(Q[0])
    assert named.mlir() == numbered.mlir()
    (call,) = named.classical_report()
    assert call.link == 1 and call.link_name == "controller"
    assert numbered.classical_report()[0].link_name == "controller"
    # An id past the listed links gets Link().
    unlisted = pulse.compile(_kernel(_function(7)), control_system=machine).lower(Q[0])
    assert "fixed_ns = 0.000000e+00" in unlisted.mlir()
    assert unlisted.classical_report()[0].link_name is None


def test_the_manifest_and_artifact_carry_link_names():
    machine = ControlSystem(links={"decoder": DECODER, "controller": CONTROLLER})
    low = pulse.compile(_kernel(_function("controller")), control_system=machine).lower(Q[0])
    (entry,) = emulation.manifest(low)["functions"]
    assert entry["link"] == 1 and entry["link_name"] == "controller"
    assert entry["slots"] == 4
    assert low.export_to_emulator()["links"]["1"]["name"] == "controller"


def test_transport_may_be_given_per_link_name():
    machine = ControlSystem(links={"controller": CONTROLLER})
    low = pulse.compile(_kernel(_function("controller")), control_system=machine).lower(Q[0])
    slow = emulation.Transport(fixed_ns=1000)
    result = emulation.emulate(low, shots_count=5, transport={"controller": slow}, seed=1)
    assert result.late_fraction == 1.0
    result = emulation.emulate(low, shots_count=5, transport={0: slow}, seed=1)
    assert result.late_fraction == 1.0


def test_unknown_names_and_bad_keys_are_errors():
    fn = _function("controller")
    with pytest.raises(CompilationError, match="no link named 'controller'.*its links: none"):
        pulse.compile(_kernel(fn)).lower(Q[0])
    with pytest.raises(CompilationError, match="its links: 'decoder'"):
        pulse.compile(_kernel(fn), control_system=ControlSystem(links={"decoder": DECODER})).lower(
            Q[0]
        )
    with pytest.raises(ValueError, match="link name must be an identifier"):
        _function("to fpga")
    with pytest.raises(ValueError, match="keyed by name"):
        ControlSystem(links={"to fpga": DECODER})
    with pytest.raises(ValueError, match="keyed by name"):
        ControlSystem(links={-1: DECODER})  # type: ignore[dict-item]


def test_names_are_part_of_a_control_systems_identity():
    a = ControlSystem(links={"decoder": DECODER, "controller": CONTROLLER})
    b = ControlSystem(links={"decoder": DECODER, "controller": CONTROLLER})
    swapped = ControlSystem(links={"controller": DECODER, "decoder": CONTROLLER})
    assert a == b and hash(a) == hash(b)
    assert a != swapped
    kernel = _kernel(_function("controller"))
    first = pulse.compile(kernel, control_system=a).lower(Q[0]).classical_report()[0]
    other = pulse.compile(kernel, control_system=swapped).lower(Q[0]).classical_report()[0]
    assert (first.link, other.link) == (1, 0)


def test_the_timeline_labels_links_by_name():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    machine = ControlSystem(links={"decoder": DECODER, "controller": CONTROLLER})
    low = pulse.compile(_kernel(_function("controller")), control_system=machine).lower(Q[0])
    figure = low.timeline()
    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert "link controller" in labels
