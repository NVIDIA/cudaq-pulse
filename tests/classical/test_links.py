# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Links named in the machine model, and functions that name them."""

from __future__ import annotations

import cudaq_pulse as pulse
import pytest
from cudaq_pulse.classical import Link, emulation
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.kernel.ir_builder import CompilationError

from tests._handles import Q

DECODER = Link(fixed_ns=100, feedback_ns=20)
CONTROLLER = Link(fixed_ns=4, max_in_flight=4)


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


def test_names_take_the_lowest_free_ids_in_order():
    machine = MachineModel(links={"decoder": DECODER, "controller": CONTROLLER})
    assert dict(machine.link_names) == {"decoder": 0, "controller": 1}
    assert machine.links[1] == CONTROLLER
    mixed = MachineModel(links={0: DECODER, "controller": CONTROLLER, 2: Link()})
    assert mixed.link_id("controller") == 1
    assert mixed.link_name(1) == "controller" and mixed.link_name(0) is None
    assert mixed.link("controller") == CONTROLLER and mixed.link(7) == Link()


def test_a_model_made_with_replace_keeps_its_link_names():
    import dataclasses

    from cudaq_pulse.controller import ControllerLimits

    mixed = MachineModel(links={0: DECODER, "controller": CONTROLLER, 2: Link(), "spare": Link()})
    changed = dataclasses.replace(mixed, controller=ControllerLimits(max_events=10))
    assert dict(changed.link_names) == {"controller": 1, "spare": 3}
    assert changed.link("controller") == CONTROLLER and dict(changed.links) == dict(mixed.links)
    assert dataclasses.replace(changed, controller=None) == mixed
    named = dataclasses.replace(mixed, links={"decoder": DECODER})
    assert dict(named.link_names) == {"decoder": 0}


def test_a_named_link_compiles_to_the_same_ir_as_its_id():
    machine = MachineModel(links={"decoder": DECODER, "controller": CONTROLLER})
    by_id = MachineModel(links={0: DECODER, 1: CONTROLLER})
    named = pulse.compile(_kernel(_function("controller")), machine=machine).lower(Q[0])
    numbered = pulse.compile(_kernel(_function(1)), machine=by_id).lower(Q[0])
    assert named.mlir() == numbered.mlir()
    (call,) = named.classical_report()
    assert call.link == 1 and call.link_name == "controller"
    assert numbered.classical_report()[0].link_name is None


def test_the_manifest_and_artifact_carry_link_names():
    machine = MachineModel(links={"decoder": DECODER, "controller": CONTROLLER})
    low = pulse.compile(_kernel(_function("controller")), machine=machine).lower(Q[0])
    (entry,) = emulation.manifest(low)["functions"]
    assert entry["link"] == 1 and entry["link_name"] == "controller"
    assert entry["max_in_flight"] == 4
    assert low.export_emulation()["links"]["1"]["name"] == "controller"


def test_transport_may_be_given_per_link_name():
    machine = MachineModel(links={"controller": CONTROLLER})
    low = pulse.compile(_kernel(_function("controller")), machine=machine).lower(Q[0])
    slow = emulation.Transport(fixed_ns=1000)
    result = emulation.emulate(low, shots=5, transport={"controller": slow}, seed=1)
    assert result.late_fraction == 1.0
    result = emulation.emulate(low, shots=5, transport={0: slow}, seed=1)
    assert result.late_fraction == 1.0


def test_unknown_names_and_bad_keys_are_errors():
    fn = _function("controller")
    with pytest.raises(CompilationError, match="no link named 'controller'.*named links: none"):
        pulse.compile(_kernel(fn)).lower(Q[0])
    with pytest.raises(CompilationError, match="named links: 'decoder'"):
        pulse.compile(_kernel(fn), machine=MachineModel(links={"decoder": DECODER})).lower(Q[0])
    with pytest.raises(ValueError, match="link name must be an identifier"):
        _function("to fpga")
    with pytest.raises(ValueError, match="names are identifiers"):
        MachineModel(links={"to fpga": DECODER})
    with pytest.raises(ValueError, match="names or link ids"):
        MachineModel(links={-1: DECODER})


def test_names_are_part_of_a_machine_models_identity():
    a = MachineModel(links={"decoder": DECODER, "controller": CONTROLLER})
    b = MachineModel(links={"decoder": DECODER, "controller": CONTROLLER})
    swapped = MachineModel(links={"controller": DECODER, "decoder": CONTROLLER})
    assert a == b and hash(a) == hash(b)
    assert a != swapped
    kernel = _kernel(_function("controller"))
    first = pulse.compile(kernel, machine=a).lower(Q[0]).classical_report()[0]
    other = pulse.compile(kernel, machine=swapped).lower(Q[0]).classical_report()[0]
    assert (first.link, other.link) == (1, 0)


def test_the_timeline_labels_links_by_name():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    machine = MachineModel(links={"decoder": DECODER, "controller": CONTROLLER})
    low = pulse.compile(_kernel(_function("controller")), machine=machine).lower(Q[0])
    figure = low.timeline()
    labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
    assert "link controller" in labels
