# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Methods that call the pulse DSL are inlined like plain helper functions.

A method is inlined with its object as ``self`` (its class as ``cls``) and
the globals of the module that defines it, whether it calls a pulse op by
name or reaches one through ``self``, ``cls`` or an object ``self`` holds.
Each kernel must trace to the same program as the op written out.
"""

from __future__ import annotations

import threading

import cudaq_pulse as pulse
import pytest

from tests._handles import Q
from tests._ir import python_ir
from tests.kernel._gate_library import Device, Doubled, Gates

_GATES = Gates(0.2)
_DOUBLED = Doubled(0.3)
_DEVICE = Device(0.15)
_STORED = _GATES.x90_wf


def _program(kernel):
    """The ops a kernel traces to, with the attributes that set them."""
    return [(op.kind, op.attrs) for op in python_ir(kernel, Q[0]).ops]


def _written_out(n, amp):
    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, square(n, amp), t)

    return k


@pulse.kernel
def plain_method(q):
    d, t = get_drive_line(q)
    drive(d, _GATES.wf(), t)


@pulse.kernel
def method_with_lines(q):
    d, t = get_drive_line(q)
    _GATES.x90(d, t)


@pulse.kernel
def through_self(q):
    d, t = get_drive_line(q)
    drive(d, _GATES.x90_wf(), t)


@pulse.kernel
def inherited(q):
    d, t = get_drive_line(q)
    drive(d, _DOUBLED.x180_wf(), t)


@pulse.kernel
def staticmethod_on_instance(q):
    d, t = get_drive_line(q)
    drive(d, _GATES.flat(40), t)


@pulse.kernel
def staticmethod_on_class(q):
    d, t = get_drive_line(q)
    drive(d, Gates.flat(40), t)


@pulse.kernel
def classmethod_through_cls(q):
    d, t = get_drive_line(q)
    drive(d, Gates.default_wf(), t)


@pulse.kernel
def callable_object(q):
    d, t = get_drive_line(q)
    drive(d, _GATES(40), t)


@pulse.kernel
def stored_bound_method(q):
    d, t = get_drive_line(q)
    drive(d, _STORED(), t)


@pulse.kernel
def bound_in_the_kernel(q):
    d, t = get_drive_line(q)
    make = _GATES.x90_wf
    drive(d, make(), t)


@pulse.kernel
def through_a_held_object(q):
    d, t = get_drive_line(q)
    drive(d, _DEVICE.pulse_wf(), t)


@pytest.mark.parametrize(
    ("kernel", "n", "amp"),
    [
        (plain_method, 40, 0.2),
        (method_with_lines, 40, 0.2),
        (through_self, 40, 0.2),
        (inherited, 80, 0.3),
        (staticmethod_on_instance, 40, 0.1),
        (staticmethod_on_class, 40, 0.1),
        (classmethod_through_cls, 40, 0.25),
        (callable_object, 40, 0.2),
        (stored_bound_method, 40, 0.2),
        (bound_in_the_kernel, 40, 0.2),
        (through_a_held_object, 40, 0.15),
    ],
)
def test_a_method_traces_like_the_op_written_out(kernel, n, amp):
    assert _program(kernel) == _program(_written_out(n, amp))
    assert [e.kind for e in pulse.compile(kernel).lower(Q[0]).schedule] == ["drive"]


def test_a_method_that_uses_no_pulse_op_runs_as_python():
    class Calibration:
        def __init__(self):
            self.calls = 0

        def amp(self):
            self.calls += 1
            return 0.2

    calibration = Calibration()

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        drive(d, square(40, calibration.amp()), t)

    assert _program(k) == _program(_written_out(40, 0.2))
    assert calibration.calls == 1


def test_host_objects_whose_methods_read_attributes_named_like_ops_run_as_python():
    class Clock:
        def __init__(self):
            self.event = threading.Event()
            self.event.set()
            self.ticks = 0

        def tick(self):
            self._wait()
            self.ticks += 1
            return 0.1

        def _wait(self):
            self.event.wait(1)

        def __call__(self):
            return self.tick()

    clock = Clock()
    barrier = threading.Barrier(1)

    @pulse.kernel
    def k(q):
        d, t = get_drive_line(q)
        barrier.wait()
        drive(d, square(40, clock.tick() + clock()), t)

    assert _program(k) == _program(_written_out(40, 0.2))
    assert clock.ticks == 2
