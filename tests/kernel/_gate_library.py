# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pulse helpers written as methods, in a module of their own.

The kernels that call them are in test_method_helpers.py; the methods read
this module's globals (``_LENGTH``), not the caller's.
"""

from cudaq_pulse.ops import drive, square

_LENGTH = 40


class Gates:
    def __init__(self, amp):
        self.amp = amp

    def wf(self, n=_LENGTH):
        return square(n, self.amp)

    def x90(self, line, tone):
        drive(line, self.wf(), tone=tone)

    def x90_wf(self):
        # No pulse op by name: it reaches one through self.
        return self.wf()

    @staticmethod
    def flat(n):
        return square(n, 0.1)

    @classmethod
    def default_wf(cls):
        return cls(0.25).wf()

    def __call__(self, n):
        return self.wf(n)


class Doubled(Gates):
    def x180_wf(self):
        return self.wf(2 * _LENGTH)


class Device:
    """Holds its gates: the kernel reaches them through two attributes."""

    def __init__(self, amp):
        self.gates = Gates(amp)

    def pulse_wf(self):
        return self.gates.x90_wf()
