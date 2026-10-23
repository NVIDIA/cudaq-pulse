# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Property-based tests of the compiler on random single-line programs."""

from __future__ import annotations

import cudaq_pulse as pulse
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property_settings import examples


def _random_kernel(amplitudes, durations, waits, phases):
    @pulse.kernel
    def k(q0):
        d0, t0 = get_drive_line(q0)
        for i in range(len(durations)):
            t0 = shift_phase(t0, phases[i % len(phases)])
            drive(d0, square(durations[i], amplitudes[i % len(amplitudes)]), t0)
            if i < len(waits):
                wait(d0, waits[i])

    return k


def _compile(kernel, **kwargs):
    return pulse.compile(kernel, **kwargs).lower(pulse.qudit_ref())


def _makespan(compiled):
    return max((e.start_vtu + e.duration_vtu for e in compiled.schedule), default=0)


_amplitudes = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=5,
)
_durations = st.lists(st.integers(min_value=4, max_value=1000), min_size=1, max_size=10)
_waits = st.lists(st.integers(min_value=1, max_value=500), max_size=3)
_phases = st.lists(st.floats(min_value=-3.0, max_value=3.0), min_size=1, max_size=3)


@given(amplitudes=_amplitudes, durations=_durations, waits=_waits, phases=_phases)
@settings(max_examples=examples(50))
def test_schedules_are_back_to_back_on_one_line(amplitudes, durations, waits, phases):
    kernel = _random_kernel(amplitudes, durations, waits, phases)
    expected = sum(durations) + sum(waits[: len(durations)])
    for policy in ("asap", "alap"):
        compiled = _compile(kernel, schedule=policy, passes=())
        events = compiled.schedule
        assert _makespan(compiled) == expected
        for before, after in zip(events, events[1:]):
            assert after.start_vtu == before.start_vtu + before.duration_vtu


@given(amplitudes=_amplitudes, durations=_durations, phases=_phases)
@settings(max_examples=examples(30))
def test_default_passes_preserve_the_schedule(amplitudes, durations, phases):
    kernel = _random_kernel(amplitudes, durations, [], phases)
    assert _makespan(_compile(kernel)) == _makespan(_compile(kernel, passes=()))


@given(durations=_durations, phases=_phases)
@settings(max_examples=examples(30))
def test_virtual_z_is_idempotent(durations, phases):
    kernel = _random_kernel([0.3], durations, [], phases)
    once = _compile(kernel, passes=("virtual_z",))
    once._module.run_passes(["pulse-virtual-z"])
    assert (
        once._module.print() == _compile(kernel, passes=("virtual_z", "virtual_z"))._module.print()
    )


@given(amplitudes=_amplitudes, durations=_durations)
@settings(max_examples=examples(30))
def test_iterating_a_list_traces_like_indexing_it(amplitudes, durations):
    pairs = list(zip(durations, amplitudes * len(durations)))

    @pulse.kernel
    def iterated(q0):
        d0, t0 = get_drive_line(q0)
        for duration, amplitude in pairs:
            drive(d0, square(duration, amplitude), t0)

    @pulse.kernel
    def indexed(q0):
        d0, t0 = get_drive_line(q0)
        for i in range(len(pairs)):
            drive(d0, square(pairs[i][0], pairs[i][1]), t0)

    def text(kernel):
        return _compile(kernel)._module.print().split("\n", 1)[1]

    assert text(iterated) == text(indexed)


def test_the_selected_hypothesis_profile_is_active():
    from tests.property_settings import PROFILE

    active = settings.default
    assert active.derandomize == settings.get_profile(PROFILE).derandomize
    assert active.deadline is None
    if PROFILE == "ci":
        assert active.derandomize
