# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Calling classical functions while the QPU runs.

``classical`` is a classical real-time dialect for co-processing at QPU
runtime: a pulse program calls GPU, host or FPGA services (decoders,
estimators, calibration updates) and uses their answers in the same shot.
``@pulse.classical`` declares a function, ``fn.submit(...)`` sends a request
on a link, and ``.result()`` takes the answer where it is needed. The
scheduler places each call with its latency budget and reports how much
slack every answer has.

This example shows:

1. Real-time decoding with feed-forward: a readout goes to a decoder, the
   decoder's answer drives a conditional correction, and a timeout falls
   back to the raw measurement when the answer is late.
2. Adaptive Bayesian T1 tracking, after F. Berritta et al., "Real-time
   adaptive tracking of fluctuating relaxation rates in superconducting
   qubits", arXiv:2506.09576 (2025). An estimator on a low-latency
   controller link keeps a gamma posterior over the decay rate, passed in
   and out of every call, and picks the next wait, which a variable
   ``wait`` plays.
3. The classical-call report, simulation with the Python bodies standing in
   for the services, and the emulation artifact and call manifest a
   controller or emulator runs the program from.

NOTE: Requires cudaq-pulse native C++ bindings (see README for build
instructions).
"""

import json
import math
import tempfile
from pathlib import Path

import cudaq_pulse as pulse
from cudaq_pulse.classical import Bits, Float, Link, emulation
from cudaq_pulse.compiler import MachineModel
from cudaq_pulse.targets import Qubit, TargetDescription

# A pi pulse: 40 virtual units at 2 GHz is 20 ns, and with
# H = amplitude * X / 2 this amplitude (rad/ns) rotates by pi.
PI_AMP = math.pi / 20

# Link 0 reaches a GPU decoder; link 1 is the controller's own FPGA logic.
GPU_LINK, FPGA_LINK = 0, 1
MACHINE = MachineModel(
    links={
        GPU_LINK: Link(fixed_ns=100, bytes_per_ns=10, feedback_ns=50),
        FPGA_LINK: Link(fixed_ns=4, feedback_ns=8, max_in_flight=4),
    }
)


# ── 1. Real-time decoding with feed-forward ─────────────────────────────────


@pulse.classical(
    args=(Bits(1),),
    results=(Bits(1),),
    min_latency_ns=200,
    nominal_latency_ns=300,
    max_latency_ns=400,
    link=GPU_LINK,
)
def decode(syndrome):
    """A toy decoder: correct when the syndrome bit is set."""
    return [syndrome[0]]


@pulse.kernel
def decoded_feed_forward(qubit):
    """Measure, decode, and correct; the raw measurement if the decoder is late."""
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    m = readout(readout_line, square(1000, 0.05), readout_tone)
    in_time, correction = decode.submit([m]).result(timeout_ns=350)
    sync(drive_line, readout_line)
    if in_time:
        if correction[0]:
            drive(drive_line, square(40, PI_AMP), drive_tone)
    else:
        if m:
            drive(drive_line, square(40, PI_AMP), drive_tone)


# ── 2. Adaptive Bayesian T1 tracking ────────────────────────────────────────

# The next wait is C_WAIT times the current T1 estimate, within the bounds.
C_WAIT = 1.0
WAIT_BOUNDS_VTU = (200, 40000)  # 0.1 us to 20 us at 2 GHz
CLOCK_GHZ = 2.0


@pulse.classical(
    args=(Bits(1), Float(), Float(), Float()),
    results=(Float(), Float(), Float()),
    max_latency_ns=200,
    link=FPGA_LINK,
)
def update_t1(survived, alpha, beta, last_wait):
    """One Bayesian update of a gamma posterior over the decay rate.

    The posterior over Gamma = 1/T1 is Gamma(alpha, beta), with mean
    alpha / beta. Simplified illustration of the paper's update: a qubit
    still excited after the wait multiplies the posterior by exp(-Gamma t),
    which adds t to beta exactly; a decay is treated as an observed event,
    adding one to alpha. The next wait is C_WAIT times the estimated T1.
    """
    t_ns = last_wait / CLOCK_GHZ
    if survived[0]:
        beta = beta + t_ns
    else:
        alpha = alpha + 1.0
    t1_estimate_ns = beta / alpha
    lo, hi = WAIT_BOUNDS_VTU
    next_wait = min(max(C_WAIT * t1_estimate_ns * CLOCK_GHZ, lo), hi)
    return alpha, beta, next_wait


@pulse.kernel
def track_t1(qubit, rounds: int, alpha: float, beta: float, first_wait: int):
    """Excite, wait, measure, update the estimate, reset; repeat."""
    drive_line, drive_tone = get_drive_line(qubit)
    readout_line, readout_tone = get_readout_line(qubit)

    drive(drive_line, square(40, PI_AMP), drive_tone)
    wait(drive_line, first_wait)
    tau = float(first_wait)
    for _ in range(rounds):
        sync(drive_line, readout_line)
        m = readout(readout_line, square(400, 0.05), readout_tone)
        alpha, beta, tau = update_t1.submit([m], alpha, beta, tau).result()
        sync(drive_line, readout_line)
        if m:  # active reset
            drive(drive_line, square(40, PI_AMP), drive_tone)
        drive(drive_line, square(40, PI_AMP), drive_tone)
        wait(drive_line, tau, bounds=WAIT_BOUNDS_VTU)
    sync(drive_line, readout_line)
    readout(readout_line, square(400, 0.05), readout_tone)


# ── 3. Report, simulation and emulation export ──────────────────────────────


def print_report(lowered):
    """Each call's issue time, the time its answer is due, and its slack
    (how much later the answer could come without delaying a pulse)."""
    clock = lowered.target.clock_ghz
    print(f"  {'function':<10} {'link':>4} {'issue ns':>9} {'ready ns':>9} {'slack ns':>9}")
    for call in lowered.classical_report():
        slack = "-" if call.slack_vtu is None else f"{call.slack_vtu / clock:.1f}"
        print(
            f"  {call.function:<10} {call.link:>4} {call.submit_vtu / clock:>9.1f} "
            f"{call.due_vtu / clock:>9.1f} {slack:>9}"
        )


def export(lowered, name, directory):
    artifact = lowered.export_emulation()
    path = Path(directory) / f"{name}.json"
    path.write_text(json.dumps(artifact, indent=1))
    print(
        f"  {artifact['format']}: events {len(artifact['events'])}, "
        f"epochs {len(artifact['epochs'])}, calls {len(artifact['calls'])} "
        f"({path.stat().st_size} bytes)"
    )
    for entry in artifact["manifest"]["functions"]:
        print(
            f"    {entry['name']}: id 0x{entry['function_id']:08x} on link {entry['link']}, "
            f"request {entry['request_bytes']} B, answer {entry['answer_bytes']} B"
        )


if __name__ == "__main__":
    # A single transmon with T1 = 10 us, for simulation.
    target = TargetDescription(
        name="classical-demo",
        qubits={
            0: Qubit(index=0, frequency_hz=5e9, anharmonicity_hz=-2e8, t1_us=10.0, t2_star_us=0.0)
        },
    )
    dev = pulse.device({"target": target})
    q = dev.qudit_handles[0]
    # A 2 ns step resolves these 20 ns pulses on one uncoupled qubit.
    dt_ns = 2.0

    print("=== 1. Decoding with feed-forward ===")
    decoded = pulse.compile(decoded_feed_forward, device=dev, machine=MACHINE).lower(q)
    print_report(decoded)
    # Branch on whether the answer is in time, with the decoder's latency
    # model (triangular over 200-400 ns, mode 300 ns): P(L <= 350) = 0.875.
    result = pulse.simulate(
        decoded, initial_state=[2**-0.5, 2**-0.5], latency_branching=True, dt_ns=dt_ns
    )
    in_time = sum(p for calls, (_, p) in zip(result.calls, result.branches) if calls[0].in_time)
    print(f"  answer in time with p = {in_time:.3f}")
    # Short of 1: the qubit can relax while the answer is in flight, and the
    # correction then excites it.
    print(f"  final |0> population: {result.density_matrix[0, 0].real:.4f}")

    print("\n=== 2. Adaptive T1 tracking (3 rounds, true T1 = 10 us) ===")
    rounds, alpha0, beta0, first_wait = 3, 1.0, 5000.0, 10000  # prior T1 = 5 us
    tracker = pulse.compile(track_t1, device=dev, machine=MACHINE).lower(
        q, rounds, alpha0, beta0, first_wait
    )
    # The static schedule places each variable wait at its lower bound; every
    # epoch after one shifts by the wait's actual length.
    print_report(tracker)
    epochs = tracker.epochs()
    print(f"  {len(epochs)} epochs, opened by: {[e.opened_by for e in epochs]}")
    result = pulse.simulate(tracker, dt_ns=dt_ns)
    print(f"  {len(result.branches)} measurement paths")
    best = max(zip(result.calls, result.branches), key=lambda pair: pair[1][1])
    print(f"  most likely path {best[1][0]} (p = {best[1][1]:.3f}):")
    for call in best[0]:
        alpha, beta, tau = call.answer
        print(
            f"    T1 estimate {beta / alpha / 1e3:5.2f} us, next wait {tau / CLOCK_GHZ / 1e3:5.2f} us"
        )
    mean_t1 = sum(
        p * (calls[-1].answer[1] / calls[-1].answer[0])
        for calls, (_, p) in zip(result.calls, result.branches)
    )
    print(f"  T1 estimate after {rounds} rounds, averaged over paths: {mean_t1 / 1e3:.2f} us")

    print("\n=== 3. Emulation artifact and call manifest ===")
    with tempfile.TemporaryDirectory() as directory:
        export(decoded, "decoded_feed_forward", directory)
        export(tracker, "track_t1", directory)
    # Run the decoder program in virtual time against its Python body, with
    # 120 ns fixed delay and 40 ns mean jitter each way on the link.
    run = emulation.emulate(
        decoded,
        shots=200,
        transport=emulation.Transport(fixed_ns=120, jitter_ns=40),
        service_time="none",
        seed=1,
    )
    print(f"  emulated 200 shots: {run.late_fraction:.1%} of answers late")
