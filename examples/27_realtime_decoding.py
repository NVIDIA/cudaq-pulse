# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-time QEC decoding with a deadline and a fallback.

Builds on ``19_classical_calls.py``. One round of a distance-3 repetition
code sends its syndrome to a CUDA-Q QEC decoder, ``nv-qldpc-decoder`` on the
GPU (belief propagation with ordered-statistics post-processing, BP+OSD),
and asks for the corrections with a deadline. If the answer arrives in time,
the correction comes from the decoder; if not, from a lookup table on the
controller. The deadline is a kernel parameter, so trading waiting against
falling back needs no change to the program.

The decoder is configured in CUDA-Q QEC, as decoder 0, and the kernel calls
it with CUDA-Q QEC's own functions, ``cudaq_qec.qec.enqueue_syndromes`` and
``get_corrections``: on hardware the requests are the messages of CUDA-Q
QEC's decoding server, here with a latency budget of 62 us typical and 75 us
at most. Those budgets are round trips measured from a host process to the
decoding server on the same workstation's GPU, over UDP loopback; they are
not a quantum controller's latencies. Over NVQLink, FPGA-to-GPU round trips
take a few microseconds.

This example shows:

1. The compile-time report: when each call is due, for an 80 us deadline.
2. The configured decoder's correction for every syndrome, called from the
   host.
3. Simulation of the round with a 1 us deadline, from the state syndrome
   extraction leaves (no error, or one flipped data qubit). Against the
   62/75 us budget, nearly every shot misses the deadline and takes the
   table; with the same decoder given 500 ns, every shot uses the decoder.
   Both restore the data qubits equally, and far better than no correction.

What is simulated: the readouts and correction pulses on a five-qudit
transmon lattice, with decay. The configured decoder answers in this
process, through CUDA-Q QEC's simulation library, as it answers a CUDA-Q
kernel on a simulator. The simulation does not wait out the 80 us deadline,
because the simulator integrates idle time step by step.

Requires cudaq-qec (``pip install cudaq-qec``; Linux, Python 3.11 to 3.13)
and a CUDA-Q compatible NVIDIA GPU for ``nv-qldpc-decoder``.
"""

import cudaq_pulse as pulse
import cudaq_qec
import numpy as np

# Decoder 0: BP+OSD on the GPU for a distance-3 repetition code (two syndrome
# bits in, one correction bit per data qubit out).
cudaq_qec.configure_decoders_from_str("""
decoders:
  - id: 0
    type: nv-qldpc-decoder
    block_size: 3
    syndrome_size: 2
    H_sparse: [0, 1, -1, 1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1]
    decoder_custom_args:
      error_rate_vec: [0.1, 0.1, 0.1]
      use_sparsity: true
      max_iterations: 50
      use_osd: true
      osd_method: 3
      osd_order: 7
""")

# Its real-time calls, sent to the decoding server over the GPU link.
enqueue_syndromes = pulse.classical(cudaq_qec.qec.enqueue_syndromes, link="gpu")
get_corrections = pulse.classical(
    cudaq_qec.qec.get_corrections, link="gpu", nominal_latency_ns=62_000, max_latency_ns=75_000
)
reset_decoder = pulse.classical(cudaq_qec.qec.reset_decoder, link="gpu")


@pulse.kernel
def qec_round(data, ancillas, deadline_ns: int):
    # ... syndrome-extraction pulses on data and ancillas go here ...
    s0 = readout(get_readout_line(ancillas[0]), square(400 * pulse.ns, 0.05))
    s1 = readout(get_readout_line(ancillas[1]), square(400 * pulse.ns, 0.05))

    enqueue_syndromes.submit(0, [s0, s1], 0)  # decoder 0, round 0
    in_time, fix = get_corrections.submit(0, 3, False).result(timeout=deadline_ns * pulse.ns)

    lookup = [s0 and not s1, s0 and s1, s1 and not s0]  # local fallback table
    for i in range(3):
        if fix[i] if in_time else lookup[i]:  # decoder's answer, or the table if late
            drive(get_drive_line(data[i]), square(20 * pulse.ns, np.pi / 20))


@pulse.kernel
def no_correction(data, ancillas):
    readout(get_readout_line(ancillas[0]), square(400 * pulse.ns, 0.05))
    readout(get_readout_line(ancillas[1]), square(400 * pulse.ns, 0.05))
    for q in data:
        wait(get_drive_line(q), 1_000 * pulse.ns)


def start_state():
    """No error, or one flipped data qubit, with the ancillas holding d0^d1 and d1^d2."""
    state = np.zeros(2**5)
    for bits in [(0, 0, 0, 0, 0), (1, 0, 0, 1, 0), (0, 1, 0, 1, 1), (0, 0, 1, 0, 1)]:
        state[sum(b << k for k, b in enumerate(bits))] = 0.5
    return state


def simulate(scheduled, label):
    res = pulse.simulate(
        scheduled, initial_state=start_state(), levels=2, latency_branching=True, dt_ns=2.0
    )
    used = sum(b.probability for b in res.branches if b.calls and b.calls[-1].in_time)
    restored = sum(res.density_matrix[i, i].real for i in range(32) if i & 0b111 == 0)
    print(f"   {label:44} decoder answer used {used:.4f}   P(data back in 000) {restored:.4f}")
    return restored


if __name__ == "__main__":
    qs = pulse.quantum_system("transmon_lattice:5")
    links = pulse.ControlSystem(links={"gpu": pulse.Link(feedback_ns=50)})
    qudits = qs.qudits
    data, ancillas = qudits[0:3], qudits[3:5]

    print("1. Compile-time report, 80 us deadline")
    compiled = pulse.compile(qec_round, quantum_system=qs, control_system=links)
    scheduled = compiled.lower(data, ancillas, 80_000)
    for call in scheduled.classical_report():
        print(f"   {call.function}: due at {call.due_cycles / qs.clock_ghz / 1e3:.1f} us")

    print("2. Decoder corrections, asked from the host")
    table = {(0, 0): [0, 0, 0], (1, 0): [1, 0, 0], (1, 1): [0, 1, 0], (0, 1): [0, 0, 1]}
    for syndrome, want in table.items():
        reset_decoder(0)
        enqueue_syndromes(0, list(syndrome), 0)
        got = get_corrections(0, 3, False)
        print(f"   syndrome {syndrome} -> {got}")
        assert got == want, (syndrome, got)

    print("3. Simulation, 1 us deadline")
    late = simulate(compiled.lower(data, ancillas, 1_000), "62/75 us budget (answers late)")

    get_corrections = pulse.classical(cudaq_qec.qec.get_corrections, link="gpu", max_latency_ns=500)
    fast = pulse.compile(qec_round, quantum_system=qs, control_system=links)
    in_time = simulate(fast.lower(data, ancillas, 1_000), "500 ns budget (answers in time)")

    none = simulate(
        pulse.compile(no_correction, quantum_system=qs).lower(data, ancillas), "no correction"
    )
    assert abs(late - in_time) < 1e-3 and late > none + 0.5
    cudaq_qec.finalize_decoders()
