#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Adiabatic sweep on a 1D Rydberg chain.

Demonstrates:
  - Creating a Rydberg target with a 1D chain geometry
  - Computing blockade radius and interaction strengths
  - Building a pulse program for global adiabatic detuning sweep
  - Compiling via ``pulse.compile()``

The adiabatic protocol starts with large negative detuning (all atoms in |g>),
ramps through resonance, and ends at large positive detuning to prepare a
many-body ordered state (Z2 antiferromagnet for appropriate spacing).
"""

import cudaq_pulse as pulse
from cudaq_pulse.targets import rydberg_chain

N_ATOMS = 7
SPACING_UM = 6.0

chain = rydberg_chain(N_ATOMS, spacing_um=SPACING_UM, global_rabi_mhz=4.0)
target = chain.to_target()

print(f"Rydberg chain: {chain.n_atoms} atoms, spacing = {SPACING_UM} um")
print(f"Blockade radius: {chain.blockade_radius():.2f} um")
print(f"Nearest-neighbor V = {chain.interaction_strength(chain.atoms[0], chain.atoms[1]):.2f} MHz")
print(f"Next-nearest V = {chain.interaction_strength(chain.atoms[0], chain.atoms[2]):.4f} MHz")

ham_terms = chain.hamiltonian_terms()
diss_terms = chain.dissipator_terms()
print(f"\nHamiltonian: {len(ham_terms)} terms")
for term in ham_terms[:5]:
    print(
        f"  {term['kind']:25s} qubits={term['qubit_indices']} coeff={term['coefficient'].real:.4e}"
    )
if len(ham_terms) > 5:
    print(f"  ... ({len(ham_terms) - 5} more)")
print(f"Dissipators: {len(diss_terms)} terms")

RAMP_STEPS = 20
STEP_DURATION = 50
amplitude = chain.global_rabi_mhz / 10.0
# The detuning ramps linearly from -DETUNING_HZ to +DETUNING_HZ, in steps.
DETUNING_HZ = 2.5 * chain.global_rabi_mhz * 1e6
DETUNING_STEP_HZ = 2 * DETUNING_HZ / (RAMP_STEPS - 1)


@pulse.kernel
def rydberg_adiabatic(atoms):
    """Drive every atom of *atoms* while the detuning sweeps through resonance."""
    lines = [get_drive_line(atom) for atom in atoms]

    for _, tone in lines:
        shift_frequency(tone, -DETUNING_HZ)

    for step in range(RAMP_STEPS):
        for line, tone in lines:
            drive(line, square(STEP_DURATION, complex(amplitude, 0)), tone)
        if step < RAMP_STEPS - 1:
            sync(*[line for line, _ in lines])
            for _, tone in lines:
                shift_frequency(tone, DETUNING_STEP_HZ)


dev = pulse.device({"target": f"rydberg_chain:{N_ATOMS}"})
compiled_kernel = pulse.compile(rydberg_adiabatic, device=dev).lower(dev.qudit_handles[:N_ATOMS])

print(
    f"\nSweep: {RAMP_STEPS} steps of {STEP_DURATION} clock cycles, detuning "
    f"{-DETUNING_HZ / 1e6:+.1f} MHz to {DETUNING_HZ / 1e6:+.1f} MHz"
)
print(compiled_kernel.mlir())

metrics = compiled_kernel.metrics
print("\n=== Compile metrics ===")
print(f"  Trace   : {metrics.trace_ms:.3f} ms")
print(f"  Passes  : {metrics.passes_ms:.3f} ms")
print(f"  Schedule: {metrics.schedule_ms:.3f} ms")
print(f"  Total   : {metrics.total_ms:.3f} ms")
