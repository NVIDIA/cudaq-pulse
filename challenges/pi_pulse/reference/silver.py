# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find the pi pulse, silver: a DRAG pulse, 20 ns long, with a narrower Gaussian.

Two things are left behind by the bronze pulse, and each has its own fix:

- about 1% stays in level 0. Level 2 pushes on the 0-1 transition while the
  drive is on, so the drive is slightly off resonance. The derivative part
  of a DRAG pulse (beta) corrects it.
- about 5e-4 ends in level 2. A Gaussian with sigma = duration / 4 starts
  and ends at 14% of its peak, and that step leaks. With sigma = 4 ns the
  pulse starts at 4% of its peak.

Score it:   python challenges/pi_pulse/score.py challenges/pi_pulse/reference/silver.py
Run it by itself to repeat the scan that found the amplitude and beta.
"""

import cudaq_pulse as pulse


@pulse.kernel
def flip(qudit, amplitude: float, beta_ns: float):
    drive_line = get_drive_line(qudit)
    shaped = drag(
        duration=20 * pulse.ns,
        amplitude=amplitude,
        sigma=4 * pulse.ns,
        beta=beta_ns * pulse.ns,
    )
    drive(drive_line, shaped)


@pulse.kernel
def pi_pulse(qudit):
    # The amplitude and beta the scan below finds.
    flip(qudit, 0.319, 0.5)


if __name__ == "__main__":
    import numpy as np

    system = pulse.quantum_system("transmon_surface17")
    qubit = system.qudit("Z4")
    ground = [1, 0, 0]

    print("beta (ns)  amplitude  level 1  level 2")
    for beta_ns in np.linspace(0, 1, 11):
        best = (0.0, 0.0, 0.0)
        for amplitude in np.linspace(0.30, 0.34, 41):
            result = pulse.simulate(flip, qubit, amplitude, beta_ns, initial_state=ground)
            levels = result.level_populations(qubit)
            best = max(best, (levels[1], levels[2], amplitude))
        print(f"{beta_ns:9.1f}  {best[2]:9.3f}  {best[0]:.5f}  {best[1]:.1e}")
    print("compilations:", flip.cache_info().compiled)
