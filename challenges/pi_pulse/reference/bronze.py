# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find the pi pulse, bronze: the tutorial's Gaussian, 20 ns long, at the amplitude a sweep picks.

Score it:   python challenges/pi_pulse/score.py challenges/pi_pulse/reference/bronze.py
Run it by itself to repeat the sweep that found the amplitude.
"""

import cudaq_pulse as pulse


@pulse.kernel
def rabi(qudit, amplitude: float):
    drive_line = get_drive_line(qudit)
    flip = gaussian(
        duration=20 * pulse.ns,
        amplitude=amplitude,
        sigma=5 * pulse.ns,
    )
    drive(drive_line, flip)


@pulse.kernel
def pi_pulse(qudit):
    # The amplitude the sweep below finds.
    rabi(qudit, 0.26)


if __name__ == "__main__":
    import numpy as np

    system = pulse.quantum_system("transmon_surface17")
    qubit = system.qudit("Z4")
    # The scorer starts the qubit in level 0, not in its thermal mixture.
    ground = [1, 0, 0]

    amplitudes = np.linspace(0, 0.4, 41)
    populations = []
    for amplitude in amplitudes:
        result = pulse.simulate(rabi, qubit, amplitude, initial_state=ground)
        populations.append(result.level_population(1))

    best = int(np.argmax(populations))
    print(f"best amplitude: {amplitudes[best]:.2f}")
    print(f"level 1:        {populations[best]:.4f}")
