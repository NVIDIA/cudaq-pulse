# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Runtime qudits: one compile for every qudit of a device.

A kernel parameter annotated ``pulse.Qudit`` is bound at each call, so the
kernel compiles once and serves any qudit, and a recalibrated device, without compiling again.
Frequencies the kernel needs beyond a qudit's own drive frequency are
ordinary float parameters: the host reads them from the device and passes
them in, and each call binds them.
"""

import cudaq_pulse as pulse


@pulse.kernel
def excite_to_two(qudit: pulse.Qudit, f12_hz: float):
    """Drive 0-1 on the qudit's own tone, then 1-2 at *f12_hz*, and read out."""
    drive_line = get_drive_line(qudit)
    drive(drive_line, gaussian(40, 0.25, 10.0))
    set_frequency(drive_line.tone, f12_hz)
    drive(drive_line, gaussian(40, 0.18, 10.0))
    readout_line = get_readout_line(qudit)
    sync(drive_line, readout_line)
    return readout(readout_line, square(400, 0.05))


if __name__ == "__main__":
    dev = pulse.device("transmon_17q")

    for i in range(4):
        f12_hz = float(dev.transition_freq[i, 1, 2])
        lowered = excite_to_two.lower(dev.qudit_handles[i], f12_hz)
        drives = [e for e in lowered.export_controller()["events"] if e["kind"] == "drive"]
        tones = ", ".join(f"{e['frequency_hz'] / 1e9:.4f}" for e in drives)
        print(f"qudit {i}: drive tones {tones} GHz")

    print("runtime parameters:", excite_to_two.runtime_parameters)
    print("compiles:", excite_to_two.cache_info().compiled)
