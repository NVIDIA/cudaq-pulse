// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The photons in a readout resonator while a readout plays and after, for
// the readout crosstalk terms of the PulseToQOp lowering.

#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace cudaq_pulse {

/// A dispersive readout resonator, as the quantum system describes it.
struct ReadoutResonator {
  /// Its frequency in Hz, unless it is at the readout tone (`atTone`),
  /// which then sits midway between the pulled frequencies. The lowering
  /// places a resonator without a stated frequency at the tone its readout
  /// line started the program at.
  double frequencyHz = 0.0;
  bool atTone = true;
  /// The dispersive shift chi / 2 pi in Hz: level k pulls it by (2k - 1) chi.
  double dispersiveShiftHz = 0.0;
  /// Its linewidth kappa / 2 pi in Hz.
  double linewidthHz = 0.0;
  /// The photons it holds, driven midway at an envelope of magnitude 1.
  double photonNumber = 0.0;
};

/// The mean photon number in each step of `stepNs` while the qudit of
/// `resonator` is in `level`: the resonator, empty at the start, is driven
/// at `toneHz` by `drive`, the complex envelope over each step scaled so
/// that 1 holds the resonator's photon number midway, and rings down after
/// the drive ends until it holds less than `floor` times the most photons
/// it held. The field follows
///
///   d alpha / dt = -(kappa / 2 - i Delta) alpha + epsilon u(t),
///   Delta = 2 pi (toneHz - f_r - (2 level - 1) chi),
///
/// solved exactly over each step, and each mean is the exact integral of
/// |alpha|^2 over its step divided by the step.
std::vector<double>
readoutPhotons(const ReadoutResonator &resonator, int level, double toneHz,
               const std::vector<std::complex<double>> &drive, double stepNs,
               double floor = 1.0e-6);

/// The most steps of `stepNs` that `readoutPhotons` adds after the drive
/// for the resonator to ring down.
size_t readoutRingDownSteps(const ReadoutResonator &resonator, double stepNs);

} // namespace cudaq_pulse
