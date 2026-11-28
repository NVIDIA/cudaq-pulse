// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "ReadoutPhotons.h"

#include <algorithm>
#include <cmath>

namespace cudaq_pulse {

namespace {

// The resonator ringing down after the drive stops for at most this many
// time constants, whatever `floor` asks.
constexpr double kMaxRingDown = 40.0;

} // namespace

std::vector<double>
readoutPhotons(const ReadoutResonator &resonator, int level, double toneHz,
               const std::vector<std::complex<double>> &drive, double stepNs,
               double floor) {
  using complex = std::complex<double>;
  std::vector<double> means;
  const double kappa = 2.0 * M_PI * resonator.linewidthHz * 1.0e-9;
  if (!(kappa > 0.0) || !(stepNs > 0.0) || drive.empty())
    return means;
  const double chi = resonator.dispersiveShiftHz;
  const double resonance = resonator.atTone ? toneHz : resonator.frequencyHz;
  const double pull = (2.0 * level - 1.0) * chi;
  const double delta = 2.0 * M_PI * (toneHz - resonance - pull) * 1.0e-9;
  // Midway, an envelope of 1 holds photonNumber photons.
  const double epsilon =
      std::sqrt(resonator.photonNumber) *
      std::abs(complex(kappa / 2.0, 2.0 * M_PI * chi * 1.0e-9));
  const complex lambda(kappa / 2.0, -delta);
  const complex decay = std::exp(-lambda * stepNs);
  const double power = std::exp(-kappa * stepNs);
  // The integral over one step of e^{-lambda t} and of e^{-kappa t}.
  const complex fieldIntegral = (1.0 - decay) / lambda;
  const double powerIntegral = (1.0 - power) / kappa;

  complex alpha = 0.0;
  double most = 0.0;
  // One step at a constant drive u: alpha(t) = a + (alpha_0 - a) e^{-lambda t}
  // with a = epsilon u / lambda, and the integral of |alpha|^2 is exact.
  auto step = [&](complex u) {
    const complex settled = epsilon * u / lambda;
    const complex rest = alpha - settled;
    const double integral =
        std::norm(settled) * stepNs +
        2.0 * std::real(std::conj(settled) * rest * fieldIntegral) +
        std::norm(rest) * powerIntegral;
    alpha = settled + rest * decay;
    const double mean = std::max(0.0, integral / stepNs);
    most = std::max(most, mean);
    means.push_back(mean);
  };
  for (complex u : drive)
    step(u);
  const size_t tail = readoutRingDownSteps(resonator, stepNs);
  for (size_t i = 0; i < tail && std::norm(alpha) > floor * most; ++i)
    step(0.0);
  return means;
}

size_t readoutRingDownSteps(const ReadoutResonator &resonator, double stepNs) {
  const double kappa = 2.0 * M_PI * resonator.linewidthHz * 1.0e-9;
  if (!(kappa > 0.0) || !(stepNs > 0.0))
    return 0;
  return static_cast<size_t>(std::ceil(kMaxRingDown / kappa / stepNs));
}

} // namespace cudaq_pulse
