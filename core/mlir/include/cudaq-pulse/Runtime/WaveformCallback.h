/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// Drive-envelope callbacks of the cuDensityMat lowering, without CUDA, so
// that the GPU runtime and the CPU reference simulator evaluate the same
// coefficients.
//
// Parameters: start (ns), duration (ns), real and imaginary amplitude,
// sigma (ns), beta, rise/fall (ns), envelope phase at start, quadrature
// (0 for the X term, 1 for Y), envelope rotation rate (rad/ns: w_q - w_tone
// for a drive), then for a pulse train (kind has
// kTrainFlag) the number of levels L and (period ns, count, phase step) per
// level, outermost first, then any samples. A train plays the pulse at
// start + sum_l k_l * period_l with phase advanced by sum_l k_l * step_l.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cudaq_pulse {

constexpr int32_t kTrainFlag = 0x100;
constexpr size_t kWaveformHeader = 10;

/// The X or Y coefficient at `time`; returns nonzero on bad parameters.
inline int32_t evaluateWaveform(int32_t kind, const double *p, size_t size,
                                double time, double &value) {
  value = 0.0;
  if (size < kWaveformHeader)
    return 2;
  double localTime = time - p[0];
  double phase = p[7];
  size_t samples = kWaveformHeader;
  if (kind & kTrainFlag) {
    if (size < kWaveformHeader + 1)
      return 2;
    const auto levels = static_cast<size_t>(p[kWaveformHeader]);
    samples = kWaveformHeader + 1 + 3 * levels;
    if (size < samples)
      return 2;
    for (size_t level = 0; level < levels; ++level) {
      const double period = p[kWaveformHeader + 1 + 3 * level];
      const double count = p[kWaveformHeader + 2 + 3 * level];
      const double step = p[kWaveformHeader + 3 + 3 * level];
      if (period <= 0.0 || count < 1.0 || localTime < 0.0)
        continue;
      const double k = std::min(std::floor(localTime / period), count - 1.0);
      localTime -= k * period;
      phase += k * step;
    }
    kind &= ~kTrainFlag;
  }
  const double duration = p[1];
  double real = 0.0;
  double imaginary = 0.0;
  if (localTime >= 0.0 && localTime < duration && duration > 0.0) {
    const double centered = localTime - duration / 2.0;
    switch (kind) {
    case 1: // square
      real = p[2];
      imaginary = p[3];
      break;
    case 2: { // Gaussian
      const double gaussian =
          std::exp(-0.5 * centered * centered / (p[4] * p[4]));
      real = p[2] * gaussian;
      break;
    }
    case 3: { // DRAG: Gaussian plus beta times its derivative.
      const double gaussian =
          std::exp(-0.5 * centered * centered / (p[4] * p[4]));
      real = p[2] * gaussian;
      imaginary = -p[5] * centered * real / (p[4] * p[4]);
      break;
    }
    case 4: // raised cosine
      real = p[2] * 0.5 * (1.0 - std::cos(2.0 * M_PI * localTime / duration));
      break;
    case 5: // smooth tanh ramp from zero to the requested amplitude
      real =
          p[2] * 0.5 * (1.0 + std::tanh((localTime - duration / 2.0) / p[4]));
      break;
    case 6: { // Gaussian-square
      const double edge = p[6];
      if (localTime < edge) {
        const double x = localTime - edge;
        real = p[2] * std::exp(-0.5 * x * x / (p[4] * p[4]));
      } else if (localTime >= duration - edge) {
        const double x = localTime - (duration - edge);
        real = p[2] * std::exp(-0.5 * x * x / (p[4] * p[4]));
      } else {
        real = p[2];
      }
      break;
    }
    case 7: { // custom samples, held piecewise constant over the duration
      const size_t sampleCount = size - samples;
      if (sampleCount > 0) {
        const auto index =
            std::min(sampleCount - 1,
                     static_cast<size_t>(localTime * sampleCount / duration));
        real = p[samples + index];
      }
      break;
    }
    default:
      return 3;
    }
  }

  const double angle = phase + p[9] * localTime;
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  const double rotatedReal = real * cosine - imaginary * sine;
  const double rotatedImaginary = real * sine + imaginary * cosine;
  // H_control = (Re(envelope) X + Im(envelope) Y) / 2.
  value = 0.5 * (p[8] == 0.0 ? rotatedReal : rotatedImaginary);
  return 0;
}

} // namespace cudaq_pulse
