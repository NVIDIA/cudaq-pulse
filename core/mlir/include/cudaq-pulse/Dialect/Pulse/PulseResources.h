// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The side-effect resources of the Pulse dialect. Lines and tones are linear:
// an op that plays on or changes one reads the version it takes and writes the
// version it returns, so its effects are on those values, not on every line
// or tone at once.

#pragma once

#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace pulse {

/// Where a line's time stands.
struct LineResource : public mlir::SideEffects::Resource::Base<LineResource> {
  llvm::StringRef getName() final { return "pulse.line"; }
};

/// A tone's frame: its frequency, phase and time.
struct ToneResource : public mlir::SideEffects::Resource::Base<ToneResource> {
  llvm::StringRef getName() final { return "pulse.tone"; }
};

} // namespace pulse
