#pragma once

#include <cstdint>

#include "tsla_lob/lobster.hpp"

namespace tsla_lob {

enum class TransitionStatus : std::uint8_t {
  exact,
  depth_censored,
  mismatch,
  unsupported,
};

[[nodiscard]] TransitionStatus audit_transition(
    const BookSnapshot& previous,
    const Message& message,
    const BookSnapshot& current);

[[nodiscard]] bool valid_snapshot(const BookSnapshot& snapshot) noexcept;

}  // namespace tsla_lob
