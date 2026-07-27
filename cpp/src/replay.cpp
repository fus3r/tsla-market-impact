#include "tsla_lob/replay.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tsla_lob {
namespace {

using Side = std::array<Level, kDepth>;
static_assert(kDepth == 2);

bool better_price(std::int32_t left, std::int32_t right, bool bids) {
  return bids ? left > right : left < right;
}

TransitionStatus audit_submission(
    const Side& previous_side,
    const Side& previous_other,
    const Side& current_side,
    const Side& current_other,
    const Message& message,
    bool bids) {
  if (message.size <= 0 || previous_other != current_other) {
    return TransitionStatus::mismatch;
  }

  Side predicted = previous_side;
  const auto existing = std::find_if(
      predicted.begin(),
      predicted.end(),
      [&message](const Level& level) { return level.price == message.price; });
  if (existing != predicted.end()) {
    const std::int64_t updated =
        static_cast<std::int64_t>(existing->size) + message.size;
    if (updated > std::numeric_limits<std::int32_t>::max()) {
      return TransitionStatus::mismatch;
    }
    existing->size = static_cast<std::int32_t>(updated);
  } else {
    std::array<Level, kDepth + 1> candidates{};
    std::copy(previous_side.begin(), previous_side.end(), candidates.begin());
    candidates.back() = {message.price, message.size};
    std::sort(
        candidates.begin(),
        candidates.end(),
        [bids](const Level& left, const Level& right) {
          return better_price(left.price, right.price, bids);
        });
    std::copy_n(candidates.begin(), kDepth, predicted.begin());
  }

  return predicted == current_side ? TransitionStatus::exact
                                   : TransitionStatus::mismatch;
}

TransitionStatus audit_removal(
    const Side& previous_side,
    const Side& previous_other,
    const Side& current_side,
    const Side& current_other,
    const Message& message) {
  if (message.size <= 0 || previous_other != current_other) {
    return TransitionStatus::mismatch;
  }

  const auto existing = std::find_if(
      previous_side.begin(),
      previous_side.end(),
      [&message](const Level& level) { return level.price == message.price; });
  if (existing == previous_side.end()) {
    return previous_side == current_side ? TransitionStatus::exact
                                         : TransitionStatus::mismatch;
  }

  const std::size_t index =
      static_cast<std::size_t>(existing - previous_side.begin());
  const std::int64_t remaining =
      static_cast<std::int64_t>(existing->size) - message.size;
  if (remaining < 0) {
    return TransitionStatus::mismatch;
  }
  if (remaining > 0) {
    Side predicted = previous_side;
    predicted[index].size = static_cast<std::int32_t>(remaining);
    return predicted == current_side ? TransitionStatus::exact
                                     : TransitionStatus::mismatch;
  }

  // A depleted level exposes a tail that was outside the requested depth.
  if (index == 0 && current_side[0] != previous_side[1]) {
    return TransitionStatus::mismatch;
  }
  if (index == 1 && current_side[0] != previous_side[0]) {
    return TransitionStatus::mismatch;
  }
  return TransitionStatus::depth_censored;
}

}  // namespace

TransitionStatus audit_transition(
    const BookSnapshot& previous,
    const Message& message,
    const BookSnapshot& current) {
  if (message.direction != -1 && message.direction != 1) {
    return TransitionStatus::unsupported;
  }

  const EventType type = message.event_type;
  if (type == EventType::hidden_execution ||
      type == EventType::cross_trade ||
      type == EventType::trading_halt) {
    return previous == current ? TransitionStatus::exact
                               : TransitionStatus::mismatch;
  }

  const bool bids = message.direction == 1;
  const Side& previous_side = bids ? previous.bids : previous.asks;
  const Side& previous_other = bids ? previous.asks : previous.bids;
  const Side& current_side = bids ? current.bids : current.asks;
  const Side& current_other = bids ? current.asks : current.bids;

  if (type == EventType::submission) {
    return audit_submission(
        previous_side,
        previous_other,
        current_side,
        current_other,
        message,
        bids);
  }
  if (type == EventType::partial_cancel ||
      type == EventType::deletion ||
      type == EventType::visible_execution) {
    return audit_removal(
        previous_side,
        previous_other,
        current_side,
        current_other,
        message);
  }
  return TransitionStatus::unsupported;
}

bool valid_snapshot(const BookSnapshot& snapshot) noexcept {
  for (std::size_t level = 0; level < kDepth; ++level) {
    if (snapshot.asks[level].price <= 0 ||
        snapshot.asks[level].size <= 0 ||
        snapshot.bids[level].price <= 0 ||
        snapshot.bids[level].size <= 0) {
      return false;
    }
  }
  if (snapshot.bids[0].price >= snapshot.asks[0].price) {
    return false;
  }
  for (std::size_t level = 1; level < kDepth; ++level) {
    if (snapshot.asks[level - 1].price >= snapshot.asks[level].price ||
        snapshot.bids[level - 1].price <= snapshot.bids[level].price) {
      return false;
    }
  }
  return true;
}

}  // namespace tsla_lob
