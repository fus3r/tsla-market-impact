#include <cstdint>

#include "tsla_lob/message.hpp"

namespace {

constexpr std::uint8_t code(tsla_lob::EventType event_type) {
  return static_cast<std::uint8_t>(event_type);
}

static_assert(code(tsla_lob::EventType::submission) == 1);
static_assert(code(tsla_lob::EventType::partial_cancel) == 2);
static_assert(code(tsla_lob::EventType::deletion) == 3);
static_assert(code(tsla_lob::EventType::visible_execution) == 4);
static_assert(code(tsla_lob::EventType::hidden_execution) == 5);
static_assert(code(tsla_lob::EventType::cross_trade) == 6);
static_assert(code(tsla_lob::EventType::trading_halt) == 7);

}  // namespace

int main() {
  const tsla_lob::Message message{
      .timestamp_ns = 43'200'000'000'001,
      .order_id = 42,
      .size = 100,
      .price = 1'234'500,
      .event_type = tsla_lob::EventType::partial_cancel,
      .direction = -1,
  };

  if (message.timestamp_ns != 43'200'000'000'001 || message.order_id != 42 ||
      message.size != 100 || message.price != 1'234'500 ||
      message.event_type != tsla_lob::EventType::partial_cancel ||
      message.direction != -1) {
    return 1;
  }

  return 0;
}
