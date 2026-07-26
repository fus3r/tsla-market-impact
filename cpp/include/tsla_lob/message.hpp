#pragma once

#include <cstdint>

namespace tsla_lob {

enum class EventType : std::uint8_t {
  submission = 1,
  partial_cancel = 2,
  deletion = 3,
  visible_execution = 4,
  hidden_execution = 5,
  cross_trade = 6,
  trading_halt = 7,
};

struct Message {
  std::uint64_t timestamp_ns{};
  std::int64_t order_id{};
  std::int32_t size{};
  std::int32_t price{};
  EventType event_type{};
  std::int8_t direction{};
  std::uint16_t reserved{};
};

static_assert(sizeof(EventType) == sizeof(std::uint8_t));
static_assert(sizeof(Message) == 32);

}  // namespace tsla_lob
