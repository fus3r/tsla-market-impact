#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "tsla_lob/message.hpp"

namespace tsla_lob {

inline constexpr std::size_t kDepth = 2;

struct Level {
  std::int32_t price{};
  std::int32_t size{};

  friend bool operator==(const Level&, const Level&) = default;
};

struct BookSnapshot {
  std::array<Level, kDepth> asks{};
  std::array<Level, kDepth> bids{};

  friend bool operator==(const BookSnapshot&, const BookSnapshot&) = default;
};

struct EventRecord {
  Message message{};
  BookSnapshot book{};
};

static_assert(sizeof(Level) == 8);
static_assert(sizeof(BookSnapshot) == 32);
static_assert(sizeof(EventRecord) == 64);

}  // namespace tsla_lob
