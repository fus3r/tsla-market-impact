#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

struct SessionFiles {
  std::string date;
  std::string message_path;
  std::string book_path;
  std::uint64_t requested_end_ms{};
};

struct SessionData {
  std::string date;
  std::vector<EventRecord> events;
};

struct Dataset {
  std::vector<SessionData> sessions;
  std::uint64_t delivered_sessions{};
  std::uint64_t declared_source_exclusions{};
  std::uint64_t input_bytes{};
  std::uint64_t events{};
};

}  // namespace tsla_lob
