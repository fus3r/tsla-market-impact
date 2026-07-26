#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "tsla_lob/csv_reader.hpp"

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
static_assert(
    std::is_same_v<
        decltype(tsla_lob::Message{}.timestamp_ns),
        std::uint64_t>);
static_assert(
    std::is_same_v<decltype(tsla_lob::Message{}.order_id), std::int64_t>);
static_assert(
    std::is_same_v<decltype(tsla_lob::Message{}.size), std::int32_t>);
static_assert(
    std::is_same_v<decltype(tsla_lob::Message{}.price), std::int32_t>);
static_assert(
    std::is_same_v<decltype(tsla_lob::Level{}.price), std::int32_t>);
static_assert(
    std::is_same_v<decltype(tsla_lob::Level{}.size), std::int32_t>);

void check(bool condition, const char* expression, int line) {
  if (!condition) {
    throw std::runtime_error(
        "check failed at line " + std::to_string(line) + ": " + expression);
  }
}

#define CHECK(expression) check((expression), #expression, __LINE__)

template <typename Callback>
void expect_parse_error(Callback&& callback) {
  try {
    callback();
  } catch (const tsla_lob::ParseError&) {
    return;
  }
  throw std::runtime_error("expected ParseError");
}

void test_paired_files(const std::filesystem::path& fixtures) {
  const std::vector<tsla_lob::EventRecord> events =
      tsla_lob::decode_paired_files(
          (fixtures / "synthetic_message.csv").string(),
          (fixtures / "synthetic_orderbook.csv").string());

  CHECK(events.size() == 2);
  CHECK(events[0].message.timestamp_ns == 43'200'000'000'001ULL);
  CHECK(
      events[0].message.event_type ==
      tsla_lob::EventType::partial_cancel);
  CHECK(events[0].message.order_id == 42);
  CHECK(events[0].message.size == 100);
  CHECK(events[0].message.price == 1'234'500);
  CHECK(events[0].message.direction == -1);
  CHECK(events[0].book.asks[0] == (tsla_lob::Level{1'234'600, 200}));
  CHECK(events[0].book.bids[1] == (tsla_lob::Level{1'234'300, 500}));

  CHECK(events[1].message.timestamp_ns == 43'200'000'000'002ULL);
  CHECK(
      events[1].message.event_type ==
      tsla_lob::EventType::visible_execution);
  CHECK(events[1].message.order_id == 43);
  CHECK(events[1].book.asks[0] == (tsla_lob::Level{1'234'600, 150}));
}

void test_rejected_inputs() {
  constexpr std::string_view message =
      "43200.000000001,2,42,100,1234500,-1\n";
  constexpr std::string_view book =
      "1234600,200,1234400,300,1234700,400,1234300,500\n";

  struct RejectedPair {
    std::string_view messages;
    std::string_view books;
  };
  constexpr std::array<RejectedPair, 4> rejected = {{
      {
          "43200.000000001,2,42,2147483648,1234500,-1\n",
          book,
      },
      {
          message,
          "1234600,200,1234400,300,1234700,400,1234300,oops\n",
      },
      {
          "43200.000000001,2,42,100,1234500,-1,code,extra\n",
          book,
      },
      {
          "43200.000000001,2,42,100,1234500,-1\n"
          "43200.000000002,4,43,50,1234400,1\n",
          book,
      },
  }};

  for (const RejectedPair& input : rejected) {
    expect_parse_error([&input] {
      (void)tsla_lob::decode_paired_csv(input.messages, input.books);
    });
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 2) {
      throw std::runtime_error("expected fixture directory argument");
    }
    test_paired_files(argv[1]);
    test_rejected_inputs();
    std::cout << "all C++ decoding tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
