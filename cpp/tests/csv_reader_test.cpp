#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include "tsla_lob/analysis_policy.hpp"
#include "tsla_lob/csv_reader.hpp"
#include "tsla_lob/replay.hpp"
#include "tsla_lob/session_integrity.hpp"

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

tsla_lob::BookSnapshot book(
    std::int32_t ask1,
    std::int32_t ask1_size,
    std::int32_t bid1,
    std::int32_t bid1_size,
    std::int32_t ask2,
    std::int32_t ask2_size,
    std::int32_t bid2,
    std::int32_t bid2_size) {
  return {
      {{{ask1, ask1_size}, {ask2, ask2_size}}},
      {{{bid1, bid1_size}, {bid2, bid2_size}}},
  };
}

tsla_lob::Message message(
    tsla_lob::EventType event_type,
    std::int8_t direction,
    std::int32_t price,
    std::int32_t size) {
  tsla_lob::Message value;
  value.event_type = event_type;
  value.direction = direction;
  value.price = price;
  value.size = size;
  return value;
}

template <typename Callback>
void expect_parse_error(Callback&& callback) {
  try {
    callback();
  } catch (const tsla_lob::ParseError&) {
    return;
  }
  throw std::runtime_error("expected ParseError");
}

template <typename Callback>
void expect_runtime_error(Callback&& callback) {
  try {
    callback();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error("expected runtime_error");
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

void test_analysis_policy(const std::filesystem::path& policy_path) {
  const tsla_lob::AnalysisPolicy policy =
      tsla_lob::load_analysis_policy(policy_path);
  tsla_lob::validate_analysis_scope(policy, "TSLA", 2019);

  CHECK(policy.expected_delivered_sessions == 252);
  CHECK(policy.expected_included_sessions == 249);
  CHECK(policy.source_exclusions.size() == 3);
  CHECK(policy.source_exclusions.contains("2019-01-09"));
  CHECK(policy.source_exclusions.contains("2019-03-08"));
  CHECK(policy.source_exclusions.contains("2019-09-18"));
  CHECK(policy.development_end == "2019-08-06");
  CHECK(policy.selection_start == "2019-08-07");
  CHECK(policy.selection_end == "2019-10-17");
  CHECK(policy.test_start == "2019-10-18");

  std::array<tsla_lob::EventRecord, 2> half_day{};
  half_day[0].message.timestamp_ns = 46'799'000'000'000ULL;
  half_day[1].message.timestamp_ns = 46'801'000'000'000ULL;
  const tsla_lob::SessionCoverage half_day_coverage =
      tsla_lob::validate_session_source(
          "2019-07-03",
          57'600'000ULL,
          half_day,
          policy);
  CHECK(half_day_coverage.scheduled_close_ns == 46'800'000'000'000ULL);
  CHECK(half_day_coverage.end_gap_ns == 1'000'000'000ULL);
  CHECK(half_day_coverage.included_events == 1);
  CHECK(half_day_coverage.events_after_scheduled_close == 1);
  CHECK(
      half_day_coverage.status ==
      tsla_lob::SessionSourceStatus::included);

  std::array<tsla_lob::EventRecord, 1> incomplete{};
  incomplete[0].message.timestamp_ns = 34'200'000'000'000ULL;
  const tsla_lob::SessionCoverage excluded =
      tsla_lob::validate_session_source(
          "2019-01-09",
          57'600'000ULL,
          incomplete,
          policy);
  CHECK(
      excluded.status ==
      tsla_lob::SessionSourceStatus::declared_source_exclusion);

  expect_runtime_error([&] {
    (void)tsla_lob::validate_session_source(
        "2019-01-02",
        57'600'000ULL,
        incomplete,
        policy);
  });
  tsla_lob::validate_analysis_universe(
      252,
      249,
      policy.source_exclusions,
      policy);
  expect_runtime_error([&] {
    tsla_lob::validate_analysis_universe(
        252,
        249,
        {"2019-01-09"},
        policy);
  });
}

void test_level_two_transition_contract() {
  const tsla_lob::BookSnapshot initial =
      book(101, 20, 99, 30, 102, 40, 98, 50);

  struct TransitionCase {
    const char* name;
    tsla_lob::BookSnapshot previous;
    tsla_lob::Message event;
    tsla_lob::BookSnapshot current;
    tsla_lob::TransitionStatus expected;
  };
  const std::array<TransitionCase, 10> cases = {{
      {
          "submission",
          initial,
          message(tsla_lob::EventType::submission, -1, 100, 10),
          book(100, 10, 99, 30, 101, 20, 98, 50),
          tsla_lob::TransitionStatus::exact,
      },
      {
          "partial cancellation",
          initial,
          message(tsla_lob::EventType::partial_cancel, -1, 101, 5),
          book(101, 15, 99, 30, 102, 40, 98, 50),
          tsla_lob::TransitionStatus::exact,
      },
      {
          "deletion",
          initial,
          message(tsla_lob::EventType::deletion, -1, 102, 10),
          book(101, 20, 99, 30, 102, 30, 98, 50),
          tsla_lob::TransitionStatus::exact,
      },
      {
          "visible execution",
          initial,
          message(tsla_lob::EventType::visible_execution, 1, 99, 10),
          book(101, 20, 99, 20, 102, 40, 98, 50),
          tsla_lob::TransitionStatus::exact,
      },
      {
          "level depletion",
          initial,
          message(tsla_lob::EventType::visible_execution, -1, 101, 20),
          book(102, 40, 99, 30, 103, 60, 98, 50),
          tsla_lob::TransitionStatus::depth_censored,
      },
      {
          "hidden execution",
          initial,
          message(tsla_lob::EventType::hidden_execution, -1, 101, 5),
          initial,
          tsla_lob::TransitionStatus::exact,
      },
      {
          "hidden execution changed the displayed book",
          initial,
          message(tsla_lob::EventType::hidden_execution, -1, 101, 5),
          book(101, 15, 99, 30, 102, 40, 98, 50),
          tsla_lob::TransitionStatus::mismatch,
      },
      {
          "depletion changed the surviving prefix",
          initial,
          message(tsla_lob::EventType::deletion, -1, 101, 20),
          book(103, 60, 99, 30, 104, 70, 98, 50),
          tsla_lob::TransitionStatus::mismatch,
      },
      {
          "unsupported direction",
          initial,
          message(tsla_lob::EventType::submission, 0, 100, 10),
          initial,
          tsla_lob::TransitionStatus::unsupported,
      },
      {
          "unsupported event type",
          initial,
          message(static_cast<tsla_lob::EventType>(8), -1, 101, 5),
          initial,
          tsla_lob::TransitionStatus::unsupported,
      },
  }};

  for (const TransitionCase& input : cases) {
    if (tsla_lob::audit_transition(
            input.previous,
            input.event,
            input.current) != input.expected) {
      throw std::runtime_error(
          "unexpected transition status for " + std::string(input.name));
    }
  }
}

void test_snapshot_invariants() {
  CHECK(tsla_lob::valid_snapshot(
      book(101, 20, 99, 30, 102, 40, 98, 50)));

  const std::array<tsla_lob::BookSnapshot, 4> invalid = {{
      book(101, 0, 99, 30, 102, 40, 98, 50),
      book(99, 20, 99, 30, 102, 40, 98, 50),
      book(102, 20, 99, 30, 101, 40, 98, 50),
      book(101, 20, 98, 30, 102, 40, 99, 50),
  }};
  for (const tsla_lob::BookSnapshot& snapshot : invalid) {
    CHECK(!tsla_lob::valid_snapshot(snapshot));
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 3) {
      throw std::runtime_error(
          "expected fixture directory and analysis policy arguments");
    }
    test_paired_files(argv[1]);
    test_rejected_inputs();
    test_analysis_policy(argv[2]);
    test_level_two_transition_contract();
    test_snapshot_invariants();
    std::cout
        << "all C++ decoding, replay, and source-integrity tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
