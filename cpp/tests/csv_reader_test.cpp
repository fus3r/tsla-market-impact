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
    std::cout << "all C++ decoding and source-integrity tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
