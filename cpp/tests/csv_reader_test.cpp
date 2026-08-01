#include <array>
#include <cmath>
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

void test_order_flow_pressure_is_causal_and_bounded() {
  const auto previous =
      book(101, 20, 99, 30, 102, 40, 98, 50);
  const auto current =
      book(101, 15, 99, 40, 102, 40, 98, 50);

  CHECK(tsla_lob::top_of_book_ofi(previous, current) == 15);
  CHECK(
      std::abs(
          tsla_lob::level_one_queue_imbalance(previous) - 0.2) <
      1e-12);
  CHECK(
      std::abs(
          tsla_lob::bounded_order_flow_pressure(50, previous) -
          0.5) <
      1e-12);
  CHECK(
      std::abs(
          tsla_lob::bounded_order_flow_pressure(-50, previous) +
          0.5) <
      1e-12);
  CHECK(
      tsla_lob::bounded_order_flow_pressure(
          5'000'000,
          previous) < 1.0);
  CHECK(
      tsla_lob::bounded_order_flow_pressure(
          -5'000'000,
          previous) > -1.0);
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

void test_replay_aggregation() {
  const tsla_lob::BookSnapshot initial =
      book(101, 20, 99, 30, 102, 40, 98, 50);
  const tsla_lob::BookSnapshot after_cancel =
      book(101, 15, 99, 30, 102, 40, 98, 50);
  const tsla_lob::BookSnapshot after_depletion =
      book(102, 40, 99, 30, 103, 60, 98, 50);
  const tsla_lob::BookSnapshot observable_mismatch =
      book(102, 39, 99, 30, 103, 60, 98, 50);

  tsla_lob::Dataset dataset;
  dataset.delivered_sessions = 3;
  dataset.declared_source_exclusions = 1;
  dataset.events = 6;
  dataset.sessions = {
      {
          "2019-01-02",
          {
              {message(tsla_lob::EventType::submission, -1, 101, 20), initial},
              {message(tsla_lob::EventType::partial_cancel, -1, 101, 5),
               after_cancel},
              {message(tsla_lob::EventType::visible_execution, -1, 101, 15),
               after_depletion},
              {message(tsla_lob::EventType::hidden_execution, -1, 102, 1),
               observable_mismatch},
              {message(tsla_lob::EventType::submission, 0, 102, 1),
               observable_mismatch},
          },
      },
      {
          "2019-01-03",
          {
              {message(tsla_lob::EventType::cross_trade, -1, 0, 1),
               book(101, 0, 99, 30, 102, 40, 98, 50)},
          },
      },
  };

  const tsla_lob::ReplayMetrics metrics =
      tsla_lob::replay_dataset(dataset);
  const tsla_lob::ReplayMetrics repeated =
      tsla_lob::replay_dataset(dataset);
  CHECK(metrics.events == 6);
  CHECK(metrics.seeded_sessions == 2);
  CHECK(metrics.exact_transitions == 1);
  CHECK(metrics.depth_censored_transitions == 1);
  CHECK(metrics.mismatches == 1);
  CHECK(metrics.unsupported == 1);
  CHECK(metrics.invalid_snapshots == 1);
  CHECK(metrics.events_by_type[1] == 2);
  CHECK(metrics.events_by_type[2] == 1);
  CHECK(metrics.events_by_type[4] == 1);
  CHECK(metrics.events_by_type[5] == 1);
  CHECK(metrics.events_by_type[6] == 1);
  CHECK(metrics.checksum != 0);
  CHECK(repeated.checksum == metrics.checksum);

  tsla_lob::Dataset changed = dataset;
  changed.sessions[0].events[0].message.order_id = 1;
  CHECK(
      tsla_lob::replay_dataset(changed).checksum !=
      metrics.checksum);

  const std::string expected_json =
      "{\n"
      "  \"dataset\": {\n"
      "    \"delivered_sessions\": 3,\n"
      "    \"included_sessions\": 2,\n"
      "    \"declared_source_exclusions\": 1,\n"
      "    \"events\": 6\n"
      "  },\n"
      "  \"audit\": {\n"
      "    \"seeded_sessions\": 2,\n"
      "    \"post_seed_transitions\": 4,\n"
      "    \"exact_transitions\": 1,\n"
      "    \"depth_censored_transitions\": 1,\n"
      "    \"mismatches\": 1,\n"
      "    \"unsupported\": 1,\n"
      "    \"invalid_snapshots\": 1,\n"
      "    \"events_by_type\": {\n"
      "      \"1\": 2,\n"
      "      \"2\": 1,\n"
      "      \"3\": 0,\n"
      "      \"4\": 1,\n"
      "      \"5\": 1,\n"
      "      \"6\": 1,\n"
      "      \"7\": 0\n"
      "    }\n"
      "  }\n"
      "}\n";
  CHECK(tsla_lob::replay_audit_json(dataset, metrics) == expected_json);

  dataset.input_bytes = 1'024;
  const tsla_lob::TimingSummary decode =
      tsla_lob::summarize_timings({2.0, 1.0, 3.0}, 1'000'000'000);
  const tsla_lob::TimingSummary replay =
      tsla_lob::summarize_timings({5.0, 4.0, 6.0}, 1'000'000'000);
  CHECK(decode.seconds == std::vector<double>({2.0, 1.0, 3.0}));
  CHECK(decode.median_seconds == 2.0);
  CHECK(decode.p95_seconds == 3.0);
  CHECK(decode.median_ns_per_event == 2.0);
  CHECK(decode.p95_ns_per_event == 3.0);
  CHECK(decode.median_million_events_per_second == 500.0);

  const std::string benchmark = tsla_lob::benchmark_json(
      dataset,
      metrics,
      decode,
      replay,
      3,
      3,
      2,
      "fixture \"host\"");
  CHECK(
      benchmark.find("\"resident_event_bytes\": 384") !=
      std::string::npos);
  CHECK(
      benchmark.find("\"machine\": \"fixture \\\"host\\\"\"") !=
      std::string::npos);
  CHECK(
      benchmark.find("\"thread_count\": 1") !=
      std::string::npos);
  CHECK(
      benchmark.find("\"seconds\": [2, 1, 3]") !=
      std::string::npos);
  CHECK(
      benchmark.find(
          "\"filesystem_cache\": "
          "\"not flushed between decode runs\"") !=
      std::string::npos);
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
    test_order_flow_pressure_is_causal_and_bounded();
    test_snapshot_invariants();
    test_replay_aggregation();
    std::cout
        << "all C++ decoding, replay, and source-integrity tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
