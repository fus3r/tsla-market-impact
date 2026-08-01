#include "tsla_lob/replay.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <numbers>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "tsla_lob/csv_reader.hpp"
#include "tsla_lob/session_integrity.hpp"

namespace tsla_lob {
namespace {

using Side = std::array<Level, kDepth>;
static_assert(kDepth == 2);

std::string regex_escape(const std::string& text) {
  static constexpr std::string_view special = R"(\.^$|()[]{}*+?)";
  std::string escaped;
  escaped.reserve(text.size() * 2);
  for (const char character : text) {
    if (special.find(character) != std::string_view::npos) {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2;
  if (values.size() % 2 == 0) {
    return (values[middle - 1] + values[middle]) / 2.0;
  }
  return values[middle];
}

double percentile95(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const double rank = std::ceil(0.95 * static_cast<double>(values.size()));
  const std::size_t index =
      std::min(
          values.size() - 1,
          static_cast<std::size_t>(std::max(1.0, rank)) - 1);
  return values[index];
}

std::string json_escape(const std::string& value) {
  std::ostringstream escaped;
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped << "\\\\";
        break;
      case '"':
        escaped << "\\\"";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        escaped << character;
    }
  }
  return escaped.str();
}

const char* compiler_name() {
#if defined(__clang__)
  return "clang " __clang_version__;
#elif defined(__GNUC__)
  return "gcc " __VERSION__;
#elif defined(_MSC_VER)
  return "msvc";
#else
  return "unknown";
#endif
}

const char* architecture_name() {
#if defined(__aarch64__) || defined(__arm64__) || defined(_M_ARM64)
  return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#else
  return "unknown";
#endif
}

bool same_best_quote(
    const BookSnapshot& left,
    const BookSnapshot& right) {
  return left.asks[0] == right.asks[0] &&
         left.bids[0] == right.bids[0];
}

std::int64_t midpoint_twice(const BookSnapshot& snapshot) {
  return static_cast<std::int64_t>(snapshot.asks[0].price) +
         snapshot.bids[0].price;
}

struct BinCount {
  std::uint64_t observations{};
  std::uint64_t up_moves{};
  std::uint64_t down_moves{};
};

constexpr std::array<std::string_view, 5> spread_bucket_names = {
    "all_spreads",
    "one_tick",
    "two_to_five_ticks",
    "six_to_ten_ticks",
    "over_ten_ticks",
};

std::size_t spread_bucket(const BookSnapshot& snapshot) {
  const std::int32_t spread =
      snapshot.asks[0].price - snapshot.bids[0].price;
  if (spread <= 100) {
    return 1;
  }
  if (spread <= 500) {
    return 2;
  }
  if (spread <= 1'000) {
    return 3;
  }
  return 4;
}

std::size_t imbalance_bin(
    const BookSnapshot& snapshot,
    std::size_t bins) {
  const double scaled =
      (std::clamp(level_one_queue_imbalance(snapshot), -1.0, 1.0) +
       1.0) *
      0.5;
  return std::min(
      bins - 1,
      static_cast<std::size_t>(scaled * static_cast<double>(bins)));
}

std::size_t signal_bin(double value, std::size_t bins) {
  const double scaled =
      (std::clamp(value, -1.0, 1.0) + 1.0) * 0.5;
  return std::min(
      bins - 1,
      static_cast<std::size_t>(scaled * static_cast<double>(bins)));
}

double bin_center(std::size_t bin, std::size_t bins) {
  return -1.0 +
         2.0 * (static_cast<double>(bin) + 0.5) /
             static_cast<double>(bins);
}

void accumulate_bin(BinCount& count, int direction) {
  ++count.observations;
  if (direction > 0) {
    ++count.up_moves;
  } else {
    ++count.down_moves;
  }
}

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

std::vector<SessionFiles> discover_sessions(
    const std::filesystem::path& raw_dir,
    const AnalysisPolicy& policy) {
  if (!std::filesystem::is_directory(raw_dir)) {
    throw std::runtime_error(
        "raw directory does not exist: " + raw_dir.string());
  }

  const std::regex pattern(
      "^" + regex_escape(policy.symbol) +
      "_(\\d{4}-\\d{2}-\\d{2})_(\\d+)_(\\d+)_(message|orderbook)_" +
      std::to_string(kDepth) + "\\.csv$");

  struct Pair {
    std::string date;
    std::string message_path;
    std::string book_path;
    std::uint64_t requested_end_ms{};
  };
  std::map<std::string, Pair> grouped;
  for (const auto& entry : std::filesystem::directory_iterator(raw_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::smatch match;
    const std::string filename = entry.path().filename().string();
    if (!std::regex_match(filename, match, pattern)) {
      continue;
    }
    const std::string date = match[1].str();
    if (!date.starts_with(std::to_string(policy.year) + '-')) {
      continue;
    }

    const std::string key =
        date + '_' + match[2].str() + '_' + match[3].str();
    Pair& pair = grouped[key];
    pair.date = date;
    pair.requested_end_ms = std::stoull(match[3].str());
    if (match[4].str() == "message") {
      pair.message_path = entry.path().string();
    } else {
      pair.book_path = entry.path().string();
    }
  }

  std::vector<SessionFiles> sessions;
  sessions.reserve(grouped.size());
  for (const auto& [key, pair] : grouped) {
    if (pair.message_path.empty() || pair.book_path.empty()) {
      throw std::runtime_error("incomplete LOBSTER pair: " + key);
    }
    sessions.push_back(
        {
            pair.date,
            pair.message_path,
            pair.book_path,
            pair.requested_end_ms,
        });
  }
  if (sessions.empty()) {
    throw std::runtime_error(
        "no " + policy.symbol + ' ' + std::to_string(policy.year) +
        " level-" + std::to_string(kDepth) +
        " sessions found in " + raw_dir.string());
  }
  return sessions;
}

Dataset load_dataset(
    const std::vector<SessionFiles>& sessions,
    const AnalysisPolicy& policy) {
  Dataset dataset;
  dataset.sessions.reserve(sessions.size());
  dataset.delivered_sessions =
      static_cast<std::uint64_t>(sessions.size());
  std::set<std::string> observed_source_exclusions;

  for (const SessionFiles& files : sessions) {
    std::vector<EventRecord> events =
        decode_paired_files(files.message_path, files.book_path);
    const SessionCoverage coverage = validate_session_source(
        files.date,
        files.requested_end_ms,
        events,
        policy);
    if (coverage.status ==
        SessionSourceStatus::declared_source_exclusion) {
      observed_source_exclusions.insert(files.date);
      continue;
    }

    std::erase_if(
        events,
        [&coverage](const EventRecord& event) {
          return event.message.timestamp_ns >
                 coverage.scheduled_close_ns;
        });
    if (events.size() != coverage.included_events) {
      throw std::runtime_error(
          "session coverage count changed while filtering " + files.date);
    }
    dataset.input_bytes +=
        static_cast<std::uint64_t>(
            std::filesystem::file_size(files.message_path)) +
        static_cast<std::uint64_t>(
            std::filesystem::file_size(files.book_path));
    dataset.events += static_cast<std::uint64_t>(events.size());
    dataset.sessions.push_back({files.date, std::move(events)});
  }

  validate_analysis_universe(
      dataset.delivered_sessions,
      static_cast<std::uint64_t>(dataset.sessions.size()),
      observed_source_exclusions,
      policy);
  dataset.declared_source_exclusions =
      static_cast<std::uint64_t>(observed_source_exclusions.size());
  return dataset;
}

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

std::int64_t top_of_book_ofi(
    const BookSnapshot& previous,
    const BookSnapshot& current) {
  const std::int64_t bid_add =
      current.bids[0].price >= previous.bids[0].price
          ? current.bids[0].size
          : 0;
  const std::int64_t bid_remove =
      current.bids[0].price <= previous.bids[0].price
          ? previous.bids[0].size
          : 0;
  const std::int64_t ask_add =
      current.asks[0].price <= previous.asks[0].price
          ? current.asks[0].size
          : 0;
  const std::int64_t ask_remove =
      current.asks[0].price >= previous.asks[0].price
          ? previous.asks[0].size
          : 0;
  return bid_add - bid_remove - ask_add + ask_remove;
}

double level_one_queue_imbalance(const BookSnapshot& snapshot) {
  const double bid = static_cast<double>(snapshot.bids[0].size);
  const double ask = static_cast<double>(snapshot.asks[0].size);
  const double displayed_depth = bid + ask;
  return displayed_depth > 0.0
             ? (bid - ask) / displayed_depth
             : 0.0;
}

double bounded_order_flow_pressure(
    std::int64_t cumulative_ofi,
    const BookSnapshot& snapshot) {
  const double displayed_depth =
      static_cast<double>(snapshot.bids[0].size) +
      snapshot.asks[0].size;
  if (displayed_depth <= 0.0) {
    return 0.0;
  }
  const double depth_units =
      static_cast<double>(cumulative_ofi) / displayed_depth;
  return 2.0 * std::atan(depth_units) /
         std::numbers::pi_v<double>;
}

ReplayMetrics replay_dataset(const Dataset& dataset) {
  ReplayMetrics metrics;
  constexpr std::uint64_t fnv_offset = 14'695'981'039'346'656'037ULL;
  constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
  metrics.checksum = fnv_offset;

  auto mix = [&metrics](std::uint64_t value) {
    metrics.checksum ^= value;
    metrics.checksum *= fnv_prime;
  };

  for (const SessionData& session : dataset.sessions) {
    if (session.events.empty()) {
      continue;
    }
    ++metrics.seeded_sessions;
    const BookSnapshot* previous = nullptr;
    for (const EventRecord& record : session.events) {
      ++metrics.events;
      const std::size_t event_type =
          static_cast<std::size_t>(record.message.event_type);
      if (event_type < metrics.events_by_type.size()) {
        ++metrics.events_by_type[event_type];
      }
      if (!valid_snapshot(record.book)) {
        ++metrics.invalid_snapshots;
      }

      if (previous != nullptr) {
        switch (audit_transition(
            *previous,
            record.message,
            record.book)) {
          case TransitionStatus::exact:
            ++metrics.exact_transitions;
            break;
          case TransitionStatus::depth_censored:
            ++metrics.depth_censored_transitions;
            break;
          case TransitionStatus::mismatch:
            ++metrics.mismatches;
            break;
          case TransitionStatus::unsupported:
            ++metrics.unsupported;
            break;
        }
      }
      mix(record.message.timestamp_ns);
      mix(static_cast<std::uint64_t>(record.message.order_id));
      mix(static_cast<std::uint32_t>(record.book.asks[0].price));
      mix(static_cast<std::uint32_t>(record.book.asks[0].size));
      mix(static_cast<std::uint32_t>(record.book.bids[0].price));
      mix(static_cast<std::uint32_t>(record.book.bids[0].size));
      previous = &record.book;
    }
  }
  return metrics;
}

TimingSummary summarize_timings(
    const std::vector<double>& seconds,
    std::uint64_t events) {
  TimingSummary summary;
  summary.seconds = seconds;
  summary.median_seconds = median(seconds);
  summary.p95_seconds = percentile95(seconds);
  if (events > 0) {
    const double observations = static_cast<double>(events);
    summary.median_ns_per_event =
        summary.median_seconds * 1e9 / observations;
    summary.p95_ns_per_event =
        summary.p95_seconds * 1e9 / observations;
    if (summary.median_seconds > 0.0) {
      summary.median_million_events_per_second =
          observations / summary.median_seconds / 1e6;
    }
  }
  return summary;
}

void write_queue_imbalance_bins(
    const Dataset& dataset,
    const std::filesystem::path& output,
    std::size_t bins) {
  if (bins < 3) {
    throw std::invalid_argument(
        "queue-imbalance bin count must be at least 3");
  }
  std::ofstream stream(output);
  if (!stream) {
    throw std::runtime_error(
        "cannot write queue-imbalance bins: " + output.string());
  }
  stream
      << "date,sample,spread_bucket,bin,bin_left,bin_right,"
         "bin_center,observations,up_moves,down_moves\n";
  stream << std::setprecision(10);

  for (const SessionData& session : dataset.sessions) {
    std::array<std::vector<BinCount>, spread_bucket_names.size()>
        all_events;
    std::array<std::vector<BinCount>, spread_bucket_names.size()>
        quote_updates;
    for (std::size_t bucket = 0;
         bucket < spread_bucket_names.size();
         ++bucket) {
      all_events[bucket].resize(bins);
      quote_updates[bucket].resize(bins);
    }

    std::size_t run_start = 0;
    while (run_start < session.events.size()) {
      const std::int64_t current_mid =
          midpoint_twice(session.events[run_start].book);
      std::size_t next_run = run_start + 1;
      while (
          next_run < session.events.size() &&
          midpoint_twice(session.events[next_run].book) == current_mid) {
        ++next_run;
      }
      if (next_run == session.events.size()) {
        break;
      }
      const int direction =
          midpoint_twice(session.events[next_run].book) > current_mid
              ? 1
              : -1;
      for (std::size_t index = run_start;
           index < next_run;
           ++index) {
        const BookSnapshot& snapshot = session.events[index].book;
        const std::size_t bin = imbalance_bin(snapshot, bins);
        const std::size_t bucket = spread_bucket(snapshot);
        accumulate_bin(all_events[0][bin], direction);
        accumulate_bin(all_events[bucket][bin], direction);
        if (
            index == 0 ||
            !same_best_quote(
                session.events[index - 1].book,
                snapshot)) {
          accumulate_bin(quote_updates[0][bin], direction);
          accumulate_bin(quote_updates[bucket][bin], direction);
        }
      }
      run_start = next_run;
    }

    const auto write_sample = [&](
                                  std::string_view name,
                                  const std::array<
                                      std::vector<BinCount>,
                                      spread_bucket_names.size()>&
                                      counts) {
      for (std::size_t bucket = 0;
           bucket < spread_bucket_names.size();
           ++bucket) {
        for (std::size_t bin = 0; bin < bins; ++bin) {
          if (counts[bucket][bin].observations == 0) {
            continue;
          }
          const double left =
              -1.0 + 2.0 * static_cast<double>(bin) /
                         static_cast<double>(bins);
          const double right =
              -1.0 + 2.0 * static_cast<double>(bin + 1) /
                         static_cast<double>(bins);
          stream
              << session.date << ',' << name << ','
              << spread_bucket_names[bucket] << ',' << bin << ','
              << left << ',' << right << ',' << (left + right) / 2.0
              << ',' << counts[bucket][bin].observations << ','
              << counts[bucket][bin].up_moves << ','
              << counts[bucket][bin].down_moves << '\n';
        }
      }
    };
    write_sample("all_events", all_events);
    write_sample("best_quote_updates", quote_updates);
  }
}

void write_order_flow_signal_bins(
    const Dataset& dataset,
    const std::filesystem::path& output,
    std::size_t bins) {
  if (bins < 3) {
    throw std::invalid_argument(
        "order-flow grid size must be at least 3");
  }
  std::ofstream stream(output);
  if (!stream) {
    throw std::runtime_error(
        "cannot write order-flow signal bins: " + output.string());
  }
  stream
      << "date,sample,spread_bucket,queue_bin,ofi_bin,queue_center,"
         "ofi_center,observations,up_moves,down_moves\n";
  stream << std::setprecision(10);

  for (const SessionData& session : dataset.sessions) {
    std::array<std::vector<BinCount>, spread_bucket_names.size()>
        counts;
    for (std::vector<BinCount>& bucket_counts : counts) {
      bucket_counts.resize(bins * bins);
    }

    std::size_t run_start = 0;
    while (run_start < session.events.size()) {
      const std::int64_t current_mid =
          midpoint_twice(session.events[run_start].book);
      std::size_t next_run = run_start + 1;
      while (
          next_run < session.events.size() &&
          midpoint_twice(session.events[next_run].book) == current_mid) {
        ++next_run;
      }
      if (next_run == session.events.size()) {
        break;
      }
      const int direction =
          midpoint_twice(session.events[next_run].book) > current_mid
              ? 1
              : -1;

      std::int64_t cumulative_ofi = 0;
      for (std::size_t index = run_start;
           index < next_run;
           ++index) {
        const BookSnapshot& snapshot = session.events[index].book;
        if (index > run_start) {
          cumulative_ofi += top_of_book_ofi(
              session.events[index - 1].book,
              snapshot);
        }
        if (
            index != run_start &&
            same_best_quote(
                session.events[index - 1].book,
                snapshot)) {
          continue;
        }

        const std::size_t queue = signal_bin(
            level_one_queue_imbalance(snapshot),
            bins);
        const std::size_t ofi = signal_bin(
            bounded_order_flow_pressure(cumulative_ofi, snapshot),
            bins);
        const std::size_t cell = queue * bins + ofi;
        const std::size_t bucket = spread_bucket(snapshot);
        accumulate_bin(counts[0][cell], direction);
        accumulate_bin(counts[bucket][cell], direction);
      }
      run_start = next_run;
    }

    for (std::size_t bucket = 0;
         bucket < spread_bucket_names.size();
         ++bucket) {
      for (std::size_t queue = 0; queue < bins; ++queue) {
        for (std::size_t ofi = 0; ofi < bins; ++ofi) {
          const BinCount& count = counts[bucket][queue * bins + ofi];
          if (count.observations == 0) {
            continue;
          }
          stream
              << session.date << ",best_quote_updates,"
              << spread_bucket_names[bucket] << ',' << queue << ','
              << ofi << ',' << bin_center(queue, bins) << ','
              << bin_center(ofi, bins) << ',' << count.observations
              << ',' << count.up_moves << ',' << count.down_moves
              << '\n';
        }
      }
    }
  }
}

std::string replay_audit_json(
    const Dataset& dataset,
    const ReplayMetrics& metrics) {
  const std::uint64_t post_seed_transitions =
      metrics.events >= metrics.seeded_sessions
          ? metrics.events - metrics.seeded_sessions
          : 0;
  std::ostringstream output;
  output
      << "{\n"
      << "  \"dataset\": {\n"
      << "    \"delivered_sessions\": "
      << dataset.delivered_sessions << ",\n"
      << "    \"included_sessions\": "
      << dataset.sessions.size() << ",\n"
      << "    \"declared_source_exclusions\": "
      << dataset.declared_source_exclusions << ",\n"
      << "    \"events\": " << dataset.events << "\n"
      << "  },\n"
      << "  \"audit\": {\n"
      << "    \"seeded_sessions\": "
      << metrics.seeded_sessions << ",\n"
      << "    \"post_seed_transitions\": "
      << post_seed_transitions << ",\n"
      << "    \"exact_transitions\": "
      << metrics.exact_transitions << ",\n"
      << "    \"depth_censored_transitions\": "
      << metrics.depth_censored_transitions << ",\n"
      << "    \"mismatches\": " << metrics.mismatches << ",\n"
      << "    \"unsupported\": " << metrics.unsupported << ",\n"
      << "    \"invalid_snapshots\": "
      << metrics.invalid_snapshots << ",\n"
      << "    \"events_by_type\": {\n";
  for (std::size_t event_type = 1;
       event_type < metrics.events_by_type.size();
       ++event_type) {
    output
        << "      \"" << event_type << "\": "
        << metrics.events_by_type[event_type]
        << (event_type + 1 < metrics.events_by_type.size()
                ? ",\n"
                : "\n");
  }
  output
      << "    }\n"
      << "  }\n"
      << "}\n";
  return output.str();
}

std::string benchmark_json(
    const Dataset& dataset,
    const ReplayMetrics& metrics,
    const TimingSummary& decode,
    const TimingSummary& replay,
    int decode_runs,
    int replay_runs,
    int warmup_runs,
    const std::string& machine) {
  const std::uint64_t auditable =
      metrics.events >= metrics.seeded_sessions
          ? metrics.events - metrics.seeded_sessions
          : 0;
  const double exact_fraction =
      auditable > 0
          ? static_cast<double>(metrics.exact_transitions) /
                static_cast<double>(auditable)
          : 0.0;
  const double censored_fraction =
      auditable > 0
          ? static_cast<double>(metrics.depth_censored_transitions) /
                static_cast<double>(auditable)
          : 0.0;

  auto timing_json = [](
      std::ostringstream& output,
      const TimingSummary& timing) {
    output
        << "{\n"
        << "      \"seconds\": [";
    for (std::size_t index = 0; index < timing.seconds.size(); ++index) {
      if (index != 0) {
        output << ", ";
      }
      output << timing.seconds[index];
    }
    output
        << "],\n"
        << "      \"median_run_seconds\": "
        << timing.median_seconds << ",\n"
        << "      \"p95_run_seconds\": "
        << timing.p95_seconds << ",\n"
        << "      \"median_run_average_ns_per_event\": "
        << timing.median_ns_per_event << ",\n"
        << "      \"p95_run_average_ns_per_event\": "
        << timing.p95_ns_per_event << ",\n"
        << "      \"median_throughput_million_events_per_second\": "
        << timing.median_million_events_per_second << "\n"
        << "    }";
  };

  std::ostringstream output;
  output << std::setprecision(12);
  output
      << "{\n"
      << "  \"dataset\": {\n"
      << "    \"delivered_sessions\": "
      << dataset.delivered_sessions << ",\n"
      << "    \"included_sessions\": "
      << dataset.sessions.size() << ",\n"
      << "    \"declared_source_exclusions\": "
      << dataset.declared_source_exclusions << ",\n"
      << "    \"events\": " << dataset.events << ",\n"
      << "    \"input_bytes\": " << dataset.input_bytes << ",\n"
      << "    \"record_bytes\": " << sizeof(EventRecord) << ",\n"
      << "    \"resident_event_bytes\": "
      << dataset.events * sizeof(EventRecord) << "\n"
      << "  },\n"
      << "  \"environment\": {\n"
      << "    \"machine\": \"" << json_escape(machine) << "\",\n"
      << "    \"architecture\": \"" << architecture_name() << "\",\n"
      << "    \"compiler\": \"" << json_escape(compiler_name()) << "\",\n"
#ifdef NDEBUG
      << "    \"build_type\": \"Release\"\n"
#else
      << "    \"build_type\": \"Debug\"\n"
#endif
      << "  },\n"
      << "  \"protocol\": {\n"
      << "    \"decode_runs\": " << decode_runs << ",\n"
      << "    \"replay_warmup_runs\": " << warmup_runs << ",\n"
      << "    \"replay_measured_runs\": " << replay_runs << ",\n"
      << "    \"thread_count\": 1,\n"
      << "    \"clock\": \"std::chrono::steady_clock\",\n"
      << "    \"filesystem_cache\": "
         "\"not flushed between decode runs\"\n"
      << "  },\n"
      << "  \"audit\": {\n"
      << "    \"seeded_sessions\": "
      << metrics.seeded_sessions << ",\n"
      << "    \"exact_transitions\": "
      << metrics.exact_transitions << ",\n"
      << "    \"depth_censored_transitions\": "
      << metrics.depth_censored_transitions << ",\n"
      << "    \"mismatches\": " << metrics.mismatches << ",\n"
      << "    \"unsupported\": " << metrics.unsupported << ",\n"
      << "    \"invalid_snapshots\": "
      << metrics.invalid_snapshots << ",\n"
      << "    \"exact_fraction_of_auditable\": "
      << exact_fraction << ",\n"
      << "    \"depth_censored_fraction_of_auditable\": "
      << censored_fraction << ",\n"
      << "    \"checksum\": " << metrics.checksum << ",\n"
      << "    \"events_by_type\": {\n";
  bool first_type = true;
  for (std::size_t type = 1;
       type < metrics.events_by_type.size();
       ++type) {
    if (metrics.events_by_type[type] == 0) {
      continue;
    }
    if (!first_type) {
      output << ",\n";
    }
    output
        << "      \"" << type << "\": "
        << metrics.events_by_type[type];
    first_type = false;
  }
  output
      << "\n    }\n"
      << "  },\n"
      << "  \"timings\": {\n"
      << "    \"decode\": ";
  timing_json(output, decode);
  output
      << ",\n"
      << "    \"in_memory_replay\": ";
  timing_json(output, replay);
  output
      << "\n"
      << "  }\n"
      << "}\n";
  return output.str();
}

}  // namespace tsla_lob
