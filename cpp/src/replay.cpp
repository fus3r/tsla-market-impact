#include "tsla_lob/replay.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
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

ReplayMetrics replay_dataset(const Dataset& dataset) {
  ReplayMetrics metrics;
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
      previous = &record.book;
    }
  }
  return metrics;
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

}  // namespace tsla_lob
