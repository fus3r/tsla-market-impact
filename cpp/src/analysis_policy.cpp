#include "tsla_lob/analysis_policy.hpp"

#include <algorithm>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string_view>

namespace tsla_lob {
namespace {

std::uint64_t unsigned_integer(std::string_view text, const std::string& field) {
  std::size_t consumed = 0;
  const std::uint64_t value = std::stoull(std::string(text), &consumed);
  if (consumed != text.size()) {
    throw std::runtime_error("invalid " + field + " in analysis policy");
  }
  return value;
}

void validate_date(const std::string& value, const std::string& field) {
  static const std::regex iso_date("^\\d{4}-\\d{2}-\\d{2}$");
  if (!std::regex_match(value, iso_date)) {
    throw std::runtime_error("invalid " + field + " date in analysis policy");
  }
}

std::uint64_t seconds_to_nanoseconds(
    std::uint64_t seconds,
    const std::string& field) {
  constexpr std::uint64_t billion = 1'000'000'000ULL;
  if (seconds == 0 ||
      seconds > std::numeric_limits<std::uint64_t>::max() / billion) {
    throw std::runtime_error("invalid " + field + " in analysis policy");
  }
  return seconds * billion;
}

}  // namespace

AnalysisPolicy load_analysis_policy(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open analysis policy: " + path.string());
  }

  AnalysisPolicy policy;
  std::set<std::string> scalar_keys;
  std::string line;
  std::uint64_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty() || line.front() == '#') {
      continue;
    }
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos || separator == 0 ||
        separator + 1 == line.size()) {
      throw std::runtime_error(
          "malformed analysis policy line " + std::to_string(line_number));
    }
    const std::string key = line.substr(0, separator);
    const std::string value = line.substr(separator + 1);
    if (key == "source_exclusion") {
      validate_date(value, key);
      if (!policy.source_exclusions.insert(value).second) {
        throw std::runtime_error("duplicate source exclusion in analysis policy");
      }
      continue;
    }
    if (key == "early_close") {
      const std::size_t comma = value.find(',');
      if (comma == std::string::npos || comma == 0 ||
          comma + 1 == value.size()) {
        throw std::runtime_error("malformed early_close in analysis policy");
      }
      const std::string date = value.substr(0, comma);
      validate_date(date, key);
      const std::uint64_t seconds =
          unsigned_integer(std::string_view(value).substr(comma + 1), key);
      if (seconds > 24ULL * 60ULL * 60ULL ||
          !policy.early_closes_ns
               .emplace(date, seconds_to_nanoseconds(seconds, key))
               .second) {
        throw std::runtime_error("invalid early_close in analysis policy");
      }
      continue;
    }
    if (!scalar_keys.insert(key).second) {
      throw std::runtime_error("duplicate analysis policy key: " + key);
    }
    if (key == "symbol") {
      policy.symbol = value;
    } else if (key == "year") {
      const std::uint64_t year = unsigned_integer(value, key);
      if (year > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid year in analysis policy");
      }
      policy.year = static_cast<int>(year);
    } else if (key == "source_availability") {
      policy.source_availability = value;
    } else if (key == "maximum_session_end_gap_seconds") {
      policy.maximum_session_end_gap_ns = seconds_to_nanoseconds(
          unsigned_integer(value, key),
          key);
    } else if (key == "expected_delivered_sessions") {
      policy.expected_delivered_sessions = unsigned_integer(value, key);
    } else if (key == "expected_included_sessions") {
      policy.expected_included_sessions = unsigned_integer(value, key);
    } else if (key == "expected_development_sessions") {
      policy.expected_development_sessions = unsigned_integer(value, key);
    } else if (key == "expected_selection_sessions") {
      policy.expected_selection_sessions = unsigned_integer(value, key);
    } else if (key == "expected_test_sessions") {
      policy.expected_test_sessions = unsigned_integer(value, key);
    } else if (key == "development_end") {
      validate_date(value, key);
      policy.development_end = value;
    } else if (key == "selection_start") {
      validate_date(value, key);
      policy.selection_start = value;
    } else if (key == "selection_end") {
      validate_date(value, key);
      policy.selection_end = value;
    } else if (key == "test_start") {
      validate_date(value, key);
      policy.test_start = value;
    } else {
      throw std::runtime_error("unknown analysis policy key: " + key);
    }
  }

  if (policy.symbol.empty() || policy.year == 0 ||
      policy.source_availability.empty() ||
      policy.maximum_session_end_gap_ns == 0 ||
      policy.expected_delivered_sessions == 0 ||
      policy.expected_included_sessions == 0 ||
      !scalar_keys.contains("expected_development_sessions") ||
      !scalar_keys.contains("expected_selection_sessions") ||
      !scalar_keys.contains("expected_test_sessions") ||
      policy.source_exclusions.empty() || policy.early_closes_ns.empty() ||
      policy.development_end.empty() || policy.selection_start.empty() ||
      policy.selection_end.empty() || policy.test_start.empty()) {
    throw std::runtime_error("analysis policy is missing required entries");
  }
  if (policy.expected_delivered_sessions <=
          policy.expected_included_sessions ||
      policy.expected_delivered_sessions -
              policy.expected_included_sessions !=
          policy.source_exclusions.size()) {
    throw std::runtime_error("analysis policy session counts are inconsistent");
  }
  if (policy.expected_development_sessions +
          policy.expected_selection_sessions +
          policy.expected_test_sessions !=
      policy.expected_included_sessions) {
    throw std::runtime_error(
        "analysis policy evaluation counts are inconsistent");
  }
  if (!(policy.development_end < policy.selection_start &&
        policy.selection_start <= policy.selection_end &&
        policy.selection_end < policy.test_start)) {
    throw std::runtime_error(
        "analysis policy calendar boundaries are not chronological");
  }
  return policy;
}

void validate_analysis_scope(
    const AnalysisPolicy& policy,
    const std::string& symbol,
    int year) {
  if (policy.symbol != symbol || policy.year != year) {
    throw std::runtime_error(
        "analysis policy scope mismatch: expected " + policy.symbol + ' ' +
        std::to_string(policy.year) + ", received " + symbol + ' ' +
        std::to_string(year));
  }
}

std::uint64_t scheduled_close_ns(
    const AnalysisPolicy& policy,
    const std::string& date,
    std::uint64_t requested_end_ms) {
  if (requested_end_ms >
      std::numeric_limits<std::uint64_t>::max() / 1'000'000ULL) {
    throw std::runtime_error("LOBSTER filename end time is outside uint64 range");
  }
  const std::uint64_t requested_end_ns = requested_end_ms * 1'000'000ULL;
  const auto close = policy.early_closes_ns.find(date);
  return close == policy.early_closes_ns.end()
             ? requested_end_ns
             : std::min(requested_end_ns, close->second);
}

}  // namespace tsla_lob
