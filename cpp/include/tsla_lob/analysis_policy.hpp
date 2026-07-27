#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <string>

namespace tsla_lob {

struct AnalysisPolicy {
  std::string symbol;
  int year{};
  std::string source_availability;
  std::uint64_t maximum_session_end_gap_ns{};
  std::uint64_t expected_delivered_sessions{};
  std::uint64_t expected_included_sessions{};
  std::set<std::string> source_exclusions;
  std::map<std::string, std::uint64_t> early_closes_ns;
  std::string development_end;
  std::string selection_start;
  std::string selection_end;
  std::string test_start;
};

[[nodiscard]] AnalysisPolicy load_analysis_policy(
    const std::filesystem::path& path);

void validate_analysis_scope(
    const AnalysisPolicy& policy,
    const std::string& symbol,
    int year);

[[nodiscard]] std::uint64_t scheduled_close_ns(
    const AnalysisPolicy& policy,
    const std::string& date,
    std::uint64_t requested_end_ms);

}  // namespace tsla_lob
