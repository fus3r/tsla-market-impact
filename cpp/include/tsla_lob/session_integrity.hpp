#pragma once

#include <cstddef>
#include <cstdint>
#include <set>
#include <span>
#include <string>

#include "tsla_lob/analysis_policy.hpp"
#include "tsla_lob/lobster.hpp"

namespace tsla_lob {

enum class SessionSourceStatus {
  included,
  declared_source_exclusion,
};

struct SessionCoverage {
  std::uint64_t scheduled_close_ns{};
  std::uint64_t end_gap_ns{};
  std::size_t included_events{};
  std::size_t events_after_scheduled_close{};
  SessionSourceStatus status{};
};

[[nodiscard]] SessionCoverage validate_session_source(
    const std::string& date,
    std::uint64_t requested_end_ms,
    std::span<const EventRecord> events,
    const AnalysisPolicy& policy);

void validate_analysis_universe(
    std::uint64_t delivered_sessions,
    std::uint64_t included_sessions,
    const std::set<std::string>& observed_source_exclusions,
    const AnalysisPolicy& policy);

}  // namespace tsla_lob
