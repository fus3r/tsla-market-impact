#include "tsla_lob/session_integrity.hpp"

#include <stdexcept>

namespace tsla_lob {

SessionCoverage validate_session_source(
    const std::string& date,
    std::uint64_t requested_end_ms,
    std::span<const EventRecord> events,
    const AnalysisPolicy& policy) {
  const std::uint64_t close_ns =
      scheduled_close_ns(policy, date, requested_end_ms);
  std::size_t included_events = 0;
  std::size_t events_after_close = 0;
  std::uint64_t last_in_session_ns = 0;
  bool found_in_session = false;
  for (const EventRecord& event : events) {
    if (event.message.timestamp_ns <= close_ns) {
      ++included_events;
      last_in_session_ns = event.message.timestamp_ns;
      found_in_session = true;
    } else {
      ++events_after_close;
    }
  }
  if (!found_in_session) {
    throw std::runtime_error(
        "LOBSTER session has no event before the scheduled close: " + date);
  }

  const std::uint64_t end_gap_ns = close_ns - last_in_session_ns;
  const bool complete = end_gap_ns <= policy.maximum_session_end_gap_ns;
  const bool declared = policy.source_exclusions.contains(date);
  if (declared && complete) {
    throw std::runtime_error(
        "declared source exclusion now passes the coverage gate: " + date);
  }
  if (!declared && !complete) {
    throw std::runtime_error(
        "undeclared incomplete LOBSTER session: " + date);
  }
  return {
      close_ns,
      end_gap_ns,
      included_events,
      events_after_close,
      declared ? SessionSourceStatus::declared_source_exclusion
               : SessionSourceStatus::included,
  };
}

void validate_analysis_universe(
    std::uint64_t delivered_sessions,
    std::uint64_t included_sessions,
    const std::set<std::string>& observed_source_exclusions,
    const AnalysisPolicy& policy) {
  if (delivered_sessions != policy.expected_delivered_sessions ||
      included_sessions != policy.expected_included_sessions ||
      observed_source_exclusions != policy.source_exclusions) {
    throw std::runtime_error(
        "analysis universe session counts do not match the shared policy");
  }
}

}  // namespace tsla_lob
