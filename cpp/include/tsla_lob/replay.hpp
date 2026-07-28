#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "tsla_lob/analysis_policy.hpp"
#include "tsla_lob/lobster.hpp"

namespace tsla_lob {

enum class TransitionStatus : std::uint8_t {
  exact,
  depth_censored,
  mismatch,
  unsupported,
};

struct ReplayMetrics {
  std::uint64_t events{};
  std::array<std::uint64_t, 8> events_by_type{};
  std::uint64_t seeded_sessions{};
  std::uint64_t exact_transitions{};
  std::uint64_t depth_censored_transitions{};
  std::uint64_t mismatches{};
  std::uint64_t unsupported{};
  std::uint64_t invalid_snapshots{};
};

[[nodiscard]] std::vector<SessionFiles> discover_sessions(
    const std::filesystem::path& raw_dir,
    const AnalysisPolicy& policy);

[[nodiscard]] Dataset load_dataset(
    const std::vector<SessionFiles>& sessions,
    const AnalysisPolicy& policy);

[[nodiscard]] ReplayMetrics replay_dataset(const Dataset& dataset);

[[nodiscard]] TransitionStatus audit_transition(
    const BookSnapshot& previous,
    const Message& message,
    const BookSnapshot& current);

[[nodiscard]] bool valid_snapshot(const BookSnapshot& snapshot) noexcept;

[[nodiscard]] std::string replay_audit_json(
    const Dataset& dataset,
    const ReplayMetrics& metrics);

}  // namespace tsla_lob
