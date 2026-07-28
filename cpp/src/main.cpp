#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "tsla_lob/analysis_policy.hpp"
#include "tsla_lob/replay.hpp"

namespace {

struct Options {
  std::filesystem::path raw_dir;
  std::filesystem::path analysis_policy{"analysis-policy.conf"};
  std::filesystem::path json_output;
};

void print_usage(std::ostream& output) {
  output
      << "Usage: lobster_replay --raw-dir PATH [options]\n"
      << "\n"
      << "Options:\n"
      << "  --analysis-policy PATH  Shared source policy (default: analysis-policy.conf)\n"
      << "  --json PATH             Write the aggregate replay audit as JSON\n"
      << "  --help                  Show this help\n";
}

std::string require_value(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(
        "missing value for " + std::string(argv[index]));
  }
  ++index;
  return argv[index];
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--raw-dir") {
      options.raw_dir = require_value(argc, argv, index);
    } else if (argument == "--analysis-policy") {
      options.analysis_policy = require_value(argc, argv, index);
    } else if (argument == "--json") {
      options.json_output = require_value(argc, argv, index);
    } else {
      throw std::invalid_argument("unknown argument: " + argument);
    }
  }
  if (options.raw_dir.empty()) {
    throw std::invalid_argument("--raw-dir is required");
  }
  return options;
}

void validate_annual_audit(
    const tsla_lob::Dataset& dataset,
    const tsla_lob::ReplayMetrics& metrics) {
  if (metrics.events != dataset.events ||
      metrics.seeded_sessions != dataset.sessions.size()) {
    throw std::runtime_error(
        "replay totals disagree with the loaded dataset");
  }

  const std::uint64_t typed_events = std::accumulate(
      metrics.events_by_type.begin() + 1,
      metrics.events_by_type.end(),
      std::uint64_t{0});
  if (typed_events != metrics.events) {
    throw std::runtime_error(
        "event-type totals disagree with the replay event count");
  }

  const std::uint64_t transitions =
      metrics.exact_transitions +
      metrics.depth_censored_transitions +
      metrics.mismatches +
      metrics.unsupported;
  if (metrics.events < metrics.seeded_sessions ||
      transitions != metrics.events - metrics.seeded_sessions) {
    throw std::runtime_error(
        "transition totals disagree with the post-seed event count");
  }

  if (metrics.mismatches != 0 ||
      metrics.unsupported != 0 ||
      metrics.invalid_snapshots != 0) {
    throw std::runtime_error(
        "annual replay audit failed: mismatches=" +
        std::to_string(metrics.mismatches) +
        ", unsupported=" + std::to_string(metrics.unsupported) +
        ", invalid_snapshots=" +
        std::to_string(metrics.invalid_snapshots));
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      print_usage(std::cout);
      return 0;
    }

    const Options options = parse_options(argc, argv);
    const tsla_lob::AnalysisPolicy policy =
        tsla_lob::load_analysis_policy(options.analysis_policy);
    const std::vector<tsla_lob::SessionFiles> sessions =
        tsla_lob::discover_sessions(options.raw_dir, policy);
    const tsla_lob::Dataset dataset =
        tsla_lob::load_dataset(sessions, policy);
    const tsla_lob::ReplayMetrics metrics =
        tsla_lob::replay_dataset(dataset);
    validate_annual_audit(dataset, metrics);

    const std::string report =
        tsla_lob::replay_audit_json(dataset, metrics);
    if (!options.json_output.empty()) {
      std::ofstream output(options.json_output);
      if (!output) {
        throw std::runtime_error(
            "cannot write replay audit: " +
            options.json_output.string());
      }
      output << report;
    }
    std::cout << report;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "lobster_replay: " << error.what() << '\n';
    return 1;
  }
}
