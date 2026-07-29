#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tsla_lob/analysis_policy.hpp"
#include "tsla_lob/replay.hpp"

namespace {

struct Options {
  std::filesystem::path raw_dir;
  std::filesystem::path analysis_policy{"analysis-policy.conf"};
  std::filesystem::path json_output;
  int decode_runs{1};
  int replay_runs{7};
  int warmup_runs{1};
  std::string machine;
};

void print_usage(std::ostream& output) {
  output
      << "Usage: lobster_replay --raw-dir PATH [options]\n"
      << "\n"
      << "Options:\n"
      << "  --analysis-policy PATH  Shared source policy (default: analysis-policy.conf)\n"
      << "  --decode-runs N          Full CSV decode repetitions (default: 1)\n"
      << "  --replay-runs N          Measured resident replay passes (default: 7)\n"
      << "  --warmup-runs N          Unmeasured replay passes (default: 1)\n"
      << "  --json PATH              Write the benchmark record as JSON\n"
      << "  --machine TEXT           Machine label stored in the benchmark\n"
      << "  --help                   Show this help\n";
}

std::string require_value(int argc, char** argv, int& index) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(
        "missing value for " + std::string(argv[index]));
  }
  ++index;
  return argv[index];
}

int positive_integer(
    const std::string& value,
    const std::string& name,
    bool allow_zero = false) {
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size() ||
      (allow_zero ? parsed < 0 : parsed < 1)) {
    throw std::invalid_argument("invalid " + name + ": " + value);
  }
  return parsed;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--raw-dir") {
      options.raw_dir = require_value(argc, argv, index);
    } else if (argument == "--analysis-policy") {
      options.analysis_policy = require_value(argc, argv, index);
    } else if (argument == "--decode-runs") {
      options.decode_runs = positive_integer(
          require_value(argc, argv, index),
          "decode runs");
    } else if (argument == "--replay-runs") {
      options.replay_runs = positive_integer(
          require_value(argc, argv, index),
          "replay runs");
    } else if (argument == "--warmup-runs") {
      options.warmup_runs = positive_integer(
          require_value(argc, argv, index),
          "warmup runs",
          true);
    } else if (argument == "--json") {
      options.json_output = require_value(argc, argv, index);
    } else if (argument == "--machine") {
      options.machine = require_value(argc, argv, index);
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

    tsla_lob::Dataset dataset;
    std::vector<double> decode_seconds;
    decode_seconds.reserve(
        static_cast<std::size_t>(options.decode_runs));
    for (int run = 0; run < options.decode_runs; ++run) {
      const auto start = std::chrono::steady_clock::now();
      tsla_lob::Dataset candidate =
          tsla_lob::load_dataset(sessions, policy);
      const auto stop = std::chrono::steady_clock::now();
      decode_seconds.push_back(
          std::chrono::duration<double>(stop - start).count());
      if (run + 1 == options.decode_runs) {
        dataset = std::move(candidate);
      }
    }
    const tsla_lob::TimingSummary decode =
        tsla_lob::summarize_timings(decode_seconds, dataset.events);

    std::uint64_t expected_checksum = 0;
    bool checksum_initialized = false;
    auto verify_checksum = [&](std::uint64_t checksum) {
      if (checksum_initialized && checksum != expected_checksum) {
        throw std::runtime_error(
            "replay checksum changed between repetitions");
      }
      expected_checksum = checksum;
      checksum_initialized = true;
    };
    for (int run = 0; run < options.warmup_runs; ++run) {
      verify_checksum(tsla_lob::replay_dataset(dataset).checksum);
    }

    tsla_lob::ReplayMetrics metrics;
    std::vector<double> replay_seconds;
    replay_seconds.reserve(
        static_cast<std::size_t>(options.replay_runs));
    for (int run = 0; run < options.replay_runs; ++run) {
      const auto start = std::chrono::steady_clock::now();
      const tsla_lob::ReplayMetrics candidate =
          tsla_lob::replay_dataset(dataset);
      const auto stop = std::chrono::steady_clock::now();
      replay_seconds.push_back(
          std::chrono::duration<double>(stop - start).count());
      verify_checksum(candidate.checksum);
      metrics = candidate;
    }
    const tsla_lob::TimingSummary replay =
        tsla_lob::summarize_timings(replay_seconds, dataset.events);
    validate_annual_audit(dataset, metrics);

    const std::string report = tsla_lob::benchmark_json(
        dataset,
        metrics,
        decode,
        replay,
        options.decode_runs,
        options.replay_runs,
        options.warmup_runs,
        options.machine);
    if (!options.json_output.empty()) {
      std::ofstream output(options.json_output);
      if (!output) {
        throw std::runtime_error(
            "cannot write benchmark JSON: " +
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
