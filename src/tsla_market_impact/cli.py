"""Command-line entry points for data preparation and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import (
    prepare_scaling_transactions,
    prepare_visible_market_orders,
    run_session_coverage_audit,
)
from .pipeline import run_analysis
from .policy import DEFAULT_ANALYSIS_POLICY, load_analysis_policy
from .queue import run_queue_analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsla-impact",
        description="Reproduce the TSLA 2019 market-impact study.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["prepare-visible", "prepare-scaling"]:
        command = subparsers.add_parser(name)
        command.add_argument("--raw-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--symbol", default="TSLA")
        command.add_argument("--year", type=int, default=2019)

    coverage = subparsers.add_parser("audit-session-coverage")
    coverage.add_argument("--raw-dir", type=Path, required=True)
    coverage.add_argument("--results", type=Path, default=Path("results"))
    coverage.add_argument("--symbol", default="TSLA")
    coverage.add_argument("--year", type=int, default=2019)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--visible", type=Path, required=True)
    analyze.add_argument("--scaling", type=Path, required=True)
    analyze.add_argument("--results", type=Path, default=Path("results"))
    analyze.add_argument("--figures", type=Path, default=Path("report/figures"))

    queue = subparsers.add_parser("analyze-queue")
    queue.add_argument("--bins", type=Path, required=True)
    queue.add_argument("--results", type=Path, default=Path("results"))
    queue.add_argument("--figures", type=Path, default=Path("report/figures"))

    for command in subparsers.choices.values():
        command.add_argument(
            "--analysis-policy",
            type=Path,
            default=DEFAULT_ANALYSIS_POLICY,
        )
    return parser


def main() -> None:
    args = _parser().parse_args()
    policy = load_analysis_policy(args.analysis_policy)
    if args.command == "audit-session-coverage":
        result = run_session_coverage_audit(
            args.raw_dir,
            args.results,
            symbol=args.symbol,
            year=args.year,
            analysis_policy=policy,
        )
    elif args.command == "prepare-visible":
        result = prepare_visible_market_orders(
            args.raw_dir,
            args.output,
            symbol=args.symbol,
            year=args.year,
            analysis_policy=policy,
        )
    elif args.command == "prepare-scaling":
        result = prepare_scaling_transactions(
            args.raw_dir,
            args.output,
            symbol=args.symbol,
            year=args.year,
            analysis_policy=policy,
        )
    elif args.command == "analyze":
        result = run_analysis(
            args.visible,
            args.scaling,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    elif args.command == "analyze-queue":
        result = run_queue_analysis(
            args.bins,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
