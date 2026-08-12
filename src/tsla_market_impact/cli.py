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
from .horizons import run_ofi_horizon_analysis
from .liquidity import run_asymmetric_liquidity_analysis
from .markouts import (
    run_marketable_markout_analysis,
    run_price_spell_landmark_analysis,
)
from .orderflow import run_order_flow_analysis, run_order_flow_grid_robustness
from .persistence import run_order_sign_persistence_analysis
from .pipeline import run_analysis
from .policy import DEFAULT_ANALYSIS_POLICY, load_analysis_policy
from .policy_audit import run_round_trip_policy_audit
from .queue import run_queue_analysis
from .round_trips import run_price_spell_round_trip_analysis
from .stability import run_signal_stability_analysis


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

    persistence = subparsers.add_parser("analyze-sign-persistence")
    persistence.add_argument("--transactions", type=Path, required=True)
    persistence.add_argument("--results", type=Path, default=Path("results"))
    persistence.add_argument(
        "--figures",
        type=Path,
        default=Path("report/figures"),
    )

    asymmetric_liquidity = subparsers.add_parser(
        "analyze-asymmetric-liquidity"
    )
    asymmetric_liquidity.add_argument("--orders", type=Path, required=True)
    asymmetric_liquidity.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
    )
    asymmetric_liquidity.add_argument(
        "--figures",
        type=Path,
        default=Path("report/figures"),
    )

    queue = subparsers.add_parser("analyze-queue")
    queue.add_argument("--bins", type=Path, required=True)
    queue.add_argument("--results", type=Path, default=Path("results"))
    queue.add_argument("--figures", type=Path, default=Path("report/figures"))

    order_flow = subparsers.add_parser("analyze-order-flow")
    order_flow.add_argument("--bins", type=Path, required=True)
    order_flow.add_argument("--results", type=Path, default=Path("results"))
    order_flow.add_argument("--figures", type=Path, default=Path("report/figures"))

    order_flow_grid = subparsers.add_parser("analyze-order-flow-grid")
    order_flow_grid.add_argument(
        "--bins",
        action="append",
        required=True,
        metavar="N=PATH",
        help="Joint signal grid, repeated for at least two resolutions.",
    )
    order_flow_grid.add_argument("--results", type=Path, default=Path("results"))

    ofi_horizons = subparsers.add_parser("analyze-ofi-horizons")
    ofi_horizons.add_argument("--bins", type=Path, required=True)
    ofi_horizons.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
    )
    ofi_horizons.add_argument(
        "--figures",
        type=Path,
        default=Path("report/figures"),
    )

    markouts = subparsers.add_parser("analyze-markouts")
    markouts.add_argument("--bins", type=Path, required=True)
    markouts.add_argument("--results", type=Path, default=Path("results"))
    markouts.add_argument(
        "--figures",
        type=Path,
        default=Path("report/figures"),
    )

    landmarks = subparsers.add_parser("analyze-landmarks")
    landmarks.add_argument("--bins", type=Path, required=True)
    landmarks.add_argument("--results", type=Path, default=Path("results"))
    landmarks.add_argument(
        "--figures",
        type=Path,
        default=Path("report/figures"),
    )

    stability = subparsers.add_parser("analyze-signal-stability")
    stability.add_argument("--bins", type=Path, required=True)
    stability.add_argument("--results", type=Path, default=Path("results"))
    stability.add_argument(
        "--figures",
        type=Path,
        default=Path("report/figures"),
    )

    round_trips = subparsers.add_parser("analyze-round-trips")
    round_trips.add_argument("--bins", type=Path, required=True)
    round_trips.add_argument("--results", type=Path, default=Path("results"))
    round_trips.add_argument(
        "--figures",
        type=Path,
        default=Path("report/figures"),
    )

    policy_audit = subparsers.add_parser("audit-round-trip-policies")
    policy_audit.add_argument("--bins", type=Path, required=True)
    policy_audit.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/price_spell_round_trip_metrics.csv"),
    )
    policy_audit.add_argument("--results", type=Path, default=Path("results"))
    policy_audit.add_argument(
        "--figures",
        type=Path,
        default=Path("report/figures"),
    )

    for command in subparsers.choices.values():
        command.add_argument(
            "--analysis-policy",
            type=Path,
            default=DEFAULT_ANALYSIS_POLICY,
        )
    return parser


def _grid_bin_specs(values: list[str]) -> list[tuple[int, Path]]:
    specs = []
    for value in values:
        size_text, separator, path_text = value.partition("=")
        if not separator or not size_text.isdigit() or not path_text:
            raise ValueError(f"invalid grid specification: {value!r}; expected N=PATH")
        specs.append((int(size_text), Path(path_text)))
    return specs


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
    elif args.command == "analyze-sign-persistence":
        result = run_order_sign_persistence_analysis(
            args.transactions,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    elif args.command == "analyze-asymmetric-liquidity":
        result = run_asymmetric_liquidity_analysis(
            args.orders,
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
    elif args.command == "analyze-order-flow":
        result = run_order_flow_analysis(
            args.bins,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    elif args.command == "analyze-order-flow-grid":
        result = run_order_flow_grid_robustness(
            _grid_bin_specs(args.bins),
            args.results,
            analysis_policy=policy,
        )
    elif args.command == "analyze-ofi-horizons":
        result = run_ofi_horizon_analysis(
            args.bins,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    elif args.command == "analyze-markouts":
        result = run_marketable_markout_analysis(
            args.bins,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    elif args.command == "analyze-landmarks":
        result = run_price_spell_landmark_analysis(
            args.bins,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    elif args.command == "analyze-signal-stability":
        result = run_signal_stability_analysis(
            args.bins,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    elif args.command == "analyze-round-trips":
        result = run_price_spell_round_trip_analysis(
            args.bins,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    elif args.command == "audit-round-trip-policies":
        result = run_round_trip_policy_audit(
            args.bins,
            args.metrics,
            args.results,
            args.figures,
            analysis_policy=policy,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
