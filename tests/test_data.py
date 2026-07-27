from pathlib import Path

import pytest

from tsla_market_impact.data import (
    DailyFiles,
    audit_session_coverage,
    discover_daily_files,
    orderbook_columns,
    prepare_visible_market_orders,
    read_visible_executions,
    run_session_coverage_audit,
)


def _analysis_policy(path: Path, exclusions: tuple[str, ...]) -> Path:
    policy = path / "analysis-policy.conf"
    policy.write_text(
        "\n".join(
            [
                "symbol=TSLA",
                "year=2019",
                (
                    "source_availability="
                    "original_project_access_ended_replacement_files_unavailable"
                ),
                "maximum_session_end_gap_seconds=60",
                "expected_delivered_sessions=2",
                "expected_included_sessions=1",
                *(f"source_exclusion={date}" for date in exclusions),
                "early_close=2019-07-03,46800",
                "development_end=2019-08-06",
                "selection_start=2019-08-07",
                "selection_end=2019-10-17",
                "test_start=2019-10-18",
            ]
        )
        + "\n"
    )
    return policy


def test_discovers_complete_pairs_in_date_order(tmp_path: Path) -> None:
    names = [
        "TSLA_2019-01-03_34200000_57600000_orderbook_2.csv",
        "TSLA_2019-01-02_34200000_57600000_message_2.csv",
        "TSLA_2019-01-03_34200000_57600000_message_2.csv",
        "TSLA_2019-01-02_34200000_57600000_orderbook_2.csv",
    ]
    for name in names:
        (tmp_path / name).write_text("")

    pairs = discover_daily_files(tmp_path)

    assert [pair.date for pair in pairs] == ["2019-01-02", "2019-01-03"]
    assert all(pair.levels == 2 for pair in pairs)


def test_rejects_incomplete_pair(tmp_path: Path) -> None:
    (tmp_path / "TSLA_2019-01-02_34200000_57600000_message_2.csv").write_text("")

    with pytest.raises(ValueError, match="missing"):
        discover_daily_files(tmp_path)


def test_applies_declared_exclusions_and_rejects_unexplained_gaps(
    tmp_path: Path,
) -> None:
    policy = _analysis_policy(tmp_path, ("2019-01-09",))
    message = tmp_path / "TSLA_2019-01-09_34200000_57600000_message_2.csv"
    orderbook = tmp_path / "TSLA_2019-01-09_34200000_57600000_orderbook_2.csv"
    message.write_text("34200,1,1,100,1000000,1\n")
    orderbook.write_text("1010000,200,990000,300,1020000,100,980000,100\n")
    early_message = tmp_path / "TSLA_2019-07-03_34200000_57600000_message_2.csv"
    early_book = tmp_path / "TSLA_2019-07-03_34200000_57600000_orderbook_2.csv"
    early_message.write_text(
        "46799.0,1,1,100,1000000,1\n"
        "46799.5,4,2,10,1010000,-1\n"
        "57599.9,4,3,10,1010000,-1\n"
    )
    early_book.write_text(
        "1010000,200,990000,300,1020000,100,980000,100\n"
        "1010000,200,990000,300,1020000,100,980000,100\n"
        "1010000,200,990000,300,1020000,100,980000,100\n"
    )

    coverage = audit_session_coverage(
        tmp_path,
        analysis_policy=policy,
    ).set_index("date")
    early_pair = next(
        pair
        for pair in discover_daily_files(tmp_path)
        if pair.date == "2019-07-03"
    )
    early_executions = read_visible_executions(early_pair, policy)

    assert coverage.loc["2019-01-09", "coverage_status"] == "incomplete"
    assert (
        coverage.loc["2019-01-09", "analysis_status"]
        == "declared_source_exclusion"
    )
    assert coverage.loc["2019-07-03", "coverage_status"] == "complete"
    assert coverage.loc["2019-07-03", "analysis_status"] == "included"
    assert coverage.loc["2019-07-03", "events_after_scheduled_close"] == 1
    assert early_executions["seconds"].tolist() == [46_799.5]

    summary = run_session_coverage_audit(
        tmp_path,
        tmp_path / "results",
        analysis_policy=policy,
    )
    prepared = prepare_visible_market_orders(
        tmp_path,
        tmp_path / "visible.parquet",
        analysis_policy=policy,
    )
    assert summary["included_sessions"] == 1
    assert summary["declared_source_exclusions"] == 1
    assert summary["unexplained_incomplete_sessions"] == 0
    assert prepared["delivered_sessions"] == 2
    assert prepared["sessions"] == 1
    assert prepared["declared_source_exclusions"] == 1

    unexplained_message = (
        tmp_path / "TSLA_2019-01-02_34200000_57600000_message_2.csv"
    )
    unexplained_book = (
        tmp_path / "TSLA_2019-01-02_34200000_57600000_orderbook_2.csv"
    )
    unexplained_message.write_text("34200,1,1,100,1000000,1\n")
    unexplained_book.write_text(
        "1010000,200,990000,300,1020000,100,980000,100\n"
    )
    output = tmp_path / "unexplained.parquet"
    with pytest.raises(ValueError, match="session coverage failure"):
        prepare_visible_market_orders(
            tmp_path,
            output,
            analysis_policy=policy,
        )
    assert not output.exists()


def test_orderbook_column_order_matches_lobster() -> None:
    assert orderbook_columns(2) == [
        "ask_price_1",
        "ask_size_1",
        "bid_price_1",
        "bid_size_1",
        "ask_price_2",
        "ask_size_2",
        "bid_price_2",
        "bid_size_2",
    ]


def test_execution_uses_the_previous_book_row(tmp_path: Path) -> None:
    message = tmp_path / "message.csv"
    orderbook = tmp_path / "orderbook.csv"
    message.write_text(
        "0,1,1,100,1000000,1\n"
        "1,4,2,50,1005000,-1\n"
    )
    orderbook.write_text(
        "1010000,200,990000,300\n"
        "1020000,180,1000000,250\n"
    )
    day = DailyFiles(
        symbol="TSLA",
        date="2019-01-02",
        start="34200000",
        end="57600000",
        levels=1,
        message_path=message,
        orderbook_path=orderbook,
    )

    execution = read_visible_executions(day).iloc[0]

    assert execution["event_row"] == 1
    assert execution["mid_price_before"] == pytest.approx(100.0)
    assert execution["spread_before"] == pytest.approx(2.0)
    assert execution["execution_price"] == pytest.approx(100.5)
    assert execution["trade_sign"] == 1


def test_execution_accepts_one_optional_message_field(tmp_path: Path) -> None:
    message = tmp_path / "message.csv"
    orderbook = tmp_path / "orderbook.csv"
    message.write_text(
        "0,1,1,100,1000000,1,null\n"
        "1,4,2,50,1005000,-1,null\n"
    )
    orderbook.write_text(
        "1010000,200,990000,300\n"
        "1020000,180,1000000,250\n"
    )
    day = DailyFiles(
        symbol="TSLA",
        date="2019-01-02",
        start="34200000",
        end="57600000",
        levels=1,
        message_path=message,
        orderbook_path=orderbook,
    )

    execution = read_visible_executions(day).iloc[0]

    assert execution["seconds"] == pytest.approx(1.0)
    assert execution["event_type"] == 4
    assert execution["order_id"] == 2
    assert execution["size"] == 50
    assert execution["execution_price"] == pytest.approx(100.5)
    assert execution["trade_sign"] == 1

    message.write_text(
        "0,1,1,100,1000000,1\n"
        "1,4,2,50,1005000,-1,null,extra\n"
    )
    with pytest.raises(ValueError, match="expected 6 or 7 fields"):
        read_visible_executions(day)
