from pathlib import Path

import pytest

from tsla_market_impact.data import (
    DailyFiles,
    discover_daily_files,
    orderbook_columns,
    read_visible_executions,
)


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
