"""LOBSTER file discovery and transaction reconstruction.

LOBSTER row ``k`` describes the event that moves the book from row ``k - 1``
to row ``k``. The functions below keep that alignment explicit and discard an
execution on the first row of a session because its pre-event state is absent.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PRICE_SCALE = 10_000.0
VISIBLE_EXECUTION_EVENT = 4
HIDDEN_EXECUTION_EVENT = 5
MESSAGE_COLUMNS = [
    "seconds",
    "event_type",
    "order_id",
    "size",
    "price_raw",
    "resting_order_direction",
]
FILE_PATTERN = re.compile(
    r"(?P<symbol>[A-Z.]+)_(?P<date>\d{4}-\d{2}-\d{2})_"
    r"(?P<start>\d+)_(?P<end>\d+)_(?P<kind>message|orderbook)_"
    r"(?P<levels>\d+)\.csv"
)


@dataclass(frozen=True)
class DailyFiles:
    """One aligned LOBSTER message and order-book pair."""

    symbol: str
    date: str
    start: str
    end: str
    levels: int
    message_path: Path
    orderbook_path: Path


def discover_daily_files(raw_dir: Path | str, symbol: str = "TSLA") -> list[DailyFiles]:
    """Return complete daily pairs in chronological order."""

    raw_path = Path(raw_dir)
    grouped: dict[tuple[str, str, str, int], dict[str, Path]] = {}
    for path in sorted(raw_path.glob(f"{symbol}_*.csv")):
        match = FILE_PATTERN.fullmatch(path.name)
        if match is None or match.group("symbol") != symbol:
            continue
        key = (
            match.group("date"),
            match.group("start"),
            match.group("end"),
            int(match.group("levels")),
        )
        grouped.setdefault(key, {})[match.group("kind")] = path

    pairs: list[DailyFiles] = []
    for (date, start, end, levels), files in sorted(grouped.items()):
        missing = {"message", "orderbook"} - files.keys()
        if missing:
            raise ValueError(f"Incomplete LOBSTER pair for {date}: missing {sorted(missing)}")
        pairs.append(
            DailyFiles(
                symbol=symbol,
                date=date,
                start=start,
                end=end,
                levels=levels,
                message_path=files["message"],
                orderbook_path=files["orderbook"],
            )
        )
    if not pairs:
        raise FileNotFoundError(f"No LOBSTER files found for {symbol} in {raw_path}")
    return pairs


def orderbook_columns(levels: int) -> list[str]:
    """Return LOBSTER order-book column names for ``levels`` price levels."""

    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def _timestamps(date: str, seconds: pd.Series) -> pd.Series:
    base = pd.Timestamp(date)
    return base + pd.to_timedelta(seconds, unit="s")


def _read_aligned_best_book(day: DailyFiles) -> tuple[pd.DataFrame, pd.DataFrame]:
    messages = pd.read_csv(
        day.message_path,
        header=None,
        usecols=range(len(MESSAGE_COLUMNS)),
        names=MESSAGE_COLUMNS,
    )
    best_columns = orderbook_columns(day.levels)[:4]
    book = pd.read_csv(
        day.orderbook_path,
        header=None,
        usecols=range(4),
        names=best_columns,
    )
    if len(messages) != len(book):
        raise ValueError(
            f"LOBSTER alignment failure for {day.date}: "
            f"{len(messages)} message rows != {len(book)} book rows"
        )
    return messages, book


def read_visible_executions(day: DailyFiles) -> pd.DataFrame:
    """Read visible fills and attach the pre-event best bid and ask."""

    messages, book = _read_aligned_best_book(day)
    event_rows = np.flatnonzero(messages["event_type"].to_numpy() == VISIBLE_EXECUTION_EVENT)
    event_rows = event_rows[event_rows > 0]
    selected = messages.iloc[event_rows].reset_index(drop=True)

    before = book.iloc[event_rows - 1].reset_index(drop=True)
    before.columns = [f"{column}_before" for column in before.columns]
    executions = pd.concat([selected, before], axis=1)
    executions.insert(0, "date", day.date)
    executions.insert(1, "timestamp", _timestamps(day.date, executions["seconds"]))
    executions.insert(3, "event_row", event_rows)
    executions["execution_price"] = executions.pop("price_raw") / PRICE_SCALE
    executions["trade_sign"] = -executions["resting_order_direction"]

    price_columns = [column for column in executions if "_price_" in column]
    executions[price_columns] = executions[price_columns] / PRICE_SCALE
    executions["mid_price_before"] = (
        executions["ask_price_1_before"] + executions["bid_price_1_before"]
    ) / 2
    executions["spread_before"] = (
        executions["ask_price_1_before"] - executions["bid_price_1_before"]
    )
    return executions


def aggregate_visible_market_orders(executions: pd.DataFrame) -> pd.DataFrame:
    """Merge visible fills with the same timestamp and initiating sign."""

    if executions.empty:
        return executions.copy()
    ordered = executions.sort_values(["date", "event_row"]).copy()
    ordered["_weighted_notional"] = ordered["execution_price"] * ordered["size"]
    grouped = ordered.groupby(["date", "seconds", "trade_sign"], sort=True, observed=True)
    market_orders = grouped.agg(
        timestamp=("timestamp", "first"),
        first_event_row=("event_row", "first"),
        last_event_row=("event_row", "last"),
        execution_count=("event_type", "size"),
        size=("size", "sum"),
        weighted_notional=("_weighted_notional", "sum"),
        mid_price_before=("mid_price_before", "first"),
        spread_before=("spread_before", "first"),
    ).reset_index()
    market_orders["execution_price_vwap"] = (
        market_orders.pop("weighted_notional") / market_orders["size"]
    )
    return market_orders.sort_values(["date", "first_event_row"]).reset_index(drop=True)


def read_scaling_transactions(day: DailyFiles) -> pd.DataFrame:
    """Build type-4/type-5 transactions for the scaling replication.

    The aggressor sign is inferred from execution price relative to the
    pre-event mid-price. Executions exactly at the midpoint are unsigned and
    are excluded.
    """

    messages, book = _read_aligned_best_book(day)
    event_rows = np.flatnonzero(
        messages["event_type"].isin([VISIBLE_EXECUTION_EVENT, HIDDEN_EXECUTION_EVENT]).to_numpy()
    )
    event_rows = event_rows[event_rows > 0]
    selected = messages.iloc[event_rows].reset_index(drop=True)
    before = book.iloc[event_rows - 1].reset_index(drop=True)
    before.columns = [f"{column}_before" for column in before.columns]

    fills = pd.concat([selected, before], axis=1)
    fills.insert(0, "date", day.date)
    fills.insert(1, "timestamp", _timestamps(day.date, fills["seconds"]))
    fills.insert(3, "event_row", event_rows)
    fills["execution_price"] = fills.pop("price_raw") / PRICE_SCALE
    fills[["ask_price_1_before", "bid_price_1_before"]] /= PRICE_SCALE
    fills["mid_price_before"] = (fills["ask_price_1_before"] + fills["bid_price_1_before"]) / 2
    fills["trade_sign"] = np.sign(fills["execution_price"] - fills["mid_price_before"]).astype(int)
    fills = fills.loc[fills["trade_sign"].ne(0)].copy()
    fills["_weighted_notional"] = fills["execution_price"] * fills["size"]

    grouped = fills.sort_values(["date", "event_row"]).groupby(
        ["date", "seconds", "trade_sign"], sort=True, observed=True
    )
    transactions = grouped.agg(
        timestamp=("timestamp", "first"),
        first_event_row=("event_row", "first"),
        last_event_row=("event_row", "last"),
        transaction_count=("event_type", "size"),
        size=("size", "sum"),
        weighted_notional=("_weighted_notional", "sum"),
        mid_price_before=("mid_price_before", "first"),
    ).reset_index()
    transactions["execution_price_vwap"] = (
        transactions.pop("weighted_notional") / transactions["size"]
    )
    return transactions.sort_values(["date", "first_event_row"]).reset_index(drop=True)


def _append_parquet(
    writer: pq.ParquetWriter | None,
    path: Path,
    frame: pd.DataFrame,
) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def _prepare_annual_table(
    raw_dir: Path | str,
    output: Path | str,
    reader: Callable[[DailyFiles], pd.DataFrame],
    symbol: str,
    year: int,
) -> dict[str, int]:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    pairs = [
        pair
        for pair in discover_daily_files(raw_dir, symbol=symbol)
        if pair.date.startswith(f"{year}-")
    ]
    if not pairs:
        raise FileNotFoundError(f"No {symbol} sessions found for {year}")

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for pair in pairs:
            frame = reader(pair)
            row_count += len(frame)
            writer = _append_parquet(writer, temporary, frame)
    finally:
        if writer is not None:
            writer.close()

    temporary.replace(output_path)
    return {"sessions": len(pairs), "rows": row_count}


def prepare_visible_market_orders(
    raw_dir: Path | str,
    output: Path | str,
    symbol: str = "TSLA",
    year: int = 2019,
) -> dict[str, int]:
    """Reconstruct and persist timestamp-aggregated visible market orders."""

    def reader(pair: DailyFiles) -> pd.DataFrame:
        return aggregate_visible_market_orders(read_visible_executions(pair))

    return _prepare_annual_table(raw_dir, output, reader, symbol, year)


def prepare_scaling_transactions(
    raw_dir: Path | str,
    output: Path | str,
    symbol: str = "TSLA",
    year: int = 2019,
) -> dict[str, int]:
    """Persist timestamp-aggregated type-4/type-5 transactions."""

    return _prepare_annual_table(raw_dir, output, read_scaling_transactions, symbol, year)


def required_columns(table: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a readable error when an input table misses its data contract."""

    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")
