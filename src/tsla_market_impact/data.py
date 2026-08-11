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

from .policy import (
    DEFAULT_ANALYSIS_POLICY,
    AnalysisPolicy,
    load_analysis_policy,
    validate_policy_scope,
)

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


def _policy(
    analysis_policy: AnalysisPolicy | Path | str,
) -> AnalysisPolicy:
    return (
        analysis_policy
        if isinstance(analysis_policy, AnalysisPolicy)
        else load_analysis_policy(analysis_policy)
    )


def _scheduled_end_seconds(day: DailyFiles, policy: AnalysisPolicy) -> float:
    requested_end = int(day.end) / 1_000
    market_close = policy.early_closes_seconds.get(day.date)
    return min(requested_end, market_close) if market_close is not None else requested_end


def _validate_message_field_counts(path: Path) -> None:
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            field_count = line.rstrip(b"\r\n").count(b",") + 1
            if field_count not in {6, 7}:
                raise ValueError(
                    f"Malformed LOBSTER message row in {path} at line "
                    f"{line_number}: expected 6 or 7 fields"
                )


def _last_message_line(path: Path) -> bytes:
    with path.open("rb") as stream:
        line = b""
        stream.seek(0, 2)
        position = stream.tell()
        blocks: list[bytes] = []
        while position > 0:
            block_size = min(position, 4_096)
            position -= block_size
            stream.seek(position)
            blocks.append(stream.read(block_size))
            lines = b"".join(reversed(blocks)).splitlines()
            if len(lines) > 1 or position == 0:
                line = lines[-1] if lines else b""
                break
    if not line:
        raise ValueError(f"Empty LOBSTER message file: {path}")
    return line


def _message_fields(line: bytes, path: Path) -> list[str]:
    try:
        fields = line.decode("ascii").split(",")
    except UnicodeDecodeError as error:
        raise ValueError(f"Non-ASCII LOBSTER message row in {path}") from error
    if len(fields) not in {6, 7}:
        raise ValueError(
            f"Malformed LOBSTER message row in {path}: expected 6 or 7 fields"
        )
    return fields


def _last_message_at_or_before(
    path: Path,
    scheduled_close_seconds: float,
) -> tuple[list[str], int]:
    last_fields = _message_fields(_last_message_line(path), path)
    if float(last_fields[0]) <= scheduled_close_seconds:
        return last_fields, 0

    last_before_close: list[str] | None = None
    events_after_close = 0
    with path.open("rb") as stream:
        for line in stream:
            fields = _message_fields(line.rstrip(b"\r\n"), path)
            if float(fields[0]) <= scheduled_close_seconds:
                last_before_close = fields
            else:
                events_after_close += 1
    if last_before_close is None:
        raise ValueError(
            f"LOBSTER message file has no event before the scheduled close: {path}"
        )
    return last_before_close, events_after_close


def _audit_daily_file_coverage(
    pairs: Iterable[DailyFiles],
    policy: AnalysisPolicy,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day in pairs:
        _validate_message_field_counts(day.message_path)
        requested_end = int(day.end) / 1_000
        scheduled_close = _scheduled_end_seconds(day, policy)
        last_fields, events_after_close = _last_message_at_or_before(
            day.message_path,
            scheduled_close,
        )
        last_event = float(last_fields[0])
        end_gap = round(max(0.0, scheduled_close - last_event), 9)
        complete = end_gap <= policy.maximum_session_end_gap_seconds
        declared_exclusion = day.date in policy.source_exclusions
        rows.append(
            {
                "date": day.date,
                "requested_end_seconds": requested_end,
                "scheduled_close_seconds": scheduled_close,
                "end_gap_seconds": end_gap,
                "events_after_scheduled_close": events_after_close,
                "coverage_status": "complete" if complete else "incomplete",
                "analysis_status": (
                    "declared_source_exclusion"
                    if declared_exclusion
                    else ("included" if complete else "unexplained_incomplete")
                ),
            }
        )
    return pd.DataFrame(rows)


def _validate_coverage_gate(
    coverage: pd.DataFrame,
    policy: AnalysisPolicy,
) -> None:
    observed_dates = set(coverage["date"])
    missing_exclusions = sorted(policy.source_exclusions - observed_dates)
    declared_complete = sorted(
        coverage.loc[
            coverage["analysis_status"].eq("declared_source_exclusion")
            & coverage["coverage_status"].eq("complete"),
            "date",
        ]
    )
    unexplained = coverage.loc[
        coverage["analysis_status"].eq("unexplained_incomplete"),
        ["date", "end_gap_seconds"],
    ]
    included_sessions = int(coverage["analysis_status"].eq("included").sum())
    delivered_sessions = len(coverage)
    counts_match = (
        delivered_sessions == policy.expected_delivered_sessions
        and included_sessions == policy.expected_included_sessions
    )
    if (
        not missing_exclusions
        and not declared_complete
        and unexplained.empty
        and counts_match
    ):
        return

    failures: list[str] = []
    if not counts_match:
        failures.append(
            "analysis universe has "
            f"{delivered_sessions} delivered and {included_sessions} included "
            "sessions; expected "
            f"{policy.expected_delivered_sessions} and "
            f"{policy.expected_included_sessions}"
        )
    if missing_exclusions:
        failures.append(f"declared exclusions absent from source: {missing_exclusions}")
    if declared_complete:
        failures.append(
            f"declared exclusions now pass the coverage gate: {declared_complete}"
        )
    if not unexplained.empty:
        details = ", ".join(
            f"{row.date} ({row.end_gap_seconds:.3f}s)"
            for row in unexplained.itertuples(index=False)
        )
        failures.append(f"undeclared incomplete sessions: {details}")
    raise ValueError("LOBSTER session coverage failure; " + "; ".join(failures))


def audit_session_coverage(
    raw_dir: Path | str,
    symbol: str = "TSLA",
    year: int = 2019,
    analysis_policy: AnalysisPolicy | Path | str = DEFAULT_ANALYSIS_POLICY,
) -> pd.DataFrame:
    """Audit whether requested daily files reach their theoretical end time."""

    policy = _policy(analysis_policy)
    validate_policy_scope(policy, symbol, year)
    pairs = [
        pair
        for pair in discover_daily_files(raw_dir, symbol=symbol)
        if pair.date.startswith(f"{year}-")
    ]
    if not pairs:
        raise FileNotFoundError(f"No {symbol} sessions found for {year}")
    return _audit_daily_file_coverage(pairs, policy)


def run_session_coverage_audit(
    raw_dir: Path | str,
    results_dir: Path | str,
    symbol: str = "TSLA",
    year: int = 2019,
    analysis_policy: AnalysisPolicy | Path | str = DEFAULT_ANALYSIS_POLICY,
) -> dict[str, object]:
    """Persist the daily coverage audit and return its compact summary."""

    policy = _policy(analysis_policy)
    coverage = audit_session_coverage(
        raw_dir,
        symbol=symbol,
        year=year,
        analysis_policy=policy,
    )
    _validate_coverage_gate(coverage, policy)
    output = Path(results_dir)
    output.mkdir(parents=True, exist_ok=True)
    coverage_path = output / "session_coverage.csv"
    temporary = coverage_path.with_suffix(".csv.tmp")
    coverage.to_csv(temporary, index=False)
    temporary.replace(coverage_path)
    included = coverage["analysis_status"].eq("included")
    declared = coverage["analysis_status"].eq("declared_source_exclusion")
    unexplained = coverage["analysis_status"].eq("unexplained_incomplete")
    return {
        "symbol": symbol,
        "year": year,
        "maximum_end_gap_seconds": policy.maximum_session_end_gap_seconds,
        "delivered_sessions": len(coverage),
        "included_sessions": int(included.sum()),
        "declared_source_exclusions": int(declared.sum()),
        "unexplained_incomplete_sessions": int(unexplained.sum()),
        "declared_source_exclusion_dates": coverage.loc[declared, "date"].tolist(),
    }


def _timestamps(date: str, seconds: pd.Series) -> pd.Series:
    base = pd.Timestamp(date)
    return base + pd.to_timedelta(seconds, unit="s")


def _read_aligned_best_book(
    day: DailyFiles,
    policy: AnalysisPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _validate_message_field_counts(day.message_path)
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
    in_session = messages["seconds"].le(_scheduled_end_seconds(day, policy))
    return (
        messages.loc[in_session].reset_index(drop=True),
        book.loc[in_session].reset_index(drop=True),
    )


def read_visible_executions(
    day: DailyFiles,
    analysis_policy: AnalysisPolicy | Path | str = DEFAULT_ANALYSIS_POLICY,
) -> pd.DataFrame:
    """Read visible fills and attach the pre-event best bid and ask."""

    messages, book = _read_aligned_best_book(day, _policy(analysis_policy))
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
    executions["opposite_best_size_before"] = np.where(
        executions["trade_sign"].eq(1),
        executions["ask_size_1_before"],
        executions["bid_size_1_before"],
    )

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
        opposite_best_size_before=("opposite_best_size_before", "first"),
    ).reset_index()
    market_orders["execution_price_vwap"] = (
        market_orders.pop("weighted_notional") / market_orders["size"]
    )
    return market_orders.sort_values(["date", "first_event_row"]).reset_index(drop=True)


def read_scaling_transactions(
    day: DailyFiles,
    analysis_policy: AnalysisPolicy | Path | str = DEFAULT_ANALYSIS_POLICY,
) -> pd.DataFrame:
    """Build type-4/type-5 transactions for the scaling analysis.

    The aggressor sign is inferred from execution price relative to the
    pre-event mid-price. Executions exactly at the midpoint are unsigned and
    are excluded.
    """

    messages, book = _read_aligned_best_book(day, _policy(analysis_policy))
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
    reader: Callable[[DailyFiles, AnalysisPolicy], pd.DataFrame],
    symbol: str,
    year: int,
    analysis_policy: AnalysisPolicy | Path | str,
) -> dict[str, int]:
    policy = _policy(analysis_policy)
    validate_policy_scope(policy, symbol, year)
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
    coverage = _audit_daily_file_coverage(pairs, policy)
    _validate_coverage_gate(coverage, policy)
    included_dates = set(
        coverage.loc[coverage["analysis_status"].eq("included"), "date"]
    )
    included_pairs = [pair for pair in pairs if pair.date in included_dates]

    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for pair in included_pairs:
            frame = reader(pair, policy)
            row_count += len(frame)
            writer = _append_parquet(writer, temporary, frame)
    finally:
        if writer is not None:
            writer.close()

    temporary.replace(output_path)
    return {
        "delivered_sessions": len(pairs),
        "sessions": len(included_pairs),
        "declared_source_exclusions": len(pairs) - len(included_pairs),
        "rows": row_count,
    }


def prepare_visible_market_orders(
    raw_dir: Path | str,
    output: Path | str,
    symbol: str = "TSLA",
    year: int = 2019,
    analysis_policy: AnalysisPolicy | Path | str = DEFAULT_ANALYSIS_POLICY,
) -> dict[str, int]:
    """Reconstruct and persist timestamp-aggregated visible market orders."""

    def reader(pair: DailyFiles, policy: AnalysisPolicy) -> pd.DataFrame:
        return aggregate_visible_market_orders(read_visible_executions(pair, policy))

    return _prepare_annual_table(
        raw_dir,
        output,
        reader,
        symbol,
        year,
        analysis_policy,
    )


def prepare_scaling_transactions(
    raw_dir: Path | str,
    output: Path | str,
    symbol: str = "TSLA",
    year: int = 2019,
    analysis_policy: AnalysisPolicy | Path | str = DEFAULT_ANALYSIS_POLICY,
) -> dict[str, int]:
    """Persist timestamp-aggregated type-4/type-5 transactions."""

    return _prepare_annual_table(
        raw_dir,
        output,
        lambda pair, policy: read_scaling_transactions(pair, policy),
        symbol,
        year,
        analysis_policy,
    )


def required_columns(table: pd.DataFrame, columns: Iterable[str]) -> None:
    """Raise a readable error when an input table misses its data contract."""

    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")
