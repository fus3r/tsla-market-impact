"""Shared source-availability and calendar policy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

DEFAULT_ANALYSIS_POLICY = Path("analysis-policy.conf")


@dataclass(frozen=True)
class AnalysisPolicy:
    """Canonical source exclusions, market closes, and evaluation boundaries."""

    symbol: str
    year: int
    source_availability: str
    maximum_session_end_gap_seconds: float
    expected_delivered_sessions: int
    expected_included_sessions: int
    expected_development_sessions: int
    expected_selection_sessions: int
    expected_test_sessions: int
    source_exclusions: frozenset[str]
    early_closes_seconds: dict[str, float]
    development_end: str
    selection_start: str
    selection_end: str
    test_start: str


def _iso_date(value: str, key: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Invalid {key} date in analysis policy: {value!r}") from error
    return parsed.isoformat()


def load_analysis_policy(path: Path | str = DEFAULT_ANALYSIS_POLICY) -> AnalysisPolicy:
    """Read the deliberately small key-value policy consumed by Python and C++."""

    policy_path = Path(path)
    values: dict[str, str] = {}
    exclusions: set[str] = set()
    early_closes: dict[str, float] = {}
    repeated = {"source_exclusion", "early_close"}
    allowed = {
        "symbol",
        "year",
        "source_availability",
        "maximum_session_end_gap_seconds",
        "expected_delivered_sessions",
        "expected_included_sessions",
        "expected_development_sessions",
        "expected_selection_sessions",
        "expected_test_sessions",
        "source_exclusion",
        "early_close",
        "development_end",
        "selection_start",
        "selection_end",
        "test_start",
    }
    for line_number, raw_line in enumerate(policy_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise ValueError(
                f"Malformed analysis policy line {line_number} in {policy_path}"
            )
        if key not in allowed:
            raise ValueError(f"Unknown analysis policy key {key!r}")
        if key not in repeated:
            if key in values:
                raise ValueError(f"Duplicate analysis policy key {key!r}")
            values[key] = value
        elif key == "source_exclusion":
            exclusion = _iso_date(value, key)
            if exclusion in exclusions:
                raise ValueError(f"Duplicate source exclusion {exclusion}")
            exclusions.add(exclusion)
        else:
            close_date, comma, seconds_text = value.partition(",")
            if not comma:
                raise ValueError(
                    f"Malformed early_close on line {line_number} in {policy_path}"
                )
            close_date = _iso_date(close_date, key)
            if close_date in early_closes:
                raise ValueError(f"Duplicate early close {close_date}")
            seconds = float(seconds_text)
            if not 0 < seconds <= 24 * 60 * 60:
                raise ValueError(f"Invalid early close time for {close_date}")
            early_closes[close_date] = seconds

    required = {
        "symbol",
        "year",
        "source_availability",
        "maximum_session_end_gap_seconds",
        "expected_delivered_sessions",
        "expected_included_sessions",
        "expected_development_sessions",
        "expected_selection_sessions",
        "expected_test_sessions",
        "development_end",
        "selection_start",
        "selection_end",
        "test_start",
    }
    missing = required - values.keys()
    if missing or not exclusions or not early_closes:
        details = sorted(missing)
        if not exclusions:
            details.append("source_exclusion")
        if not early_closes:
            details.append("early_close")
        raise ValueError(f"Analysis policy is missing required entries: {details}")

    maximum_gap = float(values["maximum_session_end_gap_seconds"])
    if maximum_gap <= 0:
        raise ValueError("maximum_session_end_gap_seconds must be positive")
    expected_delivered = int(values["expected_delivered_sessions"])
    expected_included = int(values["expected_included_sessions"])
    expected_development = int(values["expected_development_sessions"])
    expected_selection = int(values["expected_selection_sessions"])
    expected_test = int(values["expected_test_sessions"])
    if (
        expected_included < 1
        or expected_delivered <= expected_included
        or expected_delivered - expected_included != len(exclusions)
    ):
        raise ValueError("Analysis policy session counts are inconsistent")
    if (
        min(expected_development, expected_selection, expected_test) < 0
        or expected_development + expected_selection + expected_test
        != expected_included
    ):
        raise ValueError("Analysis policy evaluation counts are inconsistent")
    development_end = _iso_date(values["development_end"], "development_end")
    selection_start = _iso_date(values["selection_start"], "selection_start")
    selection_end = _iso_date(values["selection_end"], "selection_end")
    test_start = _iso_date(values["test_start"], "test_start")
    if not development_end < selection_start <= selection_end < test_start:
        raise ValueError("Analysis policy calendar boundaries are not chronological")

    return AnalysisPolicy(
        symbol=values["symbol"],
        year=int(values["year"]),
        source_availability=values["source_availability"],
        maximum_session_end_gap_seconds=maximum_gap,
        expected_delivered_sessions=expected_delivered,
        expected_included_sessions=expected_included,
        expected_development_sessions=expected_development,
        expected_selection_sessions=expected_selection,
        expected_test_sessions=expected_test,
        source_exclusions=frozenset(exclusions),
        early_closes_seconds=early_closes,
        development_end=development_end,
        selection_start=selection_start,
        selection_end=selection_end,
        test_start=test_start,
    )


def validate_policy_scope(
    policy: AnalysisPolicy,
    symbol: str,
    year: int,
) -> None:
    """Reject accidental use of the TSLA 2019 policy on another dataset."""

    if policy.symbol != symbol or policy.year != year:
        raise ValueError(
            "Analysis policy scope mismatch: "
            f"expected {policy.symbol} {policy.year}, received {symbol} {year}"
        )


def split_index_for_test_start(dates: list[str], test_start: str) -> int:
    """Return the fixed calendar split, requiring its first test date to exist."""

    try:
        split = dates.index(test_start)
    except ValueError as error:
        raise ValueError(f"Required test start date {test_start} is absent") from error
    if split == 0:
        raise ValueError("Fixed test boundary leaves no training date")
    return split


def validate_analysis_universe(
    values: Iterable[object],
    policy: AnalysisPolicy,
) -> list[str]:
    """Require an aggregate input to match the declared evaluation calendar."""

    observed: set[str] = set()
    for value in values:
        normalized = _iso_date(str(value), "analysis input")
        if date.fromisoformat(normalized).year != policy.year:
            raise ValueError(
                f"Analysis input date {normalized} is outside policy year {policy.year}"
            )
        observed.add(normalized)

    excluded = sorted(observed & policy.source_exclusions)
    if excluded:
        raise ValueError(f"Analysis input contains declared source exclusions: {excluded}")
    if len(observed) != policy.expected_included_sessions:
        raise ValueError(
            "Analysis input session count mismatch: "
            f"expected {policy.expected_included_sessions}, observed {len(observed)}"
        )

    required_boundaries = {
        policy.development_end,
        policy.selection_start,
        policy.selection_end,
        policy.test_start,
    }
    missing_boundaries = sorted(required_boundaries - observed)
    if missing_boundaries:
        raise ValueError(
            f"Analysis input is missing policy boundary sessions: {missing_boundaries}"
        )

    development = sum(value <= policy.development_end for value in observed)
    selection = sum(
        policy.selection_start <= value <= policy.selection_end for value in observed
    )
    test = sum(value >= policy.test_start for value in observed)
    assigned = development + selection + test
    expected = (
        policy.expected_development_sessions,
        policy.expected_selection_sessions,
        policy.expected_test_sessions,
    )
    observed_counts = (development, selection, test)
    if assigned != len(observed) or observed_counts != expected:
        raise ValueError(
            "Analysis input evaluation calendar mismatch: "
            f"expected development/selection/test {expected}, "
            f"observed {observed_counts}"
        )
    return sorted(observed)
