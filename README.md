# TSLA market impact and depth-aware order-book replay

This repository combines an empirical analysis of reconstructed TSLA market
orders with a C++20 level-2 replay and audit engine. The source delivery contains
252 NASDAQ sessions in 2019; the shared source policy retains 249 and declares
the three unavailable truncated pairs.

The impact question is simple: after observing signed share volume over a short
window, does the number of buyer- and seller-initiated orders add information
about the mid-price change?

It does in this sample. The fixed test period begins on 18 October, after 198
included training dates, and contains 51 dates with no trimming of the response.

| Model | Holdout R² | Holdout MSE |
|---|---:|---:|
| Signed volume | 0.118 | 147.23 |
| Nonlinear volume terms | 0.298 | 117.16 |
| Volume terms and raw signed count | 0.359 | 106.93 |
| Volume and count transforms | 0.360 | 106.86 |

The last model cuts MSE by 27.4% relative to signed volume. A bootstrap over
complete test dates gives an interval of 26.5% to 28.5% for that reduction.
Both the order-flow variables and the price change are measured inside the same
window, so this is a conditional impact model rather than a pre-trade forecast.

The separate queue-imbalance experiment is forward-looking: the displayed
level-1 state predicts the direction of the next same-session mid-price change.
The same fixed calendar boundary leaves 198 included training dates and 51 test
dates.

| Best-quote state sample | Test observations | ROC-AUC | Brier reduction |
|---|---:|---:|---:|
| All spreads | 5,489,604 | 0.538 | 0.12% |
| One-tick spread | 61,829 | 0.626 | 4.69% |

The all-spread interval is -0.03% to 0.29%, so the pooled signal is weak. The
one-tick interval is 2.32% to 7.00% under 10,000 resamples of complete test
dates. That spread-conditioned result is exploratory, and a direction score
before fees, latency, and queue position is not a trading strategy.

The scaling analysis is separate. It applies the curve-collapse construction of
Patzelt and Bouchaud to this included TSLA 2019 sample. The reported values
0.9958 and 0.9975 are in-sample fits of estimated scale parameters, not
prediction scores or an external replication.

[Read the report](report/tsla-market-impact.pdf).

## Source-data status

The source delivery contains 252 message/order-book pairs. A schedule-aware
audit finds that the pairs for 9 January, 8 March, and 18 September stop before
the Nasdaq close. The original supervised-project access has ended and
replacement files are unavailable, so
[`analysis-policy.conf`](analysis-policy.conf) declares exactly those three
dates as source exclusions.

Python and C++ consume the same policy. The Python annual gate requires 252
delivered pairs, 249 included sessions, three declared exclusions, and no
unexplained coverage failure; the C++ annual replay enforces the same close,
exclusion, and universe-count rules before producing an aggregate diagnostic.
There is no generic skip-incomplete mode. The official 13:00 closes on 3 July,
29 November, and 24 December pass the 60-second coverage threshold; rows after
13:00 are excluded from preparation and replay.

The calendar boundaries remain fixed rather than being recalculated after the
exclusions: development ends on 6 August, selection runs from 7 August through
17 October, and test starts on 18 October. This preserves 148/50/51 nested dates
and the 198/51 outer train/test split.

All committed numerical outputs were rebuilt on the 249-session universe. The
aggregate audit is in
[`results/session_coverage.csv`](results/session_coverage.csv). This repository
does not claim complete 2019 coverage or external replication.

The level-2 transition audit covers 38,516,432 included events. After one seed
per session, it classifies 28,544,808 transitions as exact and 9,971,375 as
depth-censored, with no observable mismatch, unsupported event, or invalid
snapshot. The committed
[`results/lobster_replay_audit.json`](results/lobster_replay_audit.json)
contains only these aggregate diagnostics and event-type counts.

## C++ annual replay benchmark

The benchmark separates complete CSV decoding from replay over the already
resident 64-byte event records. The replay audits the same finite-depth
transitions, validates every snapshot, and maintains a checksum so repeated
passes cannot silently disagree.

| Annual benchmark | Value |
|---|---:|
| Included events | 38,516,432 |
| Resident event payload | 2.47 GB |
| Median resident replay | 9.22 ns/event |
| p95 resident replay | 9.51 ns/event |
| Median single-thread throughput | 108.46 million events/s |
| Median CSV decode | 115.19 ns/event |

These figures are from eleven complete measured replay passes after two
warm-ups on a MacBook Pro with an Apple M4 Pro and 48 GB of memory, built with
Apple Clang 17 in Release mode. The decoder median is from three complete
passes. The operating-system page cache was not flushed: the first decode took
277.33 ns/event, and all three durations remain visible in
[`results/lobster_replay_benchmark.json`](results/lobster_replay_benchmark.json).
Each ns/event value is a full-pass duration divided by the event count. The p95
is therefore a percentile across run averages, not per-event tail latency or
wire-to-wire trading latency.

## Files

- `analysis-policy.conf` is the shared Python/C++ source and calendar policy.
- `cpp/` contains the fixed-width decoder, annual source-integrity gate, replay
  benchmark, daily queue-bin exporter, and the
  [finite-depth transition contract](cpp/README.md).
- `src/tsla_market_impact/` contains the LOBSTER reconstruction, impact study,
  and chronological next-move evaluation.
- `results/` contains aggregate tables, including the 252-row coverage audit
  and annual level-2 replay, benchmark, and queue-model results. Licensed rows
  are not included.
- `report/` contains the LaTeX source, figures, references, and compiled PDF.
- `tests/` checks source integrity, data alignment, event-time windows,
  chronological splits, complete-date bootstrap, queue evaluation, and scaling
  fits with synthetic data.

## Data

The code expects one LOBSTER message file and one order-book file per session. For example:

```text
TSLA_2019-01-02_34200000_57600000_message_2.csv
TSLA_2019-01-02_34200000_57600000_orderbook_2.csv
```

Visible executions are event type 4. Fills with the same timestamp and initiating side are grouped into reconstructed market orders. The scaling calculation also uses hidden executions, event type 5, and infers their side from the execution price relative to the pre-event midpoint.

Some delivered message files contain an optional seventh field such as `null` or a participant code. The Python and C++ readers consume the six documented fields and ignore that optional field.

LOBSTER data are licensed. Raw CSV files and row-level derived tables must stay outside this repository.

## Running the analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Audit the delivered sessions before building either annual table:

```bash
tsla-impact audit-session-coverage \
  --raw-dir data/raw/TSLA
```

The command writes only the aggregate `results/session_coverage.csv` artifact
after the exact 252/249/3 gate passes. Build the two annual transaction tables:

```bash
tsla-impact prepare-visible \
  --raw-dir data/raw/TSLA \
  --output data/processed/visible-market-orders.parquet

tsla-impact prepare-scaling \
  --raw-dir data/raw/TSLA \
  --output data/processed/scaling-transactions.parquet
```

Run the analysis. Aggregate tables go to `results/` and the vector figures go to `report/figures/` by default.

```bash
tsla-impact analyze \
  --visible data/processed/visible-market-orders.parquet \
  --scaling data/processed/scaling-transactions.parquet
```

Compile the report with Tectonic:

```bash
tectonic report/tsla-market-impact.tex --outdir report
```

The Python tests do not need the licensed data:

```bash
ruff check .
pytest
```

The C++ source-policy and decoder checks use only synthetic fixtures:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Regenerate the annual C++ benchmark after the tests pass:

```bash
./build/lobster_replay \
  --raw-dir data/raw/TSLA \
  --analysis-policy analysis-policy.conf \
  --decode-runs 3 \
  --warmup-runs 2 \
  --replay-runs 11 \
  --machine 'MacBook Pro Mac16,8; Apple M4 Pro; 48 GB; macOS 15.7.7' \
  --json results/lobster_replay_benchmark.json \
  --queue-bins data/processed/queue-imbalance-bins.csv
```

The command audits every delivered pair, refuses any undeclared coverage
failure, filters post-close rows, and exits unsuccessfully if the replay finds
an observable mismatch, unsupported event, invalid snapshot, or checksum
disagreement between repetitions. Use a truthful local machine label when
running it on another host.

Fit the queue model on the first 198 included dates and score the final 51:

```bash
tsla-impact analyze-queue \
  --bins data/processed/queue-imbalance-bins.csv
```

The daily intermediate contains only aggregate counts but remains local. The
committed JSON, metrics, calibration table, and vector figure are compact
outputs of that fixed protocol.

## Origin and credit

The work began in the CentraleSupélec course project *Liquidity Games in High-Frequency Markets*, supervised by Anastasia Bugaenko of Capital Fund Management. The group presentation was prepared by L. Cohen, R. Darwish, A. Pariente, H. Rohrbach, and R. Sithisak.

I later rebuilt the evaluation, wrote the package in this repository, and prepared the report. The original collaborative repository is separate.

No software license is granted. LOBSTER's terms govern the underlying data.
