# TSLA market impact and depth-aware order-book replay

This repository combines an empirical analysis of reconstructed TSLA market
orders with a C++20 level-2 replay and audit engine. The source delivery contains
252 NASDAQ sessions in 2019; the shared source policy retains 249 and declares
the three unavailable truncated pairs.

The first question asks whether signed order count adds information about a
same-window mid-price change after signed share volume is known. A separate
forward experiment asks whether current queue imbalance and causal top-of-book
order flow predict the direction of the next mid-price change. A marketable
markout gate then asks whether those direction scores survive the displayed
spread and an explicit decision-to-entry delay. A second gate samples one
fixed clock-time landmark per price spell so that candidate decisions no
longer share a terminal price move.

It does in this sample. The fixed test period begins on 18 October, after 198
included training dates, and contains 51 dates with no trimming of the response.

| Model | Holdout R² | Holdout MSE |
|---|---:|---:|
| Signed volume | 0.118 | 147.23 |
| Nonlinear volume terms | 0.298 | 117.16 |
| Volume terms and raw signed count | 0.359 | 106.93 |
| Volume and count transforms | 0.360 | 106.86 |

Adding raw signed count to the nonlinear volume model cuts MSE by 8.7%. A
bootstrap over complete test dates gives an interval of 8.0% to 9.5%. The full
model's total reduction relative to linear signed volume is 27.4%, but most of
that gap comes from the volume transforms. Both the order-flow variables and
the price change are measured inside the same window, so this is a conditional
impact model rather than a pre-trade forecast.

The next-move experiment is forward-looking. Queue imbalance uses the current
displayed sizes. OFI accumulates top-of-book changes only since the current
mid-price was established, then divides by current displayed depth and applies
the fixed transform `2 * atan(x) / pi`. Four logistic specifications are fit on
the first 198 included dates and scored on the final 51 without refitting.

| Spread | Signal | Test observations | ROC-AUC | Relative Brier reduction |
|---|---|---:|---:|---:|
| All | Queue | 5,489,604 | 0.538 | 0.12% |
| All | OFI | 5,489,604 | 0.616 | 5.61% |
| All | Queue + OFI | 5,489,604 | 0.627 | 5.61% |
| One tick | Queue | 61,829 | 0.627 | 4.70% |
| One tick | OFI | 61,829 | 0.694 | 12.06% |
| One tick | Queue + OFI | 61,829 | 0.752 | 18.24% |

The combined-model interval is 5.36% to 5.84% for all spreads and 16.33% to
19.97% for one-tick states under 10,000 resamples of complete test dates. Queue
imbalance adds no detectable improvement over OFI in the pooled sample. In the
exploratory one-tick subset, the combined model improves on OFI by 7.02%, with
an interval of 5.22% to 8.79%. The pooled and one-tick conclusions are unchanged
with 21, 31, and 41 bins per feature axis. Several states share one next-move
label, OFI depends on the waiting horizon, and a direction score before fees,
latency, and queue position is not a trading strategy.

The markout diagnostic converts the combined signal into buy/sell decisions,
prices a one-unit marketable order at the displayed best quote, and marks it at
the next mid-price change. Confidence cutoffs are weighted quantiles fixed on
the first 198 included dates. For the 10% train-confidence rule:

| Signal state | Entry latency | Executable test signals | Stale | Net markout |
|---|---:|---:|---:|---:|
| All spreads | 0 us | 555,039 | 0.0% | -1.254 bp [-1.364, -1.156] |
| One-tick spread | 0 us | 5,787 | 0.0% | 0.399 bp [0.298, 0.500] |
| One-tick spread | 10 us | 3,367 | 41.8% | 0.302 bp [0.135, 0.453] |
| One-tick spread | 100 us | 1,311 | 77.3% | -0.035 bp [-0.422, 0.330] |

The intervals resample complete test dates. The pre-specified 5% tail remains
positive at 100 us, at 0.167 bp [0.075, 0.267], but only 482 test signals are
executable; its 1 ms interval includes zero. These state-level opportunities
overlap within constant-mid-price spells and are not additive returns. The
diagnostic assumes a one-unit fill, omits fees, impact, inventory, capital, and
risk limits, and applies one aggregate delay to LOBSTER event time. It is not a
backtest or a deployable strategy result.

The state-level diagnostic above gives several decisions in one
constant-mid-price spell. The non-overlapping protocol instead observes the
prevailing book exactly 100 us after each spell begins and discards spells that
end at or before that landmark. This leaves 9,847,952 of 15,263,228 completed
price spells (64.5%). The combined model reaches ROC-AUC 0.555 and a 0.91% Brier
reduction [0.83%, 0.98%] across all spreads; the exploratory one-tick subset
reaches 0.650 and 7.00% [5.65%, 8.45%]. For the target 10% train-confidence
rule, grid ties raise achieved train coverage to 13.3% and 13.5% respectively:

| Landmark sample | Post-landmark latency | Executable test signals | Stale | Net markout |
|---|---:|---:|---:|---:|
| All spreads | 0 us | 248,800 | 0.0% | -1.909 bp [-2.074, -1.752] |
| One-tick spread | 0 us | 1,120 | 0.0% | 0.107 bp [0.045, 0.179] |
| One-tick spread | 10 us | 1,006 | 10.2% | 0.059 bp [0.001, 0.125] |
| One-tick spread | 100 us | 689 | 38.5% | -0.095 bp [-0.155, -0.035] |

Each eligible spell contributes at most one decision, but this is still a
markout study rather than a backtest: the terminal midpoint is not an
executable exit, and fees, impact, fill uncertainty, inventory, capital, and
risk limits remain absent. This filters short spells and changes the fitted
confidence cutoff, so it is not a paired estimate of the state-level gate. The
one-tick stratum remains exploratory on one stock-year.

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
contains only these aggregate diagnostics, 15,263,228 mid-price changes, and
event-type counts.

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
  benchmark, daily queue, joint queue/OFI, state-level markout, and price-spell
  landmark exporters, and the [finite-depth transition contract](cpp/README.md).
- `src/tsla_market_impact/` contains the LOBSTER reconstruction, impact study,
  chronological queue/OFI ablation, and marketable-markout and price-spell
  landmark analyses.
- `results/` contains aggregate tables, including the 252-row coverage audit
  and annual level-2 replay, benchmark, queue-model, order-flow, markout, and
  landmark results. Licensed rows are not included.
- `report/` contains the LaTeX source, figures, references, and compiled PDF.
- `tests/` checks source integrity, data alignment, event-time windows,
  chronological splits, complete-date bootstrap, queue and markout evaluation,
  one-signal-per-spell deadlines, and scaling fits, plus a complete CLI run on
  a hand-written redistributable session.

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

The C++ source-policy, decoder, and end-to-end CLI checks use only synthetic
fixtures:

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
  --queue-bins data/processed/queue-imbalance-bins.csv \
  --order-flow-bins data/processed/order-flow-bins-31.csv \
  --order-flow-grid 31 \
  --markout-bins data/processed/marketable-markout-bins-31.csv \
  --markout-latencies-us 0,10,100,1000,10000 \
  --landmark-bins data/processed/price-spell-landmark-bins-31.csv \
  --landmark-age-us 100 \
  --landmark-latencies-us 0,10,100,1000,10000
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

Fit the queue, OFI, and combined models on the same fixed split:

```bash
tsla-impact analyze-order-flow \
  --bins data/processed/order-flow-bins-31.csv
```

The grid-resolution check consumes independently exported aggregate grids:

```bash
tsla-impact analyze-order-flow-grid \
  --bins 21=data/processed/order-flow-bins-21.csv \
  --bins 31=data/processed/order-flow-bins-31.csv \
  --bins 41=data/processed/order-flow-bins-41.csv
```

The joint grid remains local. The committed model, comparison, calibration,
robustness, and vector-figure artifacts contain aggregate evidence only.

Apply the train-defined confidence rules to the displayed-spread and latency
diagnostic:

```bash
tsla-impact analyze-markouts \
  --bins data/processed/marketable-markout-bins-31.csv
```

The daily markout grid remains local. The committed 300-row metric table, model
metadata, and vector figure contain aggregate evidence only.

Apply the same fixed split and train-defined confidence rules to one
100-microsecond landmark per eligible price spell:

```bash
tsla-impact analyze-landmarks \
  --bins data/processed/price-spell-landmark-bins-31.csv
```

The licensed-data-derived landmark grid remains local. The committed 300-row
metric table, model metadata, and vector figure contain aggregate evidence
only.

## Origin and credit

The work began in the CentraleSupélec course project *Liquidity Games in High-Frequency Markets*, supervised by Anastasia Bugaenko of Capital Fund Management. The group presentation was prepared by L. Cohen, R. Darwish, A. Pariente, H. Rohrbach, and R. Sithisak.

I later rebuilt the evaluation, wrote the package in this repository, and prepared the report. The original collaborative repository is separate.

No software license is granted. LOBSTER's terms govern the underlying data.
