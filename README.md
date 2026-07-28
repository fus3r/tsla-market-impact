# Signed order count in aggregate price impact

This repository contains my analysis of reconstructed TSLA market orders from
249 included NASDAQ sessions in 2019. The question is simple: after observing
signed share volume over a short window, does the number of buyer- and
seller-initiated orders add information about the mid-price change?

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

## Files

- `analysis-policy.conf` is the shared Python/C++ source and calendar policy.
- `cpp/` contains the fixed-width decoder, annual source-integrity and replay
  audit, and the [finite-depth transition contract](cpp/README.md).
- `src/tsla_market_impact/` contains the LOBSTER reconstruction and analysis code.
- `results/` contains aggregate tables, including the 252-row coverage audit
  and annual level-2 replay diagnostic. Licensed rows are not included.
- `report/` contains the LaTeX source, figures, references, and compiled PDF.
- `tests/` checks source integrity, data alignment, event-time windows,
  chronological splits, bootstrap, and scaling fits with synthetic data.

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

Regenerate the annual C++ aggregate after the tests pass:

```bash
./build/lobster_replay \
  --raw-dir data/raw/TSLA \
  --analysis-policy analysis-policy.conf \
  --json results/lobster_replay_audit.json
```

The command audits every delivered pair, refuses any undeclared coverage
failure, filters post-close rows, and exits unsuccessfully if the replay finds
an observable mismatch, unsupported event, or invalid snapshot.

## Origin and credit

The work began in the CentraleSupélec course project *Liquidity Games in High-Frequency Markets*, supervised by Anastasia Bugaenko of Capital Fund Management. The group presentation was prepared by L. Cohen, R. Darwish, A. Pariente, H. Rohrbach, and R. Sithisak.

I later rebuilt the evaluation, wrote the package in this repository, and prepared the report. The original collaborative repository is separate.

No software license is granted. LOBSTER's terms govern the underlying data.
