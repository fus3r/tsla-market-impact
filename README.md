# Signed order count in aggregate price impact

This repository contains my analysis of reconstructed TSLA market orders from 252 NASDAQ sessions in 2019. The question is simple: after observing signed share volume over a short window, does the number of buyer and seller initiated orders add information about the mid-price change?

It does in this sample. The test set is the final 20% of trading dates, with no trimming of the response.

| Model | Holdout R² | Holdout MSE |
|---|---:|---:|
| Signed volume | 0.118 | 147.23 |
| Nonlinear volume terms | 0.298 | 117.18 |
| Volume terms and raw signed count | 0.359 | 106.96 |
| Volume and count transforms | 0.360 | 106.89 |

The last model cuts MSE by 27.4% relative to signed volume. A bootstrap over complete test dates gives an interval of 26.5% to 28.4% for that reduction. Both the order-flow variables and the price change are measured inside the same window, so this is a conditional impact model rather than a pre-trade forecast.

The scaling analysis is separate. It reproduces the curve-collapse construction of Patzelt and Bouchaud on TSLA 2019. The reported values 0.9963 and 0.9978 are in-sample fits of estimated scale parameters, not prediction scores.

[Read the report](report/tsla-market-impact.pdf).

## Files

- `src/tsla_market_impact/` contains the LOBSTER reconstruction and analysis code.
- `results/` contains aggregate tables. Licensed rows are not included.
- `report/` contains the LaTeX source, figures, references, and compiled PDF.
- `tests/` checks the data alignment, event-time windows, chronological splits, bootstrap, and scaling fits with synthetic data.

## Data

The code expects one LOBSTER message file and one order-book file per session. For example:

```text
TSLA_2019-01-02_34200000_57600000_message_2.csv
TSLA_2019-01-02_34200000_57600000_orderbook_2.csv
```

Visible executions are event type 4. Fills with the same timestamp and initiating side are grouped into reconstructed market orders. The scaling calculation also uses hidden executions, event type 5, and infers their side from the execution price relative to the pre-event midpoint.

Some delivered message files contain an optional seventh field such as `null` or a participant code. The Python reader consumes the six documented fields and ignores any trailing field.

LOBSTER data are licensed. Raw CSV files and row-level derived tables must stay outside this repository.

## Running the analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Build the two annual transaction tables:

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

## Origin and credit

The work began in the CentraleSupélec course project *Liquidity Games in High-Frequency Markets*, supervised by Anastasia Bugaenko of Capital Fund Management. The group presentation was prepared by L. Cohen, R. Darwish, A. Pariente, H. Rohrbach, and R. Sithisak.

I later rebuilt the evaluation, wrote the package in this repository, and prepared the report. The original collaborative repository is separate.

No software license is granted. LOBSTER's terms govern the underlying data.
