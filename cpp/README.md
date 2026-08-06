# Finite-depth replay contract

The C++ transition auditor compares each LOBSTER message with the previous and
current level-2 snapshots. It does not reconstruct order priority or the full
book.

LOBSTER snapshots contain only the requested depth. When a displayed level is
depleted, the surviving known level shifts inward, but the newly visible tail
may contain orders submitted while that price was outside level 2. Those
submissions were not observable in the preceding snapshot, so the auditor
checks the surviving prefix and then resumes from the current authoritative
snapshot.

Transitions have four outcomes:

- `exact`: the message and previous level-2 state fully determine the current
  snapshot;
- `depth_censored`: a depleted level exposes a previously unobservable tail,
  while every surviving known level agrees;
- `mismatch`: an observable price or size disagrees with the supplied snapshot;
- `unsupported`: the event type or direction is outside the replay contract.

Hidden executions, cross trades, and trading halts must leave the displayed
book unchanged. Snapshot validation separately requires positive prices and
sizes, increasing ask prices, decreasing bid prices, and a best bid below the
best ask.

## Queue-imbalance aggregates

For each state with a later same-session mid-price change, the exporter labels
the direction of that next change and calculates

```text
I = (best_bid_size - best_ask_size) /
    (best_bid_size + best_ask_size).
```

It writes counts by date, state sample, spread regime, and one of 101
equal-width imbalance bins. The `all_events` sample retains every labelled
state. The `best_quote_updates` sample retains the session's first displayed
state and later states only when a best price or size changes. Multiple states
can share one eventual price-move label, so Python resamples complete trading
dates rather than treating the states as independent trials.

Only the daily aggregate counts enter Python. Licensed message rows, order
identifiers, prices, and row-level book states remain local.

## Joint queue and order-flow aggregates

The second exporter compares queue imbalance with a causal top-of-book OFI
feature. Within each constant-mid-price run, it resets cumulative OFI to zero
and then adds the standard best-quote OFI increment after every observed
transition. At state \(t\), it reports the bounded pressure

```text
Z_t = 2 / pi * atan(cumulative_OFI_t /
                    (best_bid_size_t + best_ask_size_t)).
```

Both `Z_t` and queue imbalance lie in `[-1, 1]`. The exporter writes daily
counts on a fixed joint grid for states after best price or size updates,
separately by spread regime and next-mid-price direction. The default is 31
bins per axis. The fixed transform and grid use no late-year outcome or fitted
threshold, and only aggregate counts enter Python.

The last constant-price run in a session has no next-move label and is
excluded. Multiple states can share the same eventual direction, so the
statistical analysis resamples complete trading dates rather than treating
states as independent trials.

## Causal OFI horizon aggregates

The multi-horizon exporter tests whether the price-spell reset above can be
replaced by a fixed lookback. It emits the same queue/OFI grid and next-move
labels for nine definitions:

- cumulative OFI since the current midpoint was established;
- the most recent 1, 5, 20, or 100 best-quote OFI increments, including the
  current update;
- best-quote OFI increments in the preceding 10, 100, 1,000, or 10,000
  microseconds.

The fixed windows cross price-spell boundaries and include the update that
established the current midpoint. A clock window at time \(t\) uses increments
in \((t-h,t]\), so an increment exactly on the left boundary is excluded. Every
horizon must preserve the same date-level observation and direction counts;
Python rejects the aggregate file if that identity fails.

The analysis fits each fixed candidate through 6 August (148 included dates),
selects by Brier score from 7 August through 17 October (50 included dates),
refits the selected horizon on all 198 pre-test dates, and evaluates the final
51 dates once. The adaptive price-spell horizon is a reference rather than a
selection candidate. Only daily grid counts leave the licensed-data
environment.

## Marketable markout aggregates

The markout exporter applies the same causal queue/OFI grid to a hypothetical
one-unit marketable order at decision-to-entry latencies of 0, 10, 100, 1,000,
or 10,000 microseconds. At positive latency, a next move stamped at or before
the deadline makes the signal stale, and events stamped exactly at the deadline
do not update the entry book. Zero latency uses the signal's post-event state,
preserving file order. If the state survives, the position is marked at the
midpoint immediately after the next mid-price change.

Daily grid cells contain only counts, the sum of signed midpoint moves in basis
points, and the sum of displayed half-spreads. Python fits the direction model
and every confidence cutoff on the 198 pre-test dates, so no row-level book
state or late-date threshold enters the repository.

This is not an execution simulator. It assumes a one-unit fill at the displayed
best, omits fees and impact, and applies one aggregate delay to LOBSTER event
time without identifying feed, computation, transmission, or exchange
processing. Overlapping states remain separate diagnostics rather than additive
returns.

## Price-spell landmarks

The landmark exporter removes the within-spell overlap above. For each
constant-mid-price spell, it observes the last book state strictly before a
fixed clock-time landmark after the spell begins. A spell that ends at or
before the landmark produces no signal. Queue imbalance and cumulative OFI use
only events before that time, including the same reset and bounded transform as
the joint signal exporter.

The annual protocol fixes the landmark at 100 microseconds. Each eligible spell
contributes one signal, followed by separate entry latencies of 0, 10, 100,
1,000, and 10,000 microseconds. An entry deadline at or after the next mid-price
change is stale. Events stamped exactly at the observation or entry deadline
are excluded, preserving file order without looking past the deadline.

Price spells are sequential, so their markout intervals do not overlap. The
calculation is still not a backtest: entry assumes one unit at the displayed
best, while the terminal midpoint is a mark rather than an executable exit.
Fees, impact, fill uncertainty, inventory, capital, and risk limits remain
outside the model.

## Depth-constrained price-spell round trips

The round-trip exporter keeps the same 100-microsecond landmark, causal
features, and entry-latency grid, but replaces the terminal midpoint with an
opposite marketable order in the first snapshot after the next mid-price
change. It sweeps at most the two displayed levels at both endpoints for fixed
sizes of 1, 100, and 500 shares. A long path buys the entry asks and sells the
exit bids; a short path sells the entry bids and buys the exit asks.

Both long and short outcomes are aggregated for every signal cell before the
direction model is fitted. Python can therefore apply a train-only decision
rule without exporting row-level outcomes. A path is capacity-censored when
either leg lacks enough displayed level-2 size. The aggregate identities are

```text
signals = arrived + stale
arrived = action fills + action capacity-censored
quoted PnL = signed midpoint move - entry cost - exit cost.
```

Only the pooled sample and the pre-specified exploratory one-tick sample are
exported. The exit assumes zero reaction latency, displayed sizes are not
reserved, and fees and market impact beyond the two displayed levels are
omitted. This makes the quoted PnL an optimistic stress test rather than a fill
simulation or backtest.

## Annual replay and benchmark

The replay command discovers the level-2 pairs named by the shared
[`analysis-policy.conf`](../analysis-policy.conf), audits all 252 delivered
sessions against their scheduled close, and retains the 249 declared included
sessions. It discards rows after the three official 13:00 closes before replay.
An undeclared incomplete pair, a declared exclusion that becomes complete, or
an unexpected universe count is a hard error.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
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
  --ofi-horizon-bins data/processed/ofi-horizon-bins-31.csv \
  --ofi-quote-windows 1,5,20,100 \
  --ofi-clock-windows-us 10,100,1000,10000 \
  --markout-bins data/processed/marketable-markout-bins-31.csv \
  --markout-latencies-us 0,10,100,1000,10000 \
  --landmark-bins data/processed/price-spell-landmark-bins-31.csv \
  --landmark-age-us 100 \
  --landmark-latencies-us 0,10,100,1000,10000 \
  --round-trip-bins data/processed/price-spell-round-trip-bins-31.csv \
  --round-trip-sizes 1,100,500
```

The committed aggregate covers 38,516,432 events. Excluding the 249 session
seeds, 28,544,808 transitions are exact and 9,971,375 are depth-censored. The
audit finds no observable mismatch, unsupported event, or invalid snapshot.
The benchmark JSON contains aggregate counts, environment metadata, and
full-pass timings; licensed rows and paths remain local.

The benchmark reports two distinct costs:

- `decode` maps, parses, validates, and materializes every delivered CSV pair;
- `in_memory_replay` scans the included resident records, audits transitions,
  validates snapshots, and maintains a checksum.

The queue, order-flow, multi-horizon, markout, landmark, and round-trip exports
are separate passes after the measured replay and are not included in either
timing.

The committed Release benchmark used one thread on an Apple M4 Pro. Its
38,516,432 included events occupy 2,465,051,648 bytes as resident event
payload, excluding container and allocator overhead. Two unmeasured replay
passes preceded eleven measured passes. Every replay checksum agreed.

The operating-system page cache was not flushed between the three decode
passes, so the JSON retains every duration rather than hiding the slower first
pass. Each nanoseconds-per-event value is the elapsed time of one complete pass
divided by the included event count. The reported p95 is taken across these
full-pass averages; it is not event-level tail latency, exchange latency, or an
end-to-end trading measurement.
