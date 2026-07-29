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
  --json results/lobster_replay_benchmark.json
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
