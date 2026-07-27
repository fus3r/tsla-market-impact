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
