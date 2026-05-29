"""
Read-only diagnostic: event-collapse analysis of a completed scan.

Answers one question before we touch any filter logic:
  Of the markets the filter stage evaluated, how many DISTINCT events are
  there, and does each event contain at least one structurally-tradeable
  rung? This tells us whether the 0% pass rate is a filter-ORDERING problem
  (ladders inflate rejects; a good rung exists but is never surfaced) or a
  threshold / inventory problem (no rung in the event is tradeable at all).

It reads the persisted market_audit table for one scan_run (latest by
default). It performs NO API calls and does NOT modify any data or filter
behaviour. Thresholds are read from the live Config so the gate checks here
mirror filters.py exactly (minus the orderbook-depth gate, which only runs on
the enriched candidate subset and so is never the binding reason in a bulk
scan).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NamedTuple, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import db
from config import get_config

# Structural gates re-implemented to match filters.py priority order, minus
# status/unsafe/blocked/category (already cleared to reach the filter stage)
# and orderbook depth/age (enrichment-limited, never binding in bulk).
_GATE_ORDER = ("time", "price", "spread", "volume", "liquidity")


class Thresholds(NamedTuple):
    min_mins: float
    max_mins: float
    min_yes: int
    max_yes: int
    max_spread_cents: int
    max_spread_pct: float
    min_volume: int
    min_liquidity: float

    @classmethod
    def from_cfg(cls, cfg: Any) -> "Thresholds":
        return cls(
            min_mins=cfg.min_minutes_to_expiry,
            max_mins=cfg.max_minutes_to_expiry,
            min_yes=cfg.min_yes_price,
            max_yes=cfg.max_yes_price,
            max_spread_cents=cfg.max_spread_cents,
            max_spread_pct=cfg.max_spread_pct,
            min_volume=cfg.min_volume_24h,
            min_liquidity=cfg.min_liquidity_dollars,
        )


def _gate_failure(row: dict[str, Any], thr: Thresholds) -> Optional[str]:
    """Return the first structural gate a rung fails, or None if it passes all."""
    mtc = _f(row.get("minutes_to_close"))
    yes_ask = _i(row.get("yes_ask"))
    spread_cents = _i(row.get("spread_cents"))
    spread_pct = _f(row.get("spread_pct"))
    volume_24h = _i(row.get("volume_24h"))
    liquidity = _f(row.get("liquidity_dollars"))

    if mtc < thr.min_mins or mtc > thr.max_mins:
        return "time"
    if yes_ask <= 0 or yes_ask < thr.min_yes or yes_ask > thr.max_yes:
        return "price"
    if spread_cents > thr.max_spread_cents or spread_pct > thr.max_spread_pct:
        return "spread"
    if volume_24h < thr.min_volume:
        return "volume"
    if liquidity < thr.min_liquidity:
        return "liquidity"
    return None


def _event_tradeable(rungs: list[dict[str, Any]], thr: Thresholds) -> bool:
    return any(_gate_failure(r, thr) is None for r in rungs)


def _f(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _i(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _atm_rung(rungs: list[dict[str, Any]]) -> dict[str, Any]:
    """Rung whose YES ask is closest to 50c (the at-the-money representative)."""
    return min(rungs, key=lambda r: abs(_i(r.get("yes_ask")) - 50))


def _maxvol_rung(rungs: list[dict[str, Any]]) -> dict[str, Any]:
    """Rung with the highest 24h volume (current dedup tie-break)."""
    return max(rungs, key=lambda r: _i(r.get("volume_24h")))


def main() -> None:
    args = _parse_args()
    cfg = get_config()
    db.init(cfg.db_path)

    scan_run_id = args.scan_run_id or _latest_scan_run_id()
    if not scan_run_id:
        print("No scan runs found in the audit DB.")
        return

    rows = db.get_market_audit(scan_run_id=scan_run_id, stage="all", outcome="all", limit=100000)
    rows = [r for r in rows if r.get("stage") in ("filter_passed", "filter_skipped")]
    if not rows:
        print(f"No filter-stage audit rows for scan_run_id={scan_run_id}.")
        print("(Was this scan run before market_audit persistence, or did it fail early?)")
        return

    by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_event[str(r.get("event_ticker") or r.get("ticker") or "")].append(r)

    n_rungs = len(rows)
    n_events = len(by_event)
    thr = Thresholds.from_cfg(cfg)

    if args.sweep:
        _print_sweep(by_event, thr, scan_run_id)
        return

    # Per-event verdicts
    events_with_tradeable_rung = 0
    atm_would_pass = 0
    maxvol_would_pass = 0
    atm_binding_gate: Counter[str] = Counter()
    event_first_gate: Counter[str] = Counter()  # best-achievable gate per event
    tradeable_examples: list[str] = []

    for event, rungs in by_event.items():
        gate_results = [(_gate_failure(r, thr), r) for r in rungs]
        passing = [r for g, r in gate_results if g is None]
        if passing:
            events_with_tradeable_rung += 1
            event_first_gate["<passes>"] += 1
            if len(tradeable_examples) < 10:
                best = min(passing, key=lambda r: abs(_i(r.get("yes_ask")) - 50))
                tradeable_examples.append(
                    f"{best.get('ticker')}  yes_ask={_i(best.get('yes_ask'))}c "
                    f"spread={_i(best.get('spread_cents'))}c vol={_i(best.get('volume_24h'))} "
                    f"({len(rungs)} rungs)"
                )
        else:
            # How far did the event's BEST rung get? Take the latest gate in the
            # pipeline that any rung reached before failing (max priority index).
            best_gate = max(
                (g for g, _ in gate_results if g),
                key=lambda g: _GATE_ORDER.index(g),
            )
            event_first_gate[best_gate] += 1

        atm_gate = _gate_failure(_atm_rung(rungs), thr)
        if atm_gate is None:
            atm_would_pass += 1
        else:
            atm_binding_gate[atm_gate] += 1

        if _gate_failure(_maxvol_rung(rungs), thr) is None:
            maxvol_would_pass += 1

    rung_dist = Counter(len(v) for v in by_event.values())

    _print_report(
        scan_run_id=scan_run_id,
        n_rungs=n_rungs,
        n_events=n_events,
        events_with_tradeable_rung=events_with_tradeable_rung,
        atm_would_pass=atm_would_pass,
        maxvol_would_pass=maxvol_would_pass,
        atm_binding_gate=atm_binding_gate,
        event_first_gate=event_first_gate,
        rung_dist=rung_dist,
        by_event=by_event,
        tradeable_examples=tradeable_examples,
        top_n=args.top,
    )


def _print_report(
    *,
    scan_run_id: str,
    n_rungs: int,
    n_events: int,
    events_with_tradeable_rung: int,
    atm_would_pass: int,
    maxvol_would_pass: int,
    atm_binding_gate: Counter,
    event_first_gate: Counter,
    rung_dist: Counter,
    by_event: dict[str, list[dict[str, Any]]],
    tradeable_examples: list[str],
    top_n: int,
) -> None:
    ratio = n_rungs / n_events if n_events else 0.0
    pct = (events_with_tradeable_rung / n_events * 100) if n_events else 0.0

    print("=" * 72)
    print(f"EVENT-COLLAPSE DIAGNOSTIC  (scan_run_id={scan_run_id})")
    print("=" * 72)
    print(f"  rungs evaluated by filters : {n_rungs}")
    print(f"  distinct events            : {n_events}")
    print(f"  collapse ratio             : {ratio:.1f} rungs/event")
    print()
    print("VERDICT - does the event contain ANY structurally-tradeable rung?")
    print("  (gates: time, price, spread, volume, liquidity; depth excluded)")
    print(f"  events with >=1 tradeable rung : {events_with_tradeable_rung}/{n_events}  ({pct:.0f}%)")
    print()
    print("Per-event outcome (furthest gate the best rung reached before failing):")
    for gate in ("<passes>", *_GATE_ORDER):
        if event_first_gate.get(gate):
            print(f"    {gate:<12} {event_first_gate[gate]}")
    print()
    print("If we collapsed to ONE representative rung per event, would it pass?")
    print(f"  at-the-money rung (closest to 50c) passes : {atm_would_pass}/{n_events}")
    if atm_binding_gate:
        kills = ", ".join(f"{g}={c}" for g, c in atm_binding_gate.most_common())
        print(f"      ATM rung binding gate: {kills}")
    print(f"  highest-volume rung (current tie-break) passes : {maxvol_would_pass}/{n_events}")
    print()
    print("Rung-count distribution (rungs_per_event: num_events):")
    for size in sorted(rung_dist):
        print(f"    {size:>4} rungs : {rung_dist[size]} events")
    print()
    if tradeable_examples:
        print("Examples of events with a tradeable rung:")
        for ex in tradeable_examples:
            print(f"    {ex}")
        print()
    print(f"Largest events by rung count (top {top_n}):")
    biggest = sorted(by_event.items(), key=lambda kv: -len(kv[1]))[:top_n]
    for event, rungs in biggest:
        cat = rungs[0].get("category") or "?"
        print(f"    {len(rungs):>4} rungs  {event:<28} [{cat}]")
    print("=" * 72)


# ── Threshold sensitivity sweep ──────────────────────────────────────────────

_SPREAD_GRID = (6, 8, 10, 15)   # current=6
_VOLUME_GRID = (1000, 500, 250, 100)   # current=1000


def _print_sweep(
    by_event: dict[str, list[dict[str, Any]]],
    base: Thresholds,
    scan_run_id: str,
) -> None:
    n_events = len(by_event)
    print("=" * 72)
    print(f"THRESHOLD SENSITIVITY SWEEP  (scan_run_id={scan_run_id})")
    print(f"  {n_events} events; counts = events with >=1 tradeable rung")
    print("  (price 15-85c, time window, and liquidity held at current values;")
    print("   orderbook depth excluded -> these are UPPER bounds on candidates)")
    print("=" * 72)

    # Matrix: rows = spread cap, cols = volume floor
    header = "  spread\\vol |" + "".join(f"{v:>8}" for v in _VOLUME_GRID)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in _SPREAD_GRID:
        cells = []
        for v in _VOLUME_GRID:
            thr = base._replace(max_spread_cents=s, min_volume=v)
            cells.append(sum(_event_tradeable(rungs, thr) for rungs in by_event.values()))
        tag = " (cur)" if s == base.max_spread_cents else ""
        print(f"  {s:>3}c{tag:<7}|" + "".join(f"{c:>8}" for c in cells))
    print()

    # Effect of also widening the time window at the loosest spread/volume cell
    loose = base._replace(max_spread_cents=_SPREAD_GRID[-1], min_volume=_VOLUME_GRID[-1])
    wide_time = loose._replace(min_mins=5, max_mins=10080)  # 5 min .. 7 days
    loose_n = sum(_event_tradeable(r, loose) for r in by_event.values())
    wide_n = sum(_event_tradeable(r, wide_time) for r in by_event.values())
    print(f"At loosest cell (spread<=15c, vol>=100):")
    print(f"    current time window      : {loose_n}/{n_events} events tradeable")
    print(f"    + time window 5min..7d   : {wide_n}/{n_events} events tradeable")
    print()

    # Time-failure breakdown at current thresholds (too-close vs too-far)
    too_close = too_far = 0
    for rungs in by_event.values():
        # event is time-blocked if EVERY rung fails the time gate
        gates = [_gate_failure(r, base) for r in rungs]
        if all(g == "time" for g in gates):
            mtc = _f(rungs[0].get("minutes_to_close"))
            if mtc < base.min_mins:
                too_close += 1
            else:
                too_far += 1
    print(f"Events fully blocked by time gate: too_close={too_close}, too_far={too_far}")
    print()

    # Show which events unlock at a pragmatic loosening (spread<=10, vol>=250)
    prag = base._replace(max_spread_cents=10, min_volume=250)
    unlocked = []
    for event, rungs in by_event.items():
        if _event_tradeable(rungs, prag) and not _event_tradeable(rungs, base):
            best = min(
                (r for r in rungs if _gate_failure(r, prag) is None),
                key=lambda r: abs(_i(r.get("yes_ask")) - 50),
            )
            unlocked.append(
                f"{best.get('ticker')}  yes_ask={_i(best.get('yes_ask'))}c "
                f"spread={_i(best.get('spread_cents'))}c vol={_i(best.get('volume_24h'))} "
                f"[{best.get('category')}]"
            )
    print(f"Events unlocked by spread<=10c + vol>=250 (vs current): {len(unlocked)}")
    for u in unlocked[:20]:
        print(f"    {u}")
    print("=" * 72)


def _latest_scan_run_id() -> Optional[str]:
    runs = db.get_recent_scan_runs(limit=1)
    return runs[0]["id"] if runs else None


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only event-collapse diagnostic for a scan run.")
    p.add_argument("--scan-run-id", default=None, help="Defaults to the most recent scan run.")
    p.add_argument("--top", type=int, default=15, help="How many largest events to list.")
    p.add_argument("--sweep", action="store_true", help="Run the threshold sensitivity sweep.")
    return p.parse_args()


if __name__ == "__main__":
    main()
