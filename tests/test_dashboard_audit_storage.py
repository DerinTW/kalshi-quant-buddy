from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import db


def test_scan_run_and_market_audit_round_trip(tmp_path):
    db.init(str(tmp_path / "audit.sqlite"))
    scan_id = db.create_scan_run(mode="paper", category="crypto", limit_requested=5)

    db.insert_many_market_audits(
        [
            {
                "scan_run_id": scan_id,
                "ticker": "KXTEST-1",
                "title": "Test one",
                "category": "crypto",
                "stage": "filter_skipped",
                "outcome": "skipped",
                "skip_reason": "yes_ask=99c > max 85c",
                "skip_reason_key": "yes_ask",
                "yes_bid": 98,
                "yes_ask": 99,
                "spread_cents": 1,
                "volume_24h": 1200,
                "liquidity_dollars": 5000.0,
            },
            {
                "scan_run_id": scan_id,
                "ticker": "KXTEST-2",
                "title": "Test two",
                "category": "crypto",
                "stage": "paper_trade_opened",
                "outcome": "trade",
                "decision_action": "BUY_YES",
                "edge_cents": 8.5,
                "confidence": 0.8,
            },
        ]
    )
    db.finish_scan_run(
        scan_id,
        {
            "raw_markets": 2,
            "normalized_markets": 2,
            "passed_count": 1,
            "rejected_count": 1,
            "candidates_analyzed": 1,
            "paper_trades_inserted": 1,
            "errors": [],
        },
    )

    runs = db.get_recent_scan_runs(limit=1)
    assert runs[0]["id"] == scan_id
    assert runs[0]["summary_json"]["paper_trades_inserted"] == 1

    skipped = db.get_market_audit(scan_run_id=scan_id, outcome="skipped")
    assert len(skipped) == 1
    assert skipped[0]["ticker"] == "KXTEST-1"
    assert skipped[0]["skip_reason_key"] == "yes_ask"

    dashboard = db.get_dashboard_summary()
    assert dashboard["latest_scan_run"]["id"] == scan_id
    assert dashboard["skip_reason_counts"] == {"yes_ask": 1}
