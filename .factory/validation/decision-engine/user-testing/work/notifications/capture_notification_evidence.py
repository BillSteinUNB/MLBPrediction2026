from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(r"C:\Users\bills\Documents\Personal Projects\MLBPrediction2026")
EVIDENCE_DIR = Path(
    r"C:\Users\bills\.factory\missions\83c0c194-72d1-4821-8b08-68a3497c3590\evidence\decision-engine\notifications"
)

sys.path.insert(0, str(REPO_ROOT))

from src.notifications.discord import (  # noqa: E402
    send_drawdown_alert,
    send_failure_alert,
    send_no_picks,
    send_picks,
)


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    payloads = {
        "VAL-NOTIFY-001": send_picks(
            pipeline_date="2025-09-15",
            picks=[
                {
                    "matchup": "BOS @ NYY",
                    "scheduled_start": "2025-09-15T18:05:00+00:00",
                    "market": "f5_ml home",
                    "odds": "-110",
                    "model_probability": 0.58,
                    "edge_pct": 0.035,
                    "kelly_stake": 25.0,
                    "venue": "Yankee Stadium",
                    "weather": "72F, wind out to CF 11 mph",
                }
            ],
            bankroll_summary={
                "current_bankroll": 975.0,
                "peak_bankroll": 1000.0,
                "drawdown_pct": 0.025,
                "total_bets": 18,
                "win_rate": 0.611,
                "roi": 0.124,
            },
            dry_run=True,
        ),
        "VAL-NOTIFY-002": send_no_picks(
            pipeline_date="2025-09-15",
            reasons=["odds unavailable", "weather unavailable"],
            dry_run=True,
        ),
        "VAL-NOTIFY-003": send_failure_alert(
            pipeline_date="2025-09-15",
            error_message="model artifact missing",
            dry_run=True,
        ),
        "VAL-NOTIFY-004": send_drawdown_alert(
            pipeline_date="2025-09-15",
            drawdown_pct=0.31,
            dry_run=True,
        ),
    }

    for assertion_id, payload in payloads.items():
        output_path = EVIDENCE_DIR / f"{assertion_id}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Notification dry-run payload evidence")
    for assertion_id, payload in payloads.items():
        print(f"\n## {assertion_id}")
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
