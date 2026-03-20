from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


MISSION_DIR = Path(r"C:\Users\bills\.factory\missions\83c0c194-72d1-4821-8b08-68a3497c3590")
EVIDENCE_ROOT = MISSION_DIR / "evidence"
REPORT_PATH = Path(
    r"C:\Users\bills\Documents\Personal Projects\MLBPrediction2026\.factory\validation\decision-engine\user-testing\flows\cli-notifications.json"
)


def _ensure_exists(relative_path: str) -> str:
    path = EVIDENCE_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing evidence file: {path}")
    return relative_path


def main() -> None:
    report = {
        "groupId": "cli-notifications",
        "testedAt": datetime.now(timezone.utc).isoformat(),
        "isolation": {
            "surface": "CLI",
            "pythonInterpreter": (
                r"C:\Users\bills\Documents\Personal Projects\MLBPrediction2026\.venv\Scripts\python.exe"
            ),
            "workDirectory": (
                r"C:\Users\bills\Documents\Personal Projects\MLBPrediction2026\.factory\validation\decision-engine\user-testing\work\notifications"
            ),
            "evidenceDirectory": (
                r"C:\Users\bills\.factory\missions\83c0c194-72d1-4821-8b08-68a3497c3590\evidence\decision-engine\notifications"
            ),
        },
        "toolsUsed": ["pytest", "Python CLI"],
        "commandsRun": [
            (
                r"C:\Users\bills\Documents\Personal Projects\MLBPrediction2026\.venv\Scripts\python.exe "
                r"-m pytest C:\Users\bills\Documents\Personal Projects\MLBPrediction2026\tests\test_discord_notifications.py -v --tb=short"
            ),
            (
                r"C:\Users\bills\Documents\Personal Projects\MLBPrediction2026\.venv\Scripts\python.exe "
                r"C:\Users\bills\Documents\Personal Projects\MLBPrediction2026\.factory\validation\decision-engine\user-testing\work\notifications\capture_notification_evidence.py"
            ),
        ],
        "assertions": [
            {
                "id": "VAL-NOTIFY-001",
                "title": "Discord pick card format",
                "status": "pass",
                "steps": [
                    {
                        "action": "Run targeted pytest for tests/test_discord_notifications.py",
                        "expected": "Pick-card formatting test passes",
                        "observed": "Targeted pytest passed, including test_send_picks_formats_required_card_fields_and_bankroll_footer",
                    },
                    {
                        "action": "Generate dry-run send_picks payload with contract-shaped input",
                        "expected": "Payload contains matchup, time, market, odds, model prob, edge %, Kelly stake, venue, weather, and bankroll footer",
                        "observed": "VAL-NOTIFY-001.json contains all required pick-card fields plus bankroll footer in a green embed",
                    },
                ],
                "evidence": {
                    "terminalSnapshots": [
                        _ensure_exists("decision-engine/notifications/pytest-test_discord_notifications.txt"),
                        _ensure_exists("decision-engine/notifications/dry-run-notification-terminal.txt"),
                    ],
                    "jsonPayloads": [
                        _ensure_exists("decision-engine/notifications/VAL-NOTIFY-001.json")
                    ],
                    "network": "none (dry-run payload generation and local pytest only)",
                },
                "issues": None,
            },
            {
                "id": "VAL-NOTIFY-002",
                "title": "No-picks message",
                "status": "pass",
                "steps": [
                    {
                        "action": "Run targeted pytest for tests/test_discord_notifications.py",
                        "expected": "No-picks payload test passes",
                        "observed": "Targeted pytest passed, including test_send_no_picks_includes_reason_suffix_in_dry_run_payload",
                    },
                    {
                        "action": "Generate dry-run send_no_picks payload",
                        "expected": "Payload says 'No qualifying picks today' with reasons",
                        "observed": "VAL-NOTIFY-002.json contains 'No qualifying picks today for 2025-09-15 (odds unavailable; weather unavailable)'",
                    },
                ],
                "evidence": {
                    "terminalSnapshots": [
                        _ensure_exists("decision-engine/notifications/pytest-test_discord_notifications.txt"),
                        _ensure_exists("decision-engine/notifications/dry-run-notification-terminal.txt"),
                    ],
                    "jsonPayloads": [
                        _ensure_exists("decision-engine/notifications/VAL-NOTIFY-002.json")
                    ],
                    "network": "none (dry-run payload generation and local pytest only)",
                },
                "issues": None,
            },
            {
                "id": "VAL-NOTIFY-003",
                "title": "Failure alert",
                "status": "pass",
                "steps": [
                    {
                        "action": "Run targeted pytest for tests/test_discord_notifications.py",
                        "expected": "Failure-alert test passes",
                        "observed": "Targeted pytest passed, including test_send_failure_alert_uses_red_embed_with_error_message",
                    },
                    {
                        "action": "Generate dry-run send_failure_alert payload",
                        "expected": "Payload uses a red embed and includes the fatal error message",
                        "observed": "VAL-NOTIFY-003.json contains a red 'Pipeline failure' embed with description 'model artifact missing'",
                    },
                ],
                "evidence": {
                    "terminalSnapshots": [
                        _ensure_exists("decision-engine/notifications/pytest-test_discord_notifications.txt"),
                        _ensure_exists("decision-engine/notifications/dry-run-notification-terminal.txt"),
                    ],
                    "jsonPayloads": [
                        _ensure_exists("decision-engine/notifications/VAL-NOTIFY-003.json")
                    ],
                    "network": "none (dry-run payload generation and local pytest only)",
                },
                "issues": None,
            },
            {
                "id": "VAL-NOTIFY-004",
                "title": "Drawdown alert",
                "status": "pass",
                "steps": [
                    {
                        "action": "Run targeted pytest for tests/test_discord_notifications.py",
                        "expected": "Drawdown-alert test passes",
                        "observed": "Targeted pytest passed, including test_send_drawdown_alert_uses_red_embed_and_formats_drawdown_pct",
                    },
                    {
                        "action": "Generate dry-run send_drawdown_alert payload",
                        "expected": "Payload uses a red embed and includes the drawdown percentage",
                        "observed": "VAL-NOTIFY-004.json contains a red 'Drawdown alert' embed stating drawdown reached 31.0% and new bets are disabled",
                    },
                ],
                "evidence": {
                    "terminalSnapshots": [
                        _ensure_exists("decision-engine/notifications/pytest-test_discord_notifications.txt"),
                        _ensure_exists("decision-engine/notifications/dry-run-notification-terminal.txt"),
                    ],
                    "jsonPayloads": [
                        _ensure_exists("decision-engine/notifications/VAL-NOTIFY-004.json")
                    ],
                    "network": "none (dry-run payload generation and local pytest only)",
                },
                "issues": None,
            },
        ],
        "frictions": [],
        "blockers": [],
        "summary": "Tested 4 notification assertions: 4 passed, 0 failed, 0 blocked.",
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
