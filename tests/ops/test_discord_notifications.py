from __future__ import annotations

from typing import Any

from src.notifications.discord import (
    send_drawdown_alert,
    send_failure_alert,
    send_no_picks,
    send_picks,
)


class _StubResponse:
    def raise_for_status(self) -> None:
        return None


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def post(self, url: str, json: dict[str, Any]) -> _StubResponse:
        self.calls.append((url, json))
        return _StubResponse()


def _field_lookup(embed: dict[str, Any]) -> dict[str, str]:
    return {str(field["name"]): str(field["value"]) for field in embed["fields"]}


def test_send_picks_formats_required_card_fields_and_bankroll_footer() -> None:
    payload = send_picks(
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
    )

    assert payload["content"] == "MLB F5 picks for 2025-09-15"
    assert len(payload["embeds"]) == 1

    embed = payload["embeds"][0]
    fields = _field_lookup(embed)
    assert embed["title"] == "BOS @ NYY"
    assert fields == {
        "Time": "2025-09-15T18:05:00+00:00",
        "Market": "f5_ml home",
        "Odds": "-110",
        "Model Prob": "58.0%",
        "Edge %": "3.5%",
        "Kelly Stake": "$25.00",
        "Venue": "Yankee Stadium",
        "Weather": "72F, wind out to CF 11 mph",
    }
    assert embed["footer"]["text"] == (
        "Bankroll $975.00 | Peak $1,000.00 | Drawdown 2.5% | Bets 18 | "
        "Win 61.1% | ROI 12.4%"
    )


def test_send_picks_posts_single_webhook_for_multiple_cards() -> None:
    client = _RecordingClient()

    send_picks(
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
            },
            {
                "matchup": "ATL @ PHI",
                "scheduled_start": "2025-09-15T22:40:00+00:00",
                "market": "f5_rl away",
                "odds": "+105",
                "model_probability": 0.57,
                "edge_pct": 0.031,
                "kelly_stake": 18.5,
                "venue": "Citizens Bank Park",
                "weather": "68F, crosswind 7 mph",
            },
        ],
        bankroll_summary={"current_bankroll": 1000.0, "peak_bankroll": 1000.0, "drawdown_pct": 0.0},
        dry_run=False,
        webhook_url="https://discord.example/webhook",
        client=client,
    )

    assert len(client.calls) == 1
    assert client.calls[0][0] == "https://discord.example/webhook"
    assert len(client.calls[0][1]["embeds"]) == 2


def test_send_no_picks_includes_reason_suffix_in_dry_run_payload() -> None:
    payload = send_no_picks(
        pipeline_date="2025-09-15",
        reasons=["odds unavailable", "weather unavailable"],
        dry_run=True,
    )

    assert payload == {
        "content": "No qualifying picks today for 2025-09-15 (odds unavailable; weather unavailable)"
    }


def test_send_failure_alert_uses_red_embed_with_error_message() -> None:
    payload = send_failure_alert(
        pipeline_date="2025-09-15",
        error_message="model artifact missing",
        dry_run=True,
    )

    assert payload["content"] == ":rotating_light: Daily pipeline failed for 2025-09-15"
    assert payload["embeds"] == [
        {
            "title": "Pipeline failure",
            "description": "model artifact missing",
            "color": 0xED4245,
        }
    ]


def test_send_drawdown_alert_uses_red_embed_and_formats_drawdown_pct() -> None:
    payload = send_drawdown_alert(
        pipeline_date="2025-09-15",
        drawdown_pct=0.31,
        recommendations=[
            {
                "matchup": "BOS @ NYY",
                "scheduled_start": "2025-09-15T18:05:00+00:00",
                "market": "f5_ml home",
                "odds": "-110",
                "model_probability": 0.58,
                "edge_pct": 0.035,
                "kelly_stake": 0.0,
                "venue": "Yankee Stadium",
                "weather": "72F, wind out to CF 11 mph",
            }
        ],
        dry_run=True,
    )

    assert payload["content"] == ":warning: Kill-switch active for 2025-09-15"
    assert len(payload["embeds"]) == 2
    assert payload["embeds"][0] == {
        "title": "Drawdown alert",
        "description": "Current drawdown reached 31.0% and new bets are disabled.",
        "color": 0xED4245,
    }

    recommendation_fields = _field_lookup(payload["embeds"][1])
    assert payload["embeds"][1]["title"] == "BOS @ NYY"
    assert recommendation_fields["Market"] == "f5_ml home"
    assert recommendation_fields["Kelly Stake"] == "$0.00"
