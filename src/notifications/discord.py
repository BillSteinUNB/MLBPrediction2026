from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import httpx
from dotenv import load_dotenv

from src.config import DEFAULT_ENV_FILE


DISCORD_RED = 0xED4245
DISCORD_GREEN = 0x57F287


def _resolve_webhook_url(webhook_url: str | None = None) -> str:
    if webhook_url:
        return webhook_url

    load_dotenv(DEFAULT_ENV_FILE)
    resolved = os.getenv("DISCORD_WEBHOOK_URL")
    if not resolved:
        raise ValueError("DISCORD_WEBHOOK_URL is required to send Discord notifications")

    return resolved


def _format_percent(value: float) -> str:
    return f"{value:.1%}"


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _build_bankroll_footer(bankroll_summary: Mapping[str, Any] | None) -> dict[str, str] | None:
    if bankroll_summary is None:
        return None

    return {
        "text": (
            f"Bankroll {_format_currency(float(bankroll_summary.get('current_bankroll', 0.0)))} | "
            f"Peak {_format_currency(float(bankroll_summary.get('peak_bankroll', 0.0)))} | "
            f"Drawdown {_format_percent(float(bankroll_summary.get('drawdown_pct', 0.0)))} | "
            f"Bets {int(bankroll_summary.get('total_bets', 0) or 0)} | "
            f"Win {_format_percent(float(bankroll_summary.get('win_rate', 0.0)))} | "
            f"ROI {_format_percent(float(bankroll_summary.get('roi', 0.0)))}"
        )
    }


def _pick_embed(
    pick: Mapping[str, Any],
    *,
    bankroll_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    embed: dict[str, Any] = {
        "title": str(pick.get("matchup", "MLB F5 Pick")),
        "color": DISCORD_GREEN,
        "fields": [
            {"name": "Time", "value": str(pick.get("scheduled_start", "Unknown")), "inline": False},
            {"name": "Market", "value": str(pick.get("market", "F5")), "inline": True},
            {"name": "Odds", "value": str(pick.get("odds", "N/A")), "inline": True},
            {
                "name": "Model Prob",
                "value": _format_percent(float(pick.get("model_probability", 0.0))),
                "inline": True,
            },
            {
                "name": "Edge %",
                "value": _format_percent(float(pick.get("edge_pct", 0.0))),
                "inline": True,
            },
            {
                "name": "Kelly Stake",
                "value": f"${float(pick.get('kelly_stake', 0.0)):.2f}",
                "inline": True,
            },
            {"name": "Venue", "value": str(pick.get("venue", "Unknown venue")), "inline": True},
            {
                "name": "Weather",
                "value": str(pick.get("weather", "Weather unavailable")),
                "inline": False,
            },
        ],
    }

    footer = _build_bankroll_footer(bankroll_summary)
    if footer is not None:
        embed["footer"] = footer

    return embed


def _deliver_payload(
    payload: dict[str, Any],
    *,
    dry_run: bool,
    webhook_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    if dry_run:
        return payload

    resolved_webhook_url = _resolve_webhook_url(webhook_url)
    if client is not None:
        response = client.post(resolved_webhook_url, json=payload)
        response.raise_for_status()
        return payload

    with httpx.Client(timeout=30.0) as http_client:
        response = http_client.post(resolved_webhook_url, json=payload)
        response.raise_for_status()

    return payload


def send_picks(
    *,
    pipeline_date: str,
    picks: Sequence[Mapping[str, Any]],
    bankroll_summary: Mapping[str, Any] | None = None,
    dry_run: bool = False,
    webhook_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Send a Discord payload containing one embed per qualifying pick."""

    payload = {
        "content": f"MLB F5 picks for {pipeline_date}",
        "embeds": [
            _pick_embed(pick, bankroll_summary=bankroll_summary)
            for pick in picks
        ],
    }
    return _deliver_payload(
        payload,
        dry_run=dry_run,
        webhook_url=webhook_url,
        client=client,
    )


def send_no_picks(
    *,
    pipeline_date: str,
    reasons: Sequence[str] | None = None,
    dry_run: bool = False,
    webhook_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Send a no-picks message for a date with no qualifying recommendations."""

    suffix = f" ({'; '.join(reasons)})" if reasons else ""
    payload = {"content": f"No qualifying picks today for {pipeline_date}{suffix}"}
    return _deliver_payload(
        payload,
        dry_run=dry_run,
        webhook_url=webhook_url,
        client=client,
    )


def send_failure_alert(
    *,
    pipeline_date: str,
    error_message: str,
    dry_run: bool = False,
    webhook_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Send a Discord payload describing a fatal pipeline failure."""

    payload = {
        "content": f":rotating_light: Daily pipeline failed for {pipeline_date}",
        "embeds": [
            {
                "title": "Pipeline failure",
                "description": error_message,
                "color": DISCORD_RED,
            }
        ],
    }
    return _deliver_payload(
        payload,
        dry_run=dry_run,
        webhook_url=webhook_url,
        client=client,
    )


def send_drawdown_alert(
    *,
    pipeline_date: str,
    drawdown_pct: float,
    recommendations: Sequence[Mapping[str, Any]] | None = None,
    dry_run: bool = False,
    webhook_url: str | None = None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    """Send a kill-switch alert when bankroll drawdown suppresses all bets."""

    embeds: list[dict[str, Any]] = [
        {
            "title": "Drawdown alert",
            "description": (
                f"Current drawdown reached {_format_percent(drawdown_pct)} and new bets are disabled."
            ),
            "color": DISCORD_RED,
        }
    ]
    if recommendations:
        embeds.extend(_pick_embed(recommendation) for recommendation in recommendations)

    payload = {
        "content": f":warning: Kill-switch active for {pipeline_date}",
        "embeds": embeds,
    }
    return _deliver_payload(
        payload,
        dry_run=dry_run,
        webhook_url=webhook_url,
        client=client,
    )


__all__ = [
    "send_drawdown_alert",
    "send_failure_alert",
    "send_no_picks",
    "send_picks",
]
