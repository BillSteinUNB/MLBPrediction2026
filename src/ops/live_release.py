from __future__ import annotations

LIVE_RELEASE_NAME = "MLBPrediction2026 V1.0"
LIVE_RELEASE_VERSION = "1.0.0"
LIVE_MODEL_DISPLAY_NAME = "Precision Engine V1.0"
LIVE_STRATEGY_NAME = "Bet365 Full-Game Policy"
LIVE_STRATEGY_VERSION = "1.0"
LIVE_TECHNICAL_MODEL_VERSION = "ml=20260325T005859Z_4c611698:base;rl=20260325T005859Z_4c611698:base"
LIVE_RESEARCH_BASELINE_LABEL = "workingdb_2021_2025_flat_baseline"
LIVE_POLICY_SUMMARY = (
    "Bet365-only full-game policy with one frozen official slate per day, "
    "official picks restricted to 15%-22.5% edge, full-game ML/RL/Total support, "
    "and separate ledgers for manual bets, machine POTD, and all machine picks."
)


def live_release_metadata() -> dict[str, str]:
    return {
        "release_name": LIVE_RELEASE_NAME,
        "release_version": LIVE_RELEASE_VERSION,
        "model_display_name": LIVE_MODEL_DISPLAY_NAME,
        "strategy_name": LIVE_STRATEGY_NAME,
        "strategy_version": LIVE_STRATEGY_VERSION,
        "technical_model_version": LIVE_TECHNICAL_MODEL_VERSION,
        "research_baseline_label": LIVE_RESEARCH_BASELINE_LABEL,
        "policy_summary": LIVE_POLICY_SUMMARY,
    }
