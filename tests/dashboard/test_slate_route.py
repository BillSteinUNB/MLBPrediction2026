from src.dashboard.routes.slate import _recompute_slate_summary
from src.ops.live_release import LIVE_TECHNICAL_MODEL_VERSION, live_release_metadata


def test_recompute_slate_summary_uses_overlaid_game_state() -> None:
    payload = {
        "pick_count": 1,
        "no_pick_count": 0,
        "error_count": 0,
        "notification_type": "picks",
        "games": [
            {
                "status": "no_pick",
                "selected_decision": None,
            },
            {
                "status": "no_pick",
                "selected_decision": None,
            },
            {
                "status": "error",
                "selected_decision": None,
            },
        ],
    }

    _recompute_slate_summary(payload)

    assert payload["pick_count"] == 0
    assert payload["no_pick_count"] == 2
    assert payload["error_count"] == 1
    assert payload["notification_type"] == "failure_alert"


def test_recompute_slate_summary_prefers_picks_when_present() -> None:
    payload = {
        "games": [
            {
                "status": "pick",
                "selected_decision": {"market_type": "full_game_ml"},
            },
            {
                "status": "no_pick",
                "selected_decision": None,
            },
        ],
    }

    _recompute_slate_summary(payload)

    assert payload["pick_count"] == 1
    assert payload["no_pick_count"] == 1
    assert payload["error_count"] == 0
    assert payload["notification_type"] == "picks"


def test_live_release_metadata_exposes_current_rl_variant() -> None:
    metadata = live_release_metadata()

    assert metadata["technical_model_version"] == LIVE_TECHNICAL_MODEL_VERSION
    assert "rl=20260325T005859Z_4c611698:stacking" in metadata["technical_model_version"]
