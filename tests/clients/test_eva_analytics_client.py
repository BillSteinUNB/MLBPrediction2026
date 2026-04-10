from __future__ import annotations

import pandas as pd

from src.clients import eva_analytics_client


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


def test_fetch_eva_statcast_correlations_parses_datatable_payload(monkeypatch) -> None:
    payload = {
        "error": None,
        "dataRows": [
            {
                "columns": {
                    "Stat": "Exit Velocity (Average)",
                    "Type": "Predictive",
                    "wOBA": "0.45",
                    "HRperc": "0.57",
                    "BABIP": "-0.05",
                    "ISO": "0.54",
                    "BA": "0.11",
                    "SBperc": "0.01",
                    "SBAperc": "-0.34",
                }
            },
            {
                "columns": {
                    "Stat": "Barrel%",
                    "Type": "Descriptive",
                    "wOBA": "0.55",
                    "HRperc": "0.90",
                    "BABIP": "-0.20",
                    "ISO": "0.84",
                    "BA": "-0.04",
                    "SBperc": "-0.02",
                    "SBAperc": "-0.29",
                }
            },
        ],
    }

    monkeypatch.setattr(
        eva_analytics_client.requests,
        "post",
        lambda *args, **kwargs: _Response(payload),
    )

    result = eva_analytics_client.fetch_eva_statcast_correlations()

    assert list(result["stat"]) == ["BARREL%", "EXIT VELOCITY (AVERAGE)"]
    predictive = result.loc[result["type"] == "predictive"].iloc[0]
    assert predictive["woba_corr"] == 0.45
    assert predictive["iso_corr"] == 0.54
    assert predictive["source_page_url"] == eva_analytics_client.EVA_STATCAST_CORRELATIONS_URL
    assert predictive["datatable_id"] == eva_analytics_client.EVA_STATCAST_CORRELATIONS_DATATABLE_ID


def test_fetch_eva_statcast_correlations_returns_empty_frame_for_empty_payload(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        eva_analytics_client.requests,
        "post",
        lambda *args, **kwargs: _Response({"error": None, "dataRows": []}),
    )

    result = eva_analytics_client.fetch_eva_statcast_correlations()

    assert isinstance(result, pd.DataFrame)
    assert result.empty
