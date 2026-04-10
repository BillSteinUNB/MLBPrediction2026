from __future__ import annotations

import subprocess

import pytest

from src.clients import fangraphs_baseruns_client as client


def _sample_html() -> str:
    return """
    <html>
      <body>
        <script id="__NEXT_DATA__" type="application/json">
          {
            "props": {
              "pageProps": {
                "dehydratedState": {
                  "queries": [
                    {
                      "queryKey": ["/standings/base-runs/data", 2025],
                      "state": {
                        "data": [
                          {
                            "shortName": "Yankees",
                            "abbName": "NYY",
                            "G": 162,
                            "W": 94,
                            "L": 68,
                            "WP": 0.58,
                            "RDif": 164,
                            "RpG": 5.24,
                            "RApG": 4.23,
                            "pythW": 97.0,
                            "pythL": 65.0,
                            "pythWP": 0.601,
                            "pythWDif": -3.0,
                            "bsrW": 100.0,
                            "bsrL": 62.0,
                            "bsrWP": 0.616,
                            "bsrWDif": -6.0,
                            "bsrRDif": 187.0,
                            "bsrRpG": 5.24,
                            "bsrRApG": 4.08
                          },
                          {
                            "shortName": "Athletics",
                            "abbName": "ATH",
                            "G": 162,
                            "W": 76,
                            "L": 86,
                            "WP": 0.469,
                            "RDif": -84,
                            "RpG": 4.52,
                            "RApG": 5.04,
                            "pythW": 73.0,
                            "pythL": 89.0,
                            "pythWP": 0.448,
                            "pythWDif": 3.0,
                            "bsrW": 76.0,
                            "bsrL": 86.0,
                            "bsrWP": 0.470,
                            "bsrWDif": 0.0,
                            "bsrRDif": -50.0,
                            "bsrRpG": 4.72,
                            "bsrRApG": 5.03
                          },
                          {
                            "shortName": "White Sox",
                            "abbName": "CHW",
                            "G": 162,
                            "W": 60,
                            "L": 102,
                            "WP": 0.37,
                            "RDif": -95,
                            "RpG": 3.99,
                            "RApG": 4.58,
                            "pythW": 71.0,
                            "pythL": 91.0,
                            "pythWP": 0.437,
                            "pythWDif": -11.0,
                            "bsrW": 67.0,
                            "bsrL": 95.0,
                            "bsrWP": 0.415,
                            "bsrWDif": -7.0,
                            "bsrRDif": -128.0,
                            "bsrRpG": 3.90,
                            "bsrRApG": 4.69
                          }
                        ]
                      }
                    }
                  ]
                }
              }
            }
          }
        </script>
      </body>
    </html>
    """


def test_parse_fangraphs_baseruns_html_reads_next_data_payload() -> None:
    frame = client.parse_fangraphs_baseruns_html(_sample_html())

    assert list(frame["team"]) == ["NYY", "OAK", "CWS"]
    assert frame["season"].eq(2025).all()
    assert float(frame.loc[frame["team"] == "NYY", "baseruns_wins"].iloc[0]) == pytest.approx(100.0)
    assert float(frame.loc[frame["team"] == "OAK", "baseruns_runs_scored_per_game"].iloc[0]) == pytest.approx(
        4.72
    )


def test_parse_fangraphs_baseruns_html_rejects_missing_next_data() -> None:
    with pytest.raises(client.FanGraphsBaseRunsClientError):
        client.parse_fangraphs_baseruns_html("<html><body>no payload</body></html>")


def test_fetch_fangraphs_baseruns_html_uses_curl_fallback_when_requests_returns_challenge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        text = "<html><title>Just a moment...</title></html>"

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(client.requests, "get", lambda *args, **kwargs: _FakeResponse())
    monkeypatch.setattr(client, "_REQUESTS_CIRCUIT", client.CircuitBreaker(name="test_requests"))
    monkeypatch.setattr(client, "_CURL_CIRCUIT", client.CircuitBreaker(name="test_curl"))
    monkeypatch.setattr(client.shutil, "which", lambda _: "curl")

    completed = subprocess.CompletedProcess(
        args=["curl"],
        returncode=0,
        stdout=_sample_html(),
        stderr="",
    )
    monkeypatch.setattr(client.subprocess, "run", lambda *args, **kwargs: completed)

    html = client.fetch_fangraphs_baseruns_html(season=2025)

    assert "__NEXT_DATA__" in html
