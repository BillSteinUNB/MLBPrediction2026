from __future__ import annotations

from dataclasses import asdict, dataclass
from html import unescape
import json
import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import pandas as pd
import requests

from src.ops.error_handler import CircuitBreaker, retry


logger = logging.getLogger(__name__)

FANGRAPHS_BASERUNS_URL_TEMPLATE = (
    "https://www.fangraphs.com/standings/pythagorean-baseruns?season={season}"
)
HTTP_TIMEOUT = 30.0
NEXT_DATA_PATTERN = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
    re.DOTALL,
)
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "no-cache",
}
_REQUESTS_CIRCUIT = CircuitBreaker(name="fangraphs_baseruns_requests")
_CURL_CIRCUIT = CircuitBreaker(name="fangraphs_baseruns_curl")
_TEAM_CODE_ALIASES = {
    "ANA": "LAA",
    "ARI": "ARI",
    "ATH": "OAK",
    "AZ": "ARI",
    "CHA": "CWS",
    "CHN": "CHC",
    "CHW": "CWS",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "FLA": "MIA",
    "HOU": "HOU",
    "KCR": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "NYM": "NYM",
    "NYY": "NYY",
    "OAK": "OAK",
    "PHI": "PHI",
    "SDP": "SD",
    "SEA": "SEA",
    "SFG": "SF",
    "STL": "STL",
    "TBR": "TB",
    "TOR": "TOR",
    "WAS": "WSH",
    "WSN": "WSH",
}


class FanGraphsBaseRunsClientError(RuntimeError):
    """Raised when FanGraphs BaseRuns standings cannot be fetched or parsed."""


@dataclass(frozen=True, slots=True)
class FanGraphsBaseRunsRecord:
    season: int
    team: str
    fangraphs_team_name: str
    fangraphs_abbreviation: str
    games: int
    actual_wins: int
    actual_losses: int
    actual_win_pct: float
    actual_run_diff: float
    actual_runs_scored_per_game: float
    actual_runs_allowed_per_game: float
    pythagorean_wins: float
    pythagorean_losses: float
    pythagorean_win_pct: float
    pythagorean_win_delta: float
    baseruns_wins: float
    baseruns_losses: float
    baseruns_win_pct: float
    baseruns_win_delta: float
    baseruns_run_diff: float
    baseruns_runs_scored_per_game: float
    baseruns_runs_allowed_per_game: float


def build_fangraphs_baseruns_url(season: int) -> str:
    return FANGRAPHS_BASERUNS_URL_TEMPLATE.format(season=int(season))


def load_fangraphs_baseruns_standings(
    *,
    season: int,
    html_path: str | Path | None = None,
    save_html_path: str | Path | None = None,
    timeout: float = HTTP_TIMEOUT,
) -> pd.DataFrame:
    """Load FanGraphs season-level BaseRuns standings into a normalized DataFrame."""

    if html_path is not None:
        html = Path(html_path).read_text(encoding="utf-8")
    else:
        html = fetch_fangraphs_baseruns_html(season=season, timeout=timeout)
        if save_html_path is not None:
            resolved_save_path = Path(save_html_path)
            resolved_save_path.parent.mkdir(parents=True, exist_ok=True)
            resolved_save_path.write_text(html, encoding="utf-8")
    return parse_fangraphs_baseruns_html(html, season=season)


def fetch_fangraphs_baseruns_html(*, season: int, timeout: float = HTTP_TIMEOUT) -> str:
    """Fetch a FanGraphs season standings page, falling back to curl when needed."""

    url = build_fangraphs_baseruns_url(season)
    errors: list[str] = []

    for fetcher in (_fetch_html_with_requests, _fetch_html_with_curl):
        try:
            html = fetcher(url=url, timeout=timeout)
        except FanGraphsBaseRunsClientError as exc:
            errors.append(str(exc))
            continue
        if _looks_like_cloudflare_challenge(html):
            errors.append(
                f"{fetcher.__name__} received a Cloudflare challenge page for {url}"
            )
            continue
        if "__NEXT_DATA__" not in html:
            errors.append(f"{fetcher.__name__} did not return embedded __NEXT_DATA__ for {url}")
            continue
        return html

    joined_errors = "; ".join(errors) if errors else "unknown fetch failure"
    raise FanGraphsBaseRunsClientError(
        "Unable to fetch FanGraphs BaseRuns standings. "
        "FanGraphs appears to challenge automated requests for this page. "
        "Retry later or supply a browser-saved page via --html-input. "
        f"Details: {joined_errors}"
    )


@retry(
    logger_=logger,
    retry_exceptions=(requests.RequestException, FanGraphsBaseRunsClientError),
)
def _fetch_html_with_requests(*, url: str, timeout: float) -> str:
    def _operation() -> str:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=timeout)
        response.raise_for_status()
        return response.text

    try:
        return _REQUESTS_CIRCUIT.call(_operation)
    except requests.RequestException as exc:
        raise FanGraphsBaseRunsClientError(
            f"requests fetch failed for {url}: {exc}"
        ) from exc


@retry(
    logger_=logger,
    retry_exceptions=(subprocess.SubprocessError, FanGraphsBaseRunsClientError),
)
def _fetch_html_with_curl(*, url: str, timeout: float) -> str:
    curl_binary = shutil.which("curl")
    if not curl_binary:
        raise FanGraphsBaseRunsClientError("curl is not available on PATH")

    def _operation() -> str:
        command = [
            curl_binary,
            "-L",
            url,
            "-H",
            f"User-Agent: {REQUEST_HEADERS['User-Agent']}",
            "-H",
            f"Accept: {REQUEST_HEADERS['Accept']}",
            "-H",
            f"Accept-Language: {REQUEST_HEADERS['Accept-Language']}",
            "--max-time",
            str(int(timeout)),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.stdout

    try:
        return _CURL_CIRCUIT.call(_operation)
    except subprocess.SubprocessError as exc:
        raise FanGraphsBaseRunsClientError(f"curl fetch failed for {url}: {exc}") from exc


def parse_fangraphs_baseruns_html(html: str, *, season: int | None = None) -> pd.DataFrame:
    """Parse a FanGraphs BaseRuns standings page into a normalized DataFrame."""

    match = NEXT_DATA_PATTERN.search(html)
    if match is None:
        raise FanGraphsBaseRunsClientError(
            "FanGraphs HTML did not contain an embedded __NEXT_DATA__ payload"
        )

    payload = json.loads(unescape(match.group(1)))
    rows, payload_season = _extract_baseruns_rows(payload)
    resolved_season = int(season if season is not None else payload_season)
    if resolved_season <= 0:
        raise FanGraphsBaseRunsClientError("Unable to resolve FanGraphs standings season")

    records = [
        asdict(_coerce_baseruns_record(row, season=resolved_season))
        for row in rows
    ]
    frame = pd.DataFrame(records)
    if frame.empty:
        raise FanGraphsBaseRunsClientError(
            f"FanGraphs payload for season {resolved_season} did not contain any standings rows"
        )
    return frame.sort_values(["baseruns_wins", "actual_wins", "team"], ascending=[False, False, True]).reset_index(
        drop=True
    )


def _extract_baseruns_rows(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    queries = (
        payload.get("props", {})
        .get("pageProps", {})
        .get("dehydratedState", {})
        .get("queries", [])
    )
    for query in queries:
        query_key = query.get("queryKey")
        if not isinstance(query_key, list) or not query_key:
            continue
        if str(query_key[0]) != "/standings/base-runs/data":
            continue
        state_data = query.get("state", {}).get("data")
        if not isinstance(state_data, list):
            continue
        resolved_season = _coerce_int(query_key[1]) if len(query_key) > 1 else None
        if resolved_season is None:
            resolved_season = _infer_season_from_rows(state_data)
        return state_data, int(resolved_season or 0)
    raise FanGraphsBaseRunsClientError(
        "Embedded FanGraphs payload did not contain /standings/base-runs/data standings rows"
    )


def _infer_season_from_rows(rows: list[dict[str, Any]]) -> int | None:
    for row in rows:
        maybe_season = _coerce_int(row.get("season"))
        if maybe_season is not None:
            return maybe_season
    return None


def _coerce_baseruns_record(row: dict[str, Any], *, season: int) -> FanGraphsBaseRunsRecord:
    team_name = str(row.get("shortName") or "").strip()
    fangraphs_abbreviation = str(row.get("abbName") or "").strip().upper()
    team = _normalize_team_code(fangraphs_abbreviation=fangraphs_abbreviation, team_name=team_name)
    if not team:
        raise FanGraphsBaseRunsClientError(
            f"Unable to normalize FanGraphs team code: abb={fangraphs_abbreviation!r} name={team_name!r}"
        )

    return FanGraphsBaseRunsRecord(
        season=int(season),
        team=team,
        fangraphs_team_name=team_name,
        fangraphs_abbreviation=fangraphs_abbreviation,
        games=int(_coerce_float(row.get("G"))),
        actual_wins=int(_coerce_float(row.get("W"))),
        actual_losses=int(_coerce_float(row.get("L"))),
        actual_win_pct=_coerce_float(row.get("WP")),
        actual_run_diff=_coerce_float(row.get("RDif")),
        actual_runs_scored_per_game=_coerce_float(row.get("RpG")),
        actual_runs_allowed_per_game=_coerce_float(row.get("RApG")),
        pythagorean_wins=_coerce_float(row.get("pythW")),
        pythagorean_losses=_coerce_float(row.get("pythL")),
        pythagorean_win_pct=_coerce_float(row.get("pythWP")),
        pythagorean_win_delta=_coerce_float(row.get("pythWDif")),
        baseruns_wins=_coerce_float(row.get("bsrW")),
        baseruns_losses=_coerce_float(row.get("bsrL")),
        baseruns_win_pct=_coerce_float(row.get("bsrWP")),
        baseruns_win_delta=_coerce_float(row.get("bsrWDif")),
        baseruns_run_diff=_coerce_float(row.get("bsrRDif")),
        baseruns_runs_scored_per_game=_coerce_float(row.get("bsrRpG")),
        baseruns_runs_allowed_per_game=_coerce_float(row.get("bsrRApG")),
    )


def _normalize_team_code(*, fangraphs_abbreviation: str, team_name: str) -> str:
    normalized_abbreviation = fangraphs_abbreviation.strip().upper()
    if normalized_abbreviation in _TEAM_CODE_ALIASES:
        return _TEAM_CODE_ALIASES[normalized_abbreviation]

    normalized_name = team_name.strip().casefold()
    nickname_aliases = {
        "athletics": "OAK",
        "diamondbacks": "ARI",
        "d-backs": "ARI",
        "guardians": "CLE",
        "marlins": "MIA",
        "nationals": "WSH",
        "padres": "SD",
        "rays": "TB",
        "royals": "KC",
        "white sox": "CWS",
        "giants": "SF",
    }
    if normalized_name in nickname_aliases:
        return nickname_aliases[normalized_name]
    return normalized_abbreviation


def _looks_like_cloudflare_challenge(html: str) -> bool:
    lowered = html.casefold()
    return (
        "just a moment" in lowered
        or "_cf_chl_opt" in lowered
        or "enable javascript and cookies to continue" in lowered
    )


def _coerce_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise FanGraphsBaseRunsClientError(f"Unable to coerce numeric value from {value!r}")
    return float(numeric)


def _coerce_int(value: Any) -> int | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(numeric)


__all__ = [
    "FanGraphsBaseRunsClientError",
    "FanGraphsBaseRunsRecord",
    "HTTP_TIMEOUT",
    "REQUEST_HEADERS",
    "build_fangraphs_baseruns_url",
    "fetch_fangraphs_baseruns_html",
    "load_fangraphs_baseruns_standings",
    "parse_fangraphs_baseruns_html",
]
