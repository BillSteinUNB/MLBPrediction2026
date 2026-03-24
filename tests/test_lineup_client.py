from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from src.models.lineup import Lineup


ROTROGRINDERS_HTML = """
<div class="module game-card">
  <div class="module-header game-card-header">
    <div class="game-card-teams">
      <div class="team-nameplate">
        <span class="team-nameplate-title" data-abbr="BOS">
          <span class="team-nameplate-city">Boston</span>
          <span class="team-nameplate-mascot">Red Sox</span>
        </span>
      </div>
      <div class="team-nameplate">
        <span class="team-nameplate-title" data-abbr="NYY">
          <span class="team-nameplate-city">New York</span>
          <span class="team-nameplate-mascot">Yankees</span>
        </span>
      </div>
    </div>
  </div>
  <div class="game-card-lineups">
    <div class="lineup-card">
      <div class="lineup-card-header">
        <div class="lineup-card-pitcher break">
          <span><div><a>Chris Sale</a></div></span>
        </div>
      </div>
      <div class="lineup-card-body unconfirmed">
        <ul class="lineup-card-players">
          <li class="lineup-card-player">
            <span class="lineup-card-positions">LF</span>
            <span class="player-nameplate-name">Jarren Duran</span>
          </li>
          <li class="lineup-card-player">
            <span class="lineup-card-positions">3B</span>
            <span class="player-nameplate-name">Rafael Devers</span>
          </li>
        </ul>
      </div>
      <div class="lineup-card-unconfirmed"><span>Expected Lineup</span></div>
    </div>
    <div class="lineup-card">
      <div class="lineup-card-header">
        <div class="lineup-card-pitcher break">
          <span><div><a>Clarke Schmidt</a></div></span>
        </div>
      </div>
      <div class="lineup-card-body unconfirmed">
        <ul class="lineup-card-players">
          <li class="lineup-card-player">
            <span class="lineup-card-positions">RF</span>
            <span class="player-nameplate-name">Aaron Judge</span>
          </li>
          <li class="lineup-card-player">
            <span class="lineup-card-positions">1B</span>
            <span class="player-nameplate-name">Paul Goldschmidt</span>
          </li>
        </ul>
      </div>
      <div class="lineup-card-unconfirmed"><span>Expected Lineup</span></div>
    </div>
  </div>
</div>
"""


ROTOBALLER_HTML = """
<section class="lineup-card" data-team="Boston Red Sox">
  <div class="card-body">
    <div class="starting-pitcher">
      <strong>Chris Sale</strong>
    </div>
    <ol>
      <li data-player="Jarren Duran" data-position="LF">
        <span>Lead-off</span>
      </li>
      <li data-player="Rafael Devers" data-position="3B"></li>
    </ol>
  </div>
</section>
<section class="lineup-card" data-team="New York Yankees">
  <div class="card-body" data-pitcher="Clarke Schmidt">
    <div>
      <ol>
        <li>
          <span data-player="Aaron Judge" data-position="RF"></span>
        </li>
        <li>
          <span data-player="Paul Goldschmidt" data-position="1B"></span>
        </li>
      </ol>
    </div>
  </div>
</section>
<section class="lineup-card" data-team="Unknown Club">
  <div class="starting-pitcher">Mystery Arm</div>
</section>
"""


ROTOWIRE_HTML = """
<div class="lineup is-mlb not-in-slate">
  <div class="lineup__box">
    <div class="lineup__top">
      <div class="lineup__teams">
        <div class="lineup__team is-visit">
          <div class="lineup__abbr">BOS</div>
        </div>
        <div class="lineup__team is-home">
          <div class="lineup__abbr">NYY</div>
        </div>
      </div>
    </div>
    <div class="lineup__main">
      <ul class="lineup__list is-visit">
        <li class="lineup__player-highlight mb-0">
          <div class="lineup__player-highlight-name">
            <a href="/baseball/player/chris-sale-1">Chris Sale</a>
            <span class="lineup__throws">L</span>
          </div>
        </li>
        <li class="lineup__status is-expected">
          <div class="dot is-medium is-yellow"></div>Expected Lineup
        </li>
        <li class="lineup__player">
          <div class="lineup__pos">LF</div>
          <a title="Jarren Duran" href="/baseball/player/jarren-duran-1">J. Duran</a>
        </li>
        <li class="lineup__player">
          <div class="lineup__pos">3B</div>
          <a title="Rafael Devers" href="/baseball/player/rafael-devers-1">R. Devers</a>
        </li>
      </ul>
      <ul class="lineup__list is-home">
        <li class="lineup__player-highlight mb-0">
          <div class="lineup__player-highlight-name">
            <a href="/baseball/player/clarke-schmidt-1">Clarke Schmidt</a>
            <span class="lineup__throws">R</span>
          </div>
        </li>
        <li class="lineup__status is-expected">
          <div class="dot is-medium is-yellow"></div>Expected Lineup
        </li>
        <li class="lineup__player">
          <div class="lineup__pos">RF</div>
          <a title="Aaron Judge" href="/baseball/player/aaron-judge-1">A. Judge</a>
        </li>
        <li class="lineup__player">
          <div class="lineup__pos">1B</div>
          <a title="Paul Goldschmidt" href="/baseball/player/paul-goldschmidt-1">P. Goldschmidt</a>
        </li>
      </ul>
    </div>
  </div>
</div>
"""


def _schedule_payload() -> dict:
    return {
        "dates": [
            {
                "games": [
                    {
                        "gamePk": 12345,
                        "gameDate": "2025-09-15T23:05:00Z",
                        "officialDate": "2025-09-15",
                        "teams": {
                            "away": {
                                "team": {"name": "Boston Red Sox"},
                                "probablePitcher": {"id": 1001, "fullName": "Chris Sale"},
                            },
                            "home": {
                                "team": {"name": "New York Yankees"},
                                "probablePitcher": {"id": 1003, "fullName": "Marcus Stroman"},
                            },
                        },
                    }
                ]
            }
        ]
    }


def _live_feed_payload(*, home_confirmed: bool, away_confirmed: bool, home_pitcher_id: int) -> dict:
    home_order = [9001, 9002] if home_confirmed else []
    away_order = [8001, 8002] if away_confirmed else []
    return {
        "gameData": {
            "probablePitchers": {
                "away": {"id": 1001, "fullName": "Chris Sale"},
                "home": {"id": home_pitcher_id, "fullName": "Luke Weaver"},
            }
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {
                        "battingOrder": away_order,
                        "players": {
                            "ID8001": {
                                "person": {"id": 8001, "fullName": "Jarren Duran"},
                                "position": {"abbreviation": "LF"},
                                "battingOrder": "100",
                            },
                            "ID8002": {
                                "person": {"id": 8002, "fullName": "Rafael Devers"},
                                "position": {"abbreviation": "3B"},
                                "battingOrder": "200",
                            },
                        },
                    },
                    "home": {
                        "battingOrder": home_order,
                        "players": {
                            "ID9001": {
                                "person": {"id": 9001, "fullName": "Aaron Judge"},
                                "position": {"abbreviation": "RF"},
                                "battingOrder": "100",
                            },
                            "ID9002": {
                                "person": {"id": 9002, "fullName": "Paul Goldschmidt"},
                                "position": {"abbreviation": "1B"},
                                "battingOrder": "200",
                            },
                        },
                    },
                }
            }
        },
    }


def _pitcher_game_log_payload(*innings: str) -> dict:
    return {
        "stats": [
            {
                "splits": [
                    {
                        "date": f"2025-09-0{index + 1}",
                        "stat": {
                            "inningsPitched": value,
                            "gamesStarted": 1,
                        },
                    }
                    for index, value in enumerate(innings)
                ]
            }
        ]
    }


def _build_transport(
    *,
    schedule_payload: dict | None = None,
    live_feed_payload: dict | None = None,
    rotogrinders_status: int = 200,
    rotoballer_status: int = 200,
    rotowire_status: int = 200,
    rotogrinders_html: str = ROTROGRINDERS_HTML,
    rotoballer_html: str = "<html></html>",
    rotowire_html: str = "<html></html>",
    pitcher_logs: dict[int, dict] | None = None,
    request_log: list[str] | None = None,
) -> httpx.MockTransport:
    pitcher_logs = pitcher_logs or {}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request_log is not None:
            request_log.append(url)
        if request.url.path == "/api/v1/schedule":
            return httpx.Response(200, json=schedule_payload or _schedule_payload())
        if request.url.path == "/lineups/mlb":
            return httpx.Response(rotogrinders_status, text=rotogrinders_html)
        if request.url.host == "www.rotoballer.com":
            return httpx.Response(rotoballer_status, text=rotoballer_html)
        if request.url.host == "www.rotowire.com" and request.url.path == "/baseball/daily-lineups.php":
            return httpx.Response(rotowire_status, text=rotowire_html)
        if request.url.path == "/api/v1.1/game/12345/feed/live":
            return httpx.Response(200, json=live_feed_payload or _live_feed_payload(home_confirmed=True, away_confirmed=True, home_pitcher_id=2002))
        if request.url.path.startswith("/api/v1/people/") and request.url.path.endswith("/stats"):
            pitcher_id = int(request.url.path.split("/")[4])
            payload = pitcher_logs.get(pitcher_id, _pitcher_game_log_payload("4.0", "5.0"))
            return httpx.Response(200, json=payload)

        raise AssertionError(f"Unexpected request: {url}")

    return httpx.MockTransport(handler)


@pytest.fixture
def player_id_resolver() -> Callable[[str], int | None]:
    player_ids = {
        "Chris Sale": 1001,
        "Clarke Schmidt": 1003,
        "Marcus Stroman": 1003,
        "Luke Weaver": 2002,
        "Jarren Duran": 8001,
        "Rafael Devers": 8002,
        "Aaron Judge": 9001,
        "Paul Goldschmidt": 9002,
    }
    return player_ids.get


def test_parse_rotogrinders_html_extracts_projected_lineups(player_id_resolver: Callable[[str], int | None]) -> None:
    from src.clients.lineup_client import _parse_rotogrinders_html

    parsed = _parse_rotogrinders_html(ROTROGRINDERS_HTML, player_id_resolver=player_id_resolver)

    assert set(parsed) == {"BOS", "NYY"}
    assert parsed["BOS"].projected_starting_pitcher_id == 1001
    assert [player.player_name for player in parsed["BOS"].players] == ["Jarren Duran", "Rafael Devers"]
    assert [player.batting_order for player in parsed["NYY"].players] == [1, 2]
    assert parsed["NYY"].players[0].player_id == 9001


def test_parse_rotoballer_html_extracts_projected_lineups_and_ignores_unknown_team(
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import _parse_rotoballer_html

    parsed = _parse_rotoballer_html(ROTOBALLER_HTML, player_id_resolver=player_id_resolver)

    assert set(parsed) == {"BOS", "NYY"}
    assert parsed["BOS"].projected_starting_pitcher_id == 1001
    assert [player.player_name for player in parsed["BOS"].players] == ["Jarren Duran", "Rafael Devers"]
    assert parsed["NYY"].projected_starting_pitcher_id == 1003
    assert [player.batting_order for player in parsed["NYY"].players] == [1, 2]


def test_parse_rotowire_html_extracts_projected_lineups(
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import _parse_rotowire_html

    parsed = _parse_rotowire_html(ROTOWIRE_HTML, player_id_resolver=player_id_resolver)

    assert set(parsed) == {"BOS", "NYY"}
    assert parsed["BOS"].projected_starting_pitcher_id == 1001
    assert [player.player_name for player in parsed["BOS"].players] == ["Jarren Duran", "Rafael Devers"]
    assert parsed["NYY"].projected_starting_pitcher_id == 1003
    assert [player.player_name for player in parsed["NYY"].players] == ["Aaron Judge", "Paul Goldschmidt"]


def test_fetch_confirmed_lineups_prefers_primary_projection_and_detects_starter_change(
    monkeypatch: pytest.MonkeyPatch,
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import fetch_confirmed_lineups

    monkeypatch.setattr(
        "src.clients.lineup_client._projected_source_game_date",
        lambda: "2025-09-15",
    )

    transport = _build_transport(
        pitcher_logs={
            1001: _pitcher_game_log_payload("5.0", "6.0"),
            2002: _pitcher_game_log_payload("2.0", "2.1"),
        }
    )

    with httpx.Client(transport=transport, base_url="https://statsapi.mlb.com") as client:
        lineups = fetch_confirmed_lineups(
            "2025-09-15",
            client=client,
            player_id_resolver=player_id_resolver,
        )

    assert all(isinstance(lineup, Lineup) for lineup in lineups)
    assert {lineup.team for lineup in lineups} == {"BOS", "NYY"}

    lineup_by_team = {lineup.team: lineup for lineup in lineups}
    away_lineup = lineup_by_team["BOS"]
    home_lineup = lineup_by_team["NYY"]

    assert away_lineup.confirmed is True
    assert away_lineup.source == "mlb-api"
    assert away_lineup.projected_starting_pitcher_id == 1001
    assert away_lineup.starting_pitcher_id == 1001
    assert away_lineup.is_opener is False
    assert [player.player_id for player in away_lineup.players] == [8001, 8002]

    assert home_lineup.confirmed is True
    assert home_lineup.source == "mlb-api"
    assert home_lineup.projected_starting_pitcher_id == 1003
    assert home_lineup.starting_pitcher_id == 2002
    assert home_lineup.projected_starting_pitcher_id != home_lineup.starting_pitcher_id
    assert home_lineup.starter_avg_innings_pitched == pytest.approx(13 / 6, rel=1e-6)
    assert home_lineup.is_opener is True
    assert home_lineup.is_bullpen_game is True
    assert [player.batting_order for player in home_lineup.players] == [1, 2]


def test_fetch_confirmed_lineups_falls_back_to_mlb_api_when_primary_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import fetch_confirmed_lineups

    monkeypatch.setattr(
        "src.clients.lineup_client._projected_source_game_date",
        lambda: "2025-09-15",
    )

    transport = _build_transport(
        rotogrinders_status=503,
        rotoballer_status=503,
        pitcher_logs={
            1001: _pitcher_game_log_payload("5.0", "6.0"),
            2002: _pitcher_game_log_payload("4.0", "4.0"),
        },
    )

    with httpx.Client(transport=transport, base_url="https://statsapi.mlb.com") as client:
        lineups = fetch_confirmed_lineups(
            "2025-09-15",
            client=client,
            player_id_resolver=player_id_resolver,
        )

    assert len(lineups) == 2
    assert all(lineup.confirmed for lineup in lineups)
    assert all(lineup.source == "mlb-api" for lineup in lineups)
    assert all(lineup.players for lineup in lineups)


def test_fetch_confirmed_lineups_returns_unconfirmed_lineup_when_official_lineup_missing(
    monkeypatch: pytest.MonkeyPatch,
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import fetch_confirmed_lineups

    monkeypatch.setattr(
        "src.clients.lineup_client._projected_source_game_date",
        lambda: "2025-09-16",
    )

    transport = _build_transport(
        rotogrinders_status=503,
        rotoballer_status=503,
        live_feed_payload=_live_feed_payload(home_confirmed=False, away_confirmed=False, home_pitcher_id=2002),
        pitcher_logs={
            1001: _pitcher_game_log_payload("5.0", "6.0"),
            2002: _pitcher_game_log_payload("2.0", "2.1"),
        },
    )

    with httpx.Client(transport=transport, base_url="https://statsapi.mlb.com") as client:
        lineups = fetch_confirmed_lineups(
            "2025-09-15",
            client=client,
            player_id_resolver=player_id_resolver,
        )

    assert len(lineups) == 2
    for lineup in lineups:
        assert lineup.confirmed is False
        assert lineup.players == []
        assert lineup.source == "schedule"
        assert lineup.starting_pitcher_id is not None
        assert lineup.projected_starting_pitcher_id is not None


def test_fetch_confirmed_lineups_uses_rotoballer_when_rotogrinders_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import fetch_confirmed_lineups

    monkeypatch.setattr(
        "src.clients.lineup_client._projected_source_game_date",
        lambda: "2025-09-15",
    )

    transport = _build_transport(
        rotogrinders_status=503,
        rotoballer_html=ROTOBALLER_HTML,
        live_feed_payload=_live_feed_payload(home_confirmed=False, away_confirmed=False, home_pitcher_id=2002),
        pitcher_logs={
            1001: _pitcher_game_log_payload("5.0", "6.0"),
            1003: _pitcher_game_log_payload("4.0", "4.0"),
        },
    )

    with httpx.Client(transport=transport, base_url="https://statsapi.mlb.com") as client:
        lineups = fetch_confirmed_lineups(
            "2025-09-15",
            client=client,
            player_id_resolver=player_id_resolver,
        )

    lineup_by_team = {lineup.team: lineup for lineup in lineups}
    assert lineup_by_team["BOS"].source == "rotoballer"
    assert lineup_by_team["NYY"].source == "rotoballer"
    assert lineup_by_team["BOS"].confirmed is False
    assert [player.player_id for player in lineup_by_team["BOS"].players] == [8001, 8002]
    assert lineup_by_team["NYY"].projected_starting_pitcher_id == 1003
    assert [player.player_name for player in lineup_by_team["NYY"].players] == ["Aaron Judge", "Paul Goldschmidt"]


def test_fetch_confirmed_lineups_skips_projected_sources_for_non_matching_date(
    monkeypatch: pytest.MonkeyPatch,
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import fetch_confirmed_lineups

    monkeypatch.setattr(
        "src.clients.lineup_client._projected_source_game_date",
        lambda: "2025-09-16",
    )

    request_log: list[str] = []
    transport = _build_transport(
        live_feed_payload=_live_feed_payload(home_confirmed=False, away_confirmed=False, home_pitcher_id=2002),
        pitcher_logs={
            1001: _pitcher_game_log_payload("5.0", "6.0"),
            2002: _pitcher_game_log_payload("4.0", "4.0"),
        },
        request_log=request_log,
    )

    with httpx.Client(transport=transport, base_url="https://statsapi.mlb.com") as client:
        lineups = fetch_confirmed_lineups(
            "2025-09-15",
            client=client,
            player_id_resolver=player_id_resolver,
        )

    assert len(lineups) == 2
    assert all(lineup.source == "schedule" for lineup in lineups)
    assert all(lineup.players == [] for lineup in lineups)
    assert not any(url.endswith("/lineups/mlb") for url in request_log)
    assert not any("rotoballer.com" in url for url in request_log)
    assert not any("rotowire.com" in url for url in request_log)


def test_fetch_confirmed_lineups_uses_rotowire_for_tomorrow_projection(
    monkeypatch: pytest.MonkeyPatch,
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import fetch_confirmed_lineups

    monkeypatch.setattr(
        "src.clients.lineup_client._projected_source_game_date",
        lambda: "2025-09-15",
    )
    monkeypatch.setattr(
        "src.clients.lineup_client._projected_source_tomorrow_date",
        lambda: "2025-09-16",
    )

    transport = _build_transport(
        rotowire_html=ROTOWIRE_HTML,
        live_feed_payload=_live_feed_payload(home_confirmed=False, away_confirmed=False, home_pitcher_id=2002),
        pitcher_logs={
            1001: _pitcher_game_log_payload("5.0", "6.0"),
            1003: _pitcher_game_log_payload("4.0", "4.0"),
        },
    )

    with httpx.Client(transport=transport, base_url="https://statsapi.mlb.com") as client:
        lineups = fetch_confirmed_lineups(
            "2025-09-16",
            client=client,
            player_id_resolver=player_id_resolver,
        )

    lineup_by_team = {lineup.team: lineup for lineup in lineups}
    assert lineup_by_team["BOS"].source == "rotowire"
    assert lineup_by_team["NYY"].source == "rotowire"
    assert lineup_by_team["BOS"].confirmed is False
    assert [player.player_id for player in lineup_by_team["BOS"].players] == [8001, 8002]
    assert lineup_by_team["NYY"].projected_starting_pitcher_id == 1003


def test_fetch_confirmed_lineups_handles_rotoballer_parse_failure_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    player_id_resolver: Callable[[str], int | None],
) -> None:
    from src.clients.lineup_client import fetch_confirmed_lineups

    monkeypatch.setattr(
        "src.clients.lineup_client._projected_source_game_date",
        lambda: "2025-09-15",
    )
    monkeypatch.setattr(
        "src.clients.lineup_client._parse_rotoballer_html",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("broken rotoballer html")),
    )

    transport = _build_transport(
        rotogrinders_status=503,
        live_feed_payload=_live_feed_payload(home_confirmed=False, away_confirmed=False, home_pitcher_id=2002),
        pitcher_logs={
            1001: _pitcher_game_log_payload("5.0", "6.0"),
            2002: _pitcher_game_log_payload("4.0", "4.0"),
        },
    )

    with httpx.Client(transport=transport, base_url="https://statsapi.mlb.com") as client:
        lineups = fetch_confirmed_lineups(
            "2025-09-15",
            client=client,
            player_id_resolver=player_id_resolver,
        )

    assert len(lineups) == 2
    assert all(lineup.source == "schedule" for lineup in lineups)
    assert all(lineup.players == [] for lineup in lineups)
