from __future__ import annotations

import logging
import re
import zlib
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import date as date_type, datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Callable
from zoneinfo import ZoneInfo

import httpx

from src.clients.statcast_client import lookup_player_ids
from src.config import _load_settings_yaml
from src.models.lineup import Lineup, LineupPlayer


logger = logging.getLogger(__name__)


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
MLB_PLAYER_STATS_URL = "https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
ROTOGRINDERS_LINEUPS_URL = "https://rotogrinders.com/lineups/mlb"
ROTOBALLER_LINEUPS_URL = "https://www.rotoballer.com/fantasy-baseball-daily-projected-starting-mlb-lineups"
ROTOWIRE_LINEUPS_URL = "https://www.rotowire.com/baseball/daily-lineups.php"
HTTP_TIMEOUT = 30.0
OPENER_IP_THRESHOLD = 3.0
RECENT_START_WINDOW = 5
PROJECTED_SOURCE_TIMEZONE = ZoneInfo("America/New_York")


PlayerIdResolver = Callable[[str], int | None]


class LineupClientError(RuntimeError):
    """Base exception for lineup client failures."""


@dataclass(slots=True)
class _RawProjectedPlayer:
    name: str
    position: str | None = None


@dataclass(slots=True)
class _RawProjectedLineupCard:
    pitcher_name: str | None = None
    players: list[_RawProjectedPlayer] = field(default_factory=list)
    confirmed: bool = False


@dataclass(slots=True)
class _RawProjectedGameCard:
    teams: list[str] = field(default_factory=list)
    lineups: list[_RawProjectedLineupCard] = field(default_factory=list)


@dataclass(slots=True)
class _ProjectedLineupData:
    team: str
    source: str
    players: list[LineupPlayer] = field(default_factory=list)
    projected_starting_pitcher_id: int | None = None
    projected_starting_pitcher_name: str | None = None


_TEAM_LABEL_TO_CODE: dict[str, str] | None = None


_TEAM_ALIASES = {
    "AZ": "ARI",
    "ARI": "ARI",
    "ATH": "OAK",
    "CHW": "CWS",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WAS": "WSH",
    "WSN": "WSH",
}


class _RotoGrindersParser(HTMLParser):
    """Small HTML parser for RotoGrinders lineup cards."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.game_cards: list[_RawProjectedGameCard] = []
        self._depth = 0
        self._current_game_card: _RawProjectedGameCard | None = None
        self._current_lineup: _RawProjectedLineupCard | None = None
        self._game_card_depth: int | None = None
        self._lineup_depth: int | None = None
        self._player_depth: int | None = None
        self._pitcher_depth: int | None = None
        self._position_depth: int | None = None
        self._player_name_depth: int | None = None
        self._pitcher_buffer: list[str] = []
        self._position_buffer: list[str] = []
        self._player_name_buffer: list[str] = []
        self._current_player_name: str | None = None
        self._current_position: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._depth += 1
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if tag == "div" and {"module", "game-card"}.issubset(classes):
            self._current_game_card = _RawProjectedGameCard()
            self._game_card_depth = self._depth
            return

        if self._current_game_card is None:
            return

        if tag == "span" and "team-nameplate-title" in classes:
            team_code = _normalize_team_label(attr_map.get("data-abbr", ""))
            if team_code:
                self._current_game_card.teams.append(team_code)
            return

        if tag == "div" and "lineup-card" in classes:
            self._current_lineup = _RawProjectedLineupCard()
            self._lineup_depth = self._depth
            return

        if self._current_lineup is None:
            return

        if tag == "div" and "lineup-card-body" in classes:
            self._current_lineup.confirmed = "unconfirmed" not in classes
            return

        if tag == "div" and "lineup-card-unconfirmed" in classes:
            self._current_lineup.confirmed = False
            return

        if tag == "div" and "lineup-card-pitcher" in classes:
            self._pitcher_depth = self._depth
            self._pitcher_buffer = []
            return

        if tag == "li" and "lineup-card-player" in classes:
            self._player_depth = self._depth
            self._current_player_name = None
            self._current_position = None
            return

        if tag == "span" and "lineup-card-positions" in classes:
            self._position_depth = self._depth
            self._position_buffer = []
            return

        if "player-nameplate-name" in classes:
            self._player_name_depth = self._depth
            self._player_name_buffer = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return

        if self._pitcher_depth is not None:
            self._pitcher_buffer.append(text)

        if self._position_depth is not None:
            self._position_buffer.append(text)

        if self._player_name_depth is not None:
            self._player_name_buffer.append(text)

    def handle_endtag(self, tag: str) -> None:
        if self._player_name_depth == self._depth:
            self._current_player_name = _clean_text(" ".join(self._player_name_buffer))
            self._player_name_depth = None
            self._player_name_buffer = []

        if self._position_depth == self._depth:
            self._current_position = _clean_text(" ".join(self._position_buffer)) or None
            self._position_depth = None
            self._position_buffer = []

        if self._pitcher_depth == self._depth and self._current_lineup is not None:
            self._current_lineup.pitcher_name = _clean_text(" ".join(self._pitcher_buffer)) or None
            self._pitcher_depth = None
            self._pitcher_buffer = []

        if self._player_depth == self._depth and self._current_lineup is not None:
            if self._current_player_name:
                self._current_lineup.players.append(
                    _RawProjectedPlayer(
                        name=self._current_player_name,
                        position=self._current_position,
                    )
                )
            self._player_depth = None
            self._current_player_name = None
            self._current_position = None

        if self._lineup_depth == self._depth and self._current_game_card is not None:
            self._current_game_card.lineups.append(self._current_lineup or _RawProjectedLineupCard())
            self._current_lineup = None
            self._lineup_depth = None

        if self._game_card_depth == self._depth:
            self.game_cards.append(self._current_game_card or _RawProjectedGameCard())
            self._current_game_card = None
            self._game_card_depth = None

        self._depth -= 1


class _RotoBallerParser(HTMLParser):
    """HTML parser for RotoBaller lineup cards."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.team_cards: list[tuple[str, _RawProjectedLineupCard]] = []
        self._depth = 0
        self._current_team: str | None = None
        self._current_lineup: _RawProjectedLineupCard | None = None
        self._team_depth: int | None = None
        self._pitcher_depth: int | None = None
        self._pitcher_buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._depth += 1
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if self._current_team is None:
            team_code = _normalize_team_label(attr_map.get("data-team", ""))
            if team_code:
                self._current_team = team_code
                self._current_lineup = _RawProjectedLineupCard()
                self._team_depth = self._depth
                pitcher_name = _clean_text(attr_map.get("data-pitcher", ""))
                if pitcher_name:
                    self._current_lineup.pitcher_name = pitcher_name
            return

        if self._current_lineup is None:
            return

        pitcher_name = _clean_text(attr_map.get("data-pitcher", ""))
        if pitcher_name:
            self._current_lineup.pitcher_name = pitcher_name

        player_name = _clean_text(attr_map.get("data-player", ""))
        if player_name:
            self._current_lineup.players.append(
                _RawProjectedPlayer(
                    name=player_name,
                    position=_clean_text(attr_map.get("data-position", "")) or None,
                )
            )

        if self._current_lineup.pitcher_name is None and "starting-pitcher" in classes:
            self._pitcher_depth = self._depth
            self._pitcher_buffer = []

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if text and self._pitcher_depth is not None:
            self._pitcher_buffer.append(text)

    def handle_endtag(self, tag: str) -> None:
        if self._pitcher_depth == self._depth and self._current_lineup is not None:
            pitcher_name = _clean_text(" ".join(self._pitcher_buffer))
            if pitcher_name:
                self._current_lineup.pitcher_name = pitcher_name
            self._pitcher_depth = None
            self._pitcher_buffer = []

        if self._team_depth == self._depth:
            if self._current_team is not None and self._current_lineup is not None:
                self.team_cards.append((self._current_team, self._current_lineup))
            self._current_team = None
            self._current_lineup = None
            self._team_depth = None

        self._depth -= 1


class _RotoWireParser(HTMLParser):
    """HTML parser for RotoWire lineup cards."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.game_cards: list[_RawProjectedGameCard] = []
        self._depth = 0
        self._current_game_card: _RawProjectedGameCard | None = None
        self._game_card_depth: int | None = None
        self._team_side: str | None = None
        self._team_side_depth: int | None = None
        self._team_abbr_side: str | None = None
        self._team_abbr_depth: int | None = None
        self._team_abbr_buffer: list[str] = []
        self._current_side: str | None = None
        self._current_side_depth: int | None = None
        self._pitcher_depth: int | None = None
        self._pitcher_buffer: list[str] = []
        self._player_depth: int | None = None
        self._position_depth: int | None = None
        self._position_buffer: list[str] = []
        self._current_player_name: str | None = None
        self._current_position: str | None = None
        self._lineups_by_side: dict[str, _RawProjectedLineupCard] = {
            "away": _RawProjectedLineupCard(),
            "home": _RawProjectedLineupCard(),
        }
        self._teams_by_side: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._depth += 1
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if tag == "div" and {"lineup", "is-mlb"}.issubset(classes) and "is-tools" not in classes:
            self._current_game_card = _RawProjectedGameCard()
            self._game_card_depth = self._depth
            self._lineups_by_side = {
                "away": _RawProjectedLineupCard(),
                "home": _RawProjectedLineupCard(),
            }
            self._teams_by_side = {}
            return

        if self._current_game_card is None:
            return

        if tag == "div" and "lineup__team" in classes:
            if "is-visit" in classes:
                self._team_side = "away"
                self._team_side_depth = self._depth
            elif "is-home" in classes:
                self._team_side = "home"
                self._team_side_depth = self._depth
            return

        if tag == "div" and "lineup__abbr" in classes and self._team_side is not None:
            self._team_abbr_side = self._team_side
            self._team_abbr_depth = self._depth
            self._team_abbr_buffer = []
            return

        if tag == "ul" and "lineup__list" in classes:
            if "is-visit" in classes:
                self._current_side = "away"
                self._current_side_depth = self._depth
            elif "is-home" in classes:
                self._current_side = "home"
                self._current_side_depth = self._depth
            return

        if self._current_side is None:
            return

        lineup = self._lineups_by_side[self._current_side]

        if tag == "li" and "lineup__status" in classes:
            lineup.confirmed = "is-expected" not in classes
            return

        if tag == "div" and "lineup__player-highlight-name" in classes:
            self._pitcher_depth = self._depth
            self._pitcher_buffer = []
            return

        if tag == "li" and "lineup__player" in classes:
            self._player_depth = self._depth
            self._current_player_name = None
            self._current_position = None
            return

        if tag == "div" and "lineup__pos" in classes:
            self._position_depth = self._depth
            self._position_buffer = []
            return

        if tag == "a" and self._player_depth is not None:
            player_name = _clean_text(attr_map.get("title", ""))
            if player_name:
                self._current_player_name = player_name

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return

        if self._team_abbr_depth is not None:
            self._team_abbr_buffer.append(text)

        if self._pitcher_depth is not None:
            self._pitcher_buffer.append(text)

        if self._position_depth is not None:
            self._position_buffer.append(text)

        if self._player_depth is not None and self._current_player_name is None:
            self._current_player_name = text

    def handle_endtag(self, tag: str) -> None:
        if self._team_abbr_depth == self._depth:
            team_code = _normalize_team_label(_clean_text(" ".join(self._team_abbr_buffer)))
            if team_code and self._team_abbr_side is not None:
                self._teams_by_side[self._team_abbr_side] = team_code
            self._team_abbr_side = None
            self._team_abbr_depth = None
            self._team_abbr_buffer = []

        if self._team_side_depth == self._depth:
            self._team_side = None
            self._team_side_depth = None

        if self._position_depth == self._depth:
            self._current_position = _clean_text(" ".join(self._position_buffer)) or None
            self._position_depth = None
            self._position_buffer = []

        if self._pitcher_depth == self._depth and self._current_side is not None:
            pitcher_name = _clean_person_name(" ".join(self._pitcher_buffer)) or None
            if pitcher_name:
                self._lineups_by_side[self._current_side].pitcher_name = pitcher_name
            self._pitcher_depth = None
            self._pitcher_buffer = []

        if self._player_depth == self._depth and self._current_side is not None:
            if self._current_player_name:
                self._lineups_by_side[self._current_side].players.append(
                    _RawProjectedPlayer(
                        name=self._current_player_name,
                        position=self._current_position,
                    )
                )
            self._player_depth = None
            self._current_player_name = None
            self._current_position = None

        if self._current_side_depth == self._depth:
            self._current_side = None
            self._current_side_depth = None

        if self._game_card_depth == self._depth:
            away_team = self._teams_by_side.get("away")
            home_team = self._teams_by_side.get("home")
            if (
                self._current_game_card is not None
                and away_team
                and home_team
                and (
                    self._lineups_by_side["away"].players
                    or self._lineups_by_side["home"].players
                    or self._lineups_by_side["away"].pitcher_name
                    or self._lineups_by_side["home"].pitcher_name
                )
            ):
                self._current_game_card.teams = [away_team, home_team]
                self._current_game_card.lineups = [
                    self._lineups_by_side["away"],
                    self._lineups_by_side["home"],
                ]
                self.game_cards.append(self._current_game_card)
            self._current_game_card = None
            self._game_card_depth = None
            self._teams_by_side = {}
            self._lineups_by_side = {
                "away": _RawProjectedLineupCard(),
                "home": _RawProjectedLineupCard(),
            }

        self._depth -= 1


def _build_team_label_index() -> dict[str, str]:
    teams = _load_settings_yaml()["teams"]
    team_map: dict[str, str] = {}

    for team_code, payload in teams.items():
        city = str(payload["city"]).strip()
        nickname = str(payload["nickname"]).strip()
        full_name = str(payload["full_name"]).strip()
        for label in {
            team_code,
            city,
            nickname,
            full_name,
            f"{city} {nickname}",
        }:
            team_map[_normalize_text(label)] = team_code

    for alias, code in _TEAM_ALIASES.items():
        team_map[_normalize_text(alias)] = code

    return team_map


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().upper())


def _normalize_team_label(value: str) -> str | None:
    global _TEAM_LABEL_TO_CODE  # noqa: PLW0603
    if _TEAM_LABEL_TO_CODE is None:
        _TEAM_LABEL_TO_CODE = _build_team_label_index()
    return _TEAM_LABEL_TO_CODE.get(_normalize_text(value))


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _clean_person_name(value: str) -> str:
    cleaned = _clean_text(value)
    return re.sub(r"\s+[LRS]$", "", cleaned)


def _strip_html_tags(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value)
    return _clean_text(without_tags)


def _default_client() -> httpx.Client:
    return httpx.Client(timeout=HTTP_TIMEOUT, follow_redirects=True)


def _client_context(client: httpx.Client | None) -> Any:
    if client is not None:
        return nullcontext(client)
    return _default_client()


def _coerce_date_string(value: str | date_type | datetime) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).date().isoformat()
    if isinstance(value, date_type):
        return value.isoformat()
    return value


def _projected_source_game_date(now: datetime | None = None) -> str:
    reference = now or datetime.now(PROJECTED_SOURCE_TIMEZONE)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=PROJECTED_SOURCE_TIMEZONE)
    else:
        reference = reference.astimezone(PROJECTED_SOURCE_TIMEZONE)
    return reference.date().isoformat()


def _projected_source_tomorrow_date(now: datetime | None = None) -> str:
    reference = now or datetime.now(PROJECTED_SOURCE_TIMEZONE)
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=PROJECTED_SOURCE_TIMEZONE)
    else:
        reference = reference.astimezone(PROJECTED_SOURCE_TIMEZONE)
    return (reference.date() + timedelta(days=1)).isoformat()


def _fallback_player_id(full_name: str) -> int:
    return zlib.crc32(full_name.lower().encode("utf-8")) or 1


def _default_player_id_resolver(full_name: str) -> int | None:
    cleaned_name = _clean_text(full_name)
    parts = cleaned_name.split()
    if len(parts) < 2:
        return None

    first_name = parts[0]
    last_name = " ".join(parts[1:])
    try:
        matches = lookup_player_ids(last_name=last_name, first_name=first_name)
    except Exception:  # pragma: no cover - best-effort external lookup
        logger.debug("Player ID lookup failed for %s", cleaned_name, exc_info=True)
        return None

    if matches.empty:
        return None

    if "mlb_played_last" in matches.columns:
        matches = matches.sort_values("mlb_played_last", ascending=False)

    for _, row in matches.iterrows():
        mlbam_id = row.get("key_mlbam")
        if mlbam_id is None:
            continue
        mlbam_text = str(mlbam_id).strip()
        if mlbam_text and mlbam_text.lower() != "nan":
            return int(float(mlbam_text))

    return None


def _resolve_player_id(full_name: str, resolver: PlayerIdResolver | None = None) -> int:
    resolved = (resolver or _default_player_id_resolver)(full_name)
    if resolved is not None:
        return resolved
    return _fallback_player_id(full_name)


def _fetch_mlb_schedule(game_date: str, http_client: httpx.Client) -> list[dict[str, Any]]:
    response = http_client.get(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": game_date,
            "hydrate": "probablePitcher,team",
        },
    )
    response.raise_for_status()

    payload = response.json()
    games: list[dict[str, Any]] = []
    for date_entry in payload.get("dates", []):
        games.extend(date_entry.get("games", []))
    return games


def _fetch_game_feed(game_pk: int, http_client: httpx.Client) -> dict[str, Any]:
    try:
        response = http_client.get(MLB_GAME_FEED_URL.format(game_pk=game_pk))
        response.raise_for_status()
    except httpx.HTTPError:
        logger.info("MLB live feed unavailable for game %s", game_pk, exc_info=True)
        return {}

    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _fetch_primary_projected_lineups(
    game_date: str,
    http_client: httpx.Client,
    *,
    player_id_resolver: PlayerIdResolver | None = None,
) -> dict[str, _ProjectedLineupData]:
    today = _projected_source_game_date()
    tomorrow = _projected_source_tomorrow_date()
    request_specs: tuple[tuple[str, dict[str, str], Callable[..., dict[str, _ProjectedLineupData]]], ...]
    if game_date == today:
        request_specs = (
            (
                ROTOGRINDERS_LINEUPS_URL,
                {
                    "User-Agent": "Mozilla/5.0 (compatible; MLBPrediction2026/1.0)",
                    "Accept": "text/html,application/xhtml+xml",
                },
                _parse_rotogrinders_html,
            ),
            (
                ROTOBALLER_LINEUPS_URL,
                {
                    "User-Agent": "Mozilla/5.0 (compatible; MLBPrediction2026/1.0)",
                    "Accept": "text/html,application/xhtml+xml",
                },
                _parse_rotoballer_html,
            ),
        )
    elif game_date == tomorrow:
        request_specs = (
            (
                f"{ROTOWIRE_LINEUPS_URL}?date=tomorrow",
                {
                    "User-Agent": "Mozilla/5.0 (compatible; MLBPrediction2026/1.0)",
                    "Accept": "text/html,application/xhtml+xml",
                },
                _parse_rotowire_html,
            ),
        )
    else:
        logger.info(
            "Skipping projected lineup scrape for %s because future sources are only supported for today/tomorrow",
            game_date,
        )
        return {}

    projected_lineups: dict[str, _ProjectedLineupData] = {}

    for url, headers, parser in request_specs:
        try:
            response = http_client.get(url, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError:
            logger.info("Primary lineup source failed: %s", url, exc_info=True)
            continue

        try:
            parsed_lineups = parser(
                response.text,
                player_id_resolver=player_id_resolver,
            )
        except Exception:  # pragma: no cover - defensive parsing guard
            logger.warning("Projected lineup parser failed for %s", url, exc_info=True)
            continue

        for team, lineup in parsed_lineups.items():
            projected_lineups.setdefault(team, lineup)

    return projected_lineups


def _parse_rotogrinders_html(
    html: str,
    *,
    player_id_resolver: PlayerIdResolver | None = None,
) -> dict[str, _ProjectedLineupData]:
    parser = _RotoGrindersParser()
    parser.feed(html)

    projected_lineups: dict[str, _ProjectedLineupData] = {}
    for game_card in parser.game_cards:
        if len(game_card.teams) < 2 or len(game_card.lineups) < 2:
            continue

        for index, team_code in enumerate(game_card.teams[:2]):
            lineup_card = game_card.lineups[index]
            projected_lineups[team_code] = _build_projected_lineup_data(
                team=team_code,
                source="rotogrinders",
                pitcher_name=lineup_card.pitcher_name,
                players=lineup_card.players,
                player_id_resolver=player_id_resolver,
            )

    return projected_lineups


def _parse_rotoballer_html(
    html: str,
    *,
    player_id_resolver: PlayerIdResolver | None = None,
) -> dict[str, _ProjectedLineupData]:
    parser = _RotoBallerParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # pragma: no cover - HTMLParser is already tolerant; keep fetch resilient
        logger.warning("Failed to parse RotoBaller HTML", exc_info=True)
        return {}

    projected_lineups: dict[str, _ProjectedLineupData] = {}
    for team_code, lineup_card in parser.team_cards:
        if not lineup_card.players and lineup_card.pitcher_name is None:
            continue

        try:
            projected_lineups[team_code] = _build_projected_lineup_data(
                team=team_code,
                source="rotoballer",
                pitcher_name=lineup_card.pitcher_name,
                players=lineup_card.players,
                player_id_resolver=player_id_resolver,
            )
        except Exception:  # pragma: no cover - defensive guard for malformed cards
            logger.warning("Skipping malformed RotoBaller lineup card for %s", team_code, exc_info=True)

    return projected_lineups


def _parse_rotowire_html(
    html: str,
    *,
    player_id_resolver: PlayerIdResolver | None = None,
) -> dict[str, _ProjectedLineupData]:
    projected_lineups: dict[str, _ProjectedLineupData] = {}
    blocks = re.split(r'<div class="lineup__box">', html)[1:]

    for block in blocks:
        if "lineup__main" not in block or "lineup__abbr" not in block:
            continue

        team_codes = [
            code
            for code in (
                _normalize_team_label(match)
                for match in re.findall(r'<div class="lineup__abbr">\s*([^<]+?)\s*</div>', block)
            )
            if code
        ]
        lineup_sections = re.findall(
            r'<ul class="lineup__list is-(visit|home)">(.*?)</ul>',
            block,
            flags=re.DOTALL,
        )
        if len(team_codes) < 2 or len(lineup_sections) < 2:
            continue

        side_to_team = {"visit": team_codes[0], "home": team_codes[1]}
        for side, section_html in lineup_sections[:2]:
            team_code = side_to_team.get(side)
            if team_code is None:
                continue
            pitcher_match = re.search(
                r'<div class="lineup__player-highlight-name">\s*<a[^>]*>(.*?)</a>',
                section_html,
                flags=re.DOTALL,
            )
            pitcher_name = (
                _clean_person_name(_strip_html_tags(pitcher_match.group(1)))
                if pitcher_match
                else None
            )
            players: list[_RawProjectedPlayer] = []
            for player_match in re.finditer(
                r'<li class="lineup__player">\s*'
                r'<div class="lineup__pos">\s*(.*?)\s*</div>\s*'
                r'<a(?P<attrs>[^>]*)>(?P<label>.*?)</a>',
                section_html,
                flags=re.DOTALL,
            ):
                position = _strip_html_tags(player_match.group(1)) or None
                attrs = player_match.group("attrs")
                title_match = re.search(r'title="([^"]+)"', attrs)
                player_name = (
                    _clean_text(title_match.group(1))
                    if title_match
                    else _strip_html_tags(player_match.group("label"))
                )
                if player_name:
                    players.append(_RawProjectedPlayer(name=player_name, position=position))

            if not players and pitcher_name is None:
                continue
            projected_lineups[team_code] = _build_projected_lineup_data(
                team=team_code,
                source="rotowire",
                pitcher_name=pitcher_name,
                players=players,
                player_id_resolver=player_id_resolver,
            )

    return projected_lineups


def _build_projected_lineup_data(
    *,
    team: str,
    source: str,
    pitcher_name: str | None,
    players: list[_RawProjectedPlayer],
    player_id_resolver: PlayerIdResolver | None,
) -> _ProjectedLineupData:
    resolved_players = [
        LineupPlayer(
            batting_order=index,
            player_id=_resolve_player_id(player.name, player_id_resolver),
            player_name=player.name,
            position=player.position,
        )
        for index, player in enumerate(players[:9], start=1)
        if player.name
    ]

    projected_pitcher_id = None
    if pitcher_name and pitcher_name.upper() != "TBD":
        projected_pitcher_id = _resolve_player_id(pitcher_name, player_id_resolver)

    return _ProjectedLineupData(
        team=team,
        source=source,
        players=resolved_players,
        projected_starting_pitcher_id=projected_pitcher_id,
        projected_starting_pitcher_name=pitcher_name,
    )


def _extract_team_code(game: dict[str, Any], side: str) -> str:
    team_name = str(game.get("teams", {}).get(side, {}).get("team", {}).get("name", "")).strip()
    team_code = _normalize_team_label(team_name)
    if team_code is None:
        raise LineupClientError(f"Unable to normalize team name: {team_name!r}")
    return team_code


def _extract_schedule_probable_pitcher_id(game: dict[str, Any], side: str) -> int | None:
    pitcher_payload = game.get("teams", {}).get(side, {}).get("probablePitcher", {})
    pitcher_id = pitcher_payload.get("id")
    return int(pitcher_id) if pitcher_id is not None else None


def _extract_feed_probable_pitcher_id(feed: dict[str, Any], side: str) -> int | None:
    pitcher_payload = feed.get("gameData", {}).get("probablePitchers", {}).get(side, {})
    pitcher_id = pitcher_payload.get("id")
    return int(pitcher_id) if pitcher_id is not None else None


def _normalize_handedness_code(value: Any) -> str | None:
    code = str(value or "").strip().upper()
    return code if code in {"L", "R", "S"} else None


def _feed_player_lookup(feed: dict[str, Any], side: str) -> dict[int, dict[str, Any]]:
    players_by_id: dict[int, dict[str, Any]] = {}
    for payload in (
        feed.get("gameData", {}).get("players", {}),
        feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {}).get("players", {}),
    ):
        if not isinstance(payload, dict):
            continue
        for raw_key, player_payload in payload.items():
            if not isinstance(player_payload, dict):
                continue
            player_id: int | None = None
            person_id = player_payload.get("person", {}).get("id")
            if person_id is not None:
                player_id = int(person_id)
            elif isinstance(raw_key, str) and raw_key.startswith("ID"):
                suffix = raw_key[2:]
                if suffix.isdigit():
                    player_id = int(suffix)
            if player_id is not None:
                players_by_id[player_id] = player_payload
    return players_by_id


def _extract_feed_pitcher_throws(feed: dict[str, Any], side: str, pitcher_id: int | None) -> str | None:
    if pitcher_id is None:
        return None
    player_payload = _feed_player_lookup(feed, side).get(int(pitcher_id), {})
    if isinstance(player_payload, dict):
        return _normalize_handedness_code(player_payload.get("pitchHand", {}).get("code"))
    return None


def _enrich_players_from_feed(feed: dict[str, Any], side: str, players: list[LineupPlayer]) -> list[LineupPlayer]:
    if not players:
        return players

    player_lookup = _feed_player_lookup(feed, side)
    enriched: list[LineupPlayer] = []
    for player in players:
        player_payload = player_lookup.get(int(player.player_id), {})
        enriched.append(
            player.model_copy(
                update={
                    "bats": player.bats
                    or _normalize_handedness_code(player_payload.get("batSide", {}).get("code")),
                    "throws": player.throws
                    or _normalize_handedness_code(player_payload.get("pitchHand", {}).get("code")),
                }
            )
        )
    return enriched


def _extract_official_players(feed: dict[str, Any], side: str) -> list[LineupPlayer]:
    team_payload = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
    players_payload = team_payload.get("players", {})
    batting_order = team_payload.get("battingOrder") or []

    lineups: list[LineupPlayer] = []
    for slot, player_id in enumerate(batting_order[:9], start=1):
        player_payload = players_payload.get(f"ID{player_id}", {})
        person = player_payload.get("person", {})
        position = player_payload.get("position", {})
        lineups.append(
            LineupPlayer(
                batting_order=slot,
                player_id=int(player_id),
                player_name=str(person.get("fullName", f"Player {player_id}")),
                position=position.get("abbreviation"),
                bats=_normalize_handedness_code(player_payload.get("batSide", {}).get("code")),
                throws=_normalize_handedness_code(player_payload.get("pitchHand", {}).get("code")),
            )
        )

    return lineups


def _parse_innings_pitched(value: str) -> float:
    innings_text = str(value).strip()
    if not innings_text:
        raise ValueError("innings pitched value is empty")
    if "." not in innings_text:
        return float(innings_text)

    whole_text, remainder_text = innings_text.split(".", 1)
    whole = int(whole_text) if whole_text else 0
    remainder = int(remainder_text) if remainder_text else 0
    if remainder in {0, 1, 2}:
        return whole + remainder / 3
    return float(innings_text)


def _fetch_pitcher_avg_ip(
    pitcher_id: int,
    season: int,
    http_client: httpx.Client,
) -> float | None:
    try:
        response = http_client.get(
            MLB_PLAYER_STATS_URL.format(player_id=pitcher_id),
            params={
                "stats": "gameLog",
                "group": "pitching",
                "season": season,
            },
        )
        response.raise_for_status()
    except httpx.HTTPError:
        logger.info("Pitcher stats unavailable for pitcher %s", pitcher_id, exc_info=True)
        return None

    payload = response.json()
    starts: list[tuple[str, float]] = []
    for stat_block in payload.get("stats", []):
        for split in stat_block.get("splits", []):
            stat = split.get("stat", {})
            if int(stat.get("gamesStarted", 0) or 0) <= 0:
                continue
            innings_text = stat.get("inningsPitched")
            if innings_text is None:
                continue
            try:
                innings_value = _parse_innings_pitched(str(innings_text))
            except ValueError:
                continue
            starts.append((str(split.get("date", "")), innings_value))

    if not starts:
        return None

    recent_starts = sorted(starts, key=lambda item: item[0])[-RECENT_START_WINDOW:]
    return sum(innings for _, innings in recent_starts) / len(recent_starts)


def _build_lineup(
    *,
    game_pk: int,
    team: str,
    now: datetime,
    source: str,
    confirmed: bool,
    players: list[LineupPlayer],
    starting_pitcher_id: int | None,
    projected_starting_pitcher_id: int | None,
    starting_pitcher_throws: str | None,
    projected_starting_pitcher_throws: str | None,
    starter_avg_innings_pitched: float | None,
) -> Lineup:
    is_opener = starter_avg_innings_pitched is not None and starter_avg_innings_pitched < OPENER_IP_THRESHOLD
    return Lineup(
        game_pk=game_pk,
        team=team,
        source=source,
        confirmed=confirmed,
        as_of_timestamp=now,
        starting_pitcher_id=starting_pitcher_id,
        projected_starting_pitcher_id=projected_starting_pitcher_id,
        starting_pitcher_throws=starting_pitcher_throws,
        projected_starting_pitcher_throws=projected_starting_pitcher_throws,
        starter_avg_innings_pitched=starter_avg_innings_pitched,
        is_opener=is_opener,
        is_bullpen_game=is_opener,
        players=players,
    )


def detect_starter_changes(
    projected: list[Lineup],
    confirmed: list[Lineup],
) -> list[dict[str, Any]]:
    projected_map = {(lineup.game_pk, lineup.team): lineup for lineup in projected}
    changes: list[dict[str, Any]] = []

    for lineup in confirmed:
        prior_lineup = projected_map.get((lineup.game_pk, lineup.team))
        if prior_lineup is None:
            continue
        projected_pitcher_id = (
            prior_lineup.projected_starting_pitcher_id or prior_lineup.starting_pitcher_id
        )
        confirmed_pitcher_id = lineup.starting_pitcher_id or lineup.projected_starting_pitcher_id
        if (
            projected_pitcher_id is not None
            and confirmed_pitcher_id is not None
            and projected_pitcher_id != confirmed_pitcher_id
        ):
            changes.append(
                {
                    "game_pk": lineup.game_pk,
                    "team": lineup.team,
                    "projected_starter_id": projected_pitcher_id,
                    "confirmed_starter_id": confirmed_pitcher_id,
                }
            )

    return changes


def fetch_confirmed_lineups(
    date: str | date_type | datetime,
    *,
    client: httpx.Client | None = None,
    player_id_resolver: PlayerIdResolver | None = None,
) -> list[Lineup]:
    """Fetch lineups for all MLB games on a date using projected+official sources."""

    game_date = _coerce_date_string(date)
    season = int(game_date[:4])
    now = datetime.now(timezone.utc)

    with _client_context(client) as http_client:
        projected_lineups = _fetch_primary_projected_lineups(
            game_date,
            http_client,
            player_id_resolver=player_id_resolver,
        )

        try:
            games = _fetch_mlb_schedule(game_date, http_client)
        except httpx.HTTPError as exc:
            raise LineupClientError(f"Failed to fetch MLB schedule for {game_date}") from exc

        lineups: list[Lineup] = []
        for game in games:
            game_pk = int(game["gamePk"])
            feed = _fetch_game_feed(game_pk, http_client)

            for side in ("away", "home"):
                team = _extract_team_code(game, side)
                projected_lineup = projected_lineups.get(team)
                official_players = _extract_official_players(feed, side)
                confirmed = bool(official_players)

                projected_pitcher_id = (
                    projected_lineup.projected_starting_pitcher_id
                    if projected_lineup is not None
                    else None
                )
                projected_pitcher_id = projected_pitcher_id or _extract_schedule_probable_pitcher_id(
                    game,
                    side,
                )
                starting_pitcher_id = _extract_feed_probable_pitcher_id(feed, side) or projected_pitcher_id
                projected_pitcher_throws = _extract_feed_pitcher_throws(feed, side, projected_pitcher_id)
                starting_pitcher_throws = _extract_feed_pitcher_throws(feed, side, starting_pitcher_id)

                players = official_players
                source = "mlb-api"
                if not confirmed:
                    if projected_lineup is not None and projected_lineup.players:
                        players = _enrich_players_from_feed(feed, side, projected_lineup.players)
                        source = projected_lineup.source
                    else:
                        players = []
                        source = "schedule"

                avg_ip = None
                pitcher_for_detection = starting_pitcher_id or projected_pitcher_id
                if pitcher_for_detection is not None:
                    avg_ip = _fetch_pitcher_avg_ip(pitcher_for_detection, season, http_client)

                lineups.append(
                    _build_lineup(
                        game_pk=game_pk,
                        team=team,
                        now=now,
                        source=source,
                        confirmed=confirmed,
                        players=players,
                        starting_pitcher_id=starting_pitcher_id,
                        projected_starting_pitcher_id=projected_pitcher_id,
                        starting_pitcher_throws=starting_pitcher_throws,
                        projected_starting_pitcher_throws=projected_pitcher_throws,
                        starter_avg_innings_pitched=avg_ip,
                    )
                )

    return sorted(lineups, key=lambda lineup: (lineup.game_pk, lineup.team))


def is_lineup_confirmed(
    game_pk: int,
    *,
    client: httpx.Client | None = None,
) -> bool:
    """Return whether MLB has posted an official batting order for the game."""

    with _client_context(client) as http_client:
        feed = _fetch_game_feed(game_pk, http_client)
        return bool(_extract_official_players(feed, "away") or _extract_official_players(feed, "home"))
