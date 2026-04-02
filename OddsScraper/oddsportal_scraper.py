from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import OrderedDict
from base64 import b64decode
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

import requests
from requests.adapters import HTTPAdapter
from playwright.async_api import BrowserContext, Error as PlaywrightError, Page

try:
    from .archive.sbr_legacy import OddsRow, SEASON_DATES, SQLiteStore
except ImportError:
    from archive.sbr_legacy import OddsRow, SEASON_DATES, SQLiteStore


DEFAULT_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}


LOGGER = logging.getLogger(__name__)

ODDSPORTAL_DECRYPT_PASSWORD = "J*8sQ!p$7aD_fR2yW@gHn*3bVp#sAdLd_k".encode()
ODDSPORTAL_DECRYPT_SALT = "5b9a8f2c3e6d1a4b7c8e9d0f1a2b3c4d".encode()
ODDSPORTAL_DEFAULT_ARCHIVE_FILTER_TOKEN = (
    "X202178560X0X0X0X0X0X0X0X0X0X0X0X0X134217728X0X0X0X0X0X8X512X32X0X0X0X0X0X0X0"
    "X536870912X2560X2048X0X33554560X8519680X0X0X0X524288"
)


def _oddsportal_primary_secondary_sides(market_type: str) -> tuple[str, str]:
    if market_type.endswith("total"):
        return "over", "under"
    # OddsPortal's baseball Home/Away and Asian Handicap payloads/DOM rows surface
    # the home selection first and the away selection second.
    return "home", "away"


def _implied_probability_from_american(price: int | None) -> float | None:
    if price is None:
        return None
    if price > 0:
        return 100.0 / (price + 100.0)
    return abs(price) / (abs(price) + 100.0)


def _infer_favorite_side_from_prices(
    *,
    home_price: int | None,
    away_price: int | None,
) -> str | None:
    implied_home = _implied_probability_from_american(home_price)
    implied_away = _implied_probability_from_american(away_price)
    if implied_home is None or implied_away is None:
        return None
    if abs(implied_home - implied_away) <= 1e-9:
        return None
    return "home" if implied_home > implied_away else "away"


def _resolve_signed_runline_point(
    *,
    side: str,
    point_value: float | None,
    favorite_side: str | None,
) -> float | None:
    if point_value is None:
        return None
    # Newer OddsPortal Asian Handicap payloads already expose signed handicap values
    # for the home side. Use that directly whenever available instead of trying to
    # infer signs from moneyline favorites.
    if point_value != 0:
        return float(point_value) if side == "home" else -float(point_value)
    magnitude = abs(float(point_value))
    if favorite_side is None:
        return None
    return -magnitude if side == favorite_side else magnitude


def _extract_primary_secondary_decimal_odds(
    value: Any,
    *,
    market_type: str,
) -> tuple[float | None, float | None]:
    primary_value, secondary_value = _extract_primary_secondary_values(
        value,
        market_type=market_type,
    )
    return _coerce_optional_float(primary_value), _coerce_optional_float(secondary_value)


def _extract_primary_secondary_values(
    value: Any,
    *,
    market_type: str,
) -> tuple[Any | None, Any | None]:
    if isinstance(value, list) and len(value) >= 2:
        return value[0], value[1]
    if isinstance(value, dict):
        if market_type.endswith("ml"):
            return value.get("0"), value.get("2")
        if market_type.endswith("total"):
            return value.get("0"), value.get("1")
    return None, None


@dataclass(frozen=True)
class OddsPortalEventInfo:
    name: str
    url: str
    start_date: str | None
    home_team: str | None
    away_team: str | None


@dataclass(frozen=True)
class OddsPortalMarketScope:
    betting_type_id: int
    betting_type_name: str
    scope_id: int
    scope_name: str
    parameter_ids: tuple[str, ...]


@dataclass(frozen=True)
class OddsPortalEventInventory:
    event: OddsPortalEventInfo
    default_betting_type_id: int | None
    default_scope_id: int | None
    scopes: tuple[OddsPortalMarketScope, ...]


class OddsPortalScraper:
    """OddsPortal event-page inspector for market discovery."""

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds
        self.results_page_settle_ms = 1500
        self.results_pagination_settle_ms = 900
        self.event_page_settle_ms = 750
        self.market_switch_settle_ms = 450
        self.expand_row_settle_ms = 150
        self.collapse_row_settle_ms = 75
        self._http = requests.Session()
        self._http.headers.update(DEFAULT_REQUEST_HEADERS)
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self._http.mount("https://", adapter)
        self._http.mount("http://", adapter)

    async def _wait_for_results_rows(self, page: Page, *, season: int) -> bool:
        for _ in range(24):
            try:
                if await page.locator(".eventRow").count() > 0:
                    return True
            except PlaywrightError:
                pass
            await page.wait_for_timeout(500)
        await page.wait_for_timeout(self.results_page_settle_ms)
        try:
            if await page.locator(".eventRow").count() > 0:
                return True
        except PlaywrightError:
            pass
        LOGGER.warning("OddsPortal results rows did not appear for season %s at %s", season, page.url)
        return False

    async def _dismiss_cookie_banner(self, page: Page) -> None:
        candidates = (
            page.get_by_role("button", name="I Accept"),
            page.get_by_text("I Accept"),
            page.get_by_role("button", name="Accept"),
            page.get_by_text("Accept"),
        )
        for locator in candidates:
            try:
                if await locator.count() == 0:
                    continue
                await locator.first.click(timeout=3000)
                await page.wait_for_timeout(500)
                return
            except PlaywrightError:
                continue

    def fetch_html(self, url: str) -> str:
        response = self._http.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.text

    async def _fetch_html_async(self, url: str) -> str:
        return await asyncio.to_thread(self.fetch_html, url)

    def scan_event(self, url: str, *, event_override: OddsPortalEventInfo | None = None) -> OddsPortalEventInventory:
        html_text = self.fetch_html(url)
        return self.parse_event_inventory(html_text, event_override=event_override)

    def parse_event_inventory(
        self,
        html_text: str,
        *,
        event_override: OddsPortalEventInfo | None = None,
    ) -> OddsPortalEventInventory:
        event = event_override or self._parse_event_info(html_text)
        page_var = self._extract_page_var(html_text)
        betting_types = self._extract_json_object_after_anchor(html_text, '"bettingTypes":')
        scope_names = self._extract_json_object_after_anchor(html_text, '"scopeNames":')

        scopes: list[OddsPortalMarketScope] = []
        nav = page_var.get("nav", {}) if isinstance(page_var, dict) else {}
        for betting_type_id_str, scope_map in nav.items():
            betting_type_data = betting_types.get(str(betting_type_id_str), {})
            betting_type_name = str(
                betting_type_data.get("name")
                or betting_type_data.get("short-name")
                or betting_type_id_str
            )
            for scope_id_str, parameter_ids in scope_map.items():
                scope_name = str(scope_names.get(str(scope_id_str), scope_id_str))
                scopes.append(
                    OddsPortalMarketScope(
                        betting_type_id=int(betting_type_id_str),
                        betting_type_name=betting_type_name,
                        scope_id=int(scope_id_str),
                        scope_name=scope_name,
                        parameter_ids=tuple(str(value) for value in parameter_ids),
                    )
                )

        scopes.sort(key=lambda item: (item.betting_type_id, item.scope_id))
        return OddsPortalEventInventory(
            event=event,
            default_betting_type_id=self._coerce_optional_int(page_var.get("defaultBettingType")),
            default_scope_id=self._coerce_optional_int(page_var.get("defaultScope")),
            scopes=tuple(scopes),
        )

    def _parse_event_info(self, html_text: str) -> OddsPortalEventInfo:
        matches = re.findall(
            r'<script type="application/ld\+json">\s*(\{.*?\})\s*</script>',
            html_text,
            re.S,
        )
        for match in matches:
            try:
                payload = json.loads(match)
            except json.JSONDecodeError:
                continue
            payload_types = payload.get("@type")
            if isinstance(payload_types, list) and "SportsEvent" not in payload_types:
                continue
            home_team = payload.get("homeTeam") or {}
            away_team = payload.get("awayTeam") or {}
            return OddsPortalEventInfo(
                name=str(payload.get("name") or ""),
                url=str(payload.get("url") or ""),
                start_date=payload.get("startDate"),
                home_team=home_team.get("name") if isinstance(home_team, dict) else None,
                away_team=away_team.get("name") if isinstance(away_team, dict) else None,
            )
        raise ValueError("Could not find OddsPortal SportsEvent metadata on page")

    def _extract_page_var(self, html_text: str) -> dict[str, Any]:
        match = re.search(r"var pageVar = '(.*?)';", html_text, re.S)
        if not match:
            raise ValueError("Could not locate OddsPortal pageVar payload")
        raw = match.group(1).encode("utf-8").decode("unicode_escape")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("OddsPortal pageVar payload was not an object")
        return payload

    def _extract_json_object_after_anchor(self, html_text: str, anchor: str) -> dict[str, Any]:
        anchor_index = html_text.find(anchor)
        if anchor_index == -1:
            raise ValueError(f"Could not locate anchor {anchor!r}")
        start = html_text.find("{", anchor_index)
        if start == -1:
            raise ValueError(f"Could not locate JSON object for {anchor!r}")

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(html_text)):
            char = html_text[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
                continue
            if char == "{":
                depth += 1
                continue
            if char == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(html_text[start : index + 1])
        raise ValueError(f"Unterminated JSON object for {anchor!r}")

    def _extract_archive_request_seed(self, html_text: str) -> tuple[str, int] | None:
        match = re.search(
            r'oddsRequest&quot;:\{&quot;url&quot;:&quot;(?P<url>[^&]+?)&quot;,&quot;urlPartTz&quot;:(?P<tz>-?\d+)',
            html_text,
        )
        if not match:
            return None
        archive_partial = match.group("url").replace("\\/", "/")
        try:
            tz_part = int(match.group("tz"))
        except ValueError:
            tz_part = 0
        return archive_partial, tz_part

    def _build_default_archive_request_url(self, results_url: str) -> str | None:
        try:
            html_text = self.fetch_html(results_url)
        except Exception:
            return None
        seed = self._extract_archive_request_seed(html_text)
        if seed is None:
            return None
        archive_partial, tz_part = seed
        archive_partial = archive_partial.rstrip("/")
        return (
            urljoin(results_url, f"{archive_partial}/{ODDSPORTAL_DEFAULT_ARCHIVE_FILTER_TOKEN}/1/{tz_part}/")
            + f"?_={int(datetime.now(tz=UTC).timestamp() * 1000)}"
        )

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def enumerate_results_page_events(
        self,
        context: BrowserContext,
        *,
        season: int,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_pages: int | None = None,
    ) -> list[OddsPortalEventInfo]:
        results_url = f"https://www.oddsportal.com/baseball/usa/mlb-{season}/results/"
        default_archive_url = self._build_default_archive_request_url(results_url)
        if default_archive_url is not None:
            try:
                return self._enumerate_results_archive_events(
                    default_archive_url,
                    referer=results_url,
                    start_date=start_date,
                    end_date=end_date,
                    max_pages=max_pages,
                )
            except Exception as exc:
                LOGGER.warning(
                    "OddsPortal direct archive enumeration failed for season %s; falling back to browser path: %s",
                    season,
                    exc,
                )
        last_dom_failure = False
        for attempt in range(3):
            page = await context.new_page()
            try:
                archive_response_urls: list[str] = []
                page.on(
                    "response",
                    lambda response: archive_response_urls.append(response.url)
                    if "/ajax-sport-country-tournament-archive_/" in response.url
                    and response.status == 200
                    and response.url not in archive_response_urls
                    else None,
                )
                await page.goto(results_url, wait_until="domcontentloaded", timeout=120000)
                await self._dismiss_cookie_banner(page)
                rows_ready = await self._wait_for_results_rows(page, season=season)
                for _ in range(20):
                    if archive_response_urls:
                        break
                    await page.wait_for_timeout(250)
                if archive_response_urls:
                    try:
                        return self._enumerate_results_archive_events(
                            archive_response_urls[0],
                            referer=results_url,
                            start_date=start_date,
                            end_date=end_date,
                            max_pages=max_pages,
                        )
                    except Exception as exc:
                        LOGGER.warning(
                            "OddsPortal archive endpoint enumeration failed for season %s on attempt %s; falling back to DOM pagination: %s",
                            season,
                            attempt + 1,
                            exc,
                        )
                if not rows_ready:
                    last_dom_failure = True
                    continue

                events: list[OddsPortalEventInfo] = []
                seen_urls: set[str] = set()
                page_number = 1
                while True:
                    payload = await page.evaluate(
                        """
                        () => {
                            let currentDateHeader = '';
                            const rows = Array.from(document.querySelectorAll('.eventRow')).map((row) => {
                                const ownDateHeader = row.querySelector('[data-testid="date-header"]')?.innerText?.trim() || '';
                                if (ownDateHeader) {
                                    currentDateHeader = ownDateHeader;
                                }
                                const gameLink =
                                    row.querySelector('div[data-testid="game-row"] > a[href*="/baseball/h2h/"]') ||
                                    row.querySelector('div[data-testid="game-row"] > a[href*="/baseball/usa/mlb-"]');
                                if (!gameLink) {
                                    return null;
                                }
                                const participants = Array.from(
                                    row.querySelectorAll('[data-testid="event-participants"] .participant-name')
                                ).map((node) => node.textContent.trim());
                                return {
                                    dateHeader: currentDateHeader,
                                    href: gameLink.getAttribute('href') || '',
                                    homeTeam: participants.length > 1 ? participants[1] : null,
                                    awayTeam: participants.length > 0 ? participants[0] : null,
                                    timeText: row.querySelector('[data-testid="time-item"] p')?.textContent?.trim() || '',
                                };
                            }).filter(Boolean);
                            const nextEnabled = !!document.querySelector('.pagination-link:not(.active)') &&
                                Array.from(document.querySelectorAll('.pagination-link')).some((node) => node.textContent.trim() === 'Next');
                            return { rows, nextEnabled };
                        }
                        """
                    )
                    rows = payload.get("rows", [])
                    if not rows:
                        break

                    oldest_page_date: datetime | None = None
                    for row in rows:
                        href = str(row.get("href") or "").strip()
                        if not href:
                            continue
                        absolute_url = urljoin("https://www.oddsportal.com", href)
                        if absolute_url in seen_urls:
                            continue
                        date_text = str(row.get("dateHeader") or "").strip()
                        parsed_date = _parse_results_page_date_header(date_text)
                        if parsed_date is not None and (oldest_page_date is None or parsed_date < oldest_page_date):
                            oldest_page_date = parsed_date
                        if start_date is not None and parsed_date is not None and parsed_date < start_date:
                            continue
                        if end_date is not None and parsed_date is not None and parsed_date > end_date:
                            continue
                        event_name = (
                            f"{row.get('awayTeam')} @ {row.get('homeTeam')}"
                            if row.get("awayTeam") and row.get("homeTeam")
                            else _event_name_from_url(absolute_url)
                        )
                        seen_urls.add(absolute_url)
                        events.append(
                            OddsPortalEventInfo(
                                name=event_name,
                                url=absolute_url,
                                start_date=parsed_date.isoformat() if parsed_date is not None else None,
                                home_team=row.get("homeTeam"),
                                away_team=row.get("awayTeam"),
                            )
                        )

                    if max_pages is not None and page_number >= max_pages:
                        break
                    if start_date is not None and oldest_page_date is not None and oldest_page_date < start_date:
                        break

                    next_link = page.locator(".pagination-link").filter(has_text="Next").first
                    if await next_link.count() == 0:
                        break
                    await next_link.click()
                    rows_ready = await self._wait_for_results_rows(page, season=season)
                    if not rows_ready:
                        break
                    await page.wait_for_timeout(self.results_pagination_settle_ms)
                    page_number += 1

                return events
            finally:
                await page.close()
        if last_dom_failure:
            LOGGER.warning(
                "OddsPortal results rows did not appear after retries for season %s at %s",
                season,
                results_url,
            )
        return []

    def _fetch_results_archive_payload(self, url: str, *, referer: str) -> dict[str, Any]:
        last_payload: dict[str, Any] | None = None
        for attempt in range(3):
            request_url = _refresh_cache_buster(url)
            response = self._http.get(
                request_url,
                timeout=self.timeout_seconds,
                headers={
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": referer,
                },
            )
            response.raise_for_status()
            payload = _decrypt_match_event_payload(response.text)
            data = payload.get("d", {})
            rows = data.get("rows", []) if isinstance(data, dict) else []
            if isinstance(rows, list) and rows:
                return payload
            last_payload = payload
        return last_payload or {}

    def _enumerate_results_archive_events(
        self,
        first_archive_url: str,
        *,
        referer: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_pages: int | None = None,
    ) -> list[OddsPortalEventInfo]:
        events: list[OddsPortalEventInfo] = []
        seen_urls: set[str] = set()
        page_number = 1
        total_pages: int | None = None
        start_day = start_date.date() if start_date is not None else None
        end_day = end_date.date() if end_date is not None else None

        while True:
            archive_url = _build_results_archive_page_url(first_archive_url, page_number)
            payload = self._fetch_results_archive_payload(archive_url, referer=referer)
            data = payload.get("d", {})
            if not isinstance(data, dict):
                break
            rows = data.get("rows", [])
            if not isinstance(rows, list) or not rows:
                break
            if total_pages is None:
                total = data.get("total")
                one_page = data.get("onePage")
                try:
                    total_int = int(total)
                    one_page_int = max(1, int(one_page))
                    total_pages = max(1, (total_int + one_page_int - 1) // one_page_int)
                except (TypeError, ValueError):
                    total_pages = None

            oldest_page_date: datetime | None = None
            for row in rows:
                event = _event_from_results_archive_row(row)
                if event is None:
                    continue
                parsed_date = (
                    datetime.fromisoformat(event.start_date)
                    if event.start_date is not None
                    else None
                )
                parsed_day = parsed_date.date() if parsed_date is not None else None
                if parsed_date is not None and (
                    oldest_page_date is None or parsed_date < oldest_page_date
                ):
                    oldest_page_date = parsed_date
                if start_day is not None and parsed_day is not None and parsed_day < start_day:
                    continue
                if end_day is not None and parsed_day is not None and parsed_day > end_day:
                    continue
                if event.url in seen_urls:
                    continue
                seen_urls.add(event.url)
                events.append(event)

            if max_pages is not None and page_number >= max_pages:
                break
            if total_pages is not None and page_number >= total_pages:
                break
            if start_day is not None and oldest_page_date is not None and oldest_page_date.date() < start_day:
                break
            page_number += 1

        return events

    async def scrape_event_rows(
        self,
        context: BrowserContext,
        event_url: str,
        *,
        event_info: OddsPortalEventInfo | None = None,
        fetched_at: str | None = None,
    ) -> list[OddsRow]:
        page = await context.new_page()
        try:
            match_event_urls: list[str] = []
            page.on(
                "response",
                lambda response: match_event_urls.append(response.url)
                if "/match-event/" in response.url and response.status == 200 and response.url not in match_event_urls
                else None,
            )
            html_task = asyncio.create_task(self._fetch_html_async(event_url))
            await page.goto(event_url, wait_until="domcontentloaded", timeout=120000)
            await self._dismiss_cookie_banner(page)
            await page.wait_for_timeout(self.event_page_settle_ms)
            html_text = await html_task
            inventory = self.parse_event_inventory(html_text, event_override=event_info)
            event = inventory.event
            for _ in range(20):
                if match_event_urls:
                    break
                await page.wait_for_timeout(250)
            default_endpoint_url = _select_match_event_url(match_event_urls, event_url=event_url)
            if default_endpoint_url is None:
                raise ValueError(f"Could not capture OddsPortal match-event URL for {event_url}")
            visible_rows = await self._extract_visible_market_rows(page)
            event_start = _parse_iso_datetime(event.start_date)
            if event_start is None:
                raise ValueError(f"Could not parse event start date for {event_url}")
            commence_time_utc = event_start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            game_date = event_start.date().isoformat()
            event_id = _event_id_from_url(event_url)
            away_team_code = _normalize_oddsportal_team_code(event.away_team)
            home_team_code = _normalize_oddsportal_team_code(event.home_team)
            game_type = _classify_game_type(event_start)
            captured_at = fetched_at or datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

            default_payload = await self._fetch_match_event_payload_async(default_endpoint_url)
            provider_name_map = self._infer_provider_name_map(default_payload, visible_rows)
            favorite_sides_by_market: dict[str, dict[str, str]] = {}

            rows: list[OddsRow] = []
            target_requests: list[tuple[str, str, asyncio.Task[dict[str, Any]] | None]] = []
            for target in self._resolve_target_market_requests(inventory):
                request_url = _build_match_event_request_url(
                    default_endpoint_url,
                    betting_type_id=target["betting_type_id"],
                    scope_id=target["scope_id"],
                )
                if request_url == default_endpoint_url:
                    target_requests.append((target["market_type"], request_url, None))
                else:
                    target_requests.append(
                        (
                            target["market_type"],
                            request_url,
                            asyncio.create_task(self._fetch_match_event_payload_async(request_url)),
                        )
                    )

            for market_type, request_url, payload_task in target_requests:
                try:
                    payload = default_payload if payload_task is None else await payload_task
                except Exception as exc:
                    LOGGER.warning("Skipping OddsPortal market fetch %s for %s due to error: %s", market_type, event_id, exc)
                    continue
                if market_type.endswith("ml"):
                    favorite_sides_by_market[market_type] = self._build_bookmaker_favorite_map_from_moneyline_payload(
                        payload=payload,
                        provider_name_map=provider_name_map,
                        market_type=market_type,
                    )
                rows.extend(
                    self._build_rows_from_market_payload(
                        payload=payload,
                        provider_name_map=provider_name_map,
                        bookmaker_favorite_sides=(
                            favorite_sides_by_market.get("full_game_ml")
                            if market_type == "full_game_rl"
                            else favorite_sides_by_market.get("f5_ml")
                            if market_type == "f5_rl"
                            else None
                        ),
                        event_id=event_id,
                        game_date=game_date,
                        commence_time_utc=commence_time_utc,
                        away_team=away_team_code,
                        home_team=home_team_code,
                        game_type=game_type,
                        market_type=market_type,
                        fetched_at=captured_at,
                    )
                )
            return rows
        finally:
            await page.close()

    async def _extract_visible_market_rows(self, page: Page) -> list[dict[str, Any]]:
        payload = await page.evaluate(
            """
            () => Array.from(document.querySelectorAll('[data-testid="over-under-expanded-row"]')).map((row) => {
                const name = row.querySelector('[data-testid="outrights-expanded-bookmaker-name"]')?.textContent?.trim() || '';
                const point = row.querySelector('[data-testid="total-container"]')?.textContent?.trim() || '';
                const odds = Array.from(row.querySelectorAll('[data-testid="odd-container"] .odds-text')).map((node) => node.textContent.trim());
                return { bookmaker: name, point, odds };
            }).filter((row) => row.bookmaker && row.odds.length >= 2)
            """
        )
        return payload if isinstance(payload, list) else []

    def _fetch_match_event_payload(self, url: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(3):
            try:
                response = self._http.get(url, timeout=self.timeout_seconds)
                response.raise_for_status()
                return _decrypt_match_event_payload(response.text)
            except Exception as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def _fetch_match_event_payload_async(self, url: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._fetch_match_event_payload, url)

    def _infer_provider_name_map(
        self,
        payload: dict[str, Any],
        visible_rows: list[dict[str, Any]],
    ) -> dict[str, str]:
        market_back = payload.get("d", {}).get("oddsdata", {}).get("back", {})
        if not isinstance(market_back, dict):
            return {}
        payload_signatures: dict[tuple[float | None, int, int], list[str]] = {}
        for item in market_back.values():
            if not isinstance(item, dict):
                continue
            point_value = _coerce_optional_float(item.get("handicapValue"))
            odds_map = item.get("odds", {})
            if not isinstance(odds_map, dict):
                continue
            for provider_id, odds_pair in odds_map.items():
                if not isinstance(odds_pair, list) or len(odds_pair) < 2:
                    continue
                signature = (
                    point_value,
                    int(round(float(odds_pair[0]) * 1000)),
                    int(round(float(odds_pair[1]) * 1000)),
                )
                payload_signatures.setdefault(signature, []).append(str(provider_id))

        provider_name_map: dict[str, str] = {}
        for row in visible_rows:
            bookmaker = str(row.get("bookmaker") or "").strip()
            odds = row.get("odds") or []
            if not bookmaker or len(odds) < 2:
                continue
            signature = (
                _coerce_optional_float(row.get("point")),
                int(round(float(odds[0]) * 1000)),
                int(round(float(odds[1]) * 1000)),
            )
            provider_ids = payload_signatures.get(signature, [])
            if len(provider_ids) == 1:
                provider_name_map[provider_ids[0]] = bookmaker
        return provider_name_map

    def _resolve_target_market_requests(self, inventory: OddsPortalEventInventory) -> list[dict[str, Any]]:
        targets = [
            (("Home/Away", "1X2"), ("FT including OT", "Full Time"), "full_game_ml"),
            ("Over/Under", ("FT including OT", "Full Time"), "full_game_total"),
            ("Asian Handicap", ("FT including OT", "Full Time"), "full_game_rl"),
            (("Home/Away", "1X2"), ("1st Half", "1st Half Innings"), "f5_ml"),
            ("Over/Under", ("1st Half", "1st Half Innings"), "f5_total"),
            ("Asian Handicap", ("1st Half", "1st Half Innings"), "f5_rl"),
        ]
        resolved: list[dict[str, Any]] = []
        for betting_type_name, scope_names, market_type in targets:
            allowed_betting_type_names = (
                betting_type_name
                if isinstance(betting_type_name, tuple)
                else (betting_type_name,)
            )
            match = next(
                (
                    scope
                    for scope in inventory.scopes
                    if scope.betting_type_name in allowed_betting_type_names
                    and scope.scope_name in scope_names
                ),
                None,
            )
            if match is None:
                continue
            resolved.append(
                {
                    "market_type": market_type,
                    "betting_type_id": match.betting_type_id,
                    "scope_id": match.scope_id,
                }
            )
        return resolved

    def _build_bookmaker_favorite_map_from_moneyline_payload(
        self,
        *,
        payload: dict[str, Any],
        provider_name_map: dict[str, str],
        market_type: str,
    ) -> dict[str, str]:
        if not market_type.endswith("ml"):
            return {}
        back_data = payload.get("d", {}).get("oddsdata", {}).get("back", {})
        if not isinstance(back_data, dict):
            return {}

        favorite_sides: dict[str, str] = {}
        for item in back_data.values():
            if not isinstance(item, dict):
                continue
            odds_map = item.get("odds", {})
            if not isinstance(odds_map, dict):
                continue
            for provider_id, odds_pair in odds_map.items():
                home_decimal, away_decimal = _extract_primary_secondary_decimal_odds(
                    odds_pair,
                    market_type=market_type,
                )
                home_price = _decimal_to_american(home_decimal)
                away_price = _decimal_to_american(away_decimal)
                favorite_side = _infer_favorite_side_from_prices(
                    home_price=home_price,
                    away_price=away_price,
                )
                if favorite_side is None:
                    continue
                bookmaker_name = provider_name_map.get(str(provider_id), f"provider_{provider_id}")
                bookmaker = f"OddsPortal:{bookmaker_name}"
                favorite_sides[bookmaker] = favorite_side
        return favorite_sides

    def _build_rows_from_market_payload(
        self,
        *,
        payload: dict[str, Any],
        provider_name_map: dict[str, str],
        bookmaker_favorite_sides: dict[str, str] | None,
        event_id: str,
        game_date: str,
        commence_time_utc: str,
        away_team: str,
        home_team: str,
        game_type: str,
        market_type: str,
        fetched_at: str,
    ) -> list[OddsRow]:
        back_data = payload.get("d", {}).get("oddsdata", {}).get("back", {})
        if not isinstance(back_data, dict):
            return []

        side_a, side_b = _oddsportal_primary_secondary_sides(market_type)
        point_enabled = not market_type.endswith("ml")

        rows: list[OddsRow] = []
        seen: set[tuple[str, str, str, float | None]] = set()
        for item in back_data.values():
            if not isinstance(item, dict):
                continue
            point_value = _coerce_optional_float(item.get("handicapValue")) if point_enabled else None
            odds_map = item.get("odds", {})
            opening_odds_map = item.get("openingOdd", {})
            opening_change_time_map = item.get("openingChangeTime", {})
            if not isinstance(odds_map, dict):
                continue
            for provider_id, odds_pair in odds_map.items():
                price_a_decimal, price_b_decimal = _extract_primary_secondary_decimal_odds(
                    odds_pair,
                    market_type=market_type,
                )
                price_a = _decimal_to_american(price_a_decimal)
                price_b = _decimal_to_american(price_b_decimal)
                if price_a is None or price_b is None:
                    continue
                bookmaker_name = provider_name_map.get(str(provider_id), f"provider_{provider_id}")
                bookmaker = f"OddsPortal:{bookmaker_name}"
                favorite_side = (
                    bookmaker_favorite_sides.get(bookmaker)
                    if bookmaker_favorite_sides is not None
                    else None
                )
                resolved_point_a = (
                    _resolve_signed_runline_point(
                        side=side_a,
                        point_value=point_value,
                        favorite_side=favorite_side,
                    )
                    if market_type.endswith("rl")
                    else point_value
                )
                resolved_point_b = (
                    _resolve_signed_runline_point(
                        side=side_b,
                        point_value=point_value,
                        favorite_side=favorite_side,
                    )
                    if market_type.endswith("rl")
                    else point_value
                )
                key_a = (bookmaker, market_type, side_a, resolved_point_a)
                key_b = (bookmaker, market_type, side_b, resolved_point_b)
                opening_price_a: int | None = None
                opening_price_b: int | None = None
                opening_fetched_at_a: str | None = None
                opening_fetched_at_b: str | None = None
                if isinstance(opening_odds_map, dict):
                    opening_value = opening_odds_map.get(provider_id)
                    opening_price_a_decimal, opening_price_b_decimal = _extract_primary_secondary_decimal_odds(
                        opening_value,
                        market_type=market_type,
                    )
                    opening_price_a = _decimal_to_american(opening_price_a_decimal)
                    opening_price_b = _decimal_to_american(opening_price_b_decimal)
                if isinstance(opening_change_time_map, dict):
                    opening_time_value = opening_change_time_map.get(provider_id)
                    opening_time_a, opening_time_b = _extract_primary_secondary_values(
                        opening_time_value,
                        market_type=market_type,
                    )
                    opening_fetched_at_a = _epoch_seconds_to_iso8601(opening_time_a)
                    opening_fetched_at_b = _epoch_seconds_to_iso8601(opening_time_b)
                if opening_price_a is not None:
                    rows.append(
                        OddsRow(
                            event_id=event_id,
                            game_date=game_date,
                            commence_time_utc=commence_time_utc,
                            away_team=away_team,
                            home_team=home_team,
                            game_type=game_type,
                            fetched_at=opening_fetched_at_a or fetched_at,
                            bookmaker=bookmaker,
                            market_type=market_type,
                            side=side_a,
                            point=resolved_point_a,
                            price=opening_price_a,
                            is_opening=True,
                        )
                    )
                if opening_price_b is not None:
                    rows.append(
                        OddsRow(
                            event_id=event_id,
                            game_date=game_date,
                            commence_time_utc=commence_time_utc,
                            away_team=away_team,
                            home_team=home_team,
                            game_type=game_type,
                            fetched_at=opening_fetched_at_b or fetched_at,
                            bookmaker=bookmaker,
                            market_type=market_type,
                            side=side_b,
                            point=resolved_point_b,
                            price=opening_price_b,
                            is_opening=True,
                        )
                    )
                if key_a not in seen:
                    seen.add(key_a)
                    rows.append(
                        OddsRow(
                            event_id=event_id,
                            game_date=game_date,
                            commence_time_utc=commence_time_utc,
                            away_team=away_team,
                            home_team=home_team,
                            game_type=game_type,
                            fetched_at=fetched_at,
                            bookmaker=bookmaker,
                            market_type=market_type,
                            side=side_a,
                            point=resolved_point_a,
                            price=price_a,
                            is_opening=False,
                        )
                    )
                if key_b not in seen:
                    seen.add(key_b)
                    rows.append(
                        OddsRow(
                            event_id=event_id,
                            game_date=game_date,
                            commence_time_utc=commence_time_utc,
                            away_team=away_team,
                            home_team=home_team,
                            game_type=game_type,
                            fetched_at=fetched_at,
                            bookmaker=bookmaker,
                            market_type=market_type,
                            side=side_b,
                            point=resolved_point_b,
                            price=price_b,
                            is_opening=False,
                        )
                    )
        return rows

    async def backfill_events_to_db(
        self,
        context: BrowserContext,
        *,
        season: int,
        db: SQLiteStore,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        max_pages: int | None = None,
        max_events: int | None = None,
        existing_event_ids: set[str] | None = None,
        progress_callback: Any = None,
        concurrency: int = 4,
    ) -> tuple[int, int]:
        events = await self.enumerate_results_page_events(
            context,
            season=season,
            start_date=start_date,
            end_date=end_date,
            max_pages=max_pages,
        )

        inserted_rows = 0
        processed_events = 0
        seen_event_ids = set(existing_event_ids or ())
        pending_events = [event for event in events if _event_id_from_url(event.url) not in seen_event_ids]
        if max_events is not None:
            pending_events = pending_events[:max_events]

        grouped_events: OrderedDict[str | None, list[OddsPortalEventInfo]] = OrderedDict()
        for event in pending_events:
            grouped_events.setdefault(_event_progress_key(event), []).append(event)

        worker_limit = max(1, int(concurrency))
        semaphore = asyncio.Semaphore(worker_limit)

        async def _scrape_with_limit(event: OddsPortalEventInfo) -> tuple[OddsPortalEventInfo, list[OddsRow] | None, Exception | None]:
            async with semaphore:
                try:
                    return event, await self.scrape_event_rows(context, event.url, event_info=event), None
                except Exception as exc:
                    return event, None, exc

        for progress_key, events_in_group in grouped_events.items():
            tasks = [asyncio.create_task(_scrape_with_limit(event)) for event in events_in_group]
            for task in asyncio.as_completed(tasks):
                event, rows, exc = await task
                event_id = _event_id_from_url(event.url)
                if exc is not None:
                    if isinstance(exc, PlaywrightError):
                        LOGGER.warning("Skipping OddsPortal event %s due to Playwright error: %s", event.url, exc)
                    else:
                        LOGGER.warning("Skipping OddsPortal event %s due to unexpected error: %s", event.url, exc)
                    processed_events += 1
                    seen_event_ids.add(event_id)
                    continue
                if not rows:
                    processed_events += 1
                    seen_event_ids.add(event_id)
                    continue
                db.flag_first_seen_rows_as_opening(rows)
                db.insert_rows(rows)
                inserted_rows += len(rows)
                processed_events += 1
                seen_event_ids.add(event_id)
            if progress_callback is not None and progress_key is not None:
                progress_callback(progress_key)
        return processed_events, inserted_rows

    async def _select_market(self, page: Page, market_name: str) -> None:
        nav = page.locator('[data-testid="bet-types-nav"]')
        tab = nav.locator('[data-testid="navigation-active-tab"], [data-testid="navigation-inactive-tab"]').filter(
            has_text=market_name
        ).first
        await tab.click()

    async def _select_scope(self, page: Page, scope_name: str) -> None:
        nav = page.locator('[data-testid="kickoff-events-nav"]')
        target = nav.locator('[data-testid="sub-nav-active-tab"], [data-testid="sub-nav-inactive-tab"]').filter(
            has_text=scope_name
        ).first
        await target.click()

    async def _has_market(self, page: Page, market_name: str) -> bool:
        nav = page.locator('[data-testid="bet-types-nav"]')
        tab = nav.locator('[data-testid="navigation-active-tab"], [data-testid="navigation-inactive-tab"]').filter(
            has_text=market_name
        ).first
        return await tab.count() > 0

    async def _has_scope(self, page: Page, scope_name: str) -> bool:
        nav = page.locator('[data-testid="kickoff-events-nav"]')
        target = nav.locator('[data-testid="sub-nav-active-tab"], [data-testid="sub-nav-inactive-tab"]').filter(
            has_text=scope_name
        ).first
        return await target.count() > 0

    async def _extract_home_away_rows(
        self,
        page: Page,
        *,
        event_id: str,
        game_date: str,
        commence_time_utc: str,
        away_team: str,
        home_team: str,
        game_type: str,
        market_type: str,
        fetched_at: str,
    ) -> list[OddsRow]:
        payload = await page.evaluate(
            """
            () => Array.from(document.querySelectorAll('[data-testid="over-under-expanded-row"]')).map((row) => {
                const name = row.querySelector('[data-testid="outrights-expanded-bookmaker-name"]')?.textContent?.trim() || '';
                const odds = Array.from(row.querySelectorAll('[data-testid="odd-container"] .odds-text')).map((node) => node.textContent.trim());
                return { bookmaker: name, odds };
            })
            """
        )
        rows: list[OddsRow] = []
        for item in payload:
            bookmaker = str(item.get("bookmaker") or "").strip()
            odds = item.get("odds") or []
            if not bookmaker or len(odds) < 2:
                continue
            home_price = _decimal_to_american(odds[0])
            away_price = _decimal_to_american(odds[1])
            if home_price is None or away_price is None:
                continue
            rows.append(
                OddsRow(
                    event_id=event_id,
                    game_date=game_date,
                    commence_time_utc=commence_time_utc,
                    away_team=away_team,
                    home_team=home_team,
                    game_type=game_type,
                    fetched_at=fetched_at,
                    bookmaker=f"OddsPortal:{bookmaker}",
                    market_type=market_type,
                    side="home",
                    point=None,
                    price=home_price,
                )
            )
            rows.append(
                OddsRow(
                    event_id=event_id,
                    game_date=game_date,
                    commence_time_utc=commence_time_utc,
                    away_team=away_team,
                    home_team=home_team,
                    game_type=game_type,
                    fetched_at=fetched_at,
                    bookmaker=f"OddsPortal:{bookmaker}",
                    market_type=market_type,
                    side="away",
                    point=None,
                    price=away_price,
                )
            )
        return rows

    async def _extract_parameter_rows(
        self,
        page: Page,
        *,
        event_id: str,
        game_date: str,
        commence_time_utc: str,
        away_team: str,
        home_team: str,
        game_type: str,
        market_type: str,
        fetched_at: str,
        bookmaker_favorite_sides: dict[str, str] | None = None,
    ) -> tuple[list[OddsRow], int]:
        rows: list[OddsRow] = []
        seen_keys: set[tuple[str, str, str, float | None]] = set()
        skipped_collapsed_rows = 0
        collapsed_rows = page.locator('[data-testid="over-under-collapsed-row"]')
        count = await collapsed_rows.count()
        for index in range(count):
            row = collapsed_rows.nth(index)
            try:
                await row.scroll_into_view_if_needed(timeout=1000)
                await row.click(timeout=2500)
                await page.wait_for_timeout(self.expand_row_settle_ms)
            except PlaywrightError:
                skipped_collapsed_rows += 1
                continue
            expanded_payload = await page.evaluate(
                """
                () => Array.from(document.querySelectorAll('[data-testid="over-under-expanded-row"]')).map((row) => {
                    const name = row.querySelector('[data-testid="outrights-expanded-bookmaker-name"]')?.textContent?.trim() || '';
                    const point = row.querySelector('[data-testid="total-container"]')?.textContent?.trim() || '';
                    const odds = Array.from(row.querySelectorAll('[data-testid="odd-container"] .odds-text')).map((node) => node.textContent.trim());
                    return { bookmaker: name, point, odds };
                })
                """
            )
            for item in expanded_payload:
                bookmaker = str(item.get("bookmaker") or "").strip()
                point_value = _coerce_optional_float(item.get("point"))
                odds = item.get("odds") or []
                if not bookmaker or len(odds) < 2 or point_value is None:
                    continue
                side_a, side_b = _oddsportal_primary_secondary_sides(market_type)
                price_a = _decimal_to_american(odds[0])
                price_b = _decimal_to_american(odds[1])
                if price_a is None or price_b is None:
                    continue
                resolved_bookmaker = f"OddsPortal:{bookmaker}"
                favorite_side = (
                    bookmaker_favorite_sides.get(resolved_bookmaker)
                    if bookmaker_favorite_sides is not None
                    else None
                )
                resolved_point_a = (
                    _resolve_signed_runline_point(
                        side=side_a,
                        point_value=point_value,
                        favorite_side=favorite_side,
                    )
                    if market_type.endswith("rl")
                    else point_value
                )
                resolved_point_b = (
                    _resolve_signed_runline_point(
                        side=side_b,
                        point_value=point_value,
                        favorite_side=favorite_side,
                    )
                    if market_type.endswith("rl")
                    else point_value
                )
                key_a = (bookmaker, market_type, side_a, resolved_point_a)
                key_b = (bookmaker, market_type, side_b, resolved_point_b)
                if key_a not in seen_keys:
                    seen_keys.add(key_a)
                    rows.append(
                        OddsRow(
                            event_id=event_id,
                            game_date=game_date,
                            commence_time_utc=commence_time_utc,
                            away_team=away_team,
                            home_team=home_team,
                            game_type=game_type,
                            fetched_at=fetched_at,
                            bookmaker=resolved_bookmaker,
                            market_type=market_type,
                            side=side_a,
                            point=resolved_point_a,
                            price=price_a,
                        )
                    )
                if key_b not in seen_keys:
                    seen_keys.add(key_b)
                    rows.append(
                        OddsRow(
                            event_id=event_id,
                            game_date=game_date,
                            commence_time_utc=commence_time_utc,
                            away_team=away_team,
                            home_team=home_team,
                            game_type=game_type,
                            fetched_at=fetched_at,
                            bookmaker=resolved_bookmaker,
                            market_type=market_type,
                            side=side_b,
                            point=resolved_point_b,
                            price=price_b,
                        )
                    )
            try:
                await collapsed_rows.nth(index).click(timeout=1500)
                await page.wait_for_timeout(self.collapse_row_settle_ms)
            except PlaywrightError:
                continue
        return rows, skipped_collapsed_rows


ODDSPORTAL_TEAM_NAME_TO_CODE = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "St.Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _normalize_oddsportal_team_code(value: str | None) -> str:
    if value is None:
        return ""
    team = str(value).strip()
    mapped = ODDSPORTAL_TEAM_NAME_TO_CODE.get(team)
    if mapped is None:
        raise ValueError(f"Unknown OddsPortal team name: {team}")
    return mapped


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _parse_results_page_date_header(value: str) -> datetime | None:
    match = re.match(r"(\d{2} \w{3} \d{4})", value.strip())
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%d %b %Y")
    except ValueError:
        return None


def _event_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.fragment:
        return f"oddsportal-{parsed.fragment}"
    token = parsed.path.rstrip("/").split("-")[-1]
    return f"oddsportal-{token}"


def _raw_event_token_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.fragment:
        return parsed.fragment
    token = parsed.path.rstrip("/").split("-")[-1]
    return token or None


def _select_match_event_url(match_event_urls: list[str], *, event_url: str) -> str | None:
    event_token = _raw_event_token_from_url(event_url)
    if event_token:
        for candidate in match_event_urls:
            if f"-{event_token}-" in candidate:
                return candidate
    return match_event_urls[0] if match_event_urls else None


def _event_name_from_url(url: str) -> str:
    slug = url.rstrip("/").split("/")[-1]
    if "-" not in slug:
        return slug
    parts = slug.split("-")[:-1]
    return " ".join(part.capitalize() for part in parts)


def _event_progress_key(event: OddsPortalEventInfo) -> str | None:
    if not event.start_date:
        return None
    try:
        event_date = datetime.fromisoformat(str(event.start_date))
    except ValueError:
        return str(event.start_date)[:7]
    week_of_month = ((event_date.day - 1) // 7) + 1
    return f"{event_date.strftime('%Y-%m')} week {week_of_month}"


def _build_results_archive_page_url(first_archive_url: str, page_number: int) -> str:
    if page_number <= 1:
        return first_archive_url
    parsed = urlparse(first_archive_url)
    new_path = re.sub(r"(?:page/\d+/)?$", "", parsed.path)
    if not new_path.endswith("/"):
        new_path += "/"
    new_path = f"{new_path}page/{page_number}/"
    return parsed._replace(path=new_path).geturl()


def _refresh_cache_buster(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["_"] = str(int(datetime.now(tz=UTC).timestamp() * 1000))
    return parsed._replace(query=urlencode(query)).geturl()


def _event_from_results_archive_row(row: Any) -> OddsPortalEventInfo | None:
    if not isinstance(row, dict):
        return None
    href = str(row.get("url") or "").strip()
    if not href:
        return None
    absolute_url = urljoin("https://www.oddsportal.com", href)
    away_team = str(row.get("away-name") or "").strip() or None
    home_team = str(row.get("home-name") or "").strip() or None
    start_timestamp = row.get("date-start-base")
    start_date: str | None = None
    try:
        if start_timestamp is not None:
            start_date = (
                datetime.fromtimestamp(float(start_timestamp), tz=UTC)
                .replace(tzinfo=None)
                .isoformat()
            )
    except (TypeError, ValueError, OSError):
        start_date = None
    event_name = (
        f"{away_team} @ {home_team}"
        if away_team is not None and home_team is not None
        else _event_name_from_url(absolute_url)
    )
    return OddsPortalEventInfo(
        name=event_name,
        url=absolute_url,
        start_date=start_date,
        home_team=home_team,
        away_team=away_team,
    )


def _build_match_event_request_url(base_url: str, *, betting_type_id: int, scope_id: int) -> str:
    parsed = urlparse(base_url)
    match = re.search(r"/match-event/([^/]+)-(\d+)-(\d+)-([^.]+)\.dat$", parsed.path)
    if not match:
        raise ValueError(f"Could not parse OddsPortal match-event URL: {base_url}")
    prefix, _old_bt, _old_sc, hash_token = match.groups()
    new_path = f"/match-event/{prefix}-{betting_type_id}-{scope_id}-{hash_token}.dat"
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "geo" not in query:
        query["geo"] = "CA"
    if "lang" not in query:
        query["lang"] = "en"
    return parsed._replace(path=new_path, query=urlencode(query)).geturl()


def _decrypt_match_event_payload(encoded_text: str) -> dict[str, Any]:
    raw = b64decode(encoded_text).decode("utf-8")
    ciphertext_b64, iv_hex = raw.split(":")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=ODDSPORTAL_DECRYPT_SALT,
        iterations=1000,
        backend=default_backend(),
    )
    key = kdf.derive(ODDSPORTAL_DECRYPT_PASSWORD)
    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(bytes.fromhex(iv_hex)),
        backend=default_backend(),
    )
    decryptor = cipher.decryptor()
    padded = decryptor.update(b64decode(ciphertext_b64)) + decryptor.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()
    payload = json.loads(plaintext.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("OddsPortal decrypted payload was not an object")
    return payload


def _classify_game_type(game_datetime: datetime) -> str:
    season = SEASON_DATES.get(game_datetime.year)
    if season is None:
        return "regular_season"
    naive = game_datetime.replace(tzinfo=None)
    for gap_start, gap_end in season.get("gaps", []):
        if gap_start <= naive <= gap_end:
            return "regular_season"
    if naive < season["opening_day"]:
        return "spring_training"
    if naive >= season["postseason_start"]:
        return "postseason"
    return "regular_season"


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _decimal_to_american(value: str | float | int | None) -> int | None:
    decimal = _coerce_optional_float(value)
    if decimal is None or decimal <= 1.0:
        return None
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def _epoch_seconds_to_iso8601(value: Any) -> str | None:
    try:
        epoch_seconds = float(value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(epoch_seconds, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return None
