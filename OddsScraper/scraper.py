"""
MLB Historical Odds Scraper for SportsbookReview.com

Normalized output: one row per game × bookmaker × market × side.
Scrapes from today backwards with game_type classification.
"""

import asyncio
import csv
import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, asdict, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Page, BrowserContext

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("scraper.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class OddsRow:
    """One normalized odds observation: game x bookmaker x market x side."""

    event_id: str = ""
    game_date: str = ""
    commence_time_utc: str = ""  # ISO 8601 UTC e.g. "2025-07-05T00:05:00Z"
    away_team: str = ""
    home_team: str = ""
    game_type: str = ""  # spring_training, regular_season, postseason
    away_pitcher: str = ""
    home_pitcher: str = ""
    fetched_at: str = ""
    bookmaker: str = ""
    market_type: str = ""  # full_game_ml, full_game_rl, full_game_total, f5_ml, f5_rl, f5_total
    side: str = ""  # home, away, over, under
    point: Optional[float] = None  # spread/total value, None for moneyline
    price: int = 0  # American odds as integer
    is_opening: bool = False
    game_id: int = 0  # FK to games table


ODDS_ROW_FIELDS = [f.name for f in fields(OddsRow)]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Bet-type key → human-readable market_type
MARKET_TYPE_MAP = {
    "ml_full": "full_game_ml",
    "rl_full": "full_game_rl",
    "totals_full": "full_game_total",
    "ml_1st5": "f5_ml",
    "rl_1st5": "f5_rl",
    "totals_1st5": "f5_total",
}

# URL patterns for each bet type
URL_PATTERNS = {
    "ml_full": "https://www.sportsbookreview.com/betting-odds/mlb-baseball/?date={date}",
    "rl_full": "https://www.sportsbookreview.com/betting-odds/mlb-baseball/pointspread/full-game/?date={date}",
    "totals_full": "https://www.sportsbookreview.com/betting-odds/mlb-baseball/totals/full-game/?date={date}",
    "ml_1st5": "https://www.sportsbookreview.com/betting-odds/mlb-baseball/money-line/1st-half/?date={date}",
    "rl_1st5": "https://www.sportsbookreview.com/betting-odds/mlb-baseball/pointspread/1st-half/?date={date}",
    "totals_1st5": "https://www.sportsbookreview.com/betting-odds/mlb-baseball/totals/1st-half/?date={date}",
}

# Scrape ML pages first so pitcher info propagates to RL/Totals rows
BET_TYPE_ORDER = ["ml_full", "ml_1st5", "rl_full", "rl_1st5", "totals_full", "totals_1st5"]

# Timezone abbreviation -> UTC offset hours
TZ_OFFSETS = {
    "EDT": -4,
    "EST": -5,
    "CDT": -5,
    "CST": -6,
    "MDT": -6,
    "MST": -7,
    "PDT": -7,
    "PST": -8,
    "ET": -5,
    "CT": -6,
    "MT": -7,
    "PT": -8,
}

# Team abbreviations for validation
TEAM_ABBREVS = {
    "ARI",
    "ATL",
    "BAL",
    "BOS",
    "CHC",
    "CIN",
    "CLE",
    "COL",
    "CWS",
    "DET",
    "HOU",
    "KC",
    "LAA",
    "LAD",
    "MIA",
    "MIL",
    "MIN",
    "NYM",
    "NYY",
    "OAK",
    "PHI",
    "PIT",
    "SD",
    "SEA",
    "SF",
    "STL",
    "TB",
    "TEX",
    "TOR",
    "WSH",
    # Historical abbreviations
    "ANA",
    "FLA",
    "MON",
}

# Season boundaries per year for game_type classification.
# spring_training_start → opening_day → postseason_start → season_end
SEASON_DATES = {
    2015: {
        "spring_training_start": datetime(2015, 2, 21),
        "opening_day": datetime(2015, 4, 5),
        "postseason_start": datetime(2015, 10, 6),
        "season_end": datetime(2015, 11, 2),
    },
    2016: {
        "spring_training_start": datetime(2016, 2, 20),
        "opening_day": datetime(2016, 4, 3),
        "postseason_start": datetime(2016, 10, 4),
        "season_end": datetime(2016, 11, 3),
    },
    2017: {
        "spring_training_start": datetime(2017, 2, 22),
        "opening_day": datetime(2017, 4, 2),
        "postseason_start": datetime(2017, 10, 3),
        "season_end": datetime(2017, 11, 2),
    },
    2018: {
        "spring_training_start": datetime(2018, 2, 21),
        "opening_day": datetime(2018, 3, 29),
        "postseason_start": datetime(2018, 10, 2),
        "season_end": datetime(2018, 10, 29),
    },
    2019: {
        "spring_training_start": datetime(2019, 2, 20),
        "opening_day": datetime(2019, 3, 28),
        "postseason_start": datetime(2019, 10, 1),
        "season_end": datetime(2019, 10, 31),
    },
    2020: {
        # COVID: ST suspended Mar 12, resumed as Summer Camp Jul 1
        "spring_training_start": datetime(2020, 2, 21),
        "opening_day": datetime(2020, 7, 23),
        "postseason_start": datetime(2020, 9, 29),
        "season_end": datetime(2020, 10, 28),
        "gaps": [(datetime(2020, 3, 13), datetime(2020, 7, 22))],
    },
    2021: {
        "spring_training_start": datetime(2021, 2, 25),
        "opening_day": datetime(2021, 4, 1),
        "postseason_start": datetime(2021, 10, 5),
        "season_end": datetime(2021, 11, 3),
    },
    2022: {
        # Lockout delayed spring training to mid-March
        "spring_training_start": datetime(2022, 3, 17),
        "opening_day": datetime(2022, 4, 7),
        "postseason_start": datetime(2022, 10, 7),
        "season_end": datetime(2022, 11, 6),
    },
    2023: {
        "spring_training_start": datetime(2023, 2, 22),
        "opening_day": datetime(2023, 3, 30),
        "postseason_start": datetime(2023, 10, 3),
        "season_end": datetime(2023, 11, 2),
    },
    2024: {
        "spring_training_start": datetime(2024, 2, 21),
        "opening_day": datetime(2024, 3, 20),  # Seoul Series
        "postseason_start": datetime(2024, 10, 1),
        "season_end": datetime(2024, 11, 3),
    },
    2025: {
        "spring_training_start": datetime(2025, 2, 20),
        "opening_day": datetime(2025, 3, 27),
        "postseason_start": datetime(2025, 9, 30),
        "season_end": datetime(2025, 11, 2),
    },
    2026: {
        "spring_training_start": datetime(2026, 2, 20),
        "opening_day": datetime(2026, 3, 25),
        "postseason_start": datetime(2026, 10, 1),
        "season_end": datetime(2026, 11, 5),
    },
}

# ---------------------------------------------------------------------------
# Unified page-extraction JavaScript
# ---------------------------------------------------------------------------
# Runs in browser context — MUST be plain JS (no TypeScript).
# Reads bookmaker names from header, then for each game extracts team info
# and per-bookmaker raw cell values.  Parsing happens in Python.

EXTRACT_PAGE_JS = """
() => {
    const result = { bookmakers: [], games: [] };

    // 1. Bookmaker names from header logos
    const header = document.querySelector('[class*="Sportsbooks_sportbooks__"]');
    if (header) {
        const logos = header.querySelectorAll('img[alt*="Logo"]');
        result.bookmakers = Array.from(logos).map(
            img => img.alt.replace(' Logo', '')
        );
    }

    // 2. Each game row
    const containers = document.querySelectorAll(
        '[class*="GameRows_eventMarketGridContainer"]'
    );

    containers.forEach(container => {
        const info = container.querySelector(
            '[class*="compactBettingOptionContainer"]'
        );
        if (!info) return;

        const teamBs = info.querySelectorAll('b');
        if (teamBs.length < 2) return;

        const timeSpan = info.querySelector('span.fs-9');
        const pitchers = info.querySelectorAll('[class*="pitcherText"]');
        const link     = info.querySelector('a[href*="matchup"]');

        const game = {
            awayTeam:    teamBs[0].textContent.trim(),
            homeTeam:    teamBs[1].textContent.trim(),
            gameTime:    timeSpan ? timeSpan.textContent.trim() : '',
            awayPitcher: pitchers[0] ? pitchers[0].textContent.trim() : '',
            homePitcher: pitchers[1] ? pitchers[1].textContent.trim() : '',
            eventId:     link
                ? (link.href.match(/matchup\\/(\\d+)/)?.[1] || '')
                : '',
            columns: [],
        };

        // 3. Per-bookmaker odds columns
        const colsWrap = container.querySelector(
            '[class*="GameRows_columnsContainer"]'
        );
        if (colsWrap) {
            Array.from(colsWrap.children).forEach((col, idx) => {
                // Collect every non-empty leaf text in DOM order
                const leaves = [];
                col.querySelectorAll('*').forEach(el => {
                    if (el.children.length === 0) {
                        const t = el.textContent.trim();
                        if (t.length > 0) leaves.push(t);
                    }
                });

                // First half = away / over row, second half = home / under row
                const mid = Math.floor(leaves.length / 2);
                game.columns.push({
                    index: idx,
                    awayValues: leaves.slice(0, mid || leaves.length),
                    homeValues: mid > 0 ? leaves.slice(mid) : [],
                });
            });
        }

        // 4. Opener column (col-2, second child)
        const consensusWrap = container.querySelector(
            '[class*="consensusAndoddsContainer"]'
        );
        if (consensusWrap) {
            const col2 = consensusWrap.querySelector('[class*="col-2"]');
            if (col2 && col2.children.length >= 2) {
                const openerCol = col2.children[1]; // second col-6 = OPENER
                const oLeaves = [];
                openerCol.querySelectorAll('*').forEach(el => {
                    if (el.children.length === 0) {
                        const t = el.textContent.trim();
                        if (t.length > 0) oLeaves.push(t);
                    }
                });
                const oMid = Math.floor(oLeaves.length / 2);
                game.openerAway = oLeaves.slice(0, oMid || oLeaves.length);
                game.openerHome = oMid > 0 ? oLeaves.slice(oMid) : [];
            }
        }

        result.games.push(game);
    });

    return result;
}
"""

# ---------------------------------------------------------------------------
# Adaptive rate limiter
# ---------------------------------------------------------------------------


class AdaptiveThrottle:
    """Simple adaptive rate limiter.

    Starts fast (parallel tabs, short delays).  On errors or slow responses
    it drops to sequential with longer delays, then gradually scales back up
    after sustained healthy responses.
    """

    SLOW_THRESHOLD = 10.0  # seconds — page load considered "slow"
    BACKOFF_ERRORS = 2  # consecutive errors before backing off
    BACKOFF_SLOW = 3  # slow responses in a window before backing off
    RECOVER_AFTER = 20  # consecutive successes before trying to scale up

    def __init__(self, base_delay: float = 2.0, max_parallel: int = 3):
        self.base_delay = base_delay
        self.max_parallel = max_parallel
        # Live settings
        self.delay = base_delay
        self.parallel = max_parallel
        # Counters
        self._errors = 0
        self._successes = 0
        self._slow = 0

    def record_success(self, elapsed: float) -> None:
        self._errors = 0
        self._successes += 1
        if elapsed > self.SLOW_THRESHOLD:
            self._slow += 1
            if self._slow >= self.BACKOFF_SLOW:
                self._back_off("slow responses")
                self._slow = 0
        else:
            self._slow = max(0, self._slow - 1)
        if self._successes >= self.RECOVER_AFTER:
            self._scale_up()
            self._successes = 0

    def record_error(self) -> None:
        self._errors += 1
        self._successes = 0
        if self._errors >= self.BACKOFF_ERRORS:
            self._back_off("consecutive errors")

    def _back_off(self, reason: str) -> None:
        old_p, old_d = self.parallel, self.delay
        if self.parallel > 1:
            self.parallel = 1
        else:
            self.delay = min(self.delay * 1.5, 10.0)
        logger.warning(
            f"Throttle back-off ({reason}): parallel {old_p}->{self.parallel}, "
            f"delay {old_d:.1f}s->{self.delay:.1f}s"
        )

    def _scale_up(self) -> None:
        old_p, old_d = self.parallel, self.delay
        if self.delay > self.base_delay:
            self.delay = max(self.delay / 1.25, self.base_delay)
        elif self.parallel < self.max_parallel:
            self.parallel = min(self.parallel + 1, self.max_parallel)
        if old_p != self.parallel or abs(old_d - self.delay) > 0.1:
            logger.info(
                f"Throttle scaling up: parallel {old_p}->{self.parallel}, "
                f"delay {old_d:.1f}s->{self.delay:.1f}s"
            )

    async def wait(self) -> None:
        await asyncio.sleep(self.delay)


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------


class MLBOddsScraper:
    """Scraper producing normalized odds rows.

    Features:
    - Parallel browser tabs for concurrent bet-type scraping
    - Adaptive throttle that backs off on errors / slow responses
    - Fast page loading (domcontentloaded + selector wait)
    - Early exit when a date has no games
    """

    def __init__(
        self,
        output_dir: str = "data",
        base_delay: float = 2.0,
        max_parallel: int = 3,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.throttle = AdaptiveThrottle(
            base_delay=base_delay,
            max_parallel=max_parallel,
        )

    # ---- date classification ---------------------------------------------

    def classify_date(self, date: datetime) -> Optional[str]:
        """Return spring_training / regular_season / postseason, or None for off-season."""
        for year in sorted(SEASON_DATES.keys(), reverse=True):
            cfg = SEASON_DATES[year]
            for gap_start, gap_end in cfg.get("gaps", []):
                if gap_start <= date <= gap_end:
                    return None
            if cfg["spring_training_start"] <= date < cfg["opening_day"]:
                return "spring_training"
            if cfg["opening_day"] <= date < cfg["postseason_start"]:
                return "regular_season"
            if cfg["postseason_start"] <= date <= cfg["season_end"]:
                return "postseason"
        return None

    def _find_previous_season_end(self, date: datetime) -> Optional[datetime]:
        """Nearest season_end strictly before *date*."""
        candidates = [c["season_end"] for c in SEASON_DATES.values() if c["season_end"] < date]
        return max(candidates) if candidates else None

    # ---- value parsing ---------------------------------------------------

    @staticmethod
    def _parse_price(raw: str) -> Optional[int]:
        """Normalise a single odds string to integer.  Returns None when absent."""
        if not raw or raw in ("-", "--", ""):
            return None
        if raw.upper() == "EVEN":
            return 100
        if re.match(r"^[+-]\d{2,4}$", raw):
            return int(raw)
        return None

    @staticmethod
    def _parse_point(raw: str) -> Optional[float]:
        """Parse a spread or total value to float.  Returns None for ML (empty)."""
        if not raw or raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _parse_time_to_utc(game_date: str, time_str: str) -> str:
        """Convert '8:05 PM EDT' + '2025-07-04' -> '2025-07-05T00:05:00Z'."""
        if not time_str or time_str.upper() in ("TBD", "PPD", ""):
            return ""
        m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)\s*(\w+)", time_str.strip(), re.IGNORECASE)
        if not m:
            return ""
        hour, minute, ampm, tz_abbr = (
            int(m.group(1)),
            int(m.group(2)),
            m.group(3).upper(),
            m.group(4).upper(),
        )
        if ampm == "PM" and hour != 12:
            hour += 12
        elif ampm == "AM" and hour == 12:
            hour = 0
        offset_h = TZ_OFFSETS.get(tz_abbr, -4)  # default EDT
        dt = datetime.strptime(game_date, "%Y-%m-%d").replace(
            hour=hour,
            minute=minute,
            tzinfo=timezone(timedelta(hours=offset_h)),
        )
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _rows_from_column(
        self,
        bet_type: str,
        bookmaker: str,
        away_vals: list[str],
        home_vals: list[str],
        is_opening: bool = False,
    ) -> list[dict]:
        """Convert one bookmaker column's raw leaf texts into row dicts.

        Returns up to 2 dicts ``{bookmaker, market_type, side, point, price, is_opening}``.
        """
        mkt = MARKET_TYPE_MAP[bet_type]
        rows: list[dict] = []

        if "ml" in bet_type:
            ap = self._parse_price(away_vals[0] if away_vals else "")
            hp = self._parse_price(home_vals[0] if home_vals else "")
            if ap is not None:
                rows.append(
                    {
                        "bookmaker": bookmaker,
                        "market_type": mkt,
                        "side": "away",
                        "point": None,
                        "price": ap,
                        "is_opening": is_opening,
                    }
                )
            if hp is not None:
                rows.append(
                    {
                        "bookmaker": bookmaker,
                        "market_type": mkt,
                        "side": "home",
                        "point": None,
                        "price": hp,
                        "is_opening": is_opening,
                    }
                )

        elif "rl" in bet_type:
            if len(away_vals) >= 2:
                price = self._parse_price(away_vals[1])
                if price is not None:
                    rows.append(
                        {
                            "bookmaker": bookmaker,
                            "market_type": mkt,
                            "side": "away",
                            "point": self._parse_point(away_vals[0]),
                            "price": price,
                            "is_opening": is_opening,
                        }
                    )
            if len(home_vals) >= 2:
                price = self._parse_price(home_vals[1])
                if price is not None:
                    rows.append(
                        {
                            "bookmaker": bookmaker,
                            "market_type": mkt,
                            "side": "home",
                            "point": self._parse_point(home_vals[0]),
                            "price": price,
                            "is_opening": is_opening,
                        }
                    )

        elif "totals" in bet_type:
            if len(away_vals) >= 2:
                price = self._parse_price(away_vals[1])
                if price is not None:
                    rows.append(
                        {
                            "bookmaker": bookmaker,
                            "market_type": mkt,
                            "side": "over",
                            "point": self._parse_point(away_vals[0]),
                            "price": price,
                            "is_opening": is_opening,
                        }
                    )
            if len(home_vals) >= 2:
                price = self._parse_price(home_vals[1])
                if price is not None:
                    rows.append(
                        {
                            "bookmaker": bookmaker,
                            "market_type": mkt,
                            "side": "under",
                            "point": self._parse_point(home_vals[0]),
                            "price": price,
                            "is_opening": is_opening,
                        }
                    )

        return rows

    # ---- page loading (fast strategy) ------------------------------------

    async def _load_page(self, page: Page, url: str) -> bool:
        """Navigate with fast loading: domcontentloaded + targeted selector wait.

        Returns True if the page has game content (not 'No odds available').
        """
        t0 = time.monotonic()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            try:
                await page.wait_for_selector(
                    '[class*="eventMarketGridContainer"]',
                    timeout=6000,
                )
            except Exception:
                # No game rows appeared — might be empty date or slow load
                await asyncio.sleep(0.5)

            elapsed = time.monotonic() - t0
            content = await page.content()
            has_odds = "No odds available" not in content
            self.throttle.record_success(elapsed)
            return has_odds

        except Exception as exc:
            self.throttle.record_error()
            raise exc

    # ---- single bet-type scrape ------------------------------------------

    async def scrape_bet_type(
        self,
        page: Page,
        date: datetime,
        bet_type: str,
    ) -> list[dict]:
        """Scrape one market type for one date.

        Returns flat list of dicts ready to become OddsRow (minus game_type / fetched_at).
        """
        date_str = date.strftime("%Y-%m-%d")
        url = URL_PATTERNS[bet_type].format(date=date_str)
        logger.info(f"  Scraping {bet_type} for {date_str}...")

        try:
            has_odds = await self._load_page(page, url)
            if not has_odds:
                logger.info(f"    No {bet_type} odds for {date_str}")
                return []

            data = await page.evaluate(EXTRACT_PAGE_JS)
            bookmakers: list[str] = data.get("bookmakers", [])
            games: list[dict] = data.get("games", [])

            all_rows: list[dict] = []
            for game in games:
                if game["awayTeam"] not in TEAM_ABBREVS and game["homeTeam"] not in TEAM_ABBREVS:
                    continue

                game_base = {
                    "awayTeam": game["awayTeam"],
                    "homeTeam": game["homeTeam"],
                    "gameTime": game["gameTime"],
                    "awayPitcher": game.get("awayPitcher", ""),
                    "homePitcher": game.get("homePitcher", ""),
                    "eventId": game.get("eventId", ""),
                }

                # Regular bookmaker columns
                for col in game.get("columns", []):
                    idx = col["index"]
                    bk = bookmakers[idx] if idx < len(bookmakers) else f"book_{idx}"
                    parsed = self._rows_from_column(
                        bet_type,
                        bk,
                        col.get("awayValues", []),
                        col.get("homeValues", []),
                        is_opening=False,
                    )
                    for r in parsed:
                        r.update(game_base)
                    all_rows.extend(parsed)

                # Opener column (is_opening=True)
                opener_away = game.get("openerAway", [])
                opener_home = game.get("openerHome", [])
                if opener_away or opener_home:
                    parsed = self._rows_from_column(
                        bet_type,
                        "Opener",
                        opener_away,
                        opener_home,
                        is_opening=True,
                    )
                    for r in parsed:
                        r.update(game_base)
                    all_rows.extend(parsed)

            logger.info(f"    {bet_type}: {len(games)} games, {len(all_rows)} rows")
            return all_rows

        except Exception as e:
            logger.error(f"    Error scraping {bet_type} for {date_str}: {e}")
            return []

    # ---- full-date scrape (parallel tabs + early exit) -------------------

    async def scrape_date(
        self,
        context: BrowserContext,
        date: datetime,
    ) -> list[OddsRow]:
        """Scrape all six market types for *date*.

        - Scrapes ml_full first (to capture pitcher info + detect empty dates)
        - If no games found, skips the remaining 5 bet types (early exit)
        - Scrapes remaining bet types in parallel batches via multiple tabs
        """
        date_str = date.strftime("%Y-%m-%d")
        fetched_at = datetime.now().isoformat()
        logger.info(f"Scraping all bet types for {date_str}...")

        # ---- Phase 1: ml_full (must be first for pitcher info) -----------
        lead_page = await context.new_page()
        try:
            ml_full_rows = await self.scrape_bet_type(lead_page, date, "ml_full")
        finally:
            await lead_page.close()

        if not ml_full_rows:
            logger.info(f"  No games on {date_str} — skipping remaining bet types")
            return []

        # Build game-info lookup from ML results (pitchers, eventId)
        game_info: dict[tuple, dict] = {}
        for r in ml_full_rows:
            key = (r["awayTeam"], r["homeTeam"], r["gameTime"])
            if key not in game_info:
                game_info[key] = {
                    "awayPitcher": r.get("awayPitcher", ""),
                    "homePitcher": r.get("homePitcher", ""),
                    "eventId": r.get("eventId", ""),
                }

        # ---- Phase 2: remaining 5 bet types in parallel batches ----------
        remaining = ["ml_1st5", "rl_full", "rl_1st5", "totals_full", "totals_1st5"]

        async def _scrape_one(bet_type: str) -> list[dict]:
            pg = await context.new_page()
            try:
                return await self.scrape_bet_type(pg, date, bet_type)
            finally:
                await pg.close()

        all_raw: list[dict] = list(ml_full_rows)  # start with ml_full results

        parallel = self.throttle.parallel
        for i in range(0, len(remaining), parallel):
            batch = remaining[i : i + parallel]
            batch_results = await asyncio.gather(
                *[_scrape_one(bt) for bt in batch],
            )
            for result in batch_results:
                all_raw.extend(result)
            # Delay between batches (not after the last one)
            if i + parallel < len(remaining):
                await self.throttle.wait()

        # ---- Phase 3: build OddsRows with merged game info ---------------
        all_rows: list[OddsRow] = []
        for r in all_raw:
            key = (r["awayTeam"], r["homeTeam"], r["gameTime"])

            # Update game_info from any new data (e.g. eventId from other pages)
            if key not in game_info:
                game_info[key] = {
                    "awayPitcher": r.get("awayPitcher", ""),
                    "homePitcher": r.get("homePitcher", ""),
                    "eventId": r.get("eventId", ""),
                }
            else:
                gi = game_info[key]
                if r.get("awayPitcher") and not gi["awayPitcher"]:
                    gi["awayPitcher"] = r["awayPitcher"]
                    gi["homePitcher"] = r.get("homePitcher", "")
                if r.get("eventId") and not gi["eventId"]:
                    gi["eventId"] = r["eventId"]

            gi = game_info[key]
            all_rows.append(
                OddsRow(
                    event_id=gi.get("eventId", ""),
                    game_date=date_str,
                    commence_time_utc=self._parse_time_to_utc(date_str, r["gameTime"]),
                    away_team=r["awayTeam"],
                    home_team=r["homeTeam"],
                    game_type="",  # set by caller
                    away_pitcher=gi.get("awayPitcher", ""),
                    home_pitcher=gi.get("homePitcher", ""),
                    fetched_at=fetched_at,
                    bookmaker=r["bookmaker"],
                    market_type=r["market_type"],
                    side=r["side"],
                    point=r["point"],
                    price=r["price"],
                    is_opening=r.get("is_opening", False),
                )
            )

        logger.info(f"Total rows for {date_str}: {len(all_rows)}")
        return all_rows

    # ---- backwards date-range scrape -------------------------------------

    async def scrape_date_range_backwards(
        self,
        context: BrowserContext,
        start_date: datetime,
        end_date: datetime,
        progress_filename: str = "mlb_odds_progress.csv",
    ) -> list[OddsRow]:
        """Scrape from *start_date* backwards to *end_date*.

        - Labels every row with game_type
        - Auto-skips off-season gaps
        - Saves progress CSV every 10 dates
        """
        logger.info(
            f"Scraping backwards from {start_date.strftime('%Y-%m-%d')} "
            f"to {end_date.strftime('%Y-%m-%d')} "
            f"(parallel={self.throttle.parallel}, delay={self.throttle.delay:.1f}s)"
        )

        all_rows: list[OddsRow] = []
        current = start_date
        dates_scraped = 0
        consecutive_empty = 0

        while current >= end_date:
            game_type = self.classify_date(current)

            if game_type is None:
                prev_end = self._find_previous_season_end(current)
                if prev_end and prev_end >= end_date:
                    logger.info(
                        f"Off-season: jumping from {current.strftime('%Y-%m-%d')} "
                        f"to {prev_end.strftime('%Y-%m-%d')}"
                    )
                    current = prev_end
                    consecutive_empty = 0
                    continue
                else:
                    logger.info(f"No more seasons before {current.strftime('%Y-%m-%d')}. Done.")
                    break

            rows = await self.scrape_date(context, current)

            for r in rows:
                r.game_type = game_type

            all_rows.extend(rows)
            dates_scraped += 1

            if rows:
                consecutive_empty = 0
                logger.info(
                    f"  [{game_type}] {current.strftime('%Y-%m-%d')}: "
                    f"{len(rows)} rows (total: {len(all_rows)}) "
                    f"[p={self.throttle.parallel} d={self.throttle.delay:.1f}s]"
                )
            else:
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    logger.info(f"  {consecutive_empty} consecutive empty days")

            if dates_scraped % 10 == 0:
                self._save_progress(current, all_rows, progress_filename)

            current -= timedelta(days=1)

        self._save_progress(current, all_rows, progress_filename)
        logger.info(f"Backwards scrape complete: {len(all_rows)} rows across {dates_scraped} dates")
        return all_rows

    # ---- persistence -----------------------------------------------------

    def save_to_csv(self, rows: list[OddsRow], filename: str) -> None:
        filepath = self.output_dir / filename
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ODDS_ROW_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        logger.info(f"Saved {len(rows)} rows to {filepath}")

    def save_to_json(self, rows: list[OddsRow], filename: str) -> None:
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in rows], f, indent=2)
        logger.info(f"Saved {len(rows)} rows to {filepath}")

    def _save_progress(
        self,
        current_date: datetime,
        rows: list[OddsRow],
        csv_filename: str = "",
    ) -> None:
        progress_file = self.output_dir / "progress.json"
        progress = {
            "last_scraped_date": current_date.strftime("%Y-%m-%d"),
            "total_rows": len(rows),
            "last_updated": datetime.now().isoformat(),
        }
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)
        if csv_filename and rows:
            self.save_to_csv(rows, csv_filename)

    # ---- SQLite-backed backwards scrape ----------------------------------

    async def scrape_date_range_backwards_db(
        self,
        context: BrowserContext,
        start_date: datetime,
        end_date: datetime,
        db: "SQLiteStore",
    ) -> int:
        """Scrape from *start_date* backwards to *end_date*, writing to SQLite.

        Rows are committed per-date (no in-memory accumulation).
        Returns total rows inserted this run.

        If the database already has data, automatically resumes from the
        earliest date present (skips dates already scraped).
        """
        # Resume support: if DB has data, start from earliest date - 1 day
        resume_date = db.get_earliest_scraped_date()
        if resume_date:
            candidate = datetime.strptime(resume_date, "%Y-%m-%d") - timedelta(days=1)
            if candidate < start_date:
                logger.info(
                    f"Resuming: DB has data down to {resume_date}, "
                    f"continuing from {candidate.strftime('%Y-%m-%d')}"
                )
                start_date = candidate

        logger.info(
            f"Scraping backwards from {start_date.strftime('%Y-%m-%d')} "
            f"to {end_date.strftime('%Y-%m-%d')} -> SQLite "
            f"(parallel={self.throttle.parallel}, delay={self.throttle.delay:.1f}s)"
        )

        total_inserted = 0
        current = start_date
        dates_scraped = 0
        consecutive_empty = 0

        while current >= end_date:
            game_type = self.classify_date(current)

            if game_type is None:
                prev_end = self._find_previous_season_end(current)
                if prev_end and prev_end >= end_date:
                    logger.info(
                        f"Off-season: jumping from {current.strftime('%Y-%m-%d')} "
                        f"to {prev_end.strftime('%Y-%m-%d')}"
                    )
                    current = prev_end
                    consecutive_empty = 0
                    continue
                else:
                    logger.info(f"No more seasons before {current.strftime('%Y-%m-%d')}. Done.")
                    break

            rows = await self.scrape_date(context, current)

            for r in rows:
                r.game_type = game_type

            if rows:
                db.insert_rows(rows)
                total_inserted += len(rows)
                consecutive_empty = 0
                logger.info(
                    f"  [{game_type}] {current.strftime('%Y-%m-%d')}: "
                    f"{len(rows)} rows (db total: {db.count()}) "
                    f"[p={self.throttle.parallel} d={self.throttle.delay:.1f}s]"
                )
            else:
                consecutive_empty += 1
                if consecutive_empty >= 5:
                    logger.info(f"  {consecutive_empty} consecutive empty days")

            dates_scraped += 1
            current -= timedelta(days=1)

        logger.info(
            f"Backwards scrape complete: {total_inserted} new rows across "
            f"{dates_scraped} dates (db total: {db.count()})"
        )
        return total_inserted


# ---------------------------------------------------------------------------
# SQLite store
# ---------------------------------------------------------------------------


class SQLiteStore:
    """Lightweight SQLite wrapper for the odds warehouse.

    - Creates the odds + games tables with proper types and indexes on first use
    - Batch inserts with one transaction per date
    - Resume support via earliest-date query
    - CSV export for portability
    """

    ODDS_DDL = """
    CREATE TABLE IF NOT EXISTS odds (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id          TEXT,
        game_date         TEXT NOT NULL,
        commence_time_utc TEXT,
        away_team         TEXT NOT NULL,
        home_team         TEXT NOT NULL,
        game_type         TEXT,
        away_pitcher      TEXT,
        home_pitcher      TEXT,
        fetched_at        TEXT,
        bookmaker         TEXT NOT NULL,
        market_type       TEXT NOT NULL,
        side              TEXT NOT NULL,
        point             REAL,
        price             INTEGER NOT NULL,
        is_opening        INTEGER DEFAULT 0,
        game_id           INTEGER REFERENCES games(game_id)
    );
    """

    GAMES_DDL = """
    CREATE TABLE IF NOT EXISTS games (
        game_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        event_id          TEXT UNIQUE,
        game_date         TEXT NOT NULL,
        commence_time_utc TEXT,
        away_team         TEXT NOT NULL,
        home_team         TEXT NOT NULL,
        game_type         TEXT,
        away_pitcher      TEXT,
        home_pitcher      TEXT
    );
    """

    INDEXES = [
        "CREATE INDEX IF NOT EXISTS idx_odds_game_date     ON odds (game_date);",
        "CREATE INDEX IF NOT EXISTS idx_odds_market_type   ON odds (market_type);",
        "CREATE INDEX IF NOT EXISTS idx_odds_game_type     ON odds (game_type);",
        "CREATE INDEX IF NOT EXISTS idx_odds_bookmaker     ON odds (bookmaker);",
        "CREATE INDEX IF NOT EXISTS idx_odds_teams         ON odds (away_team, home_team);",
        "CREATE INDEX IF NOT EXISTS idx_odds_event_id      ON odds (event_id);",
        "CREATE INDEX IF NOT EXISTS idx_odds_game_date_mkt ON odds (game_date, market_type);",
        "CREATE INDEX IF NOT EXISTS idx_odds_game_id       ON odds (game_id);",
        "CREATE INDEX IF NOT EXISTS idx_odds_fetched_at    ON odds (fetched_at);",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_odds_unique  ON odds (event_id, fetched_at, bookmaker, market_type, side, COALESCE(point, -999));",
    ]

    INSERT_SQL = """
    INSERT OR IGNORE INTO odds (
        event_id, game_date, commence_time_utc, away_team, home_team,
        game_type, away_pitcher, home_pitcher, fetched_at,
        bookmaker, market_type, side, point, price, is_opening, game_id
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    GAME_UPSERT_SQL = """
    INSERT INTO games (event_id, game_date, commence_time_utc, away_team, home_team,
                       game_type, away_pitcher, home_pitcher)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(event_id) DO UPDATE SET
        commence_time_utc = COALESCE(excluded.commence_time_utc, commence_time_utc),
        game_type         = COALESCE(excluded.game_type, game_type),
        away_pitcher      = CASE WHEN excluded.away_pitcher != '' THEN excluded.away_pitcher ELSE away_pitcher END,
        home_pitcher      = CASE WHEN excluded.home_pitcher != '' THEN excluded.home_pitcher ELSE home_pitcher END
    RETURNING game_id;
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(self.GAMES_DDL)  # games first (FK target)
        self._conn.executescript(self.ODDS_DDL)
        for idx in self.INDEXES:
            try:
                self._conn.execute(idx)
            except sqlite3.OperationalError:
                pass  # index may already exist with different definition
        self._conn.commit()
        logger.info(f"SQLite store opened: {self.db_path} ({self.count()} existing rows)")

    # ---- writes ----------------------------------------------------------

    def get_or_create_game(self, row: OddsRow) -> int:
        """Insert or update a game, return its game_id."""
        cur = self._conn.execute(
            self.GAME_UPSERT_SQL,
            (
                row.event_id,
                row.game_date,
                row.commence_time_utc,
                row.away_team,
                row.home_team,
                row.game_type,
                row.away_pitcher,
                row.home_pitcher,
            ),
        )
        result = cur.fetchone()
        return result[0] if result else 0

    def insert_rows(self, rows: list[OddsRow]) -> None:
        """Batch-insert rows in a single transaction, auto-creating games."""
        # Build game_id cache for this batch
        game_cache: dict[str, int] = {}
        for r in rows:
            if r.event_id and r.event_id not in game_cache:
                game_cache[r.event_id] = self.get_or_create_game(r)

        tuples = [
            (
                r.event_id,
                r.game_date,
                r.commence_time_utc,
                r.away_team,
                r.home_team,
                r.game_type,
                r.away_pitcher,
                r.home_pitcher,
                r.fetched_at,
                r.bookmaker,
                r.market_type,
                r.side,
                r.point,
                r.price,
                1 if r.is_opening else 0,
                game_cache.get(r.event_id, 0),
            )
            for r in rows
        ]
        self._conn.executemany(self.INSERT_SQL, tuples)
        self._conn.commit()

    # ---- reads -----------------------------------------------------------

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0]

    def get_earliest_scraped_date(self) -> Optional[str]:
        row = self._conn.execute("SELECT MIN(game_date) FROM odds").fetchone()
        return row[0] if row and row[0] else None

    def get_latest_scraped_date(self) -> Optional[str]:
        row = self._conn.execute("SELECT MAX(game_date) FROM odds").fetchone()
        return row[0] if row and row[0] else None

    def date_range(self) -> tuple[Optional[str], Optional[str]]:
        return self.get_earliest_scraped_date(), self.get_latest_scraped_date()

    def summary(self) -> dict:
        c = self._conn
        total = self.count()
        if total == 0:
            return {"total_rows": 0}

        earliest, latest = self.date_range()
        dates = c.execute("SELECT COUNT(DISTINCT game_date) FROM odds").fetchone()[0]
        games = c.execute(
            "SELECT COUNT(DISTINCT game_date || away_team || home_team || commence_time_utc) FROM odds"
        ).fetchone()[0]
        game_count = c.execute("SELECT COUNT(*) FROM games").fetchone()[0]

        by_type = dict(
            c.execute("SELECT game_type, COUNT(*) FROM odds GROUP BY game_type").fetchall()
        )
        by_market = dict(
            c.execute("SELECT market_type, COUNT(*) FROM odds GROUP BY market_type").fetchall()
        )
        by_book = dict(
            c.execute("SELECT bookmaker, COUNT(*) FROM odds GROUP BY bookmaker").fetchall()
        )

        return {
            "total_rows": total,
            "date_range": f"{earliest} to {latest}",
            "unique_dates": dates,
            "unique_games": games,
            "games_table": game_count,
            "by_game_type": by_type,
            "by_market_type": by_market,
            "by_bookmaker": by_book,
        }

    # ---- export ----------------------------------------------------------

    def export_csv(self, filepath: str | Path) -> int:
        filepath = Path(filepath)
        cursor = self._conn.execute(
            "SELECT event_id, game_date, commence_time_utc, away_team, home_team, "
            "game_type, away_pitcher, home_pitcher, fetched_at, "
            "bookmaker, market_type, side, point, price, is_opening, game_id "
            "FROM odds ORDER BY game_date, commence_time_utc, away_team, market_type, bookmaker"
        )
        count = 0
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(ODDS_ROW_FIELDS)
            for row in cursor:
                writer.writerow(row)
                count += 1
        logger.info(f"Exported {count} rows to {filepath}")
        return count

    # ---- lifecycle -------------------------------------------------------

    def close(self) -> None:
        self._conn.close()
