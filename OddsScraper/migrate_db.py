"""
One-time migration of mlb_odds.db from v1 (all-text) to v2 schema.

Changes:
  - price  TEXT  -> INTEGER
  - point  TEXT  -> REAL (NULL for ML)
  - commence_time TEXT -> commence_time_utc TEXT (ISO 8601 UTC)
  - Add is_opening INTEGER DEFAULT 0
  - Add game_id INTEGER
  - Create games table, populate from odds
  - Add unique index to prevent duplicate rows
"""

import re
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

DB_PATH = Path("data/mlb_odds.db")
BACKUP_PATH = Path("data/mlb_odds_backup.db")

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


def parse_price(raw: str) -> int:
    """'+145' -> 145, '-115' -> -115, 'EVEN'/'+100' -> 100."""
    if not raw or raw in ("-", "--"):
        return 0
    raw = raw.strip()
    if raw.upper() == "EVEN":
        return 100
    try:
        return int(raw)
    except ValueError:
        return 0


def parse_point(raw: str):
    """'-1.5' -> -1.5, '8.5' -> 8.5, '' -> None."""
    if not raw or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_time_to_utc(game_date: str, time_str: str) -> str:
    """'8:05 PM EDT' + '2025-07-04' -> '2025-07-05T00:05:00Z'."""
    if not time_str or time_str.strip() == "":
        return ""
    m = re.match(r"(\d{1,2}):(\d{2})\s*(AM|PM)\s*(\w+)", time_str.strip(), re.IGNORECASE)
    if not m:
        return ""
    hour, minute = int(m.group(1)), int(m.group(2))
    ampm, tz_abbr = m.group(3).upper(), m.group(4).upper()
    if ampm == "PM" and hour != 12:
        hour += 12
    elif ampm == "AM" and hour == 12:
        hour = 0
    offset_h = TZ_OFFSETS.get(tz_abbr, -4)
    try:
        dt = datetime.strptime(game_date, "%Y-%m-%d").replace(
            hour=hour,
            minute=minute,
            tzinfo=timezone(timedelta(hours=offset_h)),
        )
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return ""


def migrate():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found")
        return

    # 1. Backup
    print(f"Backing up to {BACKUP_PATH}...")
    shutil.copy2(DB_PATH, BACKUP_PATH)

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL;")

    old_count = conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
    print(f"Existing rows: {old_count:,}")

    # 2. Check if already migrated (price is already integer)
    sample = conn.execute("SELECT price FROM odds LIMIT 1").fetchone()
    if sample and isinstance(sample[0], int):
        print("Database appears already migrated (price is integer). Skipping.")
        conn.close()
        return

    # 3. Create games table
    print("Creating games table...")
    conn.executescript("""
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
    """)

    # 4. Create new odds table
    print("Creating new odds table...")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS odds_new (
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
    """)

    # 5. Migrate rows in batches (transform in Python for price/point/time)
    print("Migrating rows (price->int, point->real, time->UTC)...")

    # Check which columns exist in old table
    cols = [row[1] for row in conn.execute("PRAGMA table_info(odds)").fetchall()]
    time_col = "commence_time_utc" if "commence_time_utc" in cols else "commence_time"

    cursor = conn.execute(f"""
        SELECT event_id, game_date, {time_col}, away_team, home_team,
               game_type, away_pitcher, home_pitcher, fetched_at,
               bookmaker, market_type, side, point, price
        FROM odds
    """)

    batch = []
    migrated = 0
    for row in cursor:
        (
            event_id,
            game_date,
            commence_time,
            away_team,
            home_team,
            game_type,
            away_pitcher,
            home_pitcher,
            fetched_at,
            bookmaker,
            market_type,
            side,
            point_raw,
            price_raw,
        ) = row

        price_int = parse_price(str(price_raw) if price_raw is not None else "")
        point_float = parse_point(str(point_raw) if point_raw is not None else "")
        time_utc = parse_time_to_utc(game_date, commence_time or "")

        if price_int == 0:
            continue  # skip rows with unparseable price

        batch.append(
            (
                event_id,
                game_date,
                time_utc,
                away_team,
                home_team,
                game_type,
                away_pitcher,
                home_pitcher,
                fetched_at,
                bookmaker,
                market_type,
                side,
                point_float,
                price_int,
                0,  # is_opening
                0,  # game_id (set later)
            )
        )
        migrated += 1

        if len(batch) >= 10000:
            conn.executemany(
                """
                INSERT INTO odds_new (
                    event_id, game_date, commence_time_utc, away_team, home_team,
                    game_type, away_pitcher, home_pitcher, fetched_at,
                    bookmaker, market_type, side, point, price, is_opening, game_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                batch,
            )
            conn.commit()
            print(f"  {migrated:,} rows...")
            batch = []

    if batch:
        conn.executemany(
            """
            INSERT INTO odds_new (
                event_id, game_date, commence_time_utc, away_team, home_team,
                game_type, away_pitcher, home_pitcher, fetched_at,
                bookmaker, market_type, side, point, price, is_opening, game_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            batch,
        )
        conn.commit()

    print(f"  Migrated {migrated:,} rows")

    # 6. Swap tables
    print("Swapping tables...")
    conn.execute("DROP TABLE odds")
    conn.execute("ALTER TABLE odds_new RENAME TO odds")
    conn.commit()

    # 7. Recreate indexes
    print("Creating indexes...")
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_odds_game_date     ON odds (game_date);",
        "CREATE INDEX IF NOT EXISTS idx_odds_market_type   ON odds (market_type);",
        "CREATE INDEX IF NOT EXISTS idx_odds_game_type     ON odds (game_type);",
        "CREATE INDEX IF NOT EXISTS idx_odds_bookmaker     ON odds (bookmaker);",
        "CREATE INDEX IF NOT EXISTS idx_odds_teams         ON odds (away_team, home_team);",
        "CREATE INDEX IF NOT EXISTS idx_odds_event_id      ON odds (event_id);",
        "CREATE INDEX IF NOT EXISTS idx_odds_game_date_mkt ON odds (game_date, market_type);",
        "CREATE INDEX IF NOT EXISTS idx_odds_game_id       ON odds (game_id);",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_odds_unique  ON odds (event_id, fetched_at, bookmaker, market_type, side, COALESCE(point, -999));",
    ]
    for idx in indexes:
        conn.execute(idx)
    conn.commit()

    # 8. Populate games table from odds
    print("Populating games table...")
    conn.execute("""
        INSERT OR IGNORE INTO games (event_id, game_date, commence_time_utc,
                                     away_team, home_team, game_type,
                                     away_pitcher, home_pitcher)
        SELECT DISTINCT event_id, game_date, commence_time_utc,
               away_team, home_team, game_type,
               MAX(away_pitcher), MAX(home_pitcher)
        FROM odds
        WHERE event_id IS NOT NULL AND event_id != ''
        GROUP BY event_id
    """)
    conn.commit()

    game_count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    print(f"  Created {game_count:,} game records")

    # 9. Update game_id in odds
    print("Linking odds to games...")
    conn.execute("""
        UPDATE odds SET game_id = (
            SELECT game_id FROM games WHERE games.event_id = odds.event_id
        )
        WHERE event_id IS NOT NULL AND event_id != ''
    """)
    conn.commit()

    linked = conn.execute("SELECT COUNT(*) FROM odds WHERE game_id > 0").fetchone()[0]
    print(f"  Linked {linked:,} odds rows to games")

    # 10. Verify
    new_count = conn.execute("SELECT COUNT(*) FROM odds").fetchone()[0]
    sample_price = conn.execute("SELECT price, typeof(price) FROM odds LIMIT 1").fetchone()
    sample_point = conn.execute(
        "SELECT point, typeof(point) FROM odds WHERE point IS NOT NULL LIMIT 1"
    ).fetchone()
    sample_time = conn.execute(
        "SELECT commence_time_utc FROM odds WHERE commence_time_utc != '' LIMIT 1"
    ).fetchone()

    print(f"\n{'=' * 50}")
    print(f"MIGRATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Rows: {old_count:,} -> {new_count:,} (dropped {old_count - new_count} unparseable)")
    print(f"Games table: {game_count:,} records")
    print(f"Price sample: {sample_price}")
    print(f"Point sample: {sample_point}")
    print(f"Time sample:  {sample_time}")
    print(f"Backup at:    {BACKUP_PATH}")

    conn.close()


if __name__ == "__main__":
    migrate()
