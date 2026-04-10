from __future__ import annotations

from contextlib import contextmanager
import sqlite3
from pathlib import Path


SCHEMA_VERSION = 8
DEFAULT_DB_PATH = Path("data") / "mlb.db"
BUILDER_SQLITE_CACHE_SIZE_KB = 64_000


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
        version INTEGER NOT NULL,
        applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS games (
        game_pk INTEGER PRIMARY KEY,
        date TEXT NOT NULL,
        home_team TEXT NOT NULL,
        away_team TEXT NOT NULL,
        home_starter_id INTEGER,
        away_starter_id INTEGER,
        venue TEXT NOT NULL,
        is_dome INTEGER NOT NULL DEFAULT 0 CHECK (is_dome IN (0, 1)),
        is_abs_active INTEGER NOT NULL DEFAULT 1 CHECK (is_abs_active IN (0, 1)),
        f5_home_score INTEGER,
        f5_away_score INTEGER,
        final_home_score INTEGER,
        final_away_score INTEGER,
        status TEXT NOT NULL CHECK (
            status IN ('scheduled', 'final', 'suspended', 'postponed', 'cancelled')
        )
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS features (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_pk INTEGER NOT NULL,
        feature_name TEXT NOT NULL,
        feature_value REAL NOT NULL,
        window_size INTEGER,
        as_of_timestamp TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (game_pk) REFERENCES games (game_pk),
        UNIQUE (game_pk, feature_name, window_size, as_of_timestamp)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS team_platoon_splits (
        team_abbr TEXT NOT NULL,
        season INTEGER NOT NULL,
        vs_hand TEXT NOT NULL CHECK (vs_hand IN ('L', 'R')),
        woba REAL,
        xwoba REAL,
        k_pct REAL,
        bb_pct REAL,
        pa INTEGER,
        PRIMARY KEY (team_abbr, season, vs_hand)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_pk INTEGER NOT NULL,
        model_version TEXT NOT NULL,
        f5_ml_home_prob REAL NOT NULL CHECK (f5_ml_home_prob BETWEEN 0 AND 1),
        f5_ml_away_prob REAL NOT NULL CHECK (f5_ml_away_prob BETWEEN 0 AND 1),
        f5_rl_home_prob REAL NOT NULL CHECK (f5_rl_home_prob BETWEEN 0 AND 1),
        f5_rl_away_prob REAL NOT NULL CHECK (f5_rl_away_prob BETWEEN 0 AND 1),
        predicted_at TEXT NOT NULL,
        FOREIGN KEY (game_pk) REFERENCES games (game_pk),
        UNIQUE (game_pk, model_version)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS odds_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_pk INTEGER NOT NULL,
        book_name TEXT NOT NULL,
        market_type TEXT NOT NULL CHECK (
            market_type IN (
                'f5_ml',
                'f5_rl',
                'f5_total',
                'full_game_ml',
                'full_game_rl',
                'full_game_total',
                'full_game_team_total_home',
                'full_game_team_total_away'
            )
        ),
        home_odds INTEGER NOT NULL,
        away_odds INTEGER NOT NULL,
        home_point REAL,
        away_point REAL,
        fetched_at TEXT NOT NULL,
        is_frozen INTEGER NOT NULL DEFAULT 0 CHECK (is_frozen IN (0, 1)),
        FOREIGN KEY (game_pk) REFERENCES games (game_pk)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_pk INTEGER NOT NULL,
        market_type TEXT NOT NULL CHECK (
            market_type IN (
                'f5_ml',
                'f5_rl',
                'f5_total',
                'full_game_ml',
                'full_game_rl',
                'full_game_total',
                'full_game_team_total_home',
                'full_game_team_total_away'
            )
        ),
        side TEXT NOT NULL CHECK (side IN ('home', 'away', 'over', 'under')),
        book_name TEXT,
        source_model TEXT,
        source_model_version TEXT,
        model_probability REAL CHECK (model_probability BETWEEN 0 AND 1),
        fair_probability REAL CHECK (fair_probability BETWEEN 0 AND 1),
        edge_pct REAL NOT NULL,
        ev REAL,
        kelly_stake REAL NOT NULL CHECK (kelly_stake >= 0),
        odds_at_bet INTEGER NOT NULL,
        line_at_bet REAL,
        result TEXT NOT NULL DEFAULT 'PENDING' CHECK (
            result IN ('WIN', 'LOSS', 'PUSH', 'NO_ACTION', 'PENDING')
        ),
        settled_at TEXT,
        profit_loss REAL,
        FOREIGN KEY (game_pk) REFERENCES games (game_pk)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bankroll_ledger (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT NOT NULL CHECK (
            event_type IN ('bet_placed', 'bet_settled', 'drawdown_alert', 'kill_switch')
        ),
        amount REAL NOT NULL,
        running_balance REAL NOT NULL,
        notes TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bet_performance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        bet_id INTEGER NOT NULL UNIQUE,
        game_pk INTEGER NOT NULL,
        market_type TEXT NOT NULL CHECK (
            market_type IN (
                'f5_ml',
                'f5_rl',
                'f5_total',
                'full_game_ml',
                'full_game_rl',
                'full_game_total',
                'full_game_team_total_home',
                'full_game_team_total_away'
            )
        ),
        side TEXT NOT NULL CHECK (side IN ('home', 'away', 'over', 'under')),
        book_name TEXT NOT NULL DEFAULT 'manual',
        model_probability REAL NOT NULL CHECK (model_probability BETWEEN 0 AND 1),
        market_probability REAL NOT NULL CHECK (market_probability BETWEEN 0 AND 1),
        edge_pct REAL NOT NULL,
        odds_at_bet INTEGER NOT NULL,
        stake REAL NOT NULL CHECK (stake >= 0),
        result TEXT NOT NULL DEFAULT 'PENDING' CHECK (
            result IN ('WIN', 'LOSS', 'PUSH', 'NO_ACTION', 'PENDING')
        ),
        profit_loss REAL,
        closing_odds INTEGER,
        closing_probability REAL CHECK (
            closing_probability IS NULL OR (closing_probability BETWEEN 0 AND 1)
        ),
        clv REAL,
        placed_at TEXT NOT NULL,
        settled_at TEXT,
        FOREIGN KEY (bet_id) REFERENCES bets (id),
        FOREIGN KEY (game_pk) REFERENCES games (game_pk)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS retrosheet_game_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        season INTEGER NOT NULL,
        game_date TEXT NOT NULL,
        away_team TEXT NOT NULL,
        home_team TEXT NOT NULL,
        matchup_sequence INTEGER NOT NULL,
        doubleheader_code TEXT NOT NULL DEFAULT '0',
        site TEXT,
        away_score INTEGER,
        home_score INTEGER,
        plate_umpire_id TEXT,
        plate_umpire_name TEXT,
        row_order INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (season, game_date, away_team, home_team, matchup_sequence)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS pitcher_siera_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        season INTEGER NOT NULL,
        pitcher_name TEXT NOT NULL,
        team TEXT,
        pitcher_id INTEGER,
        fangraphs_id INTEGER,
        siera REAL NOT NULL,
        xfip REAL,
        era REAL,
        fip REAL,
        k_pct REAL,
        bb_pct REAL,
        fetched_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_features_game_pk ON features (game_pk)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_game_pk ON predictions (game_pk)",
    "CREATE INDEX IF NOT EXISTS idx_odds_snapshots_game_pk ON odds_snapshots (game_pk)",
    "CREATE INDEX IF NOT EXISTS idx_bets_game_pk ON bets (game_pk)",
    "CREATE INDEX IF NOT EXISTS idx_bet_performance_game_market ON bet_performance (game_pk, market_type)",
    """
    CREATE INDEX IF NOT EXISTS idx_pitcher_siera_cache_season
    ON pitcher_siera_cache (season)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_retrosheet_game_logs_lookup
    ON retrosheet_game_logs (season, game_date, home_team, away_team, matchup_sequence)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_retrosheet_game_logs_umpire
    ON retrosheet_game_logs (plate_umpire_id, game_date)
    """,
)


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _column_exists(connection: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    return any(
        row[1] == column_name
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    )


def _apply_migrations(connection: sqlite3.Connection) -> None:
    _refresh_odds_snapshots_market_type_constraint(connection)
    if _table_exists(connection, "bets") and not _column_exists(connection, "bets", "book_name"):
        connection.execute("ALTER TABLE bets ADD COLUMN book_name TEXT")
    if _table_exists(connection, "bets") and not _column_exists(connection, "bets", "source_model"):
        connection.execute("ALTER TABLE bets ADD COLUMN source_model TEXT")
    if _table_exists(connection, "bets") and not _column_exists(
        connection,
        "bets",
        "source_model_version",
    ):
        connection.execute("ALTER TABLE bets ADD COLUMN source_model_version TEXT")
    if _table_exists(connection, "bets") and not _column_exists(
        connection,
        "bets",
        "model_probability",
    ):
        connection.execute("ALTER TABLE bets ADD COLUMN model_probability REAL")
    if _table_exists(connection, "bets") and not _column_exists(
        connection,
        "bets",
        "fair_probability",
    ):
        connection.execute("ALTER TABLE bets ADD COLUMN fair_probability REAL")
    if _table_exists(connection, "bets") and not _column_exists(connection, "bets", "ev"):
        connection.execute("ALTER TABLE bets ADD COLUMN ev REAL")
    if _table_exists(connection, "bets") and not _column_exists(connection, "bets", "line_at_bet"):
        connection.execute("ALTER TABLE bets ADD COLUMN line_at_bet REAL")
    if _table_exists(connection, "bet_performance") and not _column_exists(
        connection,
        "bet_performance",
        "book_name",
    ):
        connection.execute(
            "ALTER TABLE bet_performance ADD COLUMN book_name TEXT NOT NULL DEFAULT 'manual'"
        )
    if _table_exists(connection, "odds_snapshots") and not _column_exists(
        connection,
        "odds_snapshots",
        "home_point",
    ):
        connection.execute("ALTER TABLE odds_snapshots ADD COLUMN home_point REAL")
    if _table_exists(connection, "odds_snapshots") and not _column_exists(
        connection,
        "odds_snapshots",
        "away_point",
    ):
        connection.execute("ALTER TABLE odds_snapshots ADD COLUMN away_point REAL")
    _refresh_betting_table_constraints(connection)


def _refresh_odds_snapshots_market_type_constraint(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "odds_snapshots"):
        return
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'odds_snapshots'"
    ).fetchone()
    create_sql = str(row[0]) if row and row[0] is not None else ""
    required_market_types = (
        "f5_total",
        "full_game_ml",
        "full_game_rl",
        "full_game_total",
        "full_game_team_total_home",
        "full_game_team_total_away",
    )
    if all(token in create_sql for token in required_market_types):
        return

    connection.execute("ALTER TABLE odds_snapshots RENAME TO odds_snapshots_legacy")
    connection.execute(
        """
        CREATE TABLE odds_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_pk INTEGER NOT NULL,
            book_name TEXT NOT NULL,
            market_type TEXT NOT NULL CHECK (
                market_type IN (
                    'f5_ml',
                    'f5_rl',
                    'f5_total',
                    'full_game_ml',
                    'full_game_rl',
                    'full_game_total',
                    'full_game_team_total_home',
                    'full_game_team_total_away'
                )
            ),
            home_odds INTEGER NOT NULL,
            away_odds INTEGER NOT NULL,
            home_point REAL,
            away_point REAL,
            fetched_at TEXT NOT NULL,
            is_frozen INTEGER NOT NULL DEFAULT 0 CHECK (is_frozen IN (0, 1)),
            FOREIGN KEY (game_pk) REFERENCES games (game_pk)
        )
        """
    )
    legacy_columns = {
        row_info[1]
        for row_info in connection.execute("PRAGMA table_info(odds_snapshots_legacy)").fetchall()
    }
    select_home_point = "home_point" if "home_point" in legacy_columns else "NULL"
    select_away_point = "away_point" if "away_point" in legacy_columns else "NULL"
    connection.execute(
        f"""
        INSERT INTO odds_snapshots (
            id,
            game_pk,
            book_name,
            market_type,
            home_odds,
            away_odds,
            home_point,
            away_point,
            fetched_at,
            is_frozen
        )
        SELECT
            id,
            game_pk,
            book_name,
            market_type,
            home_odds,
            away_odds,
            {select_home_point},
            {select_away_point},
            fetched_at,
            is_frozen
        FROM odds_snapshots_legacy
        """
    )
    connection.execute("DROP TABLE odds_snapshots_legacy")


def _refresh_betting_table_constraints(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "bets") and not _table_exists(connection, "bet_performance"):
        return

    bets_create_sql = ""
    if _table_exists(connection, "bets"):
        bets_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'bets'"
        ).fetchone()
        bets_create_sql = str(bets_row[0]) if bets_row and bets_row[0] is not None else ""
    performance_create_sql = ""
    if _table_exists(connection, "bet_performance"):
        performance_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'bet_performance'"
        ).fetchone()
        performance_create_sql = (
            str(performance_row[0]) if performance_row and performance_row[0] is not None else ""
        )

    required_market_types = (
        "f5_total",
        "full_game_ml",
        "full_game_rl",
        "full_game_total",
        "full_game_team_total_home",
        "full_game_team_total_away",
    )
    bets_ready = all(token in bets_create_sql for token in required_market_types) and all(
        token in bets_create_sql for token in ("'over'", "'under'")
    )
    performance_ready = all(
        token in performance_create_sql for token in required_market_types
    ) and all(token in performance_create_sql for token in ("'over'", "'under'"))
    if bets_ready and performance_ready:
        return

    connection.commit()
    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        if _table_exists(connection, "bets"):
            connection.execute("ALTER TABLE bets RENAME TO bets_legacy")
        if _table_exists(connection, "bet_performance"):
            connection.execute("ALTER TABLE bet_performance RENAME TO bet_performance_legacy")

        connection.execute(
            """
            CREATE TABLE bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_pk INTEGER NOT NULL,
                market_type TEXT NOT NULL CHECK (
                    market_type IN (
                        'f5_ml',
                        'f5_rl',
                        'f5_total',
                        'full_game_ml',
                        'full_game_rl',
                        'full_game_total',
                        'full_game_team_total_home',
                        'full_game_team_total_away'
                    )
                ),
                side TEXT NOT NULL CHECK (side IN ('home', 'away', 'over', 'under')),
                book_name TEXT,
                source_model TEXT,
                source_model_version TEXT,
                model_probability REAL CHECK (model_probability BETWEEN 0 AND 1),
                fair_probability REAL CHECK (fair_probability BETWEEN 0 AND 1),
                edge_pct REAL NOT NULL,
                ev REAL,
                kelly_stake REAL NOT NULL CHECK (kelly_stake >= 0),
                odds_at_bet INTEGER NOT NULL,
                line_at_bet REAL,
                result TEXT NOT NULL DEFAULT 'PENDING' CHECK (
                    result IN ('WIN', 'LOSS', 'PUSH', 'NO_ACTION', 'PENDING')
                ),
                settled_at TEXT,
                profit_loss REAL,
                FOREIGN KEY (game_pk) REFERENCES games (game_pk)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE bet_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_id INTEGER NOT NULL UNIQUE,
                game_pk INTEGER NOT NULL,
                market_type TEXT NOT NULL CHECK (
                    market_type IN (
                        'f5_ml',
                        'f5_rl',
                        'f5_total',
                        'full_game_ml',
                        'full_game_rl',
                        'full_game_total',
                        'full_game_team_total_home',
                        'full_game_team_total_away'
                    )
                ),
                side TEXT NOT NULL CHECK (side IN ('home', 'away', 'over', 'under')),
                book_name TEXT NOT NULL DEFAULT 'manual',
                model_probability REAL NOT NULL CHECK (model_probability BETWEEN 0 AND 1),
                market_probability REAL NOT NULL CHECK (market_probability BETWEEN 0 AND 1),
                edge_pct REAL NOT NULL,
                odds_at_bet INTEGER NOT NULL,
                stake REAL NOT NULL CHECK (stake >= 0),
                result TEXT NOT NULL DEFAULT 'PENDING' CHECK (
                    result IN ('WIN', 'LOSS', 'PUSH', 'NO_ACTION', 'PENDING')
                ),
                profit_loss REAL,
                closing_odds INTEGER,
                closing_probability REAL CHECK (
                    closing_probability IS NULL OR (closing_probability BETWEEN 0 AND 1)
                ),
                clv REAL,
                placed_at TEXT NOT NULL,
                settled_at TEXT,
                FOREIGN KEY (bet_id) REFERENCES bets (id),
                FOREIGN KEY (game_pk) REFERENCES games (game_pk)
            )
            """
        )

        if _table_exists(connection, "bets_legacy"):
            legacy_columns = {
                row_info[1] for row_info in connection.execute("PRAGMA table_info(bets_legacy)").fetchall()
            }
            select_book_name = "book_name" if "book_name" in legacy_columns else "NULL"
            select_source_model = "source_model" if "source_model" in legacy_columns else "NULL"
            select_source_model_version = (
                "source_model_version" if "source_model_version" in legacy_columns else "NULL"
            )
            select_model_probability = (
                "model_probability" if "model_probability" in legacy_columns else "NULL"
            )
            select_fair_probability = (
                "fair_probability" if "fair_probability" in legacy_columns else "NULL"
            )
            select_ev = "ev" if "ev" in legacy_columns else "NULL"
            select_line_at_bet = "line_at_bet" if "line_at_bet" in legacy_columns else "NULL"
            connection.execute(
                f"""
                INSERT INTO bets (
                    id,
                    game_pk,
                    market_type,
                    side,
                    book_name,
                    source_model,
                    source_model_version,
                    model_probability,
                    fair_probability,
                    edge_pct,
                    ev,
                    kelly_stake,
                    odds_at_bet,
                    line_at_bet,
                    result,
                    settled_at,
                    profit_loss
                )
                SELECT
                    id,
                    game_pk,
                    market_type,
                    side,
                    {select_book_name},
                    {select_source_model},
                    {select_source_model_version},
                    {select_model_probability},
                    {select_fair_probability},
                    edge_pct,
                    {select_ev},
                    kelly_stake,
                    odds_at_bet,
                    {select_line_at_bet},
                    result,
                    settled_at,
                    profit_loss
                FROM bets_legacy
                """
            )
            connection.execute("DROP TABLE bets_legacy")

        if _table_exists(connection, "bet_performance_legacy"):
            legacy_columns = {
                row_info[1]
                for row_info in connection.execute(
                    "PRAGMA table_info(bet_performance_legacy)"
                ).fetchall()
            }
            select_book_name = (
                "book_name" if "book_name" in legacy_columns else "'manual'"
            )
            connection.execute(
                f"""
                INSERT INTO bet_performance (
                    id,
                    bet_id,
                    game_pk,
                    market_type,
                    side,
                    book_name,
                    model_probability,
                    market_probability,
                    edge_pct,
                    odds_at_bet,
                    stake,
                    result,
                    profit_loss,
                    closing_odds,
                    closing_probability,
                    clv,
                    placed_at,
                    settled_at
                )
                SELECT
                    id,
                    bet_id,
                    game_pk,
                    market_type,
                    side,
                    {select_book_name},
                    model_probability,
                    market_probability,
                    edge_pct,
                    odds_at_bet,
                    stake,
                    result,
                    profit_loss,
                    closing_odds,
                    closing_probability,
                    clv,
                    placed_at,
                    settled_at
                FROM bet_performance_legacy
                """
            )
            connection.execute("DROP TABLE bet_performance_legacy")
        connection.commit()
    finally:
        connection.execute("PRAGMA foreign_keys = ON")


def _patch_data_builder_team_platoon_splits_fetcher() -> None:
    import sys

    data_builder_module = sys.modules.get("src.model.data_builder")
    if data_builder_module is None:
        return

    from src.features.offense import build_db_backed_team_batting_splits_fetcher

    if (
        getattr(data_builder_module, "_build_cached_team_batting_splits_fetcher", None)
        is build_db_backed_team_batting_splits_fetcher
    ):
        return

    setattr(
        data_builder_module,
        "_build_cached_team_batting_splits_fetcher",
        build_db_backed_team_batting_splits_fetcher,
    )


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """Initialize the SQLite schema and record the current schema version."""

    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")

        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)

        _apply_migrations(connection)
        _patch_data_builder_team_platoon_splits_fetcher()

        connection.execute(
            """
            INSERT INTO schema_version (singleton_id, version)
            VALUES (1, ?)
            ON CONFLICT(singleton_id) DO UPDATE SET
                version = excluded.version,
                applied_at = CURRENT_TIMESTAMP
            """,
            (SCHEMA_VERSION,),
        )

        connection.commit()

    return database_path


def configure_sqlite_connection(
    connection: sqlite3.Connection,
    *,
    builder_optimized: bool = False,
) -> sqlite3.Connection:
    """Apply standard SQLite pragmas and optional builder-focused performance tuning."""

    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    if builder_optimized:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.execute(f"PRAGMA cache_size = {-BUILDER_SQLITE_CACHE_SIZE_KB}")
    else:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def sqlite_connection(
    db_path: str | Path,
    *,
    builder_optimized: bool = False,
):
    """Open a SQLite connection with standard repo pragmas applied."""

    connection = sqlite3.connect(db_path)
    try:
        yield configure_sqlite_connection(connection, builder_optimized=builder_optimized)
    finally:
        connection.close()
