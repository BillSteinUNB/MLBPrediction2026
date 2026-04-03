"""
Compatibility wrapper for the archived SportsbookReview scraper.

The legacy SBR implementation lives under ``OddsScraper/archive`` so new
OddsPortal work can be added beside it without mixing both sources in one file.
"""

try:
    from .archive.sbr_legacy import MLBOddsScraper, OddsRow, SEASON_DATES, SQLiteStore
except ImportError:
    from archive.sbr_legacy import MLBOddsScraper, OddsRow, SEASON_DATES, SQLiteStore

__all__ = [
    "MLBOddsScraper",
    "OddsRow",
    "SEASON_DATES",
    "SQLiteStore",
]
