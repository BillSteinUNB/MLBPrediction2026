# Lineup client notes

- Public projected-lineup pages currently consumed by `src/clients/lineup_client.py` should be treated as **current-day only** sources.
- `fetch_confirmed_lineups(date)` now skips RotoGrinders/RotoBaller scraping when `date` does not match the current Eastern Time calendar date, preventing historical/future requests from being polluted with today's projections.
- RotoBaller fallback parsing is now DOM-based (`HTMLParser`) instead of regex-based so malformed nested markup degrades to `{}`/schedule fallback instead of crashing the fetch path.
