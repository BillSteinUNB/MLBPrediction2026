# Environment

Environment variables, external dependencies, and setup notes.

**What belongs here:** Required env vars, external API keys/services, dependency quirks, platform-specific notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Required Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Source |
|----------|-------------|--------|
| `ODDS_API_KEY` | The Odds API key | https://the-odds-api.com/ |
| `OPENWEATHER_API_KEY` | OpenWeatherMap API key | https://openweathermap.org/api |
| `DISCORD_WEBHOOK_URL` | Discord channel webhook | Discord server settings |

## External APIs

### The Odds API
- **Endpoint:** `https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/`
- **Free tier:** 500 requests/month
- **Markets needed:** F5 h2h (moneyline), F5 spreads (run line)
- **Rate limiting:** Client tracks usage in SQLite, fails before exceeding limit
- **Implementation note:** Current client flow lists events via `/v4/sports/baseball_mlb/events` and then fetches per-event odds via `/v4/sports/baseball_mlb/events/{id}/odds`
- **Quota accounting:** Current implementation reads `x-requests-used` and `x-requests-last` response headers when available

### OpenWeatherMap
- **Endpoint:** `api.openweathermap.org/data/2.5/forecast`
- **Free tier:** 1000 calls/day
- **Usage:** One call per open-air stadium per day

### MLB Stats API
- **Endpoint:** `statsapi.mlb.com`
- **Auth:** None required
- **Usage:** Game schedules, lineups

### pybaseball / FanGraphs
- **Auth:** None required
- **Caching:** Enable with `cache.enable()` for production
- **Statcast limit:** 30K rows per query, chunk date ranges
- **Current environment note:** The repo's installed `pybaseball` currently reports version `2.2.7`, while mission text still references `pybaseball>=2.3`; treat that as a known contract mismatch during review/fix work

### Discord Webhook
- **Format:** POST JSON to webhook URL
- **Rate limit:** Max 1 message per pipeline run (batch all picks)
- **Embeds:** Use for formatted pick cards

## Platform Notes

- **Windows:** Use PowerShell for commands, `schtasks` for scheduler
- **Linux/macOS:** Use bash, cron for scheduler
- **Python version:** 3.11+ required (using 3.13.12 in current environment)

### Windows shell gotcha

- In this repo, `bash .factory/init.sh` under Git Bash did not resolve `pip` after activation on Windows.
- Prefer invoking the project interpreter directly for installs and validation commands, e.g. `& .venv\Scripts\python.exe -m pip install -e ".[dev]"` and `& .venv\Scripts\python.exe -m pytest tests/ -v --tb=short`.
- `python -m py_compile src/**/*.py` is not reliable in PowerShell; expand files with `Get-ChildItem` or use the Windows-specific commands in `.factory/services.yaml`.

## Data Storage

- `data/mlb.db` - SQLite database (embedded, no server)
- `data/raw/statcast/` - Parquet files for raw Statcast data
- `data/models/` - Saved model files (.joblib)
- `data/logs/` - Pipeline logs (daily rotation, 30-day retention)
- `data/training/` - Training data Parquet files
