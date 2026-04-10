# src/clients — External API Wrappers

**9 Python files.** One file per external data source. All wrap HTTP calls with error handling.

## WHERE TO LOOK

| Client | File | Source | API Key |
|--------|------|--------|---------|
| Live odds | `odds_client.py` | the-odds-api.com | `ODDS_API_KEY` |
| Statcast data | `statcast_client.py` | pybaseball | — |
| Weather | `weather_client.py` | OpenWeatherMap | `OPENWEATHER_API_KEY` |
| Lineups | `lineup_client.py` | MLB stats API | — |
| Retrosheet | `retrosheet_client.py` | Chadwick/Retrosheet | — |
| Chadwick utility | `chadwick_client.py` | Chadwick register | — |
| Historical odds | `historical_odds_client.py` | Scraped/Fanduel | — |
| Historical F5 | `historical_f5_acquirer.py` | Custom source | — |

## PATTERN (follow odds_client.py)

```python
# Each client:
# 1. Gets logger = logging.getLogger(__name__)
# 2. Wraps HTTP in CircuitBreaker/retry from src.ops.error_handler
# 3. Returns domain objects from src/models/ (not raw dicts)
# 4. Has graceful fallback for missing data (cached/neutral values)
```

## ANTI-PATTERNS

- Do NOT skip CircuitBreaker/retry wrappers on external fetches
- Do NOT return raw API responses — always map to domain models (`src/models/`)
- Do NOT hardcode team codes — use `config/settings.yaml`
- Weather and lineups must have fallback paths (cached/neutral, team averages)

## NOTES

- `historical_odds_client.py` and `historical_f5_acquirer.py` have `__main__` guards for one-off data acquisition
- pybaseball has a pandas 3 compat issue (workaround in place)
- Odds API has rate limits — client handles 429 responses
