# PROJECT KNOWLEDGE BASE

**Generated:** 2026-03-30
**Commit:** 03f03b3
**Branch:** main

## OVERVIEW

MLB First Five Innings (F5) betting prediction system. Python backend (pandas/xgboost/lightgbm/SQLAlchemy) + React/TS dashboard. Daily pipeline fetches schedule → builds sabermetric features → runs stacked ML ensemble → calculates de-vigged edge → sizes via Quarter Kelly → alerts via Discord.

## STRUCTURE

```
MLBPrediction2026/
├── src/
│   ├── pipeline/       # Daily orchestrator (daily.py) + narrative builder
│   ├── model/          # Training, stacking, calibration, MCMC, run-count  [→ AGENTS.md]
│   ├── features/       # Feature engineering (offense/pitching/bullpen/defense/adjustments)
│   ├── clients/        # External API wrappers (odds, statcast, weather, lineup, retrosheet)
│   ├── models/         # Domain data classes (Game, Bet, Prediction, Lineup, Weather)
│   ├── engine/         # Edge calc, Kelly bankroll, settlement
│   ├── dashboard/      # FastAPI routes/adapters serving React frontend
│   ├── ops/            # Error handling, logging, experiment tracking, AutoResearch
│   ├── backtest/       # Walk-forward backtest runner
│   ├── notifications/  # Discord webhook
│   ├── config.py       # Settings loader (pydantic-settings + settings.yaml)
│   ├── db.py           # SQLite init/helpers
│   └── display.py      # Rich CLI output
├── dashboard/          # React + TypeScript + Vite frontend [→ AGENTS.md]
├── tests/              # pytest suite (features/, model/, ops/, pipeline/, integration/)
├── scripts/            # CLI utilities (setup, scheduler, training wrappers)
├── AutoResearch/       # Overnight research loop (launcher.py, train.py, agent.py)
├── OddsScraper/        # Playwright-based historical odds scraper
├── config/
│   └── settings.yaml   # Team codes, park factors, ABS exceptions, Marcel weights
└── data/               # mlb.db, models/, training/, reports/ (gitignored artifacts)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Daily pipeline entry | `src/pipeline/daily.py` | `python -m src.pipeline.daily` — main orchestrator |
| Add a new API client | `src/clients/` | One file per service; follow odds_client.py pattern |
| Feature engineering | `src/features/` | offense.py, pitching.py, bullpen.py, defense.py, adjustments/ |
| Train ML model | `src/model/` | xgboost_trainer → stacking → calibration flow |
| Change betting logic | `src/engine/` | edge_calculator.py, bankroll.py, settlement.py |
| Domain data models | `src/models/` | Pydantic-like dataclasses (Game, Bet, Lineup, Prediction) |
| Dashboard API | `src/dashboard/` | FastAPI app in main.py, routes/ dir |
| Dashboard frontend | `dashboard/src/` | React/TS pages, components, charts |
| Config & park factors | `config/settings.yaml` | Authoritative for team codes, stadium metadata |
| Error handling | `src/ops/error_handler.py` | CircuitBreaker, retry, graceful degradation |
| Logging setup | `src/ops/logging_config.py` | configure_logging() — sentinel-marked handlers |
| Experiment tracking | `src/ops/experiment_tracker.py` | Model experiment metadata |
| Backtesting | `src/backtest/run.py` | Walk-forward runner |

## CRITICAL DISTINCTIONS

- **`src/model/`** (singular) = ML training code: trainers, stacking, calibration, MCMC. 21 Python files.
- **`src/models/`** (plural) = Domain data classes: Game, Bet, Lineup, Prediction, Weather. Runtime schemas.
- **`dashboard/`** = React/TS frontend (Vite, port 5173).
- **`src/dashboard/`** = Python FastAPI backend serving the frontend.

## CONVENTIONS

- **Packaging**: pyproject.toml (PEP 621), setuptools backend. Install: `pip install -e ".[dev]"`
- **Linting**: ruff only. line-length=100, target=py311. No black/isort/flake8.
- **Testing**: pytest. `testpaths=["tests"]`, `pythonpath=["."]`. Name files `test_*.py`.
- **Mocking**: Prefer `monkeypatch.setattr()` over `unittest.mock.patch()`.
- **Logging**: `logging.getLogger(__name__)` with %-format. Call `src.ops.logging_config.configure_logging()` at process start.
- **CLI output**: Use `src/display.py` (Rich console), not raw `print()`.
- **Config**: Team/stadium/park-factor metadata lives in `config/settings.yaml`. Do not hardcode.
- **Dependencies**: No requirements.txt. Everything in pyproject. pybaseball installed from git URI.

## ANTI-PATTERNS

- Do NOT infer event-level challenge counts from ABS proxy features (`src/features/adjustments/abs_adjustment.py`).
- Do NOT skip CircuitBreaker/retry wrappers on external fetches (clients wrap schedule, history, lineups, odds).
- Do NOT use `print()` for CLI output — use the Rich console via `src/display.py`.
- Narrative assembly must build the edge sentence first (`src/pipeline/narrative.py`).
- Fatal errors must go through `notify_fatal_error()` to ensure Discord alerts fire.

## COMMANDS

```bash
# Install
pip install -e ".[dev]"                    # dev deps (pytest, ruff)
pip install -e ".[dev,.dashboard]"         # + fastapi, playwright

# Daily pipeline
python -m src.pipeline.daily --date today --mode backtest --dry-run
python -m src.pipeline.daily --date today --mode prod

# Training
python -m src.model.xgboost_trainer --training-data data/training/training_data.parquet
python -m src.model.stacking --training-data data/training/training_data.parquet
python -m src.model.calibration --training-data data/training/training_data.parquet

# Backtesting
python -m src.backtest.run --start 2022-01-01 --end 2023-12-31

# Tests
pytest tests/ -v
pytest tests/ --cov=src --cov-report=term-missing

# Lint
ruff .

# Dashboard backend
python -m uvicorn src.dashboard.main:app --host 127.0.0.1 --port 8000

# Dashboard frontend
cd dashboard && npm run dev          # dev server on :5173
cd dashboard && npm run build        # production build

# E2E tests (requires both backend + frontend + playwright)
pytest tests/integration/dashboard_e2e -v

# Scheduler setup
python scripts/setup_scheduler.py --test
python scripts/setup_scheduler.py --platform cron --install

# DB init
python -c "from src.db import init_db; init_db('data/mlb.db')"
```

## ENVIRONMENT VARIABLES

```env
ODDS_API_KEY=...           # the-odds-api.com
OPENWEATHER_API_KEY=...    # openweathermap.org
DISCORD_WEBHOOK_URL=...    # Discord channel webhook
```

## NOTES

- No CI/CD pipeline configured (no .github/workflows, no Dockerfile).
- Python version: pyproject says >=3.11, README says 3.12+. pyproject is authoritative.
- Weather falls back to cached/neutral values when API unavailable.
- Lineups fall back to team averages when scratches detected.
- pybaseball has a pandas 3 compat issue (workaround in place).
- Model artifacts saved to `data/models/<experiment_name>/`.
- OddsScraper has its own SQLite DB at `OddsScraper/data/mlb_odds.db`.
