# MLB F5 Betting Prediction System — 2026 Season

---

## 🔄 PROGRESS TRACKER

**Mission ID:** `83c0c194-72d1-4821-8b08-68a3497c3590`
**Started:** 2026-03-19
**Status:** IN PROGRESS

### Milestone Progress

| Milestone | Status | Tasks | Notes |
|-----------|--------|-------|-------|
| foundation | 🔄 In Progress | T1-T8 | Project scaffolding started; remaining foundation tasks pending |
| feature-engineering | 🔲 Not Started | T9-T15 | Rolling features, adjustments, baselines |
| ml-pipeline | 🔲 Not Started | T16-T20 | Training data, XGBoost, stacking, calibration, backtest |
| decision-engine | 🔲 Not Started | T21-T26 | Edge calc, Kelly, settlement, orchestrator, Discord |
| automation-testing | 🔲 Not Started | T27-T31 | Scheduler, pytest suite, performance tracker, error handling |
| final-review | 🔲 Not Started | F1-F4 | Plan compliance, code quality, manual QA, scope check |

### Task Progress

| Task | Status | Commit | Notes |
|------|--------|--------|-------|
| T1: Scaffolding | ✅ Completed | pending commit | Project layout, pyproject, .gitignore, .env.example, and scaffold tests added |
| T2: Config Module | 🔲 Pending | - | |
| T3: SQLite Schema | 🔲 Pending | - | |
| T4: Pydantic Models | 🔲 Pending | - | |
| T5: Odds API Client | 🔲 Pending | - | |
| T6: pybaseball Ingestion | 🔲 Pending | - | |
| T7: MLB Lineup Client | 🔲 Pending | - | |
| T8: Weather Client | 🔲 Pending | - | |
| T9: Offensive Features | 🔲 Pending | - | |
| T10: Pitching Features | 🔲 Pending | - | |
| T11: Defense Features | 🔲 Pending | - | |
| T12: Bullpen Features | 🔲 Pending | - | |
| T13: Park/ABS Adjustments | 🔲 Pending | - | |
| T14: Weather Engine | 🔲 Pending | - | |
| T15: Pythagorean/Log5 | 🔲 Pending | - | |
| T16: Training Data Builder | 🔲 Pending | - | |
| T17: XGBoost Training | 🔲 Pending | - | |
| T18: LR Stacking | 🔲 Pending | - | |
| T19: Isotonic Calibration | 🔲 Pending | - | |
| T20: Walk-Forward Backtest | 🔲 Pending | - | |
| T21: Edge Calculator | 🔲 Pending | - | |
| T22: Quarter Kelly | 🔲 Pending | - | |
| T23: Settlement Rules | 🔲 Pending | - | |
| T24: Daily Orchestrator | 🔲 Pending | - | |
| T25: Marcel Blend | 🔲 Pending | - | |
| T26: Discord Webhook | 🔲 Pending | - | |
| T27: Scheduler Setup | 🔲 Pending | - | |
| T28: Data Integrity Tests | 🔲 Pending | - | |
| T29: Financial Tests | 🔲 Pending | - | |
| T30: Performance Tracker | 🔲 Pending | - | |
| T31: Error Handling | 🔲 Pending | - | |
| F1: Plan Compliance | 🔲 Pending | - | |
| F2: Code Quality | 🔲 Pending | - | |
| F3: Manual QA | 🔲 Pending | - | |
| F4: Scope Check | 🔲 Pending | - | |

### Key Files Reference

- **Mission Directory:** `C:\Users\bills\.factory\missions\83c0c194-72d1-4821-8b08-68a3497c3590`
- **Validation Contract:** `{missionDir}/validation-contract.md`
- **Features List:** `{missionDir}/features.json`
- **Worker Skills:** `{repo}/.factory/skills/`
- **Services Manifest:** `{repo}/.factory/services.yaml`

### Resume Instructions

If resuming from a new session:
1. Read this plan file for full context
2. Check the Task Progress table above for current status
3. Read `{missionDir}/validation-state.json` for assertion status
4. Read `{missionDir}/features.json` for feature completion
5. Continue from first pending task in current milestone

---

## TL;DR

> **Quick Summary**: Build an end-to-end Python MLB betting system targeting First Five Innings (F5) Moneyline and Run Line markets for the 2026 season. The system uses advanced sabermetrics (wRC+, wOBA, xFIP, xERA, DRS, OAA), a stacked XGBoost→LR→Isotonic calibration ensemble, and Quarter Kelly bankroll management to identify +EV bets with ≥3% edge, delivered daily via Discord webhook.
> 
> **Deliverables**:
> - Automated daily data pipeline (Statcast, FanGraphs, lineups, odds, weather)
> - Feature engineering engine with multi-window (7/14/30/60 game) rolling sabermetrics
> - Stacked ML model with walk-forward backtested calibration
> - Decision engine with de-vig edge calculation + Quarter Kelly sizing
> - Discord webhook notifications with formatted pick cards
> - pytest suite for data integrity, settlement logic, and calibration quality gates
> - Windows Task Scheduler automation for daily unattended operation
> 
> **Estimated Effort**: XL (30+ tasks across 5 waves)
> **Parallel Execution**: YES — 5 waves, up to 8 concurrent tasks
> **Critical Path**: Task 1→3→9→16→17→19→21→24→27→Final

---

## Context

### Original Request
Build a highly profitable MLB betting model for the 2026 season, starting with First Five Innings (F5) markets to isolate starting pitching and core lineups. The model uses purely advanced sabermetrics, accounts for the new ABS Challenge System strike zone, adjusts for park factors and weather, and manages bankroll via Quarter Kelly staking.

### Interview Summary
**Key Discussions**:
- **Target markets**: F5 Moneyline + F5 Run Line (-1.5/+1.5); full-game expansion deferred to v2
- **Sabermetrics philosophy**: Reject traditional stats; use wRC+, wOBA, xFIP, xERA (proxy for SIERA), DRS, OAA
- **ABS impact**: Zone ~2" smaller in 2026; framing 70-80% retained in low leverage; model must handle ABS exception venues (Mexico City, Field of Dreams, Little League Classic)
- **Venue correction**: Rays return to Tropicana Field in 2026 (NOT Steinbrenner); only Sutter Health Park (A's) needs special treatment
- **ML architecture**: Stacked ensemble (XGBoost primary → LR stacking → Isotonic calibration)
- **Data persistence**: SQLite for structured data + Parquet for raw Statcast bulk
- **Notifications**: Discord webhook with formatted pick cards
- **Backtesting**: Walk-forward (6-month train / 1-month test), strict anti-leakage enforcement
- **Early season**: Marcel-style regression blending prior-year stats weighted by games played
- **Edge threshold**: ≥3% edge over de-vigged fair probability to trigger a bet

**Research Findings**:
- pybaseball provides wRC+, wOBA, FIP, xFIP, DRS, OAA but NOT SIERA → resolved with xFIP + xERA proxies
- Statcast: 30K rows/query limit, 700K+ pitches/season, enable caching for production
- ABS quantified: 52.2% overturn rate, 4.1 challenges/game, batters 4-5x more likely to challenge on 2 strikes
- Log5 is biased for batter-pitcher matchups (SABR confirmed) → use for team strengths only
- Isotonic regression > Platt scaling for sports betting calibration
- keeks library provides production Quarter Kelly implementation
- Feature engineering quality > model complexity for baseball (confirmed by correlation studies)
- Sutter Health Park: 2nd in MLB for HR/runs in 2025; use MLB-level park factors only

### Metis Review
**Identified Gaps** (addressed):
- **Settlement rules**: F5 tie = push (refund), game < 5 innings = no action, starter scratch = no action — implemented as first-class module with exhaustive tests
- **Vig removal**: Proportional de-vig applied to convert raw odds to fair probabilities before edge comparison
- **Odds snapshot policy**: Best available odds at runtime from The Odds API; frozen once Discord notification sent
- **As-of timestamps**: Every feature row stores computation timestamp; tests enforce no future data leakage
- **Canonical IDs**: Use MLB `game_pk` for games, FanGraphs IDs for players, standard team abbreviations
- **Failure policy**: Missing data → explicit "NO PICK (reason)" row; never guess silently; alert Discord on pipeline failures
- **Doubleheader handling**: Disambiguate via `game_pk` (unique per game including DH game 1/2)
- **Opener/bullpen games**: Detect via innings pitched threshold; apply team pitching composite instead of starter metrics
- **Max exposure**: No same-team F5 ML + F5 RL stacking (correlates → count as single bet for Kelly purposes)
- **Secrets**: All API keys via `.env` file; `.env` in `.gitignore`; fail loudly if missing

---

## Work Objectives

### Core Objective
Build a production-ready, automated MLB betting system that identifies +EV F5 bets daily using advanced sabermetrics, calibrated machine learning, and rigorous bankroll management.

### Concrete Deliverables
- `src/` — Python package with all pipeline modules
- `data/` — Parquet raw cache + SQLite database
- `tests/` — pytest suite with 90%+ critical-path coverage
- `config/` — YAML configuration for API keys, thresholds, stadium data
- Daily automated pipeline producing Discord pick cards
- Walk-forward backtest producing verified ROI/Brier score reports

### Definition of Done
- [ ] `python -m src.pipeline.daily --date 2026-04-15 --mode prod` exits 0 with predictions
- [ ] `python -m src.backtest.run --start 2019-03-20 --end 2025-10-31` produces Brier < 0.25
- [ ] pytest suite passes with 0 failures
- [ ] Discord webhook delivers formatted picks with edge %, Kelly stake, and confidence
- [ ] Bankroll tracker maintains correct ledger with settlement logic

### Must Have
- Anti-leakage enforcement with `as_of_timestamp` on every feature row
- Multi-window rolling features (7/14/30/60 games)
- ABS strike zone adjustment factor (league-wide + venue-specific exceptions)
- Park factor adjustments (2025 MLB-level data for Sutter Health Park)
- Weather adjustments for open-air stadiums (skip domed venues)
- F5 settlement rules: tie=push, <5 innings=no action, starter scratch=no action
- Proportional de-vig before edge calculation
- Quarter Kelly with 30% max drawdown kill-switch
- Walk-forward backtesting with reproducible cached data
- Discord failure notifications (not just pick notifications)

### Must NOT Have (Guardrails)
- ❌ Traditional stats (ERA, pitcher W-L, batting average) — sabermetrics only
- ❌ Batter-vs-pitcher Log5 features — biased per SABR research; team-only
- ❌ Full-game markets in v1 — F5 ML + F5 RL only
- ❌ Props, live betting, or automated bet placement
- ❌ Neural network layers — dataset too small (~17K games)
- ❌ Over-engineering ABS per-player adjustments before baseline validation
- ❌ Hard-coding API keys anywhere (must use `.env`)
- ❌ Silent fallbacks for missing data — always explicit "NO PICK" with reason
- ❌ Manual verification steps in acceptance criteria

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (fresh project)
- **Automated tests**: YES (Tests-after) — pytest for pipeline integrity
- **Framework**: pytest + pytest-cov
- **Coverage target**: 90%+ on settlement logic, edge calculation, anti-leakage; 70%+ overall

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Data Pipeline**: Use Bash (Python REPL / pytest) — import modules, verify outputs, check schemas
- **ML Model**: Use Bash (Python scripts) — train, predict, verify calibration metrics
- **Discord**: Use Bash (`--dry-run` mode) — verify JSON payload structure
- **Integration**: Use Bash (full pipeline) — end-to-end run with test data

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — start immediately, 8 parallel tasks):
├── Task 1:  Project scaffolding + dependencies [quick]
├── Task 2:  Configuration module (constants, mappings, coords) [quick]
├── Task 3:  SQLite schema design + initialization [quick]
├── Task 4:  Type definitions / data models (Pydantic) [quick]
├── Task 5:  The Odds API client [unspecified-high]
├── Task 6:  pybaseball data ingestion module [unspecified-high]
├── Task 7:  MLB API lineup client [unspecified-high]
├── Task 8:  OpenWeatherMap client [quick]

Wave 2 (Feature Engineering — after Wave 1, 7 parallel tasks):
├── Task 9:  Offensive features: wRC+, wOBA multi-window rolling (depends: 3,4,6) [deep]
├── Task 10: Pitching features: xFIP, xERA multi-window rolling (depends: 3,4,6) [deep]
├── Task 11: Defense features: DRS, OAA (depends: 3,4,6) [unspecified-high]
├── Task 12: Bullpen fatigue features: PC L3/L5, rest, IR% (depends: 3,4,6) [deep]
├── Task 13: Park factor + ABS zone adjustments (depends: 2,3) [unspecified-high]
├── Task 14: Weather adjustment engine (depends: 2,8) [unspecified-high]
├── Task 15: Pythagorean WP + Log5 team strength baselines (depends: 3,4) [unspecified-high]

Wave 3 (ML Pipeline — after Wave 2, 5 tasks, some parallel):
├── Task 16: Historical training data builder + anti-leakage (depends: 9-15) [deep]
├── Task 17: XGBoost model training + hyperparameter tuning (depends: 16) [deep]
├── Task 18: LR stacking meta-learner (depends: 17) [unspecified-high]
├── Task 19: Isotonic calibration layer (depends: 18) [unspecified-high]
├── Task 20: Walk-forward backtesting framework (depends: 16-19) [deep]

Wave 4 (Decision Engine + Integration — after Wave 3, 6 parallel tasks):
├── Task 21: Edge calculator: de-vig, implied prob, EV (depends: 5,19) [deep]
├── Task 22: Quarter Kelly bankroll manager (depends: 3,21) [unspecified-high]
├── Task 23: F5 settlement rules module (depends: 3,4) [unspecified-high]
├── Task 24: Daily pipeline orchestrator (depends: 6-8,9-15,17-19,21-23) [deep]
├── Task 25: Early-season Marcel blend module (depends: 9,10) [unspecified-high]
├── Task 26: Discord webhook formatter + notifier (depends: 21,22) [unspecified-high]

Wave 5 (Testing, Scheduling, Polish — after Wave 4, 5 parallel tasks):
├── Task 27: Windows Task Scheduler + .env setup (depends: 24) [quick]
├── Task 28: pytest suite: data integrity + anti-leakage tests (depends: 16,24) [unspecified-high]
├── Task 29: pytest suite: settlement + edge calc + Kelly tests (depends: 21-23) [unspecified-high]
├── Task 30: Performance tracker + CLV logging (depends: 3,5,24) [unspecified-high]
├── Task 31: Error handling, retry logic, failure alerts (depends: 24,26) [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA — end-to-end pipeline test (unspecified-high)
├── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay

Critical Path: T1 → T3 → T9 → T16 → T17 → T18 → T19 → T21 → T24 → T27 → F1-F4 → user okay
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 8 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 2-8 | 1 |
| 2 | 1 | 13,14 | 1 |
| 3 | 1 | 9-13,15,16,22,23,30 | 1 |
| 4 | 1 | 9-12,15,16,23 | 1 |
| 5 | 1 | 21,30 | 1 |
| 6 | 1 | 9-12,24 | 1 |
| 7 | 1 | 24 | 1 |
| 8 | 1 | 14 | 1 |
| 9 | 3,4,6 | 16,25 | 2 |
| 10 | 3,4,6 | 16,25 | 2 |
| 11 | 3,4,6 | 16 | 2 |
| 12 | 3,4,6 | 16 | 2 |
| 13 | 2,3 | 16 | 2 |
| 14 | 2,8 | 16 | 2 |
| 15 | 3,4 | 16 | 2 |
| 16 | 9-15 | 17,20,28 | 3 |
| 17 | 16 | 18,20 | 3 |
| 18 | 17 | 19 | 3 |
| 19 | 18 | 20,21 | 3 |
| 20 | 16-19 | — | 3 |
| 21 | 5,19 | 22,24,26,29 | 4 |
| 22 | 3,21 | 24,26,29 | 4 |
| 23 | 3,4 | 24,29 | 4 |
| 24 | 6-8,9-15,17-19,21-23 | 27,28,30,31 | 4 |
| 25 | 9,10 | 24 | 4 |
| 26 | 21,22 | 27,31 | 4 |
| 27 | 24,26 | — | 5 |
| 28 | 16,24 | — | 5 |
| 29 | 21-23 | — | 5 |
| 30 | 3,5,24 | — | 5 |
| 31 | 24,26 | — | 5 |

### Agent Dispatch Summary

- **Wave 1**: **8 tasks** — T1-T4 → `quick`, T5-T7 → `unspecified-high`, T8 → `quick`
- **Wave 2**: **7 tasks** — T9,T10,T12 → `deep`, T11,T13,T14,T15 → `unspecified-high`
- **Wave 3**: **5 tasks** — T16,T17,T20 → `deep`, T18,T19 → `unspecified-high`
- **Wave 4**: **6 tasks** — T21,T24 → `deep`, T22,T23,T25,T26 → `unspecified-high`
- **Wave 5**: **5 tasks** — T27 → `quick`, T28-T31 → `unspecified-high`
- **FINAL**: **4 tasks** — F1 → `oracle`, F2,F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.

- [ ] 1. Project Scaffolding + Dependencies

  **What to do**:
  - Create project directory structure:
    ```
    MLBPrediction2026/
    ├── pyproject.toml          # Project metadata + dependencies
    ├── .env.example            # Template for API keys
    ├── .gitignore              # Exclude .env, __pycache__, data/, .sisyphus/evidence/
    ├── src/
    │   ├── __init__.py
    │   ├── clients/            # External API clients
    │   ├── features/           # Feature engineering
    │   ├── model/              # ML pipeline
    │   ├── engine/             # Decision engine
    │   ├── pipeline/           # Daily orchestrator
    │   ├── notifications/      # Discord webhook
    │   ├── ops/                # Scheduling, retry, logging
    │   └── models/             # Pydantic data models
    ├── config/                 # YAML configuration files
    ├── data/                   # Parquet cache + SQLite DB
    ├── tests/                  # pytest suite
    └── scripts/                # Utility scripts
    ```
  - Create `pyproject.toml` with dependencies:
    - Core: `pybaseball>=2.3`, `pandas>=2.0`, `numpy`, `scikit-learn>=1.3`, `xgboost>=2.0`
    - Data: `sqlalchemy`, `pydantic>=2.0`, `pyyaml`, `python-dotenv`
    - API: `requests`, `httpx`
    - Dev: `pytest`, `pytest-cov`, `ruff`
  - Initialize `.gitignore` (exclude `.env`, `data/*.db`, `data/raw/`, `__pycache__/`)
  - Create `.env.example` with placeholder keys: `ODDS_API_KEY`, `OPENWEATHER_API_KEY`, `DISCORD_WEBHOOK_URL`

  **Must NOT do**:
  - Do not install SIERA-specific scrapers
  - Do not add neural network libraries (torch, tensorflow)
  - Do not create a web framework (flask, streamlit)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Standard scaffolding, no domain complexity

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-8)
  - **Blocks**: Tasks 2-8 (all depend on project structure existing)
  - **Blocked By**: None (start immediately)

  **References**:
  - **External**: pybaseball PyPI page for latest version
  - **Pattern**: Standard Python package layout with src/ directory

  **Acceptance Criteria**:
  - [ ] `pip install -e ".[dev]"` completes without errors
  - [ ] `python -c "import src"` succeeds
  - [ ] `.env.example` exists with all 3 API key placeholders
  - [ ] `.gitignore` excludes `.env`, `data/*.db`, `__pycache__/`

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Clean install from scratch
    Tool: Bash
    Preconditions: Fresh Python 3.11+ virtual environment
    Steps:
      1. Run `python -m venv .venv && .venv\Scripts\activate && pip install -e ".[dev]"`
      2. Run `python -c "import src; import pybaseball; import xgboost; import sklearn; print('OK')"`
      3. Run `pip list | findstr pybaseball` — verify version >= 2.3
    Expected Result: All imports succeed, no missing dependencies
    Failure Indicators: ImportError, ModuleNotFoundError, pip install failure
    Evidence: .sisyphus/evidence/task-1-clean-install.txt

  Scenario: Gitignore prevents secret leakage
    Tool: Bash
    Preconditions: .env file created with dummy key
    Steps:
      1. Create `.env` with `ODDS_API_KEY=test123`
      2. Run `git status` — verify `.env` does NOT appear in untracked files
      3. Run `git status` — verify `data/` directory is ignored
    Expected Result: .env and data/ excluded from git tracking
    Failure Indicators: .env appears in `git status` output
    Evidence: .sisyphus/evidence/task-1-gitignore-check.txt
  ```

  **Commit**: YES (group C1)
  - Message: `feat(scaffold): project structure, dependencies, env template`
  - Files: `pyproject.toml`, `.gitignore`, `.env.example`, `src/__init__.py`, directory tree
  - Pre-commit: `pip install -e ".[dev]" && python -c "import src"`

- [ ] 2. Configuration Module (Constants, Mappings, Stadium Coordinates)

  **What to do**:
  - Create `config/settings.yaml` with:
    - `teams`: All 30 MLB team abbreviations → full names, league, division
    - `stadiums`: Team → lat/lon coordinates, dome/open-air flag, capacity, park name
    - `abs_exceptions`: List of venue/game identifiers where ABS is NOT active (Mexico City Series, Field of Dreams, Little League Classic)
    - `thresholds`: edge_min=0.03, kelly_fraction=0.25, max_drawdown=0.30, min_games_rolling=7
    - `rolling_windows`: [7, 14, 30, 60]
    - `pythagorean_exponent`: 1.83
    - `season_range`: {train_start: 2019, train_end: 2025}
  - Create `src/config.py` module that:
    - Loads `.env` for API keys (python-dotenv)
    - Loads `settings.yaml` for constants
    - Provides typed access via Pydantic Settings model
    - Validates all required env vars present at startup (fail loudly if missing)
  - Include Sutter Health Park (Sacramento) with 2025 MLB park factors (Runs: 1.25, HR: 1.30)
  - Include stadium geometry data for wind interaction modeling (orientation, open/closed sides)

  **Must NOT do**:
  - Do not hard-code API keys anywhere
  - Do not include Steinbrenner Field for Rays (they're at Tropicana in 2026)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Configuration file creation, no complex logic

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-8)
  - **Blocks**: Tasks 13, 14
  - **Blocked By**: Task 1 (needs project structure)

  **References**:
  - **Data**: Sutter Health Park dimensions: LF 330, CF 403, RF 325; park factor ~1.25 runs (2nd in MLB 2025)
  - **Data**: Tropicana Field: dome, turf, known park factors (pitcher-friendly historically)
  - **Data**: ABS exceptions: Mexico City (Apr 25-26), Field of Dreams (Aug 13), Little League Classic (Aug 23)
  - **External**: python-dotenv docs for .env loading pattern
  - **External**: Pydantic Settings for typed configuration

  **Acceptance Criteria**:
  - [ ] `settings.yaml` contains all 30 teams with coordinates
  - [ ] `src/config.py` loads env vars and YAML; raises on missing keys
  - [ ] Sutter Health Park correctly listed with park factors
  - [ ] `abs_exceptions` list includes all 3 known exception events

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Config loads with valid env
    Tool: Bash
    Preconditions: .env file with valid keys
    Steps:
      1. Run `python -c "from src.config import Settings; s = Settings(); print(s.teams['NYY']); print(s.stadiums['OAK']['park_name'])"`
      2. Verify output includes "New York Yankees" and "Sutter Health Park"
      3. Run `python -c "from src.config import Settings; s = Settings(); print(len(s.teams))"`
      4. Verify output is "30"
    Expected Result: 30 teams loaded, Sutter Health Park present for OAK/A's
    Failure Indicators: KeyError, FileNotFoundError, validation error
    Evidence: .sisyphus/evidence/task-2-config-loads.txt

  Scenario: Config fails on missing API key
    Tool: Bash
    Preconditions: .env file WITHOUT ODDS_API_KEY
    Steps:
      1. Remove ODDS_API_KEY from .env
      2. Run `python -c "from src.config import Settings; Settings()"` and capture stderr
      3. Verify it raises a ValidationError or SystemExit with clear message about missing key
    Expected Result: Loud failure mentioning ODDS_API_KEY
    Failure Indicators: Silent success or generic error without key name
    Evidence: .sisyphus/evidence/task-2-missing-key-error.txt
  ```

  **Commit**: YES (group C1)
  - Message: `feat(config): settings YAML, env loader, stadium coordinates, ABS exceptions`
  - Files: `config/settings.yaml`, `src/config.py`
  - Pre-commit: `python -c "from src.config import Settings; Settings()"`

- [ ] 3. SQLite Schema Design + Initialization

  **What to do**:
  - Design and create SQLite database schema in `src/db.py`:
    - **games** table: `game_pk` (PRIMARY KEY), date, home_team, away_team, home_starter_id, away_starter_id, venue, is_dome, is_abs_active, f5_home_score, f5_away_score, final_home_score, final_away_score, status (scheduled/final/suspended/postponed)
    - **features** table: `game_pk` (FK), feature_name, feature_value, window_size, as_of_timestamp (CRITICAL for anti-leakage), created_at
    - **predictions** table: `game_pk` (FK), model_version, f5_ml_home_prob, f5_ml_away_prob, f5_rl_home_prob, f5_rl_away_prob, predicted_at
    - **odds_snapshots** table: `game_pk` (FK), book_name, market_type (f5_ml/f5_rl), home_odds, away_odds, fetched_at, is_frozen (bool — frozen once Discord notification sent)
    - **bets** table: `id` (AUTO), game_pk (FK), market_type, side (home/away), edge_pct, kelly_stake, odds_at_bet, result (win/loss/push/no_action/pending), settled_at, profit_loss
    - **bankroll_ledger** table: `id` (AUTO), timestamp, event_type (bet_placed/bet_settled/drawdown_alert/kill_switch), amount, running_balance, notes
  - Create `init_db()` function that creates tables if not exist
  - Add migration support (version table for schema changes)
  - Store `as_of_timestamp` on features table to enable anti-leakage testing

  **Must NOT do**:
  - Do not use ORM (keep raw SQL for transparency and performance)
  - Do not store Statcast raw data in SQLite (that goes in Parquet)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: SQL schema design, straightforward

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4-8)
  - **Blocks**: Tasks 9-13, 15, 16, 22, 23, 30
  - **Blocked By**: Task 1

  **References**:
  - **Pattern**: SQLite with raw SQL (no ORM) for betting systems — common in OSS betting repos
  - **Data**: `game_pk` is MLB's unique game identifier, disambiguates doubleheaders
  - **External**: Python sqlite3 stdlib docs

  **Acceptance Criteria**:
  - [ ] `init_db()` creates all 6 tables with correct schemas
  - [ ] `as_of_timestamp` column exists on features table
  - [ ] `is_frozen` column exists on odds_snapshots table
  - [ ] Re-running `init_db()` is idempotent (CREATE IF NOT EXISTS)
  - [ ] Schema version tracking works

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Database initialization from scratch
    Tool: Bash
    Preconditions: No existing database file
    Steps:
      1. Delete data/mlb.db if exists
      2. Run `python -c "from src.db import init_db; init_db('data/test.db')"`
      3. Run `python -c "import sqlite3; conn = sqlite3.connect('data/test.db'); tables = conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall(); print([t[0] for t in tables])"`
      4. Verify output contains: games, features, predictions, odds_snapshots, bets, bankroll_ledger
    Expected Result: All 6 tables created
    Failure Indicators: Missing tables, sqlite3 errors
    Evidence: .sisyphus/evidence/task-3-db-init.txt

  Scenario: Anti-leakage column enforcement
    Tool: Bash
    Preconditions: Database initialized
    Steps:
      1. Run `python -c "import sqlite3; conn = sqlite3.connect('data/test.db'); cols = conn.execute('PRAGMA table_info(features)').fetchall(); print([c[1] for c in cols])"`
      2. Verify 'as_of_timestamp' appears in column list
      3. Attempt to insert a feature row without as_of_timestamp
      4. Verify it fails (NOT NULL constraint)
    Expected Result: as_of_timestamp is NOT NULL, enforced at DB level
    Failure Indicators: Column missing or allows NULL
    Evidence: .sisyphus/evidence/task-3-antileak-column.txt
  ```

  **Commit**: YES (group C1)
  - Message: `feat(db): SQLite schema with games, features, predictions, odds, bets, bankroll`
  - Files: `src/db.py`, `data/` directory
  - Pre-commit: `python -c "from src.db import init_db; init_db('data/test.db')"`

- [ ] 4. Type Definitions / Data Models (Pydantic)

  **What to do**:
  - Create `src/models/` package with Pydantic v2 models:
    - `game.py`: `Game` (game_pk, date, home/away team, starters, venue, dome flag, abs_active)
    - `features.py`: `GameFeatures` (offensive, pitching, defense, bullpen, park, weather, baseline — all with as_of_timestamp)
    - `prediction.py`: `Prediction` (game_pk, f5_ml_probs, f5_rl_probs, model_version, calibration_method)
    - `odds.py`: `OddsSnapshot` (game_pk, book, market, home/away odds, implied_prob_home/away, devigged_prob_home/away, fetched_at)
    - `bet.py`: `BetDecision` (game_pk, market, side, edge_pct, kelly_fraction, stake_amount, odds), `BetResult` (result enum: WIN/LOSS/PUSH/NO_ACTION/PENDING)
    - `lineup.py`: `Lineup` (game_pk, team, batting_order list of player_ids, confirmed bool, source)
  - All models must have validators ensuring:
    - Probabilities ∈ [0, 1]
    - Edge percentages can be negative (no bet) or positive
    - Odds are in American format (e.g., -110, +150)
    - `as_of_timestamp` is always UTC

  **Must NOT do**:
  - Do not add batter-vs-pitcher matchup models
  - Do not add full-game market types

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Data model definition, well-defined structure

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5-8)
  - **Blocks**: Tasks 9-12, 15, 16, 23
  - **Blocked By**: Task 1

  **References**:
  - **External**: Pydantic v2 documentation for model_validator, field_validator
  - **Data**: American odds format: negative = favorite (e.g., -110 means bet $110 to win $100), positive = underdog (e.g., +150 means bet $100 to win $150)

  **Acceptance Criteria**:
  - [ ] All 6 model files created with correct fields
  - [ ] Probability validators reject values outside [0,1]
  - [ ] BetResult enum has exactly: WIN, LOSS, PUSH, NO_ACTION, PENDING
  - [ ] All models serialize/deserialize to JSON correctly

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Probability validation rejects invalid values
    Tool: Bash
    Preconditions: Models module importable
    Steps:
      1. Run `python -c "from src.models.prediction import Prediction; Prediction(game_pk=1, f5_ml_home_prob=1.5, f5_ml_away_prob=0.5, model_version='v1', calibration_method='isotonic', predicted_at='2026-04-01T12:00:00Z')"` and capture stderr
      2. Verify ValidationError raised for f5_ml_home_prob > 1.0
      3. Run `python -c "from src.models.prediction import Prediction; p = Prediction(game_pk=1, f5_ml_home_prob=0.55, f5_ml_away_prob=0.45, model_version='v1', calibration_method='isotonic', predicted_at='2026-04-01T12:00:00Z'); print(p.model_dump_json())"` 
      4. Verify valid JSON output
    Expected Result: Invalid prob rejected, valid prob serializes to JSON
    Failure Indicators: No validation error on 1.5, or serialization failure on valid input
    Evidence: .sisyphus/evidence/task-4-prob-validation.txt

  Scenario: BetResult enum completeness
    Tool: Bash
    Preconditions: Models module importable
    Steps:
      1. Run `python -c "from src.models.bet import BetResult; print([e.value for e in BetResult])"`
      2. Verify output contains exactly: ['WIN', 'LOSS', 'PUSH', 'NO_ACTION', 'PENDING']
    Expected Result: All 5 result types present
    Failure Indicators: Missing enum value or extra unexpected values
    Evidence: .sisyphus/evidence/task-4-bet-result-enum.txt
  ```

  **Commit**: YES (group C1)
  - Message: `feat(models): Pydantic data models for games, features, predictions, odds, bets, lineups`
  - Files: `src/models/*.py`
  - Pre-commit: `python -c "from src.models import game, features, prediction, odds, bet, lineup"`

- [ ] 5. The Odds API Client

  **What to do**:
  - Create `src/clients/odds_client.py`:
    - `fetch_mlb_odds(date, markets=['h2h_f5', 'spreads_f5'])` → list of `OddsSnapshot`
    - Parse American odds from API response; compute implied probabilities
    - Implement **proportional de-vig**: `fair_prob = implied_prob / sum(implied_probs)` per market
    - Store raw + de-vigged odds in SQLite via odds_snapshots table
    - Handle missing F5 markets gracefully (log warning, return empty)
    - Implement rate limiting (free tier: 500 requests/month)
    - Track API usage count in SQLite to prevent overage
  - The Odds API endpoint: `https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/`
    - Parameters: `apiKey`, `regions=us`, `markets=h2h,spreads`, `oddsFormat=american`
    - Note: F5 markets may use different market keys — discover via API exploration
  - Implement `freeze_odds(game_pk)` — marks odds as frozen after Discord notification sent

  **Must NOT do**:
  - Do not scrape odds from websites (use API only)
  - Do not store odds for full-game markets in v1

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: API integration with rate limiting, de-vig math, error handling

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-4, 6-8)
  - **Blocks**: Tasks 21, 30
  - **Blocked By**: Task 1

  **References**:
  - **External**: The Odds API docs: https://the-odds-api.com/liveapi/guides/v4/
  - **Formula**: Proportional de-vig: `fair_p = implied_p / (implied_p_home + implied_p_away)`
  - **Formula**: American odds to implied prob: negative odds → `|odds| / (|odds| + 100)`, positive odds → `100 / (odds + 100)`

  **Acceptance Criteria**:
  - [ ] Client fetches MLB odds and parses American format correctly
  - [ ] De-vig produces probabilities summing to 1.0
  - [ ] Rate limit tracking prevents exceeding 500/month
  - [ ] Missing F5 markets return empty list (no crash)
  - [ ] `freeze_odds()` sets `is_frozen=True` in DB

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: De-vig math correctness
    Tool: Bash
    Preconditions: Odds client importable
    Steps:
      1. Run `python -c "from src.clients.odds_client import american_to_implied, devig; h = american_to_implied(-150); a = american_to_implied(+130); fair_h, fair_a = devig(h, a); print(f'{fair_h:.4f} + {fair_a:.4f} = {fair_h+fair_a:.4f}')"`
      2. Verify -150 implied ≈ 0.6000, +130 implied ≈ 0.4348
      3. Verify de-vigged sum = 1.0000
    Expected Result: De-vigged probabilities sum to exactly 1.0
    Failure Indicators: Sum != 1.0, negative probabilities, division by zero
    Evidence: .sisyphus/evidence/task-5-devig-math.txt

  Scenario: Missing market handling
    Tool: Bash
    Preconditions: Mock API response without F5 markets
    Steps:
      1. Create mock response with only full-game h2h (no F5 keys)
      2. Call `parse_odds(mock_response, market_type='h2h_f5')`
      3. Verify returns empty list, no exception raised
      4. Verify warning logged
    Expected Result: Graceful empty return, logged warning
    Failure Indicators: Exception, crash, or silent success without warning
    Evidence: .sisyphus/evidence/task-5-missing-market.txt
  ```

  **Commit**: YES (group C2)
  - Message: `feat(odds): The Odds API client with de-vig, rate limiting, freeze support`
  - Files: `src/clients/odds_client.py`

- [ ] 6. pybaseball Data Ingestion Module

  **What to do**:
  - Create `src/clients/statcast_client.py`:
    - `fetch_statcast_range(start_date, end_date)` → saves to Parquet in `data/raw/statcast/`
    - `fetch_pitcher_stats(season, min_ip=20)` → FanGraphs pitching leaderboard (xFIP, xERA, K%, BB%, GB%)
    - `fetch_batting_stats(season, min_pa=50)` → FanGraphs batting leaderboard (wRC+, wOBA, ISO, BABIP)
    - `fetch_fielding_stats(season)` → DRS from FanGraphs + OAA from Statcast
    - `fetch_catcher_framing(season)` → Statcast catcher framing data
    - `fetch_team_game_logs(season, team)` → game-by-game results for run scoring
  - Enable pybaseball caching: `cache.enable()` with Parquet format
  - Implement incremental updates (only fetch new data since last pull)
  - Handle Statcast 30K row limit by chunking date ranges
  - Store raw data in Parquet (compressed, columnar, fast reads)
  - Create player ID lookup utility (FanGraphs ID ↔ MLB ID ↔ BBREF ID)

  **Must NOT do**:
  - Do not fetch traditional stats (ERA, W-L, BA)
  - Do not scrape SIERA (using xFIP + xERA as proxies)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Data pipeline with multiple API sources, chunking, caching

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-5, 7-8)
  - **Blocks**: Tasks 9-12, 24
  - **Blocked By**: Task 1

  **References**:
  - **External**: pybaseball docs: `statcast()`, `batting_stats()`, `pitching_stats()`, `statcast_outs_above_average()`
  - **Pattern**: Enable caching with `from pybaseball import cache; cache.enable()`
  - **Limit**: Statcast queries >30K rows auto-split; >42 days triggers warning
  - **Data**: pybaseball `playerid_lookup(last, first)` returns cross-reference IDs

  **Acceptance Criteria**:
  - [ ] Statcast data fetched and saved to Parquet files
  - [ ] FanGraphs batting stats include wRC+, wOBA columns
  - [ ] FanGraphs pitching stats include xFIP columns
  - [ ] OAA data fetched via `statcast_outs_above_average()`
  - [ ] Incremental updates skip already-fetched date ranges
  - [ ] pybaseball caching is enabled

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Statcast data retrieval and Parquet storage
    Tool: Bash
    Preconditions: pybaseball installed, internet access
    Steps:
      1. Run `python -c "from src.clients.statcast_client import fetch_statcast_range; df = fetch_statcast_range('2025-09-01', '2025-09-03'); print(f'Rows: {len(df)}, Cols: {len(df.columns)}')"` (timeout: 120s)
      2. Verify rows > 0 and columns > 80
      3. Verify Parquet file created in data/raw/statcast/
      4. Run `python -c "import pandas as pd; df = pd.read_parquet('data/raw/statcast/'); print(df.columns.tolist()[:10])"` — verify columns include launch_speed, release_speed
    Expected Result: Data fetched, stored as Parquet, key columns present
    Failure Indicators: Empty DataFrame, missing Parquet file, network timeout
    Evidence: .sisyphus/evidence/task-6-statcast-fetch.txt

  Scenario: FanGraphs advanced metrics available
    Tool: Bash
    Preconditions: pybaseball installed
    Steps:
      1. Run `python -c "from src.clients.statcast_client import fetch_batting_stats; df = fetch_batting_stats(2025); print('wRC+' in df.columns, 'wOBA' in df.columns)"` (timeout: 60s)
      2. Verify both print True
    Expected Result: wRC+ and wOBA columns present in batting stats
    Failure Indicators: False for either column check
    Evidence: .sisyphus/evidence/task-6-fangraphs-metrics.txt
  ```

  **Commit**: YES (group C2)
  - Message: `feat(statcast): pybaseball data ingestion with Parquet caching, FanGraphs stats`
  - Files: `src/clients/statcast_client.py`

- [ ] 7. MLB API Lineup Client

  **What to do**:
  - Create `src/clients/lineup_client.py`:
    - `fetch_confirmed_lineups(date)` → list of `Lineup` models per game
    - Primary source: Scrape RotoBaller/RotoGrinders projected lineups (available 2-4 hrs pre-game)
    - Fallback: MLB Stats API (`statsapi.mlb.com`) for officially confirmed lineups (~1 hr pre-game)
    - `is_lineup_confirmed(game_pk)` → bool (check if MLB API has posted official lineup)
    - Detect starter changes: compare projected vs confirmed; if starter differs, flag for re-evaluation
    - Handle missing lineups: return `Lineup(confirmed=False)` with projected data
    - Detect "opener" / bullpen games: if projected starter has < 3.0 avg IP in recent starts
  - Include player ID mapping to FanGraphs IDs for metric lookup

  **Must NOT do**:
  - Do not build a custom lineup projection model
  - Do not automate bet placement based on lineup changes

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Web scraping + API integration, fallback logic, opener detection

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-6, 8)
  - **Blocks**: Task 24
  - **Blocked By**: Task 1

  **References**:
  - **External**: MLB Stats API (statsapi.mlb.com) — free, no auth required for basic lineup data
  - **Pattern**: RotoGrinders lineups page scraping for early lineup availability
  - **Data**: `mlb-statsapi` Python package for official API access

  **Acceptance Criteria**:
  - [ ] Lineups fetched for a given date with player IDs
  - [ ] Fallback from RotoBaller to MLB API works when primary fails
  - [ ] Opener/bullpen game detection flags games with < 3.0 avg IP starters
  - [ ] Starter change detection compares projected vs confirmed

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Lineup retrieval with confirmed flag
    Tool: Bash
    Preconditions: Internet access, lineup client importable
    Steps:
      1. Run `python -c "from src.clients.lineup_client import fetch_confirmed_lineups; lineups = fetch_confirmed_lineups('2025-09-15'); print(f'Games: {len(lineups)}')"` (timeout: 30s)
      2. Verify at least 1 game returned (September 15 is mid-season)
      3. Verify each lineup has batting_order list and confirmed boolean
    Expected Result: Multiple games with lineup data
    Failure Indicators: Empty result, missing batting_order field, crash
    Evidence: .sisyphus/evidence/task-7-lineup-fetch.txt

  Scenario: Missing lineup graceful handling
    Tool: Bash
    Preconditions: Mock MLB API returning empty lineup
    Steps:
      1. Call `fetch_confirmed_lineups()` with mock returning empty data for one game
      2. Verify that game returns `Lineup(confirmed=False)` instead of crashing
    Expected Result: Unconfirmed lineup returned, no exception
    Failure Indicators: Exception raised, None returned instead of Lineup object
    Evidence: .sisyphus/evidence/task-7-missing-lineup.txt
  ```

  **Commit**: YES (group C2)
  - Message: `feat(lineups): MLB lineup client with RotoBaller scraping + API fallback`
  - Files: `src/clients/lineup_client.py`

- [ ] 8. OpenWeatherMap Client

  **What to do**:
  - Create `src/clients/weather_client.py`:
    - `fetch_game_weather(team_abbr, game_datetime)` → WeatherData model
    - Use stadium lat/lon from config to query OpenWeatherMap forecast API
    - Return: temperature (°F), humidity (%), wind_speed (mph), wind_direction (degrees), pressure (hPa), air_density (calculated)
    - Calculate **air density**: `ρ = (P × M) / (R × T)` adjusted for humidity (affects ball flight)
    - Skip weather fetch for domed stadiums (return neutral/default values)
    - Cache weather data in SQLite to avoid redundant API calls
    - Handle API failures gracefully (return default neutral weather, log warning)
  - Create wind interaction calculator:
    - Cross-reference wind direction (degrees) with stadium orientation (CF direction from config)
    - Output: `wind_factor` — positive = wind blowing out (favors HR), negative = wind blowing in

  **Must NOT do**:
  - Do not build complex fluid dynamics models
  - Do not factor weather into domed stadium predictions

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Simple API client with caching, straightforward physics formula

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-7)
  - **Blocks**: Task 14
  - **Blocked By**: Task 1

  **References**:
  - **External**: OpenWeatherMap forecast API: `api.openweathermap.org/data/2.5/forecast`
  - **Formula**: Air density: `ρ = (P_dry / (R_dry × T)) + (P_vapor / (R_vapor × T))` where P in Pa, T in Kelvin
  - **Data**: Stadium orientations (CF direction in degrees) from config/settings.yaml
  - **Research**: Temperature +1°F ≈ +0.1% HR distance; wind 10mph out ≈ +5% HR rate

  **Acceptance Criteria**:
  - [ ] Weather fetched for open-air stadiums with temp, humidity, wind
  - [ ] Domed stadiums return default/neutral weather without API call
  - [ ] Air density calculated correctly from pressure + humidity + temperature
  - [ ] Wind factor calculated relative to stadium orientation
  - [ ] API failures return defaults (not crash)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Weather for open-air vs dome
    Tool: Bash
    Preconditions: Weather client importable, config loaded
    Steps:
      1. Run `python -c "from src.clients.weather_client import fetch_game_weather; w = fetch_game_weather('NYY', '2025-07-15T19:00:00'); print(f'Temp: {w.temperature}, Wind: {w.wind_speed}')"` (timeout: 15s)
      2. Verify temperature and wind_speed are realistic values (not 0 or None)
      3. Run `python -c "from src.clients.weather_client import fetch_game_weather; w = fetch_game_weather('TB', '2025-07-15T19:00:00'); print(f'Is default: {w.is_dome_default}')"` (TB = Tropicana Field, dome)
      4. Verify dome returns default weather (is_dome_default = True)
    Expected Result: Open-air gets real weather; dome gets defaults
    Failure Indicators: API error, dome calling weather API, zero values
    Evidence: .sisyphus/evidence/task-8-weather-fetch.txt
  ```

  **Commit**: YES (group C2)
  - Message: `feat(weather): OpenWeatherMap client with air density, wind factor, dome detection`
  - Files: `src/clients/weather_client.py`

- [ ] 9. Offensive Features: wRC+, wOBA Multi-Window Rolling

  **What to do**:
  - Create `src/features/offense.py`:
    - For each team on a given date, compute rolling averages over [7, 14, 30, 60] game windows:
      - Team wRC+ (weighted runs created plus)
      - Team wOBA (weighted on-base average)
      - Team ISO (isolated power)
      - Team BABIP (batting avg on balls in play)
      - Team K% and BB%
    - For the **starting lineup specifically** (not full roster): compute lineup-weighted versions of above metrics
    - Implement Marcel-style early-season blending: `blended = (current_year × games_played + prior_year × regression_weight) / (games_played + regression_weight)` where regression_weight = 30 (approximately 1 month of games)
    - All features must be tagged with `as_of_timestamp` (the date BEFORE the game, not including that game's results)
    - Store computed features in SQLite features table
  - **Anti-leakage rule**: When computing rolling average for game on date D, only use data from games before D (exclusive of D). This is the "lagged" approach.

  **Must NOT do**:
  - Do not compute individual batter-vs-pitcher splits (biased per SABR research)
  - Do not use traditional batting average

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Complex rolling window logic with anti-leakage enforcement, Marcel blending

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 10-15)
  - **Blocks**: Tasks 16, 25
  - **Blocked By**: Tasks 3, 4, 6

  **References**:
  - **Formula**: Marcel regression: `blended = (curr × GP + prior × 30) / (GP + 30)` — standard 30-game regression weight
  - **Data**: wRC+ league average = 100 by definition; wOBA league average ~0.310-0.320 depending on year
  - **Pattern**: Use `pandas.DataFrame.rolling()` with `min_periods` parameter for early-season stability
  - **External**: FanGraphs library definitions for wRC+, wOBA formulas

  **Acceptance Criteria**:
  - [ ] 4 rolling windows (7/14/30/60) computed for each offensive metric
  - [ ] Lineup-weighted versions computed using confirmed lineup player IDs
  - [ ] Marcel blend activates when games_played < 30
  - [ ] No future data leakage: feature for game on date D uses only data strictly before D
  - [ ] Features stored in SQLite with as_of_timestamp

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Anti-leakage verification
    Tool: Bash
    Preconditions: Historical data loaded for 2025 season
    Steps:
      1. Compute offensive features for NYY on 2025-07-15
      2. Verify the as_of_timestamp is 2025-07-14 (day before)
      3. Verify the 30-game rolling wRC+ does NOT include 2025-07-15 game stats
      4. Manually check: latest game in the rolling window is 2025-07-14 or earlier
    Expected Result: No data from July 15 or later included in features computed for July 15's game
    Failure Indicators: Feature includes current-day stats
    Evidence: .sisyphus/evidence/task-9-antileak-offense.txt

  Scenario: Marcel blend early season
    Tool: Bash
    Preconditions: 2025 and 2024 season data available
    Steps:
      1. Compute features for a team on 2025-04-10 (only ~10 games played)
      2. Verify the blended wRC+ is closer to 2024 season final value than raw 2025 10-game average
      3. Verify regression weight is applied: `(curr × 10 + prior × 30) / (10 + 30)`
    Expected Result: Early-season features blend toward prior year
    Failure Indicators: Raw noisy 10-game average used without regression
    Evidence: .sisyphus/evidence/task-9-marcel-blend.txt
  ```

  **Commit**: YES (group C3)
  - Message: `feat(features): offensive rolling features (wRC+, wOBA) with Marcel blend`
  - Files: `src/features/offense.py`

- [ ] 10. Pitching Features: xFIP, xERA Multi-Window Rolling

  **What to do**:
  - Create `src/features/pitching.py`:
    - For the **starting pitcher** of each game, compute rolling averages over [7, 14, 30, 60] game-starts windows:
      - xFIP (expected fielding independent pitching) — from FanGraphs via pybaseball
      - xERA (expected ERA from Statcast) — from `statcast_pitcher_expected_stats()`
      - K% (strikeout rate), BB% (walk rate)
      - GB% (ground ball rate), HR/FB (home run to fly ball ratio)
      - Avg fastball velocity (from Statcast pitch-level data)
      - Pitch mix entropy (variety of pitch types used — higher entropy = less predictable)
    - Marcel-style early-season blending (same formula as offense, regression_weight=15 starts for pitchers)
    - For **opposing starter** too (mirror features for the other team's starter)
    - Handle opener/bullpen games: if projected starter has < 3.0 avg IP in recent starts, use team pitching composite instead

  **Must NOT do**:
  - Do not use ERA (traditional, defense-dependent)
  - Do not scrape SIERA (using xFIP + xERA as proxies)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Complex starter identification, opener detection, rolling with Marcel

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 9, 11-15)
  - **Blocks**: Tasks 16, 25
  - **Blocked By**: Tasks 3, 4, 6

  **References**:
  - **Data**: xFIP available from `pitching_stats()` via pybaseball; xERA from `statcast_pitcher_expected_stats()`
  - **Formula**: Pitch mix entropy: `H = -Σ(p_i × log2(p_i))` where p_i = fraction of each pitch type
  - **Pattern**: Pitcher rolling windows count by starts (not calendar games)
  - **Data**: Marcel regression for pitchers: 15 starts regression weight (pitchers have smaller sample sizes)

  **Acceptance Criteria**:
  - [ ] 4 rolling windows per pitching metric, counted by starts
  - [ ] xFIP and xERA both present (proxying SIERA)
  - [ ] Opener/bullpen game detection working
  - [ ] Marcel blend with 15-start regression weight
  - [ ] Home and away starter features computed for each game

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Pitcher rolling by starts not calendar days
    Tool: Bash
    Preconditions: 2025 pitching data loaded
    Steps:
      1. Pick a specific pitcher (e.g., Gerrit Cole) and compute 7-start rolling xFIP on 2025-07-15
      2. Verify the 7 games included are the 7 most recent STARTS (not relief appearances)
      3. Count starts backwards from July 14 — verify exactly 7 starts included
    Expected Result: Rolling window counts starts only, not all team games
    Failure Indicators: Window includes non-start appearances or wrong count
    Evidence: .sisyphus/evidence/task-10-pitcher-rolling.txt

  Scenario: Opener game detection
    Tool: Bash
    Preconditions: Data for a known opener game (e.g., Rays 2025)
    Steps:
      1. Identify a game where the "starter" pitched < 2 innings
      2. Verify `detect_opener()` flags this game
      3. Verify team pitching composite is used instead of individual starter metrics
    Expected Result: Opener detected, team composite applied
    Failure Indicators: Opener not detected, individual stats used for 1-inning "starter"
    Evidence: .sisyphus/evidence/task-10-opener-detect.txt
  ```

  **Commit**: YES (group C3)
  - Message: `feat(features): pitching rolling features (xFIP, xERA, velocity) with opener detection`
  - Files: `src/features/pitching.py`

- [ ] 11. Defense Features: DRS, OAA

  **What to do**:
  - Create `src/features/defense.py`:
    - Compute team-level defensive metrics:
      - DRS (Defensive Runs Saved) — from FanGraphs fielding stats
      - OAA (Outs Above Average) — from `statcast_outs_above_average()`
      - Team defensive efficiency (proportion of balls in play converted to outs)
    - These are slower-moving metrics; compute season-to-date and 30/60-game rolling
    - Weight by position importance for F5 (up-the-middle defense matters more in first 5 innings)
    - Include catcher framing runs — with ABS depreciation factor:
      - `adjusted_framing = raw_framing × abs_retention_factor` where retention_factor = 0.75 (mid-estimate from research: 70-80% retained in low-mid leverage)

  **Must NOT do**:
  - Do not compute individual fielder DRS (team-level only for v1)
  - Do not over-weight framing in ABS era (apply depreciation)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Multiple data source aggregation, ABS adjustment

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 9, 10, 12-15)
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 3, 4, 6

  **References**:
  - **Data**: `statcast_outs_above_average()` returns OAA per fielder; aggregate to team
  - **Data**: `statcast_catcher_framing()` returns framing runs per catcher
  - **Research**: ABS framing retention: 70-80% low/mid leverage, 60-75% high leverage → use 0.75 default
  - **Data**: `fielding_stats()` from pybaseball for DRS

  **Acceptance Criteria**:
  - [ ] Team DRS and OAA computed per game date
  - [ ] Catcher framing adjusted by ABS retention factor (0.75)
  - [ ] Season-to-date and 30/60 rolling windows computed
  - [ ] Features stored with as_of_timestamp

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: ABS framing depreciation applied
    Tool: Bash
    Steps:
      1. Compute raw catcher framing for a team (e.g., 5.0 framing runs)
      2. Verify adjusted_framing = 5.0 × 0.75 = 3.75
      3. Verify the depreciation factor is configurable in settings.yaml
    Expected Result: Framing runs reduced by 25% with ABS adjustment
    Evidence: .sisyphus/evidence/task-11-abs-framing.txt
  ```

  **Commit**: YES (group C3)
  - Message: `feat(features): defense features (DRS, OAA, framing with ABS depreciation)`
  - Files: `src/features/defense.py`

- [ ] 12. Bullpen Fatigue Features: PC L3/L5, Rest Days, IR%

  **What to do**:
  - Create `src/features/bullpen.py`:
    - For each team on a given date, compute:
      - **PC L3**: Total bullpen pitch count over last 3 days
      - **PC L5**: Total bullpen pitch count over last 5 days
      - **Avg rest days**: Mean days of rest for top 5 relievers (by usage)
      - **IR%**: Inherited Runners Scored percentage (bullpen's tendency to allow inherited runners to score) — rolling 30-game
      - **Bullpen xFIP**: Team bullpen's composite xFIP (weighted by innings pitched)
      - **High-leverage availability**: Count of relievers with 0-1 days rest who are likely unavailable
    - Source pitch counts from Statcast pitch-level data (group by pitcher, game date, count pitches)
    - While these are primarily for full-game predictions, they ALSO matter for F5 because:
      - Starter may be on short leash → bullpen used in inning 4-5
      - Opponent's bullpen fatigue → may affect managerial decisions on starter pull

  **Must NOT do**:
  - Do not project individual reliever usage (team composite only)
  - Do not compute bullpen features for the first 3 days of the season (insufficient data)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Complex aggregation of pitch-level data across multiple days and relievers

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 9-11, 13-15)
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 3, 4, 6

  **References**:
  - **Data**: Statcast pitch-level data has `pitcher` ID and `game_date` — count rows per pitcher per game for pitch counts
  - **Data**: `game_type` field distinguishes regular season from spring training
  - **Formula**: IR% = inherited runners scored / total inherited runners (from Baseball Reference game logs)
  - **Pattern**: Use `pd.DataFrame.groupby(['pitcher', 'game_date']).size()` for pitch counts

  **Acceptance Criteria**:
  - [ ] PC L3 and PC L5 computed correctly from Statcast pitch counts
  - [ ] IR% rolling 30-game computed from game-level bullpen data
  - [ ] High-leverage availability count calculated
  - [ ] Returns empty/default for first 3 days of season
  - [ ] Features tagged with as_of_timestamp

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Pitch count aggregation correctness
    Tool: Bash
    Steps:
      1. For a known team/date, compute PC L3 from Statcast data
      2. Manually verify by summing pitches for all relievers in last 3 game dates
      3. Verify the automated value matches the manual calculation
    Expected Result: Automated PC L3 matches manual sum
    Evidence: .sisyphus/evidence/task-12-pitch-count.txt

  Scenario: Early season default handling
    Tool: Bash
    Steps:
      1. Attempt to compute bullpen features for 2025-03-28 (day 2 of season)
      2. Verify returns default/neutral values (not crash)
      3. Verify PC L3 and PC L5 handle < 3 or < 5 days of data
    Expected Result: Graceful defaults for insufficient data
    Evidence: .sisyphus/evidence/task-12-early-season.txt
  ```

  **Commit**: YES (group C3)
  - Message: `feat(features): bullpen fatigue features (PC L3/L5, rest, IR%, availability)`
  - Files: `src/features/bullpen.py`

- [ ] 13. Park Factor + ABS Zone Adjustments

  **What to do**:
  - Create `src/features/adjustments/park_factors.py`:
    - Load park factors from config (2025 MLB data for all 30 stadiums)
    - Sutter Health Park (A's): use 2025 MLB-level factors (Runs ~1.25, HR ~1.30)
    - Tropicana Field (Rays): use historical dome factors
    - `adjust_for_park(metric, park_factor)` → scaled metric
    - Handle new/temporary venues: if park factor unavailable, use league average (1.00)
  - Create `src/features/adjustments/abs_adjustment.py`:
    - League-wide ABS adjustment factors:
      - Walk rate: +4% (midpoint of 3-6% range from research)
      - Strikeout rate: -3% (midpoint of 2-5% range)
      - Catcher framing: ×0.75 retention (already used in Task 11, but standardized here)
    - Venue-specific ABS toggle: check if game is in `abs_exceptions` list
      - If exception venue → use pre-ABS historical adjustments (no ABS effect)
      - If standard venue → apply ABS adjustments
    - Time-decay factor: as season progresses, ABS impact may stabilize; allow for configurable decay

  **Must NOT do**:
  - Do not use MiLB park factors for Sutter Health Park (use 2025 MLB data only)
  - Do not model per-batter ABS impact (league-wide only in v1)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Straightforward adjustment calculations with configuration-driven logic

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 9-12, 14-15)
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 2, 3

  **References**:
  - **Data**: Sutter Health Park 2025: Runs factor ~1.25, HR factor ~1.30 (2nd in MLB)
  - **Research**: ABS walk rate impact: +3-6% (KBO study + Triple-A data); K rate: -2-5%
  - **Data**: ABS exceptions 2026: Mexico City (Apr 25-26), Field of Dreams (Aug 13), Little League Classic (Aug 23)
  - **Config**: All factors configurable in settings.yaml for easy tuning

  **Acceptance Criteria**:
  - [ ] Park factors loaded for all 30 stadiums
  - [ ] ABS exceptions correctly identified and handled
  - [ ] Walk/K rate adjustments applied league-wide
  - [ ] Sutter Health Park uses 2025 MLB factors (not MiLB)
  - [ ] All adjustments configurable via settings.yaml

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: ABS exception venue handling
    Tool: Bash
    Steps:
      1. Check ABS status for a Mexico City Series game (Apr 25-26)
      2. Verify abs_active = False for this venue
      3. Verify no ABS adjustments applied to features for this game
      4. Check a regular venue game on same date → verify abs_active = True
    Expected Result: Exception venues skip ABS adjustments
    Evidence: .sisyphus/evidence/task-13-abs-exceptions.txt
  ```

  **Commit**: YES (group C4)
  - Message: `feat(adjustments): park factors + ABS zone adjustments with exception handling`
  - Files: `src/features/adjustments/park_factors.py`, `src/features/adjustments/abs_adjustment.py`

- [ ] 14. Weather Adjustment Engine

  **What to do**:
  - Create `src/features/adjustments/weather.py`:
    - `compute_weather_features(weather_data, stadium_config)` → dict of weather feature values
    - Features to compute:
      - `temp_factor`: Temperature adjustment (HR distance increases ~1.5ft per 10°F above 70°F)
      - `air_density_factor`: Based on air density calculation from weather client (lower density = more HR)
      - `humidity_factor`: Higher humidity = slightly less HR (contrary to popular belief, humid air is less dense — this is counterintuitive but correct)
      - `wind_factor`: Wind speed × cos(angle between wind direction and CF orientation); positive = blowing out
      - `rain_risk`: Boolean from forecast (if rain probability > 60%, flag for potential delay/shortened game)
    - For domed stadiums: all factors = 1.0 (neutral), rain_risk = False
    - Combine into single `weather_composite` score for model input

  **Must NOT do**:
  - Do not build stadium-specific aerodynamic models
  - Do not factor weather into domed stadiums

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Physics calculations combined with config data

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 9-13, 15)
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 2, 8

  **References**:
  - **Research**: Temperature: +1°F above 70°F ≈ +0.1% HR distance (Alan Nathan research)
  - **Research**: Air density: ball travels ~4% farther in Denver vs sea level (Coors effect)
  - **Research**: Humid air is LESS dense than dry air at same temp/pressure (molecular weight of water < N2/O2)
  - **Formula**: Wind effect: `wind_factor = wind_speed × cos(wind_direction - stadium_cf_azimuth)`

  **Acceptance Criteria**:
  - [ ] All 5 weather features computed for open-air stadiums
  - [ ] Domed stadiums return neutral (1.0) factors
  - [ ] Wind factor correctly uses cosine of angle difference
  - [ ] Air density calculation accounts for humidity

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Wind blowing out vs in
    Tool: Bash
    Steps:
      1. Simulate wind at 15mph directly toward CF (angle diff = 0°)
      2. Verify wind_factor is positive (blowing out)
      3. Simulate wind at 15mph directly FROM CF (angle diff = 180°)
      4. Verify wind_factor is negative (blowing in)
    Expected Result: Correct sign for wind direction relative to stadium
    Evidence: .sisyphus/evidence/task-14-wind-factor.txt
  ```

  **Commit**: YES (group C4)
  - Message: `feat(adjustments): weather engine with temp, density, humidity, wind factors`
  - Files: `src/features/adjustments/weather.py`

- [ ] 15. Pythagorean WP + Log5 Team Strength Baselines

  **What to do**:
  - Create `src/features/baselines.py`:
    - `pythagorean_wp(runs_scored, runs_allowed, exponent=1.83)` → expected win percentage
      - Formula: `RS^exp / (RS^exp + RA^exp)`
      - Compute for both teams using their rolling run totals
    - `log5_matchup(team_a_wp, team_b_wp)` → probability team A wins
      - Formula: `(pA × (1 - pB)) / (pA × (1 - pB) + pB × (1 - pA))`
      - Use team-level Pythagorean WP as inputs (NOT batter-pitcher matchups)
    - Compute **F5-specific Pythagorean**: use only runs scored/allowed in innings 1-5 (not full game)
      - This is a key innovation: F5 run differentials are different from full-game
    - Rolling windows: 30 and 60 games for Pythagorean; Log5 computed fresh each game from latest Pythagorean

  **Must NOT do**:
  - Do not use Log5 for batter-vs-pitcher matchups (biased per SABR research)
  - Do not use exponent = 2 (use 1.83, the Baseball Reference standard)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Mathematical formulas with F5-specific twist

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 9-14)
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 3, 4

  **References**:
  - **Formula**: Pythagorean: `WP = RS^1.83 / (RS^1.83 + RA^1.83)`
  - **Formula**: Log5: `P(A) = (pA(1-pB)) / (pA(1-pB) + pB(1-pA))`
  - **Research**: SABR confirmed Log5 biased for batter-pitcher matchups (overestimates extreme mismatches)
  - **Innovation**: F5-specific Pythagorean uses innings 1-5 run data only

  **Acceptance Criteria**:
  - [ ] Pythagorean WP computed with exponent 1.83
  - [ ] Log5 produces probabilities in [0,1] that sum correctly for both teams
  - [ ] F5-specific Pythagorean uses only innings 1-5 runs (verified against full-game version)
  - [ ] Log5 NOT applied to batter-pitcher level (team-only enforcement)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Pythagorean + Log5 math verification
    Tool: Bash
    Steps:
      1. Compute Pythagorean WP for team with RS=500, RA=450 → verify ≈ 0.549
      2. Compute Pythagorean WP for team with RS=400, RA=450 → verify ≈ 0.447
      3. Apply Log5: team A (0.549) vs team B (0.447) → verify P(A) ≈ 0.601
      4. Verify P(A) + P(B) ≈ 1.000
    Expected Result: Math matches expected values within 0.001 tolerance
    Evidence: .sisyphus/evidence/task-15-pyth-log5-math.txt

  Scenario: F5 vs full-game Pythagorean divergence
    Tool: Bash
    Steps:
      1. Compute full-game Pythagorean for a team using all 9 innings data
      2. Compute F5 Pythagorean for same team using only innings 1-5 data
      3. Verify the values are DIFFERENT (F5 isolates starting pitching quality)
    Expected Result: F5 and full-game Pythagorean values diverge
    Evidence: .sisyphus/evidence/task-15-f5-vs-fullgame.txt
  ```

  **Commit**: YES (group C4)
  - Message: `feat(baselines): Pythagorean WP (1.83) + Log5 with F5-specific run scoring`
  - Files: `src/features/baselines.py`

- [ ] 16. Historical Training Data Builder + Anti-Leakage Enforcement

  **What to do**:
  - Create `src/model/data_builder.py`:
    - `build_training_dataset(start_year=2019, end_year=2025)` → Parquet file with one row per game
    - For each historical game, assemble ALL features (offense, pitching, defense, bullpen, park, weather, baselines) using ONLY data available BEFORE that game
    - Target variables:
      - `f5_ml_result`: 1 if home team leads after 5 innings, 0 if away leads (exclude ties for ML; ties handled as push in settlement)
      - `f5_rl_result`: 1 if home team leads by 2+ after 5 innings, 0 otherwise
    - Anti-leakage enforcement:
      - Every feature's `as_of_timestamp` must be STRICTLY before game start time
      - Add assertion: `assert all(feature.as_of < game.start_time)` during build
      - Store the complete build timestamp and data version hash for reproducibility
    - Handle missing data: if a feature is unavailable for a game (e.g., first week of season), fill with Marcel-blended prior year value or league average (never drop the game row)
    - Output: `data/training/training_data_2019_2025.parquet` with ~17,000 rows

  **Must NOT do**:
  - Do not include current game's stats in features (this IS the leakage)
  - Do not drop games with partial features (fill with Marcel blend / league avg)
  - Do not include spring training or postseason games in training data

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Critical anti-leakage logic, complex data assembly across all feature modules

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (must wait for all Wave 2 features)
  - **Blocks**: Tasks 17, 20, 28
  - **Blocked By**: Tasks 9-15

  **References**:
  - **Pattern**: Lagged rolling averages — subtract current day's stats from cumulative total (user requirement)
  - **Data**: ~2,430 regular season games/year × 7 years = ~17,000 training samples
  - **External**: Walk-forward validation requires this data to be chronologically ordered

  **Acceptance Criteria**:
  - [ ] Training dataset has ~17,000 rows (±500 for missing/excluded games)
  - [ ] Every row has `as_of_timestamp` strictly before game time
  - [ ] Anti-leakage assertion passes on full dataset
  - [ ] No spring training or postseason games included
  - [ ] Missing features filled (not dropped)
  - [ ] Data version hash stored for reproducibility

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Anti-leakage assertion on full dataset
    Tool: Bash (timeout: 300s)
    Steps:
      1. Build training dataset for 2019-2025
      2. Run anti-leakage check: for each row, verify all feature timestamps < game start
      3. Report: total rows, rows passing check, rows failing check
    Expected Result: 0 rows failing anti-leakage check
    Failure Indicators: Any row with feature timestamp >= game start
    Evidence: .sisyphus/evidence/task-16-antileak-full.txt

  Scenario: Data completeness check
    Tool: Bash
    Steps:
      1. Load training dataset, check for NaN values in target columns
      2. Verify f5_ml_result and f5_rl_result have no NaN
      3. Check feature columns — verify NaN rate < 5% (Marcel fill should handle most)
    Expected Result: Zero NaN in targets, < 5% NaN in features
    Evidence: .sisyphus/evidence/task-16-completeness.txt
  ```

  **Commit**: YES (group C5)
  - Message: `feat(data): historical training data builder with anti-leakage enforcement`
  - Files: `src/model/data_builder.py`

- [ ] 17. XGBoost Model Training + Hyperparameter Tuning

  **What to do**:
  - Create `src/model/xgboost_trainer.py`:
    - Train TWO separate XGBoost classifiers:
      - `f5_ml_model`: Predicts F5 Moneyline outcome (home team leads after 5)
      - `f5_rl_model`: Predicts F5 Run Line outcome (home team leads by 2+)
    - Hyperparameter tuning via `sklearn.model_selection.TimeSeriesSplit` (NOT random k-fold — respects temporal order):
      - Search space: `max_depth` [3-8], `n_estimators` [100-500], `learning_rate` [0.01-0.1], `subsample` [0.6-0.9], `colsample_bytree` [0.6-0.9], `reg_alpha` [0-1], `reg_lambda` [0-1]
      - Use `xgboost.XGBClassifier` with `eval_metric='logloss'`
    - Feature importance extraction: SHAP values or XGBoost built-in importance
    - Save trained models to `data/models/` as joblib files with version tag
    - Log training metrics: accuracy, log loss, AUC, and feature importance rankings

  **Must NOT do**:
  - Do not use random k-fold cross-validation (violates temporal order)
  - Do not train neural network models
  - Do not tune on test data (use separate validation fold)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Hyperparameter optimization, temporal cross-validation, model serialization

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (depends on training data)
  - **Blocks**: Tasks 18, 20
  - **Blocked By**: Task 16

  **References**:
  - **External**: XGBoost docs for `XGBClassifier`, `eval_metric`, early stopping
  - **Pattern**: `TimeSeriesSplit(n_splits=5)` for temporal cross-validation
  - **Research**: OSS MLB models report ~57-67% accuracy with XGBoost on similar feature sets

  **Acceptance Criteria**:
  - [ ] Two models trained (F5 ML and F5 RL)
  - [ ] TimeSeriesSplit used (not random k-fold)
  - [ ] Best hyperparameters logged
  - [ ] Feature importance rankings extracted
  - [ ] Models saved as versioned joblib files

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Model training completes with valid metrics
    Tool: Bash (timeout: 600s)
    Steps:
      1. Train F5 ML model on 2019-2024 data
      2. Evaluate on 2025 holdout
      3. Verify accuracy > 50% (better than random), log loss < 0.69 (better than uninformed)
      4. Verify model file saved to data/models/
    Expected Result: Model trained, metrics reasonable, file persisted
    Failure Indicators: Accuracy < 50%, missing model file, training error
    Evidence: .sisyphus/evidence/task-17-xgb-training.txt
  ```

  **Commit**: YES (group C5)
  - Message: `feat(model): XGBoost F5 ML/RL classifiers with temporal cross-validation`
  - Files: `src/model/xgboost_trainer.py`

- [ ] 18. Logistic Regression Stacking Meta-Learner

  **What to do**:
  - Create `src/model/stacking.py`:
    - Implement stacking ensemble: XGBoost predictions become features for LR meta-learner
    - Input features to LR: XGBoost predicted probabilities + select raw features (Pythagorean WP, Log5, park factor)
    - Train `sklearn.linear_model.LogisticRegression` with `C=1.0` (regularization) and `solver='lbfgs'`
    - Use out-of-fold predictions from XGBoost to prevent leakage in stacking
    - Output: calibrated probability from LR (inherently better calibrated than XGBoost)
    - Save stacking model alongside base models

  **Must NOT do**:
  - Do not train LR on in-fold predictions (causes overfitting to XGBoost outputs)
  - Do not add all raw features to LR (only select baselines + XGBoost probs)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Standard stacking pattern, moderate complexity

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (needs XGBoost out-of-fold predictions)
  - **Blocks**: Task 19
  - **Blocked By**: Task 17

  **References**:
  - **Pattern**: Stacking with out-of-fold predictions: use `cross_val_predict(xgb, method='predict_proba')` to generate meta-features without leakage
  - **External**: sklearn `StackingClassifier` or manual implementation

  **Acceptance Criteria**:
  - [ ] LR meta-learner trained on out-of-fold XGBoost predictions
  - [ ] Stacking uses select baselines + XGBoost probs as features
  - [ ] LR produces probabilities in [0,1]
  - [ ] Stacking model saved

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Stacking improves or maintains calibration
    Tool: Bash
    Steps:
      1. Compare Brier score of raw XGBoost vs stacked XGBoost+LR on 2025 holdout
      2. Verify stacked model Brier score <= XGBoost Brier score (not worse)
    Expected Result: Stacking maintains or improves calibration
    Evidence: .sisyphus/evidence/task-18-stacking-brier.txt
  ```

  **Commit**: YES (group C5)
  - Message: `feat(model): LR stacking meta-learner on XGBoost out-of-fold predictions`
  - Files: `src/model/stacking.py`

- [ ] 19. Isotonic Calibration Layer

  **What to do**:
  - Create `src/model/calibration.py`:
    - `calibrate_model(model, X_cal, y_cal)` → fitted `IsotonicRegression` calibrator
    - Use a dedicated calibration set (10% of training data, chronologically last portion)
    - Apply `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')` to map stacked model outputs to perfectly calibrated probabilities
    - Evaluate calibration quality:
      - Brier score (primary): target < 0.25
      - Reliability diagram: predicted vs actual probabilities in 10 bins
      - Expected calibration error (ECE): target < 0.05
    - `predict_calibrated(model, calibrator, X)` → calibrated probability in [0,1]
    - Save calibrator alongside models

  **Must NOT do**:
  - Do not use Platt scaling (isotonic is better for this domain per research)
  - Do not calibrate on training data (use separate calibration holdout)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Standard calibration pipeline with quality metrics

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (needs stacking output)
  - **Blocks**: Tasks 20, 21
  - **Blocked By**: Task 18

  **References**:
  - **External**: sklearn `IsotonicRegression` docs
  - **Formula**: Brier score = `mean((predicted - actual)^2)`, lower is better, 0.25 = random baseline
  - **Formula**: ECE = `Σ(|bin_accuracy - bin_confidence| × bin_count) / total`
  - **Research**: Isotonic > Platt for sports betting calibration (consensus from research agents)

  **Acceptance Criteria**:
  - [ ] Isotonic calibration fitted on dedicated holdout set
  - [ ] Brier score < 0.25 on test data
  - [ ] Reliability diagram shows calibration curve near diagonal
  - [ ] ECE < 0.05
  - [ ] Calibrator saved

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Calibration quality gates pass
    Tool: Bash
    Steps:
      1. Train full pipeline (XGBoost → LR → Isotonic) on 2019-2024
      2. Evaluate on 2025 holdout: compute Brier score and ECE
      3. Verify Brier < 0.25 and ECE < 0.05
      4. Generate reliability diagram data (10 bins of predicted prob vs actual freq)
    Expected Result: Calibration metrics within thresholds
    Failure Indicators: Brier >= 0.25, ECE >= 0.05, S-curve in reliability diagram
    Evidence: .sisyphus/evidence/task-19-calibration-quality.txt
  ```

  **Commit**: YES (group C5)
  - Message: `feat(model): isotonic calibration with Brier score and reliability evaluation`
  - Files: `src/model/calibration.py`

- [ ] 20. Walk-Forward Backtesting Framework

  **What to do**:
  - Create `src/backtest/walk_forward.py`:
    - `run_backtest(start_date, end_date, train_window_months=6, test_window_months=1)` → BacktestResult
    - Walk-forward process:
      1. Train on first 6 months of data
      2. Predict next 1 month
      3. Slide forward 1 month, retrain, predict next month
      4. Repeat until end_date
    - For each test window, record:
      - Predictions vs actuals
      - Brier score per window
      - Simulated bets (using edge calculator + Quarter Kelly)
      - Cumulative ROI and bankroll evolution
    - Anti-leakage: recompute features at each step using only data available at that point
    - **Reproducibility**: pin all random seeds, cache raw data pulls, store data version hashes
    - Output: summary CSV with metrics per window + aggregate, bankroll evolution chart data
  - Entry point: `python -m src.backtest.run --start 2019-03-20 --end 2025-10-31`

  **Must NOT do**:
  - Do not use look-ahead features during backtesting
  - Do not use full dataset for feature normalization (must be train-window only)
  - Do not report in-sample metrics as performance indicators

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Complex temporal pipeline with anti-leakage at every step

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (needs complete ML pipeline)
  - **Blocks**: —
  - **Blocked By**: Tasks 16-19

  **References**:
  - **Pattern**: Walk-forward validation: train on [T-6mo, T], test on [T, T+1mo], slide 1 month
  - **Research**: Monthly retraining captures seasonal shifts in MLB (cold April → hot July → September callups)
  - **External**: Backtesting frameworks from WalrusQuant NFL study as reference pattern

  **Acceptance Criteria**:
  - [ ] Walk-forward produces results for each 1-month window
  - [ ] Aggregate Brier score < 0.25
  - [ ] Simulated ROI > 0% on 3%+ edge bets
  - [ ] Reproducible: same inputs → same outputs (pinned seeds, cached data)
  - [ ] No look-ahead bias in any test window

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Full walk-forward backtest execution
    Tool: Bash (timeout: 900s)
    Steps:
      1. Run `python -m src.backtest.run --start 2022-04-01 --end 2023-09-30` (shorter range for speed)
      2. Verify output CSV created with columns: window, brier_score, accuracy, roi, bet_count
      3. Verify aggregate Brier < 0.25
      4. Verify no test window has metrics from a future window's data
    Expected Result: Backtest completes, metrics within thresholds, anti-leakage verified
    Evidence: .sisyphus/evidence/task-20-walkforward-backtest.txt

  Scenario: Reproducibility check
    Tool: Bash (timeout: 900s)
    Steps:
      1. Run backtest twice with identical parameters
      2. Compare output CSVs — they must be byte-identical
    Expected Result: Two runs produce identical results
    Evidence: .sisyphus/evidence/task-20-reproducibility.txt
  ```

  **Commit**: YES (group C6)
  - Message: `feat(backtest): walk-forward framework with anti-leakage, reproducibility, ROI tracking`
  - Files: `src/backtest/walk_forward.py`, `src/backtest/__main__.py`

- [ ] 21. Edge Calculator: De-Vig, Implied Probability, Expected Value

  **What to do**:
  - Create `src/engine/edge_calculator.py`:
    - `calculate_edge(model_prob, odds_snapshot)` → EdgeResult (edge_pct, ev, is_positive_ev)
    - Pipeline: raw American odds → implied probability → proportional de-vig → fair probability → edge
    - `edge = model_prob - fair_prob`; only recommend bet when edge ≥ 0.03 (3% threshold)
    - `expected_value = (model_prob × potential_profit) - ((1 - model_prob) × stake)`
    - Handle edge cases in odds conversion:
      - Even odds (100 / -100): implied = 0.50
      - Heavy favorites (-300+): implied > 0.75; need higher model confidence for edge
      - Heavy underdogs (+300+): implied < 0.25; these are where biggest edges hide
    - Calculate for both F5 ML and F5 RL markets per game
    - Log all edge calculations to SQLite for performance tracking

  **Must NOT do**:
  - Do not use raw implied probability (must de-vig first)
  - Do not recommend bets below 3% edge threshold

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Critical financial calculations, edge case handling, vig removal math

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 22-26)
  - **Blocks**: Tasks 22, 24, 26, 29
  - **Blocked By**: Tasks 5, 19

  **References**:
  - **Formula**: American to implied: negative → `|odds|/(|odds|+100)`, positive → `100/(odds+100)`
  - **Formula**: De-vig (proportional): `fair_p = implied_p / (sum_implied_p)`
  - **Formula**: EV = `(model_p × profit) - ((1-model_p) × stake)`, where profit from American odds: positive → `odds/100 × stake`, negative → `100/|odds| × stake`
  - **Config**: edge_min = 0.03 from settings.yaml

  **Acceptance Criteria**:
  - [ ] Odds conversion handles all American format edge cases
  - [ ] De-vig produces fair probabilities summing to 1.0
  - [ ] Edge correctly calculated as model_prob - fair_prob
  - [ ] Only bets with edge ≥ 3% recommended
  - [ ] EV calculation accounts for American odds payout structure
  - [ ] All calculations logged to SQLite

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Edge calculation accuracy
    Tool: Bash
    Steps:
      1. Model prob = 0.58, odds = -130 (implied ≈ 0.565, fair after devig ≈ 0.545)
      2. Verify edge ≈ 0.58 - 0.545 = 0.035 (3.5% → should recommend bet)
      3. Model prob = 0.55, odds = -150 (implied ≈ 0.600, fair ≈ 0.575)
      4. Verify edge ≈ 0.55 - 0.575 = -0.025 (negative → should NOT recommend bet)
    Expected Result: Correct edge sign and magnitude, correct bet/no-bet decision
    Evidence: .sisyphus/evidence/task-21-edge-calc.txt

  Scenario: Even odds edge case
    Tool: Bash
    Steps:
      1. Test odds = +100 (even money): verify implied = 0.50
      2. Test odds = -100 (also even money): verify implied = 0.50
      3. Both should de-vig to 0.50 each
    Expected Result: Even odds handled symmetrically
    Evidence: .sisyphus/evidence/task-21-even-odds.txt
  ```

  **Commit**: YES (group C7)
  - Message: `feat(engine): edge calculator with de-vig, EV computation, 3% threshold`
  - Files: `src/engine/edge_calculator.py`

- [ ] 22. Quarter Kelly Bankroll Manager

  **What to do**:
  - Create `src/engine/bankroll.py`:
    - `calculate_kelly_stake(edge, odds, bankroll, fraction=0.25)` → stake amount
    - Full Kelly: `f = (b×p - q) / b` where b=decimal odds-1, p=model_prob, q=1-p
    - Quarter Kelly: `stake = bankroll × f × 0.25`
    - Constraints:
      - Maximum single bet: 5% of current bankroll (hard cap even if Kelly suggests more)
      - Minimum bet: $0 (never negative)
      - No same-team ML + RL stacking (if both F5 ML and F5 RL on same game have edge, treat as single bet with higher Kelly for the better edge)
    - `update_bankroll(bet_result)` → updates bankroll_ledger in SQLite
    - **30% max drawdown kill-switch**: if bankroll drops 30% below peak, stop all betting, send Discord alert
    - Track: current bankroll, peak bankroll, current drawdown %, total bets, win rate, ROI

  **Must NOT do**:
  - Do not implement full Kelly (too aggressive)
  - Do not allow bets exceeding 5% of bankroll
  - Do not continue betting after 30% drawdown kill-switch triggers

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Financial calculations with safety constraints

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 21, 23-26)
  - **Blocks**: Tasks 24, 26, 29
  - **Blocked By**: Tasks 3, 21

  **References**:
  - **Formula**: Full Kelly: `f = (b×p - q) / b`; Quarter Kelly: `f × 0.25`
  - **External**: keeks library `FractionalKellyCriterion(fraction=0.25)` as reference
  - **Research**: Quarter Kelly provides 75% of growth with 25% of volatility

  **Acceptance Criteria**:
  - [ ] Quarter Kelly stake calculated correctly
  - [ ] 5% bankroll hard cap enforced
  - [ ] Same-team correlation handled (single bet for both ML+RL)
  - [ ] Kill-switch triggers at 30% drawdown
  - [ ] Bankroll ledger updated in SQLite with each transaction

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Kelly stake calculation and caps
    Tool: Bash
    Steps:
      1. Bankroll = $1000, edge = 5%, odds = -110
      2. Calculate Quarter Kelly → verify stake is reasonable ($10-25 range)
      3. Verify stake does not exceed $50 (5% of $1000)
      4. Test with extreme edge (20%) → verify cap at 5% of bankroll
    Expected Result: Kelly sized appropriately, hard cap enforced
    Evidence: .sisyphus/evidence/task-22-kelly-stake.txt

  Scenario: Drawdown kill-switch
    Tool: Bash
    Steps:
      1. Set initial bankroll = $1000 (peak)
      2. Simulate losses bringing bankroll to $690 (31% drawdown)
      3. Verify kill-switch triggers (returns stake=0 for all subsequent bets)
      4. Verify Discord alert would be triggered
    Expected Result: Kill-switch activates at 30% drawdown
    Evidence: .sisyphus/evidence/task-22-killswitch.txt
  ```

  **Commit**: YES (group C7)
  - Message: `feat(engine): Quarter Kelly bankroll manager with drawdown kill-switch`
  - Files: `src/engine/bankroll.py`

- [ ] 23. F5 Settlement Rules Module

  **What to do**:
  - Create `src/engine/settlement.py`:
    - `settle_bet(bet, game_result)` → BetResult (WIN/LOSS/PUSH/NO_ACTION)
    - **F5 ML settlement**:
      - Home team leads after 5 → home bet WINS, away bet LOSES
      - Away team leads after 5 → away bet WINS, home bet LOSES
      - Tied after 5 → PUSH (refund)
    - **F5 RL settlement** (-1.5 / +1.5):
      - Home leads by 2+ → home -1.5 WINS, away +1.5 LOSES
      - Home leads by exactly 1 → home -1.5 LOSES, away +1.5 WINS
      - Tied after 5 → away +1.5 WINS, home -1.5 LOSES
      - Away leads → away +1.5 WINS, home -1.5 LOSES
    - **No Action triggers**:
      - Game does not reach 5 full innings (rain, suspended): NO_ACTION
      - Listed starting pitcher scratched before first pitch: NO_ACTION
      - Game played at ABS exception venue when model assumed ABS: flag for review
    - **Doubleheader handling**: Use `game_pk` to disambiguate Game 1 vs Game 2
    - Update bets table and bankroll_ledger in SQLite upon settlement

  **Must NOT do**:
  - Do not settle partial games as WIN/LOSS (must be NO_ACTION)
  - Do not auto-settle before game completion (require final status confirmation)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Critical business logic with extensive edge cases

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 21, 22, 24-26)
  - **Blocks**: Tasks 24, 29
  - **Blocked By**: Tasks 3, 4

  **References**:
  - **Rules**: Standard F5 ML settlement: tie = push (most US sportsbooks)
  - **Rules**: F5 RL -1.5: requires 2+ run lead after 5 complete innings
  - **Edge case**: Game suspended and completed on different day — settle based on score when game reaches 5 innings (may be on resumption day)

  **Acceptance Criteria**:
  - [ ] F5 ML: home win / away win / tie=push all handled correctly
  - [ ] F5 RL: 2+ run lead / 1 run lead / tie / deficit all handled correctly
  - [ ] No Action for: game < 5 innings, starter scratched
  - [ ] Doubleheaders disambiguated by game_pk
  - [ ] Bankroll updated correctly for each settlement type

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Exhaustive F5 settlement logic
    Tool: Bash
    Steps:
      1. Test F5 ML: home leads 3-1 after 5 → verify home WIN, away LOSS
      2. Test F5 ML: tied 2-2 after 5 → verify PUSH
      3. Test F5 RL -1.5: home leads 4-2 (2+ run lead) → verify home -1.5 WIN
      4. Test F5 RL -1.5: home leads 3-2 (1 run lead) → verify home -1.5 LOSS
      5. Test game < 5 innings → verify NO_ACTION
      6. Test starter scratched → verify NO_ACTION
    Expected Result: All 6 scenarios produce correct settlement
    Failure Indicators: Wrong result for any scenario
    Evidence: .sisyphus/evidence/task-23-settlement-logic.txt
  ```

  **Commit**: YES (group C7)
  - Message: `feat(engine): F5 settlement rules (ML tie=push, RL, no-action triggers)`
  - Files: `src/engine/settlement.py`

- [ ] 24. Daily Pipeline Orchestrator

  **What to do**:
  - Create `src/pipeline/daily.py`:
    - Main entry point: `python -m src.pipeline.daily --date YYYY-MM-DD --mode [prod|backtest|dry-run]`
    - Pipeline steps (sequential):
      1. **Ingest**: Fetch today's schedule (MLB API), confirmed lineups, odds, weather
      2. **Validate**: Check data completeness; for any game missing starter/odds/lineup → mark "NO PICK (reason)"
      3. **Feature**: Compute all features for each game using latest available data
      4. **Predict**: Run stacked model → calibrated probabilities for F5 ML and F5 RL
      5. **Edge**: Compare model probabilities vs de-vigged market odds → filter by 3% threshold
      6. **Size**: Apply Quarter Kelly to each qualifying bet
      7. **Store**: Save all predictions, odds snapshots, and bet decisions to SQLite
      8. **Notify**: Send Discord webhook with pick cards (or dry-run to console)
      9. **Freeze**: Mark odds as frozen for notified picks
    - Error handling: if any step fails, log error, send Discord failure alert, continue with remaining games
    - `--dry-run` mode: executes full pipeline but prints Discord payload to console instead of sending
    - Logging: comprehensive logging with timestamps for debugging

  **Must NOT do**:
  - Do not silently skip games with missing data (explicit NO PICK)
  - Do not send duplicate Discord notifications (check if already notified for this date)
  - Do not place actual bets (advisory only)

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []
  - Reason: Complex orchestration tying all modules together, error handling, mode switching

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (depends on most Wave 4 tasks)
  - **Blocks**: Tasks 27, 28, 30, 31
  - **Blocked By**: Tasks 6-8, 9-15, 17-19, 21-23, 25

  **References**:
  - **Pattern**: Pipeline pattern with each step producing artifacts stored in SQLite
  - **Pattern**: `--dry-run` flag for testing without side effects

  **Acceptance Criteria**:
  - [ ] Pipeline runs end-to-end for a given date
  - [ ] Missing data produces explicit "NO PICK" rows (not silent skip)
  - [ ] Dry-run mode outputs valid JSON without sending to Discord
  - [ ] Errors in one game don't crash entire pipeline
  - [ ] All predictions and decisions stored in SQLite

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Full pipeline dry-run
    Tool: Bash (timeout: 120s)
    Steps:
      1. Run `python -m src.pipeline.daily --date 2025-09-15 --mode backtest --dry-run`
      2. Verify exit code 0
      3. Verify console output contains JSON payload with pick cards
      4. Verify SQLite has predictions for that date
    Expected Result: Pipeline completes, valid output, data persisted
    Evidence: .sisyphus/evidence/task-24-pipeline-dryrun.txt

  Scenario: Missing odds graceful degradation
    Tool: Bash
    Steps:
      1. Run pipeline for a date where odds data is deliberately missing for one game
      2. Verify that game produces "NO PICK (odds unavailable)" 
      3. Verify remaining games still get predictions
    Expected Result: Partial success — some picks, explicit no-pick for missing data
    Evidence: .sisyphus/evidence/task-24-missing-odds.txt
  ```

  **Commit**: YES (group C8)
  - Message: `feat(pipeline): daily orchestrator with ingestion, prediction, edge, Discord notification`
  - Files: `src/pipeline/daily.py`, `src/pipeline/__main__.py`

- [ ] 25. Early-Season Marcel Blend Module

  **What to do**:
  - Create `src/features/marcel_blend.py`:
    - Centralized Marcel blending logic (used by offense and pitching features):
    - `marcel_blend(current_season_value, prior_season_value, games_played, regression_weight)` → blended value
    - Formula: `blended = (current × games_played + prior × regression_weight) / (games_played + regression_weight)`
    - Default regression weights:
      - Offensive stats (wRC+, wOBA): 30 games
      - Pitching stats (xFIP, xERA): 15 starts
      - Defense (DRS, OAA): 50 games (slower-moving metrics)
    - Progressive blend: as season progresses, current-year weight naturally increases
    - Handle first year in league: if no prior season, use league average as prior
    - Handle team roster changes: if >40% roster turnover, reduce prior-year weight by 50%

  **Must NOT do**:
  - Do not use projection systems (ZiPS, Steamer) — build own Marcel from data

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Statistical regression logic, roster change detection

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 21-24, 26)
  - **Blocks**: Task 24 (integrated into feature pipeline)
  - **Blocked By**: Tasks 9, 10

  **References**:
  - **Formula**: Marcel: `blended = (curr × GP + prior × R) / (GP + R)` where R = regression weight
  - **Research**: Marcel method named after Tom Tango's "Marcel the Monkey" — simplest reliable projection

  **Acceptance Criteria**:
  - [ ] Marcel blend produces values between prior and current season
  - [ ] Regression weights configurable per metric type
  - [ ] First-year players use league average as prior
  - [ ] Blend approaches current-year value as games_played increases

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Marcel blend convergence
    Tool: Bash
    Steps:
      1. Prior year wRC+ = 100, current = 120, after 10 games
      2. Verify blend ≈ (120×10 + 100×30)/(10+30) = 105
      3. Same values after 60 games: verify blend ≈ (120×60 + 100×30)/(60+30) = 113.3
      4. After 162 games: verify blend ≈ (120×162 + 100×30)/(162+30) ≈ 116.9 (mostly current year)
    Expected Result: Blend shifts from prior-weighted to current-weighted over season
    Evidence: .sisyphus/evidence/task-25-marcel-convergence.txt
  ```

  **Commit**: YES (group C7)
  - Message: `feat(features): Marcel blend module for early-season regression toward prior year`
  - Files: `src/features/marcel_blend.py`

- [ ] 26. Discord Webhook Formatter + Notifier

  **What to do**:
  - Create `src/notifications/discord.py`:
    - `send_picks(picks, bankroll_summary)` → sends formatted embed to Discord webhook
    - Pick card format (Discord embed):
      ```
      🔥 MLB F5 PICKS — April 15, 2026
      ─────────────────────
      ⚾ NYY vs BOS (7:05 PM ET)
      📊 F5 ML: NYY -130 → Model: 58.2% | Edge: 3.5% | Kelly: $25
      📊 F5 RL: NYY -1.5 +145 → Model: 38.5% | Edge: 4.1% | Kelly: $18
      🏟️ Yankee Stadium | 72°F | Wind: 8mph Out
      ─────────────────────
      💰 Bankroll: $1,150 | ROI: +15.0% | W-L: 45-38
      ```
    - Include: game matchup, time, market, odds, model prob, edge %, Kelly stake, venue, weather
    - `send_no_picks(reason)` → sends "No qualifying picks today" message
    - `send_failure_alert(error_message)` → sends pipeline failure notification (red embed)
    - `send_drawdown_alert(drawdown_pct)` → sends kill-switch activation (red embed)
    - Support `--dry-run`: print payload to console without sending
    - Rate limit: max 1 webhook call per pipeline run (batch all picks into single message)

  **Must NOT do**:
  - Do not include automated bet placement links
  - Do not send more than 1 message per pipeline run

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Discord API integration, embed formatting, multiple notification types

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 21-25)
  - **Blocks**: Tasks 27, 31
  - **Blocked By**: Tasks 21, 22

  **References**:
  - **External**: Discord webhook API: POST `https://discord.com/api/webhooks/{id}/{token}` with JSON embed payload
  - **Pattern**: Discord embeds support: title, description, color, fields, footer, timestamp

  **Acceptance Criteria**:
  - [ ] Pick cards sent with all required info (odds, model prob, edge, Kelly, venue, weather)
  - [ ] No-picks message sent on days with no qualifying bets
  - [ ] Failure alerts sent on pipeline errors
  - [ ] Drawdown alert sent when kill-switch activates
  - [ ] Dry-run prints valid JSON without sending

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Discord payload structure validation
    Tool: Bash
    Steps:
      1. Run pipeline in dry-run mode
      2. Capture JSON payload from console output
      3. Validate: has "embeds" array, each embed has "fields" with game info
      4. Verify bankroll summary in footer
    Expected Result: Valid Discord webhook JSON with all required fields
    Evidence: .sisyphus/evidence/task-26-discord-payload.txt
  ```

  **Commit**: YES (group C8)
  - Message: `feat(discord): webhook formatter with pick cards, alerts, dry-run support`
  - Files: `src/notifications/discord.py`

- [ ] 27. Windows Task Scheduler + .env Setup

  **What to do**:
  - Create `scripts/setup_scheduler.py`:
    - Creates Windows Task Scheduler task to run pipeline daily
    - Schedule: 3 hours before typical first pitch (e.g., 10:00 AM ET for day games, 4:00 PM ET for evening slates)
    - Task action: `python -m src.pipeline.daily --date today --mode prod`
    - Working directory: project root
    - Trigger: daily at configured time
    - Settings: run whether user is logged in or not, wake computer if asleep
  - Create `scripts/run_daily.bat` batch file as Task Scheduler target
  - Create `scripts/setup_env.py`:
    - Interactive script to populate `.env` from `.env.example`
    - Prompts user for each API key
    - Validates API keys by making test requests (Odds API, OpenWeatherMap, Discord webhook)
    - Creates `data/` directory structure if not exists

  **Must NOT do**:
  - Do not store credentials in Task Scheduler (read from .env at runtime)
  - Do not schedule more than once per day

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: System admin scripting, straightforward

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 28-31)
  - **Blocks**: —
  - **Blocked By**: Tasks 24, 26

  **References**:
  - **External**: Windows Task Scheduler via `schtasks` command or `win32com.client` Python API
  - **Pattern**: `schtasks /create /tn "MLB_Daily" /tr "cmd /c run_daily.bat" /sc daily /st 16:00`

  **Acceptance Criteria**:
  - [ ] Task Scheduler task created successfully
  - [ ] Batch file runs pipeline with correct working directory
  - [ ] .env setup script validates API keys with test requests
  - [ ] Pipeline runs unattended (no user interaction required)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Scheduler setup and batch execution
    Tool: Bash
    Steps:
      1. Run `python scripts/setup_scheduler.py --test` (creates task in test mode)
      2. Run `scripts\run_daily.bat --dry-run` manually
      3. Verify pipeline executes and produces output
    Expected Result: Batch file triggers pipeline successfully
    Evidence: .sisyphus/evidence/task-27-scheduler-setup.txt
  ```

  **Commit**: YES (group C9)
  - Message: `feat(ops): Windows Task Scheduler setup, batch runner, env configuration`
  - Files: `scripts/setup_scheduler.py`, `scripts/run_daily.bat`, `scripts/setup_env.py`

- [ ] 28. pytest Suite: Data Integrity + Anti-Leakage Tests

  **What to do**:
  - Create `tests/test_data_integrity.py`:
    - Test: training data has no NaN in target columns
    - Test: all feature `as_of_timestamp` values are before game start time
    - Test: no spring training or postseason games in training set
    - Test: game_pk is unique (no duplicates)
    - Test: features table schema matches expected columns
  - Create `tests/test_antileak.py`:
    - Test: for 100 random games, verify no feature uses data from game day or later
    - Test: rolling window calculations exclude current game
    - Test: Marcel blend uses only prior-year final values (not in-season updates from current year's future)
  - Create `tests/test_feature_engineering.py`:
    - Test: wRC+ values are reasonable (50-200 range for most teams)
    - Test: xFIP values are reasonable (2.0-6.0 range for most pitchers)
    - Test: park factors applied correctly (Sutter Health Park > Tropicana)
    - Test: ABS exception venues have no ABS adjustment applied

  **Must NOT do**:
  - Do not require live API access for tests (use fixtures/mocks)
  - Do not test on production database

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Test design across multiple modules

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 27, 29-31)
  - **Blocks**: —
  - **Blocked By**: Tasks 16, 24

  **References**:
  - **Pattern**: pytest fixtures for test database and mock data
  - **External**: pytest-cov for coverage reporting

  **Acceptance Criteria**:
  - [ ] All data integrity tests pass
  - [ ] Anti-leakage tests verify 100 random games
  - [ ] Feature engineering tests verify reasonable value ranges
  - [ ] Tests use fixtures (not live APIs)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Test suite execution
    Tool: Bash
    Steps:
      1. Run `pytest tests/test_data_integrity.py tests/test_antileak.py tests/test_feature_engineering.py -v`
      2. Verify 0 failures
      3. Run `pytest --cov=src tests/ --cov-report=term-missing`
      4. Verify coverage on critical modules > 70%
    Expected Result: All tests pass, coverage meets threshold
    Evidence: .sisyphus/evidence/task-28-data-tests.txt
  ```

  **Commit**: YES (group C10)
  - Message: `test(integrity): data integrity, anti-leakage, and feature engineering tests`
  - Files: `tests/test_data_integrity.py`, `tests/test_antileak.py`, `tests/test_feature_engineering.py`

- [ ] 29. pytest Suite: Settlement + Edge Calc + Kelly Tests

  **What to do**:
  - Create `tests/test_settlement.py`:
    - Test all F5 ML scenarios: home win, away win, tie=push
    - Test all F5 RL scenarios: 2+ run lead, 1 run lead, tie, deficit
    - Test no-action: game < 5 innings, starter scratched
    - Test doubleheader disambiguation by game_pk
  - Create `tests/test_edge_calculator.py`:
    - Test American odds conversion (positive, negative, even)
    - Test de-vig: probabilities sum to 1.0 after de-vig
    - Test edge calculation with known inputs → expected outputs
    - Test 3% threshold filtering
    - Edge cases: -100, +100, -300, +300 odds
  - Create `tests/test_bankroll.py`:
    - Test Quarter Kelly stake calculation
    - Test 5% bankroll hard cap
    - Test 30% drawdown kill-switch
    - Test same-team correlation handling (ML + RL same game)
    - Test bankroll ledger updates

  **Must NOT do**:
  - Do not use approximate assertions for financial calculations (use exact or Decimal)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Critical financial logic testing, exhaustive edge cases

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 27, 28, 30, 31)
  - **Blocks**: —
  - **Blocked By**: Tasks 21-23

  **References**:
  - **Pattern**: pytest parametrize for exhaustive settlement scenarios
  - **External**: `pytest.approx()` for floating-point comparison with tolerance

  **Acceptance Criteria**:
  - [ ] All settlement scenarios covered and passing
  - [ ] Edge calculation tests with exact expected values
  - [ ] Kelly + bankroll safety tests passing
  - [ ] 0 test failures

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Settlement + financial test execution
    Tool: Bash
    Steps:
      1. Run `pytest tests/test_settlement.py tests/test_edge_calculator.py tests/test_bankroll.py -v`
      2. Verify 0 failures
      3. Count total tests — should be > 20 (comprehensive coverage)
    Expected Result: All financial logic tests pass
    Evidence: .sisyphus/evidence/task-29-financial-tests.txt
  ```

  **Commit**: YES (group C10)
  - Message: `test(engine): settlement, edge calculator, and bankroll safety tests`
  - Files: `tests/test_settlement.py`, `tests/test_edge_calculator.py`, `tests/test_bankroll.py`

- [ ] 30. Performance Tracker + CLV Logging

  **What to do**:
  - Create `src/ops/performance_tracker.py`:
    - Track per-bet: model prob, market prob, edge, result, profit/loss
    - Track aggregate: total bets, win rate, ROI, Brier score, log loss
    - **Closing Line Value (CLV)**: compare model's odds at bet time vs closing line odds
      - Fetch closing odds from The Odds API near game start
      - `clv = closing_implied - opening_implied` (positive = model caught a line move early)
      - CLV is the gold standard for model quality — even more than ROI
    - Generate daily/weekly/monthly performance reports
    - Store all tracking data in SQLite
    - `export_report(period)` → CSV with performance metrics

  **Must NOT do**:
  - Do not build a web dashboard (CSV reports for v1)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Performance analytics, CLV calculation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 27-29, 31)
  - **Blocks**: —
  - **Blocked By**: Tasks 3, 5, 24

  **References**:
  - **Formula**: CLV = `closing_implied_prob - bet_time_implied_prob`; positive CLV = beating the market
  - **Research**: Winning sports bettors consistently show positive CLV even during losing streaks

  **Acceptance Criteria**:
  - [ ] CLV computed by comparing bet-time vs closing odds
  - [ ] Performance metrics stored in SQLite per bet
  - [ ] Aggregate reports generated (ROI, Brier, win rate, CLV)
  - [ ] CSV export works

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: CLV computation
    Tool: Bash
    Steps:
      1. Bet placed at odds -130 (implied 0.565), closing line -140 (implied 0.583)
      2. Verify CLV = 0.583 - 0.565 = +0.018 (1.8% positive CLV — good)
      3. Verify CLV stored in SQLite
    Expected Result: Positive CLV correctly computed and stored
    Evidence: .sisyphus/evidence/task-30-clv-tracking.txt
  ```

  **Commit**: YES (group C9)
  - Message: `feat(ops): performance tracker with CLV logging, aggregate metrics, CSV export`
  - Files: `src/ops/performance_tracker.py`

- [ ] 31. Error Handling, Retry Logic, Failure Alerts

  **What to do**:
  - Create `src/ops/error_handler.py`:
    - Retry decorator with exponential backoff for API calls (max 3 retries, 2/4/8 second delays)
    - Circuit breaker: if an API fails 5 times in a row, mark it as unavailable for this run
    - Graceful degradation: if weather API fails → use neutral defaults; if odds API fails → no picks for affected games
    - Error classification: FATAL (pipeline can't continue), WARNING (partial data), INFO (expected condition)
  - Create `src/ops/logging_config.py`:
    - Configure Python logging: file handler (data/logs/pipeline.log) + console handler
    - Log rotation: daily, keep 30 days
    - Log format: `[timestamp] [level] [module] message`
  - Integrate with Discord notifications:
    - FATAL errors → immediate Discord red alert
    - WARNING → included in daily pick notification footer
    - All errors logged to file regardless

  **Must NOT do**:
  - Do not silently swallow errors
  - Do not retry indefinitely (max 3 attempts)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Production reliability patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Tasks 27-30)
  - **Blocks**: —
  - **Blocked By**: Tasks 24, 26

  **References**:
  - **Pattern**: Exponential backoff: `sleep(2^attempt)` seconds
  - **Pattern**: Circuit breaker pattern for API clients
  - **External**: Python logging with RotatingFileHandler

  **Acceptance Criteria**:
  - [ ] Retry with exponential backoff works (3 attempts max)
  - [ ] Circuit breaker activates after 5 consecutive failures
  - [ ] FATAL errors trigger Discord alert
  - [ ] Log files rotate daily with 30-day retention
  - [ ] Pipeline continues on WARNING (graceful degradation)

  **QA Scenarios (MANDATORY)**:
  ```
  Scenario: Retry and circuit breaker
    Tool: Bash
    Steps:
      1. Mock weather API to fail 3 times then succeed
      2. Verify retry decorator attempts 3 times with exponential backoff
      3. Mock API to fail 5 times consecutively
      4. Verify circuit breaker activates (no further attempts)
    Expected Result: Retries work, circuit breaker prevents infinite loops
    Evidence: .sisyphus/evidence/task-31-retry-circuitbreaker.txt
  ```

  **Commit**: YES (group C9)
  - Message: `feat(ops): error handling with retry, circuit breaker, logging, failure alerts`
  - Files: `src/ops/error_handler.py`, `src/ops/logging_config.py`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns (`ERA` as feature, hard-coded API keys, batter-pitcher Log5). Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `python -m py_compile src/**/*.py` + `pytest --tb=short`. Review all files for: `# type: ignore`, bare `except:`, `print()` in production code, unused imports, missing docstrings on public functions. Check for AI slop: excessive comments, over-abstraction, generic variable names.
  Output: `Build [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Run `python -m src.pipeline.daily --date 2025-09-15 --mode backtest --dry-run`. Verify: data ingestion completes, features computed, predictions generated, edge calculations correct, Discord payload valid JSON, bankroll ledger updated. Test failure paths: remove odds data → verify "NO PICK" output. Test early-season: run with 2025-04-05 → verify Marcel blend activates.
  Output: `Pipeline [PASS/FAIL] | Features [N computed] | Predictions [N generated] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual code. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Detect full-game market code, ERA features, neural network imports. Flag unaccounted files.
  Output: `Tasks [N/N compliant] | Scope [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Wave | Commit | Message | Files |
|------|--------|---------|-------|
| 1 | C1 | `feat(scaffold): project structure, config, SQLite schema, data models` | pyproject.toml, src/__init__.py, config/, src/models/ |
| 1 | C2 | `feat(clients): Odds API, pybaseball ingestion, MLB lineup, weather clients` | src/clients/ |
| 2 | C3 | `feat(features): offensive, pitching, defense, bullpen rolling features` | src/features/ |
| 2 | C4 | `feat(adjustments): park factors, ABS zone, weather, Pythagorean/Log5` | src/features/adjustments/ |
| 3 | C5 | `feat(model): XGBoost training, LR stacking, isotonic calibration` | src/model/ |
| 3 | C6 | `feat(backtest): walk-forward framework with anti-leakage validation` | src/backtest/ |
| 4 | C7 | `feat(engine): edge calculator, Kelly bankroll, settlement, Marcel blend` | src/engine/ |
| 4 | C8 | `feat(pipeline): daily orchestrator + Discord webhook` | src/pipeline/, src/notifications/ |
| 5 | C9 | `feat(ops): scheduler, error handling, retry, CLV tracking` | src/ops/, tasks/ |
| 5 | C10 | `test(suite): data integrity, settlement, edge, calibration tests` | tests/ |

---

## Success Criteria

### Verification Commands
```bash
# Install and verify
pip install -e ".[dev]"                    # Expected: clean install
pytest tests/ -v --tb=short               # Expected: 0 failures

# Backtest
python -m src.backtest.run --start 2019-03-20 --end 2025-10-31
# Expected: Brier score < 0.25, ROI > 0% on 3%+ edge bets

# Daily pipeline (dry run)
python -m src.pipeline.daily --date 2025-09-15 --mode backtest --dry-run
# Expected: exit 0, predictions CSV, Discord payload JSON

# Calibration check
python -m src.model.calibrate --eval
# Expected: reliability diagram within 5% of diagonal
```

### Final Checklist
- [ ] All "Must Have" present (anti-leakage, multi-window, ABS, park, weather, settlement, Kelly, backtest, Discord)
- [ ] All "Must NOT Have" absent (no ERA, no batter-pitcher Log5, no full-game, no NN, no hardcoded keys)
- [ ] All tests pass with 0 failures
- [ ] Walk-forward Brier score < 0.25
- [ ] Discord dry-run produces valid JSON payload
- [ ] Bankroll kill-switch triggers at 30% drawdown
- [ ] Pipeline handles missing data with explicit "NO PICK" rows
