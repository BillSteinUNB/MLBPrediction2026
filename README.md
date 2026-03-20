# MLB F5 Betting Prediction System

A production-ready Python system for predicting First Five Innings (F5) Moneyline and Run Line outcomes in MLB games. The system uses advanced sabermetrics, a stacked ML ensemble, and disciplined bankroll management to identify +EV bets.

## Features

- **Data Pipeline**: Automated ingestion from Statcast, FanGraphs, Odds API, OpenWeather, and lineup sources
- **Feature Engineering**: Multi-window rolling sabermetrics (wRC+, xFIP, DRS, OAA) with Marcel blending
- **ML Pipeline**: XGBoost → Logistic Regression stacking → Platt calibration
- **Decision Engine**: De-vigged edge calculation with Quarter Kelly sizing
- **Bankroll Management**: Kill-switch at 30% drawdown, same-team correlation handling
- **Notifications**: Discord webhook alerts with formatted pick cards
- **Automation**: Cross-platform scheduler (Windows Task Scheduler / cron / systemd)

## Quick Start

### Prerequisites

- Python 3.12+
- SQLite 3
- API Keys: Odds API, OpenWeather, Discord Webhook (optional for dry-run mode)

### Installation

```bash
# Clone and install
git clone <repo-url>
cd MLBPrediction2026
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix
pip install -e ".[dev]"

# Initialize database
python -c "from src.db import init_db; init_db('data/mlb.db')"

# Configure environment
python scripts/setup_env.py  # Interactive .env setup
```

### Daily Usage

```bash
# Dry-run (prints picks to console without betting)
python -m src.pipeline.daily --date today --mode backtest --dry-run

# Production run (sends Discord notifications)
python -m src.pipeline.daily --date today --mode prod
```

### Scheduler Setup

```bash
# Windows Task Scheduler
python scripts/setup_scheduler.py --test

# Unix cron
python scripts/setup_scheduler.py --platform cron --install
```

## Project Structure

```
MLBPrediction2026/
├── config/
│   └── settings.yaml       # Team configs, park factors, ABS exceptions
├── data/
│   ├── mlb.db              # SQLite database
│   ├── models/             # Saved model artifacts
│   └── training/           # Training data parquets
├── scripts/
│   ├── setup_scheduler.py  # Cross-platform scheduler setup
│   ├── setup_env.py        # Interactive .env configuration
│   ├── run_daily.bat       # Windows runner
│   └── run_daily.sh        # Unix runner
├── src/
│   ├── backtest/           # Walk-forward backtest
│   ├── clients/            # External API clients
│   ├── db/                 # Database layer
│   ├── engine/             # Edge calc, Kelly, settlement
│   ├── features/           # Feature engineering modules
│   ├── model/              # Training, stacking, calibration
│   ├── notifications/      # Discord webhook
│   ├── ops/                # Error handling, logging, performance
│   └── pipeline/           # Daily orchestrator
└── tests/                  # pytest suite (210+ tests)
```

## Configuration

### Environment Variables (.env)

```env
ODDS_API_KEY=your_odds_api_key
OPENWEATHER_API_KEY=your_openweather_key
DISCORD_WEBHOOK_URL=your_discord_webhook
```

### Key Settings (config/settings.yaml)

- **30 MLB teams**: Stadium coordinates, park factors
- **ABS exceptions**: Venues without Automatic Ball-Strike (Mexico City, MiLB parks)
- **Marcel weights**: Regression-to-mean blending parameters

## ML Pipeline

1. **Training Data**: 7 seasons (2018-2025, skipping 2020), anti-leakage safe
2. **Base Model**: XGBoost with temporal cross-validation
3. **Stacking**: Logistic regression on XGBoost probabilities + baselines
4. **Calibration**: Platt scaling on dedicated 10% holdout
5. **Quality Gates**: Brier < 0.25, ECE < 0.05

### Model Training

```bash
# Train from scratch
python -m src.model.xgboost_trainer --training-data data/training/training_data.parquet
python -m src.model.stacking --training-data data/training/training_data.parquet
python -m src.model.calibration --training-data data/training/training_data.parquet
```

### Backtesting

```bash
python -m src.backtest.run --start 2022-01-01 --end 2023-12-31
```

## Betting Strategy

- **Edge Threshold**: Minimum 3% de-vigged edge
- **Stake Sizing**: Quarter Kelly, max 5% bankroll
- **Kill-Switch**: 30% drawdown halts betting
- **Correlation**: Same-game ML+RL treated as single bet

## Testing

```bash
# Full suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=term-missing

# Targeted
pytest tests/test_data_integrity.py tests/test_antileak.py tests/test_feature_engineering.py
```

## Known Limitations

- **pybaseball team_game_logs**: Pandas 3 compatibility issue (worked around)
- **Weather**: Uses cached/neutral values when API unavailable
- **Lineups**: Falls back to team averages when scratches detected

## Roadmap / Future Work

- [ ] Live CLV tracking with closing line updates
- [ ] Bookmaker identity in performance tracking
- [ ] Enhanced same-day feature recomputation
- [ ] Expanded anti-leakage test coverage for real training data

## License

MIT

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    DAILY PIPELINE                       │
                    │                                                         │
┌─────────────┐     │  ┌──────────┐   ┌──────────┐   ┌──────────┐           │
│   Schedule  │─────┼──▶│  Lineups │──▶│   Odds   │──▶│ Weather  │           │
│   (Statcast)│     │  └──────────┘   └──────────┘   └──────────┘           │
└─────────────┘     │        │              │              │                 │
                    │        ▼              ▼              ▼                 │
                    │  ┌───────────────────────────────────────────┐        │
                    │  │            FEATURE ENGINEERING              │        │
                    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │        │
                    │  │  │ Offense │ │ Pitching│ │ Defense │       │        │
                    │  │  └─────────┘ └─────────┘ └─────────┘       │        │
                    │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐       │        │
                    │  │  │Bullpen  │ │ Weather │ │Baseline │       │        │
                    │  │  └─────────┘ └─────────┘ └─────────┘       │        │
                    │  └───────────────────────────────────────────┘        │
                    │                         │                              │
                    │                         ▼                              │
                    │  ┌───────────────────────────────────────────┐        │
                    │  │              ML ENSEMBLE                    │        │
                    │  │  XGBoost ──▶ LR Stacking ──▶ Platt Calib  │        │
                    │  └───────────────────────────────────────────┘        │
                    │                         │                              │
                    │                         ▼                              │
                    │  ┌───────────────────────────────────────────┐        │
                    │  │            DECISION ENGINE                 │        │
                    │  │  Edge Calc ──▶ Kelly Sizing ──▶ Kill-SW   │        │
                    │  └───────────────────────────────────────────┘        │
                    │                         │                              │
                    │                         ▼                              │
                    │  ┌──────────┐    ┌──────────┐    ┌──────────┐         │
                    │  │  SQLite  │    │ Discord  │    │  Logs    │         │
                    │  └──────────┘    └──────────┘    └──────────┘         │
                    └─────────────────────────────────────────────────────────┘
```
