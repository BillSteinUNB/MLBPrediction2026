# MLBPrediction2026

MLB first-five and run-count prediction system with a Python modeling pipeline, FastAPI backend, and React dashboard.

## Repo Layout

- `src/`
  - Main Python application code: pipeline, model training, features, engine, dashboard API, ops, and notifications.
- `dashboard/`
  - Vite + React frontend.
- `tests/`
  - Pytest suites grouped by area such as `features/`, `model/`, `ops/`, and `integration/`.
- `scripts/`
  - Operational entrypoints, data builders, research runners, and export utilities.
- `config/`
  - YAML configuration and environment-specific presets.
- `docs/`
  - Human-written documentation, handoffs, repo maps, runbooks, and research notes.
- `data/`
  - Versioned project data, reports, metadata, and local runtime database files.
- `archive/`
  - Preserved historical material that is intentionally kept but not part of the active workflow.
- `AutoResearch/`
  - Overnight research orchestration subsystem.
- `OddsScraper/`
  - Historical odds scraping subsystem.

## Start Here

- General orientation: `docs/INDEX.md`
- Repo layout: `docs/repo/REPO_MAP.md`
- Run-count workflow: `docs/runbooks/RUN_COUNT_RESEARCH_WORKFLOW.md`
- Active research assumptions: `docs/research/MLB_RUN_MODEL_RESEARCH_ASSUMPTIONS.md`
- Historical comparison tracker: `docs/research/MODEL_COMPARISON_TRACKER.md`

## Common Commands

```bash
pip install -e ".[dev]"
python -m src.pipeline.daily --date today --mode backtest --dry-run
python -m src.pipeline.daily --date today --mode prod
pytest tests/ -v
ruff .
python -m uvicorn src.dashboard.main:app --host 127.0.0.1 --port 8000
cd dashboard && npm run dev
```

## Notes

- Python packaging and tooling are defined in `pyproject.toml`.
- The repo intentionally keeps important research outputs on GitHub; local cache and temp spill are ignored.
- Large historical or superseded material is moved under `archive/` to keep the active tree readable.
