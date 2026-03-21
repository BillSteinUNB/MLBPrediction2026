# Experiment Dashboard — Local ML Experiment Index UI

## TL;DR

> **Quick Summary**: Build a 3-layer local experiment dashboard (FastAPI backend + React/Vite/Plotly.js frontend) that sits on top of existing CSV/JSON training artifacts and answers: *What changed? Did it help? Which model should I trust right now?*
> 
> **Deliverables**:
> - FastAPI backend at `src/dashboard/` with 6 API endpoints (health, runs, lanes, compare, run detail, promotions)
> - React+Vite frontend at `dashboard/` with 5 views (Overview, Lane Explorer, Compare, Run Detail, Promotion Board)
> - Backend pytest test suite with fixture data matching real schemas
> - Promotions tracking via `data/experiments/promotions.json`
> - Playwright smoke tests for all 5 views
> 
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 5 waves
> **Critical Path**: Task 1 → Task 5 → Task 8 → Task 13 → Task 17 → Task 22 → F1-F4

---

## Context

### Original Request
Build a local "experiment index" layer between raw training artifacts and a human-friendly UI. The system should answer 3 questions: What changed? Did it help? Which model should I trust right now? Uses 3 layers: source-of-truth artifacts, local backend adapter, and React frontend with dashboards/comparisons/trend lines.

### Interview Summary
**Key Discussions**:
- **5 data concepts**: experiment_runs, metrics, config, artifacts, promotions
- **Lane concept**: (holdout_season, target_column, variant) — deltas only meaningful within same lane
- **Immutability rule**: Runs never overwritten, always appended, every promotion is separate record
- **Visual emphasis**: Main cards (roc_auc, log_loss, brier, accuracy), warnings (ece, reliability_gap), change indicators (delta_vs_prev_*, comparison_*_delta)
- **Direction rule**: higher=better (accuracy, roc_auc), lower=better (log_loss, brier, ece, reliability_gap)

**User Decisions**:
- **Project location**: Monorepo subfolder — `dashboard/` for frontend, `src/dashboard/` for backend
- **Chart library**: Plotly.js (full-featured, good for reliability diagrams, Python ecosystem alignment)
- **Promotions storage**: `data/experiments/promotions.json` — append-only array
- **Test strategy**: Backend pytest only, Playwright for UI verification
- **Data layer**: CSV/JSON reading initially, no database

**Research Findings**:
- Pure Python project — no frontend code exists (greenfield)
- `experiment_metrics.csv` — 30 rows, 31 columns, lanes/deltas already computed by `experiment_report.py`
- `experiment_report.build_experiment_metrics_dataframe()` implements exact groupby/delta/is_best logic needed
- 5 experiment dirs, 45 JSON artifacts, rich schemas (training: feature importance + hyperparams, calibration: reliability diagrams + quality gates, stacking: base vs stacked comparison)
- `experiment_tracker.py` exists with config field logging (start_year, end_year, search_iterations) but JSONL log files not yet generated
- Variant values in data: `"base"`, `"stacked"`, `"identity"`, `"platt"` (NOT `"calibrated"`)
- `summary_path` is unique per row — usable as canonical run identifier

### Metis Review
**Identified Gaps** (addressed):
- Backend MUST import `experiment_report.build_experiment_metrics_dataframe()` — not reimplement logic
- Run ID scheme needed: use `summary_path` as canonical key, URL-encoded for API routes
- Backend should regenerate DataFrame from JSONs on each request (with mtime caching) — not read stale CSV
- Promotions are per-lane (holdout_season + target_column + variant)
- CSV-level data for list/overview views; direct JSON reading for Run Detail rich data (separate code paths)
- All metric fields must be `Optional[float]` — many legitimately null
- Must handle missing data files gracefully (fresh clone = 200 with empty arrays, not 500)
- `.gitignore` needs `dashboard/node_modules/` and `dashboard/dist/` entries
- FastAPI deps added to pyproject.toml under `[project.optional-dependencies]` dashboard extra
- Feature columns (210 items) excluded from list endpoints — only in Run Detail
- 14 edge cases documented and addressed in individual tasks

---

## Work Objectives

### Core Objective
Create a local experiment dashboard that normalizes existing training artifacts (CSV/JSON) into a browsable UI, enabling quick answers to "what changed?", "did it help?", and "which model to trust?" — organized by lanes (holdout_season × target_column × variant).

### Concrete Deliverables
- `src/dashboard/` — FastAPI backend package (main.py, schemas.py, adapters.py, routes/, dependencies.py)
- `dashboard/` — React+Vite frontend (5 views with Plotly.js charts)
- `data/experiments/promotions.json` — created on first promotion
- `tests/dashboard/` — Backend pytest suite
- `tests/dashboard_e2e/` — Playwright smoke tests

### Definition of Done
- [ ] `uvicorn src.dashboard.main:app` starts and serves API at localhost:8000
- [ ] `npm run dev` (in dashboard/) starts Vite dev server at localhost:5173 with API proxy
- [ ] All 5 views render with real experiment data
- [ ] All backend pytest tests pass: `pytest tests/dashboard/ -v`
- [ ] Playwright smoke tests pass for all 5 views
- [ ] Promotions can be created and displayed
- [ ] Empty data directory returns empty arrays (not errors)

### Must Have
- All 5 views: Overview, Lane Explorer, Experiment Compare, Run Detail, Promotion Board
- Lane-scoped metrics and deltas (never mix lanes)
- Backend reuses `experiment_report.build_experiment_metrics_dataframe()` for consistency
- Plotly.js charts: metric trends, reliability diagrams, feature importance bars
- Color-coded delta indicators (green=improvement, red=regression) with correct direction per metric
- Warning badges for ece > 0.05 or reliability_gap > 0.05
- Promotions tracking with explicit promote/read API
- Graceful handling of missing data files

### Must NOT Have (Guardrails)
- No database (no SQLite, DuckDB, migrations)
- No WebSocket, SSE, file watching, or auto-refresh
- No `.joblib` file loading in backend
- No authentication, sessions, or user management
- No Redux, Zustand, TanStack Query, or state management libraries
- No Jest, Vitest, or frontend unit tests
- No dark mode, themes, CSS framework (Tailwind/MUI), or customization
- No data export (CSV download, PDF), no advanced table sorting/filtering
- No Docker, CI/CD, or deployment config
- No model inference or training triggers from dashboard
- No config editing from UI — read-only except promotions
- No custom D3 or canvas rendering — use Plotly defaults
- Backend must NOT write to any file except `promotions.json`
- Feature columns (210 items) must NOT appear in list/overview endpoints

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: YES (pytest for Python, no JS test infra)
- **Automated tests**: YES (backend endpoint tests with pytest)
- **Framework**: pytest + FastAPI TestClient for backend, Playwright for frontend
- **Test-first for backend**: Each backend task writes the pytest test BEFORE the endpoint implementation (red-green pattern). Write a failing test for the expected API response shape, then implement the route to make it pass. This ensures the API contract is locked before code is written.
- **No frontend unit tests**: Frontend verification via Playwright smoke tests only (Task 21)

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Backend API**: Use Bash (curl) — Send requests, assert status + response fields
- **Frontend UI**: Use Playwright (playwright skill) — Navigate, interact, assert DOM, screenshot
- **Python modules**: Use Bash (pytest) — Run test suite, verify pass/fail

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Foundation — scaffolding, schemas, project setup):
├── Task 1: Update pyproject.toml + .gitignore for dashboard deps [quick]
├── Task 2: Scaffold FastAPI app with health endpoint [quick]
├── Task 3: Scaffold React+Vite app with proxy config [quick]
├── Task 4: Define Pydantic schemas for all 5 data concepts [quick]
└── Task 5: Build data adapter module (CSV + JSON reading) [deep]

Wave 2 (Backend API — all endpoints, MAX PARALLEL):
├── Task 6: Runs list endpoint + test (depends: 4, 5) [quick]
├── Task 7: Lanes endpoint + test (depends: 4, 5) [quick]
├── Task 8: Run detail endpoint + JSON artifact reading + test (depends: 4, 5) [unspecified-high]
├── Task 9: Compare endpoint + test (depends: 4, 5) [quick]
├── Task 10: Promotions CRUD endpoint + test (depends: 4, 5) [unspecified-high]
└── Task 11: Overview/summary endpoint + test (depends: 4, 5) [quick]

Wave 3 (Frontend Core — layout, shared components, Plotly wrappers):
├── Task 12: Layout shell + React Router for 5 views (depends: 3) [visual-engineering]
├── Task 13: Shared components — MetricCard, DeltaIndicator, WarningBadge (depends: 12) [visual-engineering]
├── Task 14: Plotly chart wrappers — MetricTrend, ReliabilityDiagram, FeatureImportance, MetricComparison (depends: 12) [visual-engineering]
└── Task 15: API client module + TypeScript types matching backend schemas (depends: 3, 4) [quick]

Wave 4 (Frontend Views — all 5 pages, MAX PARALLEL):
├── Task 16: Overview page (depends: 11, 13, 14, 15) [visual-engineering]
├── Task 17: Lane Explorer page (depends: 7, 13, 14, 15) [visual-engineering]
├── Task 18: Run Detail page (depends: 8, 13, 14, 15) [visual-engineering]
├── Task 19: Experiment Compare page (depends: 9, 13, 14, 15) [visual-engineering]
└── Task 20: Promotion Board page (depends: 10, 13, 15) [visual-engineering]

Wave 5 (Integration + Verification):
├── Task 21: Playwright smoke tests for all 5 views (depends: 16-20) [unspecified-high]
└── Task 22: Backend test suite finalization + edge cases (depends: 6-11) [unspecified-high]

Wave FINAL (After ALL tasks — 4 parallel reviews, then user okay):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)
-> Present results -> Get explicit user okay
```

Critical Path: Task 1 → Task 5 → Task 8 → Task 13 → Task 17 → Task 21 → F1-F4 → user okay
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 6 (Waves 2 & 4)

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| 1 | — | 2, 3, 5 |
| 2 | 1 | 6-11 |
| 3 | 1 | 12, 15 |
| 4 | — | 5, 6-11, 15 |
| 5 | 1, 4 | 6-11 |
| 6-11 | 4, 5 | 16-20 |
| 12 | 3 | 13, 14, 16-20 |
| 13 | 12 | 16-20 |
| 14 | 12 | 16-19 |
| 15 | 3, 4 | 16-20 |
| 16-20 | 13, 14, 15, respective backend tasks | 21 |
| 21 | 16-20 | F1-F4 |
| 22 | 6-11 | F1-F4 |

### Agent Dispatch Summary

- **Wave 1**: **5** — T1 → `quick`, T2 → `quick`, T3 → `quick`, T4 → `quick`, T5 → `deep`
- **Wave 2**: **6** — T6 → `quick`, T7 → `quick`, T8 → `unspecified-high`, T9 → `quick`, T10 → `unspecified-high`, T11 → `quick`
- **Wave 3**: **4** — T12 → `visual-engineering`, T13 → `visual-engineering`, T14 → `visual-engineering`, T15 → `quick`
- **Wave 4**: **5** — T16-T20 → `visual-engineering`
- **Wave 5**: **2** — T21 → `unspecified-high`, T22 → `unspecified-high`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. Update pyproject.toml + .gitignore for dashboard dependencies

  **What to do**:
  - Add `[project.optional-dependencies]` section `dashboard` to `pyproject.toml` with: `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `python-multipart`
  - Add to `.gitignore`: `dashboard/node_modules/`, `dashboard/dist/`, `dashboard/.vite/`
  - Run `pip install -e ".[dashboard,dev]"` to verify install

  **Must NOT do**:
  - Do not add frontend (npm) packages to pyproject.toml
  - Do not add database dependencies (sqlalchemy already exists for other purposes — do not add sqlite/duckdb extras)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4)
  - **Blocks**: Tasks 2, 3, 5
  - **Blocked By**: None

  **References**:
  - `pyproject.toml` (lines 1-43) — current project config, add dashboard extras under `[project.optional-dependencies]`
  - `.gitignore` — add dashboard-specific entries

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Dashboard deps install successfully
    Tool: Bash
    Preconditions: Virtual environment active
    Steps:
      1. Run: pip install -e ".[dashboard,dev]"
      2. Run: python -c "import fastapi; import uvicorn; print('OK')"
    Expected Result: Both commands exit 0, "OK" printed
    Evidence: .sisyphus/evidence/task-1-deps-install.txt

  Scenario: .gitignore includes dashboard entries
    Tool: Bash
    Preconditions: .gitignore exists
    Steps:
      1. Run: grep "dashboard/node_modules" .gitignore
      2. Run: grep "dashboard/dist" .gitignore
    Expected Result: Both greps return matches (exit 0)
    Evidence: .sisyphus/evidence/task-1-gitignore.txt
  ```

  **Commit**: YES (group 1)
  - Message: `feat(dashboard): add pyproject.toml dashboard extras and gitignore entries`
  - Files: `pyproject.toml`, `.gitignore`

- [x] 2. Scaffold FastAPI app with health endpoint

  **What to do**:
  - Create `src/dashboard/__init__.py` (empty)
  - Create `src/dashboard/main.py` with FastAPI app, CORS middleware (localhost origins), and `/api/health` endpoint
  - Create `src/dashboard/dependencies.py` with `get_models_dir()` and `get_experiments_dir()` dependency functions (return Path, configurable via env vars with defaults to `data/models` and `data/experiments`)
  - Health endpoint returns: `{"status": "ok", "data_available": bool, "run_count": int}` — check if experiment_metrics.csv exists and count rows
  - Create `tests/dashboard/__init__.py` and `tests/dashboard/conftest.py` with shared fixtures: `tmp_models_dir`, `sample_training_json`, `sample_metrics_csv`
  - Create `tests/dashboard/test_health.py` — test health with data present and without

  **Must NOT do**:
  - No authentication middleware
  - No WebSocket or startup events that watch files
  - CORS only allows localhost origins (127.0.0.1:5173, localhost:5173)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4)
  - **Blocks**: Tasks 6-11
  - **Blocked By**: Task 1

  **References**:
  - `src/ops/experiment_report.py:11-12` — `DEFAULT_MODELS_DIR` and `DEFAULT_REPORT_DIR` path constants
  - `src/config.py` — existing pydantic-settings pattern for settings management
  - `pyproject.toml:37-39` — pytest config, testpaths and pythonpath

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Health endpoint returns status with data
    Tool: Bash (curl)
    Preconditions: Backend running with real data directory
    Steps:
      1. Start: uvicorn src.dashboard.main:app --port 8000 &
      2. Wait 3 seconds
      3. Run: curl -s http://127.0.0.1:8000/api/health
      4. Kill uvicorn
    Expected Result: JSON contains "status":"ok", "data_available":true, "run_count":30
    Evidence: .sisyphus/evidence/task-2-health-with-data.txt

  Scenario: Health endpoint handles missing data
    Tool: Bash (pytest)
    Steps:
      1. Run: pytest tests/dashboard/test_health.py -v
    Expected Result: All tests pass, including test for missing data returning {"data_available": false, "run_count": 0}
    Evidence: .sisyphus/evidence/task-2-health-test.txt
  ```

  **Commit**: YES (group 2)
  - Message: `feat(dashboard-api): scaffold FastAPI app with health endpoint and test`
  - Files: `src/dashboard/__init__.py`, `src/dashboard/main.py`, `src/dashboard/dependencies.py`, `tests/dashboard/__init__.py`, `tests/dashboard/conftest.py`, `tests/dashboard/test_health.py`
  - Pre-commit: `pytest tests/dashboard/ -v`

- [x] 3. Scaffold React+Vite app with proxy config

  **What to do**:
  - Run `npm create vite@latest dashboard -- --template react-ts` from project root
  - Configure `vite.config.ts` with API proxy: `/api` → `http://127.0.0.1:8000`
  - Install dependencies: `npm install react-router-dom react-plotly.js plotly.js-basic-dist`
  - Install dev dependencies: `npm install -D @types/react-plotly.js`
  - Update `dashboard/src/App.tsx` with a minimal "Dashboard loading..." placeholder
  - Verify `npm run build` succeeds

  **Must NOT do**:
  - No CSS framework (no Tailwind, no MUI) — use plain CSS
  - No state management library (no Redux, no Zustand)
  - No test framework (no Jest, no Vitest)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4)
  - **Blocks**: Tasks 12, 15
  - **Blocked By**: Task 1 (.gitignore must have dashboard entries first)

  **References**:
  - Vite docs: proxy config at `server.proxy` in vite.config.ts
  - react-plotly.js with plotly.js-basic-dist factory pattern for smaller bundle

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Vite app builds successfully
    Tool: Bash
    Preconditions: Node.js 18+ installed
    Steps:
      1. Run: cd dashboard && npm run build
    Expected Result: Exit code 0, dist/ directory created with index.html
    Evidence: .sisyphus/evidence/task-3-vite-build.txt

  Scenario: Vite proxy config present
    Tool: Bash
    Steps:
      1. Run: grep -c "proxy" dashboard/vite.config.ts
    Expected Result: At least 1 match found
    Evidence: .sisyphus/evidence/task-3-proxy-config.txt
  ```

  **Commit**: YES (group 3)
  - Message: `feat(dashboard): scaffold React+Vite app with proxy config`
  - Files: `dashboard/*`
  - Pre-commit: `cd dashboard && npm run build`

- [x] 4. Define Pydantic schemas for all 5 data concepts

  **What to do**:
  - Create `src/dashboard/schemas.py` with Pydantic v2 BaseModel classes:
    - `RunSummary` — list-level: experiment_name, summary_path, run_kind, model_name, target_column, holdout_season (int), model_version, variant, accuracy (Optional[float]), log_loss (Optional[float]), roc_auc (Optional[float]), brier (Optional[float]), ece (Optional[float]), reliability_gap (Optional[float]), delta_vs_prev_roc_auc (Optional[float]), delta_vs_prev_log_loss (Optional[float]), delta_vs_prev_brier (Optional[float]), delta_vs_prev_accuracy (Optional[float]), comparison_brier_delta (Optional[float]), comparison_log_loss_delta (Optional[float]), comparison_roc_auc_delta (Optional[float]), comparison_accuracy_delta (Optional[float]), is_best_accuracy (bool), is_best_log_loss (bool), is_best_roc_auc (bool), is_best_brier (bool), run_timestamp (str), feature_column_count (Optional[int])
    - `RunDetail` — extends RunSummary with: feature_importance (list of {feature: str, importance: float}), best_params (dict), reliability_diagram (Optional list of bin objects), quality_gates (Optional dict), meta_feature_columns (Optional list of str), calibration_method (Optional str), train_row_count (Optional int), holdout_row_count (Optional int), stacking_metrics (Optional dict with base vs stacked comparison)
    - `Lane` — holdout_season (int), target_column (str), variant (str), run_count (int), best_run (RunSummary), latest_run (RunSummary)
    - `Promotion` — lane_holdout_season (int), lane_target_column (str), lane_variant (str), promoted_summary_path (str), promoted_at (str ISO datetime), reason (str), promoted_by (str, default "manual")
    - `PromotionRequest` — summary_path (str), reason (str)
    - `CompareResult` — run_a (RunSummary), run_b (RunSummary), metric_deltas (dict mapping metric name to float delta)
    - `HealthResponse` — status (str), data_available (bool), run_count (int)
    - `OverviewResponse` — total_runs (int), latest_run (RunSummary), best_per_lane (list of Lane), biggest_improvements (list of RunSummary), biggest_regressions (list of RunSummary)
  - All metric fields MUST be `Optional[float]` — training runs have null brier/ece/reliability_gap
  - Use `model_config = ConfigDict(from_attributes=True)` for ORM-mode compat

  **Must NOT do**:
  - No feature_columns list (210 items) in RunSummary — only in RunDetail
  - No data_version_hash in RunSummary (internal implementation detail)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3)
  - **Blocks**: Tasks 5, 6-11, 15
  - **Blocked By**: None

  **References**:
  - `data/experiments/experiment_metrics.csv` (line 1) — all 31 column names for RunSummary fields
  - `data/models/2024-umpires-identity/training_run_20260321T163829Z_b3922ce1.json` — feature_importance_rankings schema (line 241-342), best_params (line 224-233)
  - `data/models/2024-umpires-identity/calibration_run_20260321T163829Z_b3922ce1.json` — reliability_diagram schema (lines 38-128), quality_gates (lines 223-227)
  - `data/models/2024-umpires-identity/stacking_run_20260321T163829Z_b3922ce1.json` — meta_feature_columns (lines 251-256), stacked vs base metrics (lines 231-260)

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Schemas can be instantiated with real data
    Tool: Bash
    Steps:
      1. Run: python -c "from src.dashboard.schemas import RunSummary, Lane, Promotion; print('Schemas imported OK')"
    Expected Result: "Schemas imported OK" printed
    Evidence: .sisyphus/evidence/task-4-schemas-import.txt

  Scenario: RunSummary handles null metrics
    Tool: Bash
    Steps:
      1. Run: python -c "from src.dashboard.schemas import RunSummary; r = RunSummary(experiment_name='test', summary_path='test.json', run_kind='training', model_name='f5_ml_model', target_column='f5_ml_result', holdout_season=2024, model_version='v1', variant='base', run_timestamp='20260321T163829Z', accuracy=0.58, log_loss=0.67, roc_auc=0.61, brier=None, ece=None, reliability_gap=None, is_best_accuracy=True, is_best_log_loss=False, is_best_roc_auc=True, is_best_brier=False); print(f'brier={r.brier}, ece={r.ece}')"
    Expected Result: "brier=None, ece=None" — no validation error
    Evidence: .sisyphus/evidence/task-4-null-metrics.txt
  ```

  **Commit**: YES (group 4)
  - Message: `feat(dashboard-api): add Pydantic schemas for all 5 data concepts`
  - Files: `src/dashboard/schemas.py`
  - Pre-commit: `pytest tests/dashboard/ -v`

- [x] 5. Build data adapter module (CSV + JSON reading)

  **What to do**:
  - Create `src/dashboard/adapters.py` with:
    - `ExperimentDataAdapter` class that wraps data directory paths
    - `get_all_runs(models_dir) -> list[RunSummary]` — calls `experiment_report.build_experiment_metrics_dataframe(models_dir)`, converts DataFrame rows to `RunSummary` Pydantic objects. Implements mtime caching: stores last modification time of models dir, only rebuilds DataFrame if directory changed.
    - `get_run_detail(models_dir, summary_path) -> RunDetail` — reads the specific JSON artifact file referenced by `summary_path`, extracts rich data (feature_importance, reliability_diagram, quality_gates, best_params, stacking metrics), merges with CSV-level metrics
    - `get_lanes(runs) -> list[Lane]` — groups runs by (holdout_season, target_column, variant), computes best and latest per lane
    - `get_overview(runs) -> OverviewResponse` — latest run, best per lane, top 5 improvements (highest delta_vs_prev_roc_auc), top 5 regressions (lowest delta_vs_prev_roc_auc)
    - `compare_runs(run_a, run_b) -> CompareResult` — computes metric deltas between two RunSummary objects
    - `read_promotions(experiments_dir) -> list[Promotion]` — reads `promotions.json`, returns empty list if file missing
    - `write_promotion(experiments_dir, promotion) -> Promotion` — appends to `promotions.json`, creates file with `[]` if missing
  - Path normalization: convert all backslash paths to forward slashes
  - Handle missing files gracefully: return empty lists, not exceptions
  - Handle malformed JSON: wrap `json.loads` in try/except, skip with warning log

  **Must NOT do**:
  - Do not load .joblib files
  - Do not reimplement groupby/delta/is_best logic — use `build_experiment_metrics_dataframe()`
  - Do not add any database layer

  **Recommended Agent Profile**:
  - **Category**: `deep`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential — depends on Task 1 (deps) and Task 4 (schemas)
  - **Blocks**: Tasks 6-11
  - **Blocked By**: Tasks 1, 4

  **References**:
  - `src/ops/experiment_report.py:15-136` — `build_experiment_metrics_dataframe()` function — the core logic to reuse. Import as: `from src.ops.experiment_report import build_experiment_metrics_dataframe`
  - `src/ops/experiment_report.py:121-136` — sorting, delta computation, is_best marking logic
  - `data/models/2024-umpires-identity/training_run_20260321T163829Z_b3922ce1.json` — training artifact schema (full file)
  - `data/models/2024-umpires-identity/calibration_run_20260321T163829Z_b3922ce1.json` — calibration artifact schema (full file)
  - `data/models/2024-umpires-identity/stacking_run_20260321T163829Z_b3922ce1.json` — stacking artifact schema (full file)
  - `src/dashboard/schemas.py` — Pydantic models to populate (created in Task 4)

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Adapter loads real experiment data
    Tool: Bash (pytest)
    Steps:
      1. Create tests/dashboard/test_adapters.py with test that creates fixture JSON files matching real schema, calls get_all_runs(), asserts correct run count and field types
      2. Run: pytest tests/dashboard/test_adapters.py -v
    Expected Result: All tests pass, runs list is non-empty, all RunSummary fields validate
    Evidence: .sisyphus/evidence/task-5-adapter-test.txt

  Scenario: Adapter handles empty data directory
    Tool: Bash (pytest)
    Steps:
      1. Test calls get_all_runs() with empty tmp_path
    Expected Result: Returns empty list, no exception
    Evidence: .sisyphus/evidence/task-5-empty-data.txt

  Scenario: Adapter reads run detail with feature importance
    Tool: Bash (pytest)
    Steps:
      1. Test creates fixture training JSON with feature_importance_rankings
      2. Calls get_run_detail() with that summary_path
    Expected Result: RunDetail object has non-empty feature_importance list
    Evidence: .sisyphus/evidence/task-5-run-detail.txt
  ```

  **Commit**: YES (group 5)
  - Message: `feat(dashboard-api): add data adapter with experiment_report integration`
  - Files: `src/dashboard/adapters.py`, `tests/dashboard/test_adapters.py`
  - Pre-commit: `pytest tests/dashboard/ -v`

- [x] 6. Runs list endpoint + test

  **What to do**:
  - Create `src/dashboard/routes/__init__.py` and `src/dashboard/routes/runs.py`
  - **Test-first**: Write `tests/dashboard/test_runs.py` FIRST — define expected response shape for list endpoint (status 200, returns list of RunSummary-shaped dicts), filter test, and empty-data test. Tests should fail initially.
  - Then implement `GET /api/runs` — returns `list[RunSummary]` from adapter. Supports optional query params: `?holdout_season=2024`, `?target_column=f5_ml_result`, `?variant=base` for filtering
  - `GET /api/runs/{summary_path:path}` — returns `RunDetail` for specific run (delegates to Task 8)
  - Register router in `main.py`
  - Verify all tests pass

  **Must NOT do**: No pagination (data is small), no sorting params

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 7, 8, 9, 10, 11)
  - **Blocks**: Tasks 16, 17
  - **Blocked By**: Tasks 4, 5

  **References**:
  - `src/dashboard/adapters.py` — `get_all_runs()` function (Task 5)
  - `src/dashboard/schemas.py` — `RunSummary` model (Task 4)
  - `src/dashboard/dependencies.py` — `get_models_dir()` dependency (Task 2)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Runs list returns all 30 runs
    Tool: Bash (curl)
    Steps:
      1. Start backend with real data
      2. curl -s http://127.0.0.1:8000/api/runs | python -c "import sys,json; d=json.load(sys.stdin); print(len(d))"
    Expected Result: Prints "30"
    Evidence: .sisyphus/evidence/task-6-runs-list.txt

  Scenario: Filter by holdout_season
    Tool: Bash (curl)
    Steps:
      1. curl -s "http://127.0.0.1:8000/api/runs?holdout_season=2025" | python -c "import sys,json; d=json.load(sys.stdin); print(len(d)); assert all(r['holdout_season']==2025 for r in d)"
    Expected Result: Returns only 2025 holdout runs (6 runs), assertion passes
    Evidence: .sisyphus/evidence/task-6-runs-filter.txt
  ```
  **Commit**: YES (group 6) — `feat(dashboard-api): add runs list and detail endpoints with tests`

- [x] 7. Lanes endpoint + test

  **What to do**:
  - Create `src/dashboard/routes/lanes.py`
  - `GET /api/lanes` — returns `list[Lane]` with each lane containing: holdout_season, target_column, variant, run_count, best_run (by roc_auc), latest_run (by run_timestamp)
  - `GET /api/lanes/{holdout_season}/{target_column}/{variant}/runs` — returns filtered runs for specific lane, sorted by run_timestamp
  - Register router in `main.py`
  - Create `tests/dashboard/test_lanes.py`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 17
  - **Blocked By**: Tasks 4, 5

  **References**:
  - `src/ops/experiment_report.py:121-136` — groupby `["holdout_season", "target_column", "variant"]` — same grouping for lanes
  - `src/dashboard/adapters.py` — `get_lanes()` function (Task 5)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Lanes endpoint returns distinct lanes
    Tool: Bash (curl)
    Steps:
      1. curl -s http://127.0.0.1:8000/api/lanes | python -c "import sys,json; d=json.load(sys.stdin); print(len(d)); print(set(l['variant'] for l in d))"
    Expected Result: Multiple lanes, variants include "base", "stacked", "platt", "identity"
    Evidence: .sisyphus/evidence/task-7-lanes.txt

  Scenario: Lane runs endpoint filters correctly
    Tool: Bash (curl)
    Steps:
      1. curl -s "http://127.0.0.1:8000/api/lanes/2024/f5_ml_result/base/runs" | python -c "import sys,json; d=json.load(sys.stdin); assert all(r['variant']=='base' and r['target_column']=='f5_ml_result' for r in d); print(f'{len(d)} runs')"
    Expected Result: All runs match lane, assertion passes
    Evidence: .sisyphus/evidence/task-7-lane-runs.txt
  ```
  **Commit**: YES (group 6)

- [x] 8. Run detail endpoint + JSON artifact reading + test

  **What to do**:
  - Add `GET /api/runs/detail` endpoint (accepts `?summary_path=...` query param) — returns `RunDetail`
  - Reads the JSON artifact file referenced by summary_path
  - Extracts rich data based on run_kind:
    - training: feature_importance_rankings, best_params, cv_best_log_loss, train/holdout row counts
    - stacking: base vs stacked metrics (base_brier, stacked_brier, stacked_brier_improvement), meta_feature_columns, oof counts
    - calibration: **Both** `stacked_reliability_diagram` (pre-calibration) and `reliability_diagram` (post-calibration) exist in the JSON. Adapter maps `reliability_diagram` → response field `reliability_diagram` (use the post-calibration version). Also extract: quality_gates, calibration_method, calibration_fraction
  - Merges with CSV-level metrics from adapter
  - Returns 404 if summary_path not found
  - Create `tests/dashboard/test_run_detail.py`

  **Must NOT do**: Do not return feature_columns array (210 items) — omit from response

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 18
  - **Blocked By**: Tasks 4, 5

  **References**:
  - `data/models/2024-umpires-identity/training_run_20260321T163829Z_b3922ce1.json` — training JSON schema: lines 217-475 (models.f5_ml_model.feature_importance_rankings, best_params, holdout_metrics)
  - `data/models/2024-umpires-identity/calibration_run_20260321T163829Z_b3922ce1.json` — calibration JSON: lines 130-220 (`reliability_diagram` — post-calibration version), lines 38-128 (`stacked_reliability_diagram` — pre-calibration version). Use `reliability_diagram` for the API response. Lines 223-227 (`quality_gates`)
  - `data/models/2024-umpires-identity/stacking_run_20260321T163829Z_b3922ce1.json` — stacking JSON: lines 231-260 (holdout_metrics with base_* and stacked_* comparison)
  - `src/dashboard/adapters.py` — `get_run_detail()` function (Task 5)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Run detail returns feature importance for training run
    Tool: Bash (curl)
    Steps:
      1. curl -s "http://127.0.0.1:8000/api/runs/detail?summary_path=data/models/2024-umpires-identity/training_run_20260321T163829Z_b3922ce1.json" | python -c "import sys,json; d=json.load(sys.stdin); print(f'features: {len(d.get(\"feature_importance\", []))}'); assert len(d.get('feature_importance',[])) > 0"
    Expected Result: feature_importance has >0 items
    Evidence: .sisyphus/evidence/task-8-detail-training.txt

  Scenario: Run detail returns reliability diagram for calibration run (uses reliability_diagram key, not stacked_reliability_diagram)
    Tool: Bash (curl)
    Steps:
      1. curl -s "http://127.0.0.1:8000/api/runs/detail?summary_path=data/models/2024-umpires-identity/calibration_run_20260321T163829Z_b3922ce1.json" | python -c "import sys,json; d=json.load(sys.stdin); bins=d.get('reliability_diagram',[]); print(f'reliability bins: {len(bins)}'); assert len(bins)==10; assert d.get('quality_gates') is not None"
    Expected Result: reliability_diagram has exactly 10 bins (mapped from JSON key `reliability_diagram`), quality_gates present
    Evidence: .sisyphus/evidence/task-8-detail-calibration.txt

  Scenario: Nonexistent run returns 404
    Tool: Bash (curl)
    Steps:
      1. curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8000/api/runs/detail?summary_path=data/models/fake/nonexistent.json"
    Expected Result: HTTP 404
    Evidence: .sisyphus/evidence/task-8-detail-404.txt
  ```
  **Commit**: YES (group 7)

- [x] 9. Compare endpoint + test

  **What to do**:
  - Create `src/dashboard/routes/compare.py`
  - `GET /api/compare?run_a={summary_path}&run_b={summary_path}` — returns `CompareResult` with both RunSummary objects and computed metric deltas (run_b - run_a for each metric)
  - Include lane match indicator: `same_lane: bool` — true if both runs share (holdout_season, target_column, variant)
  - Return 404 if either run not found
  - Create `tests/dashboard/test_compare.py`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 19
  - **Blocked By**: Tasks 4, 5

  **References**:
  - `src/dashboard/adapters.py` — `compare_runs()` function (Task 5)
  - `src/dashboard/schemas.py` — `CompareResult` model (Task 4)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Compare two runs in same lane
    Tool: Bash (curl)
    Steps:
      1. curl two real summary_paths from same lane (2024/f5_ml_result/base)
      2. Assert same_lane is true, metric_deltas has roc_auc, log_loss, accuracy keys
    Expected Result: JSON with run_a, run_b, metric_deltas, same_lane=true
    Evidence: .sisyphus/evidence/task-9-compare-same-lane.txt

  Scenario: Compare runs from different lanes shows warning
    Tool: Bash (curl)
    Steps:
      1. curl two summary_paths from different lanes
    Expected Result: same_lane is false
    Evidence: .sisyphus/evidence/task-9-compare-diff-lane.txt
  ```
  **Commit**: YES (group 7)

- [x] 10. Promotions CRUD endpoint + test

  **What to do**:
  - Create `src/dashboard/routes/promotions.py`
  - `GET /api/promotions` — returns `list[Promotion]` from `promotions.json`. Empty list if file doesn't exist.
  - `POST /api/promotions` — accepts `PromotionRequest` body, validates run exists, appends to `promotions.json`, returns 201 with created Promotion. Creates file with `[]` if missing.
  - `GET /api/promotions/current` — returns current promoted run per lane (latest promotion per unique lane key). This is the "which model should I trust right now?" answer.
  - Validate: 404 if referenced run doesn't exist, 409 if same run already promoted for same lane
  - Create `tests/dashboard/test_promotions.py` — test create, read, duplicate rejection, missing file creation

  **Must NOT do**: No delete/revoke endpoint in v1 (append-only)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 20
  - **Blocked By**: Tasks 4, 5

  **References**:
  - `src/dashboard/schemas.py` — `Promotion`, `PromotionRequest` models (Task 4)
  - `src/dashboard/adapters.py` — `read_promotions()`, `write_promotion()` (Task 5)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Create promotion and read it back
    Tool: Bash (curl)
    Steps:
      1. POST /api/promotions with {"summary_path": "data\\models\\2024-umpires-identity\\calibration_run_20260321T163829Z_b3922ce1.json", "reason": "Best brier in lane"}
      2. GET /api/promotions
    Expected Result: POST returns 201, GET returns array containing the new promotion
    Evidence: .sisyphus/evidence/task-10-promotion-create.txt

  Scenario: Promotions file missing returns empty array
    Tool: Bash (pytest)
    Steps:
      1. Test with tmp_path that has no promotions.json
      2. GET /api/promotions
    Expected Result: 200 with empty array
    Evidence: .sisyphus/evidence/task-10-promotion-empty.txt
  ```
  **Commit**: YES (group 7)

- [x] 11. Overview/summary endpoint + test

  **What to do**:
  - Create `src/dashboard/routes/overview.py`
  - `GET /api/overview` — returns `OverviewResponse` with:
    - total_runs count
    - latest_run (most recent by run_timestamp)
    - best_per_lane (list of Lane objects — best roc_auc per lane)
    - biggest_improvements (top 5 runs by delta_vs_prev_roc_auc, descending — only runs with positive delta)
    - biggest_regressions (top 5 runs by delta_vs_prev_roc_auc, ascending — only runs with negative delta)
  - Register router in `main.py`
  - Create `tests/dashboard/test_overview.py`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 16
  - **Blocked By**: Tasks 4, 5

  **References**:
  - `src/dashboard/adapters.py` — `get_overview()` function (Task 5)
  - `src/dashboard/schemas.py` — `OverviewResponse` (Task 4)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Overview returns structured summary
    Tool: Bash (curl)
    Steps:
      1. curl -s http://127.0.0.1:8000/api/overview | python -c "import sys,json; d=json.load(sys.stdin); print(f'total={d[\"total_runs\"]}, lanes={len(d[\"best_per_lane\"])}, improvements={len(d[\"biggest_improvements\"])}')"
    Expected Result: total_runs=30, best_per_lane has multiple entries, biggest_improvements has entries
    Evidence: .sisyphus/evidence/task-11-overview.txt
  ```
  **Commit**: YES (group 6)

- [ ] 12. Layout shell + React Router for 5 views

  **What to do**:
  - Create `dashboard/src/Layout.tsx` — sidebar nav with links to 5 views, main content area, app title "MLB Experiment Dashboard"
  - Create `dashboard/src/pages/` directory with 5 placeholder page components: `OverviewPage.tsx`, `LaneExplorerPage.tsx`, `ComparePage.tsx`, `RunDetailPage.tsx`, `PromotionBoardPage.tsx`
  - Set up React Router in `App.tsx`:
    - `/` → OverviewPage
    - `/lanes` → LaneExplorerPage
    - `/compare` → ComparePage
    - `/runs/:summaryPath` → RunDetailPage (URL-encoded summary_path)
    - `/promotions` → PromotionBoardPage
  - Style with plain CSS: clean sidebar, responsive layout. Use a neutral color scheme matching the user's existing HTML dashboard (white background, light gray borders, Arial/sans-serif)
  - Each placeholder page shows page title and "Loading..." text

  **Must NOT do**: No CSS framework, no dark mode, no theme system

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 start
  - **Blocks**: Tasks 13, 14, 16-20
  - **Blocked By**: Task 3

  **References**:
  - `data/experiments/experiment_dashboard.html` — existing HTML dashboard styling to match (lines 229-249 — CSS styles)
  - `dashboard/src/App.tsx` — placeholder from Task 3 to replace

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: All 5 routes render without errors
    Tool: Playwright
    Steps:
      1. Navigate to http://localhost:5173/
      2. Assert: sidebar nav visible with 5 links
      3. Click each link: /, /lanes, /compare, /runs/test, /promotions
      4. Assert: no console errors on any page
    Expected Result: All 5 pages render with titles visible
    Evidence: .sisyphus/evidence/task-12-layout-routes.png
  ```
  **Commit**: YES (group 8) — `feat(dashboard-ui): add layout shell and routing for 5 views`

- [ ] 13. Shared components — MetricCard, DeltaIndicator, WarningBadge

  **What to do**:
  - Create `dashboard/src/components/MetricCard.tsx` — displays a metric name, value (formatted to 4 decimal places), and optional delta value. Props: `name`, `value`, `delta`, `higherIsBetter`. Color-codes: green if improvement, red if regression (direction-aware per `higherIsBetter` prop)
  - Create `dashboard/src/components/DeltaIndicator.tsx` — displays a delta value with arrow (↑/↓) and color. Props: `value`, `higherIsBetter`. Green+↑ for positive improvement, red+↓ for negative regression. Shows "—" for null
  - Create `dashboard/src/components/WarningBadge.tsx` — shows orange/red warning badge when metric exceeds threshold. Props: `label`, `value`, `threshold`. Shows if value > threshold
  - Create `dashboard/src/components/RunTable.tsx` — reusable table component for displaying lists of RunSummary. Shows: experiment_name, variant, roc_auc, log_loss, brier, accuracy, delta indicators. Clickable rows link to Run Detail
  - Style with plain CSS: match existing dashboard aesthetic (cards with borders, rounded corners, color coding for good/bad)

  **Must NOT do**: No sortable/filterable table library — just plain HTML table with CSS

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 14, 15)
  - **Blocks**: Tasks 16-20
  - **Blocked By**: Task 12

  **References**:
  - User spec: "higher is better: accuracy, roc_auc" / "lower is better: log_loss, brier, ece, reliability_gap"
  - `src/ops/experiment_report.py:499-529` — `_panel_card()`, `_fmt()`, `_fmt_signed()`, `_delta_class()` functions — formatting patterns to follow in React
  - `data/experiments/experiment_dashboard.html` — existing card/panel CSS (lines 234-249)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: MetricCard renders with correct color coding
    Tool: Playwright
    Steps:
      1. Navigate to a page that uses MetricCard (Overview after Task 16)
      2. Assert: cards for roc_auc, log_loss, brier, accuracy are visible
      3. Assert: delta values show green for improvements, red for regressions
    Expected Result: At least 4 metric cards visible with formatted values
    Evidence: .sisyphus/evidence/task-13-metric-cards.png
  ```
  **Commit**: YES (group 8)

- [ ] 14. Plotly chart wrappers — MetricTrend, ReliabilityDiagram, FeatureImportance, MetricComparison

  **What to do**:
  - Create `dashboard/src/charts/createPlotlyComponent.ts` — factory function using `react-plotly.js` with `plotly.js-basic-dist` for smaller bundle. Lazy-loaded with React.lazy + Suspense
  - Create `dashboard/src/charts/MetricTrendChart.tsx` — line chart showing metric values over time (x=run_timestamp, y=metric value). Props: `runs` (array), `metricKey` (string), `higherIsBetter`. Shows experiment_name as hover label
  - Create `dashboard/src/charts/ReliabilityDiagramChart.tsx` — bar chart showing calibration bins. X=bin_lower-bin_upper range, Y=empirical_positive_rate vs mean_predicted_probability. Include diagonal reference line for perfect calibration
  - Create `dashboard/src/charts/FeatureImportanceChart.tsx` — horizontal bar chart of top N features. Props: `features` (array of {feature, importance}), `limit` (default 15). Sorted by importance descending
  - Create `dashboard/src/charts/MetricComparisonChart.tsx` — grouped bar chart comparing metrics between two runs. Props: `runA`, `runB`, `metrics`. Shows delta as annotation

  **Must NOT do**: No custom D3 or canvas — use Plotly defaults. No zooming/panning customization.

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 13, 15)
  - **Blocks**: Tasks 16-19
  - **Blocked By**: Task 12

  **References**:
  - `src/ops/experiment_report.py:436-496` — `_render_metric_svg()` — existing SVG chart logic showing how metrics are plotted over time (line + points). Follow same data flow in Plotly
  - `data/models/2024-umpires-identity/calibration_run_20260321T163829Z_b3922ce1.json:130-220` — `reliability_diagram` (post-calibration) bin schema: bin_index, bin_lower, bin_upper, count, mean_predicted_probability, empirical_positive_rate, absolute_gap. Note: `stacked_reliability_diagram` (lines 38-128) is the pre-calibration version — the chart should use `reliability_diagram`
  - `data/models/2024-umpires-identity/training_run_20260321T163829Z_b3922ce1.json:241-342` — feature_importance_rankings schema (feature, importance)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Plotly chart renders without errors
    Tool: Playwright
    Steps:
      1. Navigate to Lane Explorer (after Task 17) with data
      2. Assert: at least one Plotly chart container is visible (class "js-plotly-plot")
      3. Assert: no console errors
    Expected Result: Chart SVG visible in DOM
    Evidence: .sisyphus/evidence/task-14-plotly-chart.png
  ```
  **Commit**: YES (group 9) — `feat(dashboard-ui): add Plotly chart wrappers and API client`

- [ ] 15. API client module + TypeScript types matching backend schemas

  **What to do**:
  - Create `dashboard/src/api/types.ts` — TypeScript interfaces matching backend Pydantic schemas: `RunSummary`, `RunDetail`, `Lane`, `Promotion`, `PromotionRequest`, `CompareResult`, `HealthResponse`, `OverviewResponse`
  - Create `dashboard/src/api/client.ts` — fetch wrapper functions:
    - `fetchHealth(): Promise<HealthResponse>`
    - `fetchRuns(filters?): Promise<RunSummary[]>`
    - `fetchLanes(): Promise<Lane[]>`
    - `fetchLaneRuns(holdout, target, variant): Promise<RunSummary[]>`
    - `fetchRunDetail(summaryPath): Promise<RunDetail>`
    - `fetchCompare(pathA, pathB): Promise<CompareResult>`
    - `fetchOverview(): Promise<OverviewResponse>`
    - `fetchPromotions(): Promise<Promotion[]>`
    - `fetchCurrentPromotions(): Promise<Promotion[]>`
    - `createPromotion(request): Promise<Promotion>`
  - All fetches go to `/api/*` (proxied by Vite to backend)
  - Handle errors: throw on non-2xx with error message from response body

  **Must NOT do**: No axios, no TanStack Query — plain fetch only

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 13, 14)
  - **Blocks**: Tasks 16-20
  - **Blocked By**: Tasks 3, 4

  **References**:
  - `src/dashboard/schemas.py` — Pydantic schemas to mirror in TypeScript (Task 4)

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: TypeScript compiles without errors
    Tool: Bash
    Steps:
      1. cd dashboard && npx tsc --noEmit
    Expected Result: Exit code 0, no type errors
    Evidence: .sisyphus/evidence/task-15-types-check.txt
  ```
  **Commit**: YES (group 9)

- [ ] 16. Overview page

  **What to do**:
  - Implement `dashboard/src/pages/OverviewPage.tsx`:
    - Calls `fetchOverview()` on mount
    - Shows 4 summary cards: total runs, latest experiment name, best roc_auc (any lane), latest delta_vs_prev_roc_auc
    - Shows "Best Run Per Lane" section — table of Lane objects with best_run metrics
    - Shows "Biggest Improvements" section — top 5 runs with largest positive delta_vs_prev_roc_auc, displayed as RunTable
    - Shows "Biggest Regressions" section — top 5 runs with largest negative delta_vs_prev_roc_auc
    - Loading state: "Loading overview..." spinner/text
    - Error state: "Failed to load data" message
    - Empty state: "No experiment data found. Run training first." message

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Tasks 17-20)
  - **Blocks**: Task 21
  - **Blocked By**: Tasks 11, 13, 14, 15

  **References**:
  - `src/dashboard/schemas.py` — `OverviewResponse` shape
  - `dashboard/src/components/MetricCard.tsx`, `RunTable.tsx` (Task 13)
  - `data/experiments/experiment_dashboard.html` — existing dashboard layout to loosely follow

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Overview page shows real data
    Tool: Playwright
    Steps:
      1. Start backend + frontend
      2. Navigate to http://localhost:5173/
      3. Wait for data to load (wait for text "total" or "runs" to appear)
      4. Assert: at least one MetricCard visible
      5. Assert: "Best Run Per Lane" section has table rows
      6. Screenshot
    Expected Result: Page populated with real experiment data
    Evidence: .sisyphus/evidence/task-16-overview.png

  Scenario: Overview handles empty data
    Tool: Playwright
    Steps:
      1. Start backend with empty data directory
      2. Navigate to http://localhost:5173/
      3. Assert: "No experiment data" message visible
    Expected Result: Graceful empty state message
    Evidence: .sisyphus/evidence/task-16-overview-empty.png
  ```
  **Commit**: YES (group 10)

- [ ] 17. Lane Explorer page

  **What to do**:
  - Implement `dashboard/src/pages/LaneExplorerPage.tsx`:
    - Calls `fetchLanes()` on mount to populate lane selector
    - 3 dropdown selectors: holdout_season, target_column, variant (populated from available lanes)
    - When lane selected: calls `fetchLaneRuns()` to get runs in that lane, sorted by run_timestamp
    - Displays 4 MetricTrendCharts: roc_auc, log_loss, brier, accuracy over time for runs in lane
    - Displays RunTable showing all runs in lane with delta indicators
    - Each metric chart has correct `higherIsBetter` prop (true for roc_auc/accuracy, false for log_loss/brier)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 21
  - **Blocked By**: Tasks 7, 13, 14, 15

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Lane Explorer shows filtered metrics with charts
    Tool: Playwright
    Steps:
      1. Navigate to http://localhost:5173/lanes
      2. Select holdout_season=2024, target_column=f5_ml_result, variant=base
      3. Wait for charts to render
      4. Assert: at least 1 Plotly chart visible
      5. Assert: run table shows only matching runs
    Expected Result: Charts and table populated with lane-specific data
    Evidence: .sisyphus/evidence/task-17-lane-explorer.png
  ```
  **Commit**: YES (group 10)

- [ ] 18. Run Detail page

  **What to do**:
  - Implement `dashboard/src/pages/RunDetailPage.tsx`:
    - Reads summary_path from URL params
    - Calls `fetchRunDetail(summaryPath)` on mount
    - Shows header: experiment_name, run_kind, variant, holdout_season, run_timestamp
    - Shows 4 main MetricCards: roc_auc, log_loss, brier, accuracy (with deltas)
    - Shows 2 WarningBadges: ece (threshold 0.05), reliability_gap (threshold 0.05)
    - Conditional sections based on run_kind:
      - Training: FeatureImportanceChart (top 15), best_params table, train/holdout row counts
      - Stacking: base vs stacked comparison (MetricComparisonChart), meta_feature_columns list
      - Calibration: ReliabilityDiagramChart, quality_gates status badges (pass/fail)
    - "Config" section: best_params displayed as key-value table
    - "Artifact Links" section: summary_path displayed as clickable text (not functional link — just shows the path)

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 21
  - **Blocked By**: Tasks 8, 13, 14, 15

  **References**:
  - `data/models/2024-umpires-identity/training_run_20260321T163829Z_b3922ce1.json` — feature_importance shape
  - `data/models/2024-umpires-identity/calibration_run_20260321T163829Z_b3922ce1.json` — reliability_diagram and quality_gates shape

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Training run detail shows feature importance
    Tool: Playwright
    Steps:
      1. Navigate to run detail for a training run
      2. Assert: FeatureImportance chart visible (horizontal bars)
      3. Assert: best_params table has rows (subsample, learning_rate, etc.)
    Expected Result: Feature importance chart and config table visible
    Evidence: .sisyphus/evidence/task-18-detail-training.png

  Scenario: Calibration run detail shows reliability diagram
    Tool: Playwright
    Steps:
      1. Navigate to run detail for a calibration run
      2. Assert: ReliabilityDiagram chart visible
      3. Assert: quality_gates badges visible (brier_lt_0_25, ece_lt_0_05)
    Expected Result: Reliability diagram and quality gate indicators visible
    Evidence: .sisyphus/evidence/task-18-detail-calibration.png
  ```
  **Commit**: YES (group 10)

- [ ] 19. Experiment Compare page

  **What to do**:
  - Implement `dashboard/src/pages/ComparePage.tsx`:
    - Two run selectors (dropdowns populated from `fetchRuns()`)
    - "Compare" button that calls `fetchCompare(pathA, pathB)`
    - Shows side-by-side metric cards for both runs
    - Shows MetricComparisonChart (grouped bars) for roc_auc, log_loss, brier, accuracy
    - Shows delta values between runs with color coding (green=improvement, red=regression)
    - Shows lane match indicator: green "Same lane" or orange "Different lanes — comparison may not be meaningful"
    - Config diff section: if both runs have best_params, show parameter differences

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 21
  - **Blocked By**: Tasks 9, 13, 14, 15

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Compare page shows side-by-side metrics
    Tool: Playwright
    Steps:
      1. Navigate to http://localhost:5173/compare
      2. Select two runs from dropdowns
      3. Click "Compare"
      4. Assert: two columns of metrics visible
      5. Assert: delta values visible with color coding
    Expected Result: Comparison table/chart renders with deltas
    Evidence: .sisyphus/evidence/task-19-compare.png
  ```
  **Commit**: YES (group 10)

- [ ] 20. Promotion Board page

  **What to do**:
  - Implement `dashboard/src/pages/PromotionBoardPage.tsx`:
    - Calls `fetchCurrentPromotions()` on mount
    - Shows table of current promoted models per lane: lane (holdout_season / target_column / variant), promoted experiment name, key metrics (roc_auc, brier), promoted_at timestamp, reason
    - "Promote Run" section: dropdown to select a run, text input for reason, "Promote" button
    - After promotion: refresh table, show success toast/message
    - If no promotions: show "No runs promoted yet. Use the form below to promote your best model."
    - Validation: prevent promoting if no run selected or no reason provided

  **Recommended Agent Profile**:
  - **Category**: `visual-engineering`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 21
  - **Blocked By**: Tasks 10, 13, 15

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Promotion Board allows promoting a run
    Tool: Playwright
    Steps:
      1. Navigate to http://localhost:5173/promotions
      2. Select a run from dropdown
      3. Enter reason: "Best brier score in 2024 ML lane"
      4. Click "Promote"
      5. Assert: new promotion appears in table
    Expected Result: Promotion created and displayed
    Evidence: .sisyphus/evidence/task-20-promotion.png

  Scenario: Empty promotions shows CTA
    Tool: Playwright
    Steps:
      1. Navigate with no promotions.json
      2. Assert: "No runs promoted yet" message visible
    Expected Result: Empty state message with form visible
    Evidence: .sisyphus/evidence/task-20-promotion-empty.png
  ```
  **Commit**: YES (group 10) — `feat(dashboard-ui): implement all 5 view pages`

- [ ] 21. Playwright smoke tests for all 5 views

  **What to do**:
  - Create `tests/dashboard_e2e/` directory with Playwright test suite
  - Install Playwright: `pip install playwright pytest-playwright && playwright install chromium`
  - Create `tests/dashboard_e2e/conftest.py` — fixture to start backend (uvicorn) and frontend (npm run dev) as subprocesses, wait for health check, tear down after
  - Create `tests/dashboard_e2e/test_views.py` with 5 smoke tests:
    1. Overview: loads, shows metric cards, no console errors
    2. Lane Explorer: loads, lane selector has options, selecting lane shows chart
    3. Run Detail: loads for specific run, shows metric cards, conditional chart visible
    4. Compare: loads, can select two runs, compare renders metrics
    5. Promotion Board: loads, shows form, can submit (or shows empty state)
  - Each test captures screenshot as evidence

  **Must NOT do**: No component-level tests, no mock data — test against real backend serving real data

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: [`playwright`]

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 16-20

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: All 5 Playwright tests pass
    Tool: Bash
    Steps:
      1. Start backend: uvicorn src.dashboard.main:app --port 8000 &
      2. Start frontend: cd dashboard && npm run dev &
      3. Wait for both to be ready
      4. Run: pytest tests/dashboard_e2e/ -v --headed
    Expected Result: 5 tests pass, screenshots captured
    Evidence: .sisyphus/evidence/task-21-e2e-results.txt
  ```
  **Commit**: YES (group 11) — `test(dashboard): add Playwright smoke tests for all 5 views`

- [ ] 22. Backend test suite finalization + edge cases

  **What to do**:
  - Review all tests in `tests/dashboard/` and add missing edge cases:
    - Empty data directory → all endpoints return 200 with empty/default responses
    - Malformed JSON in data/models → skipped gracefully, other runs still returned
    - Lane with only 1 run → delta_vs_prev fields are null
    - Null metrics for training runs → brier/ece/reliability_gap are null in response
    - Promotion for nonexistent run → 404
    - Duplicate promotion → 409
    - Path normalization → Windows backslashes converted to forward slashes
  - Ensure `conftest.py` fixtures cover all 3 run types (training, stacking, calibration) with realistic JSON matching actual schema
  - Run full backend test suite: `pytest tests/dashboard/ -v --tb=short`

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 5 (with Task 21)
  - **Blocks**: F1-F4
  - **Blocked By**: Tasks 6-11

  **QA Scenarios (MANDATORY):**
  ```
  Scenario: Full backend test suite passes including edge cases
    Tool: Bash
    Steps:
      1. Run: pytest tests/dashboard/ -v --tb=short
    Expected Result: All tests pass, 0 failures, covers 7+ edge case scenarios
    Evidence: .sisyphus/evidence/task-22-backend-tests.txt
  ```
  **Commit**: YES (group 12) — `test(dashboard-api): finalize backend test suite with edge cases`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Present consolidated results to user and get explicit "okay" before completing.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `pytest tests/dashboard/ -v`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check Python files follow ruff config (line-length=100, py311). Check Pydantic models use `Optional[float]` for nullable metrics. Check no `.joblib` loading. Check no state management libraries in package.json.
  Output: `Tests [N pass/N fail] | Python lint [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill)
  Start from clean state. Start backend (`uvicorn src.dashboard.main:app`), start frontend (`npm run dev` in dashboard/). Navigate all 5 views with Playwright. Verify: Overview shows latest run, Lane Explorer filters by lane, Run Detail shows feature importance chart, Compare shows delta coloring, Promotion Board allows promotion. Test edge case: curl API with nonexistent run ID (expect 404). Save screenshots.
  Output: `Views [5/5 pass] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual files created. Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance: no DB, no WebSocket, no auth, no state management, no CSS framework, no dark mode, no data export. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Creep [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

| # | Message | Files | Pre-commit |
|---|---------|-------|------------|
| 1 | `feat(dashboard): add pyproject.toml dashboard extras and gitignore entries` | pyproject.toml, .gitignore | — |
| 2 | `feat(dashboard-api): scaffold FastAPI app with health endpoint and test` | src/dashboard/*.py, tests/dashboard/*.py | pytest tests/dashboard/ |
| 3 | `feat(dashboard): scaffold React+Vite app with proxy config` | dashboard/* | npm run build (in dashboard/) |
| 4 | `feat(dashboard-api): add Pydantic schemas for all 5 data concepts` | src/dashboard/schemas.py | pytest tests/dashboard/ |
| 5 | `feat(dashboard-api): add data adapter with experiment_report integration` | src/dashboard/adapters.py, tests/dashboard/test_adapters.py | pytest tests/dashboard/ |
| 6 | `feat(dashboard-api): add runs and lanes API endpoints with tests` | src/dashboard/routes/*.py, tests/dashboard/test_*.py | pytest tests/dashboard/ |
| 7 | `feat(dashboard-api): add compare, detail, promotions endpoints with tests` | src/dashboard/routes/*.py, tests/dashboard/test_*.py | pytest tests/dashboard/ |
| 8 | `feat(dashboard-ui): add layout shell, routing, shared components` | dashboard/src/* | npm run build |
| 9 | `feat(dashboard-ui): add Plotly chart wrappers and API client` | dashboard/src/* | npm run build |
| 10 | `feat(dashboard-ui): implement all 5 view pages` | dashboard/src/pages/* | npm run build |
| 11 | `test(dashboard): add Playwright smoke tests for all 5 views` | tests/dashboard_e2e/* | — |
| 12 | `test(dashboard-api): finalize backend test suite with edge cases` | tests/dashboard/*.py | pytest tests/dashboard/ |

---

## Success Criteria

### Verification Commands
```bash
# Backend starts
uvicorn src.dashboard.main:app --host 127.0.0.1 --port 8000  # Expected: "Uvicorn running on http://127.0.0.1:8000"

# Health check
curl http://127.0.0.1:8000/api/health  # Expected: {"status":"ok","data_available":true,"run_count":30}

# Runs endpoint
curl http://127.0.0.1:8000/api/runs  # Expected: JSON array with 30 objects

# Lanes endpoint
curl http://127.0.0.1:8000/api/lanes  # Expected: JSON array of lane objects

# Frontend builds
cd dashboard && npm run build  # Expected: exit code 0

# Backend tests pass
pytest tests/dashboard/ -v  # Expected: all pass

# Frontend dev server
cd dashboard && npm run dev  # Expected: Vite dev server on localhost:5173
```

### Final Checklist
- [ ] All 5 views render with real data
- [ ] All "Must Have" features present
- [ ] All "Must NOT Have" guardrails respected
- [ ] All backend tests pass
- [ ] Playwright smoke tests pass
- [ ] Empty data directory handled gracefully
- [ ] Promotions create + read works
