---
name: data-worker
description: Handles project scaffolding, configuration, database schema, data models, and external API clients
---

# Data Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use this worker for features involving:
- Project scaffolding and dependency setup (pyproject.toml, .env, .gitignore)
- Configuration modules (settings.yaml, constants, mappings)
- SQLite schema design and initialization
- Pydantic data models
- External API clients (The Odds API, pybaseball, MLB API, OpenWeatherMap)

## Windows-Specific Setup

On Windows, `.factory/init.sh` may fail in Git Bash due to `pip` not being on PATH after activation. Use direct interpreter invocation instead:

```powershell
# Install dependencies
& ".venv\Scripts\python.exe" -m pip install -e ".[dev]"

# Run tests
& ".venv\Scripts\python.exe" -m pytest tests/ -v --tb=short

# Syntax check (PowerShell)
$env:PYTHONDONTWRITEBYTECODE=1; & ".venv\Scripts\python.exe" -m py_compile (Get-ChildItem -Path "src" -Recurse -Filter "*.py" | ForEach-Object { $_.FullName })

# Lint
& ".venv\Scripts\python.exe" -m ruff check src tests
```

See `.factory/library/environment.md` for additional Windows notes.

## Work Procedure

1. **Read existing context**: Check `.factory/library/` for established patterns, `config/settings.yaml` for existing configuration, `src/db.py` for schema.

2. **Write tests first (TDD)**:
   - For schema: test table creation, column constraints, idempotency
   - For models: test validators (probability bounds, odds format, enum completeness)
   - For API clients: test de-vig math, rate limiting, missing data handling
   - Run tests to confirm they FAIL (red phase)

3. **Implement**:
   - Follow existing patterns in the codebase
   - Use `as_of_timestamp` for all feature-related data
   - All API keys from environment variables (never hardcode)
   - Use type hints and Pydantic for validation

4. **Verify tests pass (green phase)**:
   - Run `pytest tests/ -v --tb=short`
   - All tests must pass before proceeding

5. **Manual verification**:
   - For API clients: make a real API call (or mock if no key)
   - For database: run `init_db()` and verify schema with sqlite3
   - For config: verify loading with Python import

6. **Run linters and type checks**:
   - `python -m py_compile src/**/*.py` (use PowerShell variant on Windows)
   - Ensure no syntax errors

## Example Handoff

```json
{
  "salientSummary": "Implemented SQLite schema with games, features, predictions, odds_snapshots, bets, bankroll_ledger tables. Added as_of_timestamp NOT NULL constraint on features table. All 6 tests pass.",
  "whatWasImplemented": "Created src/db.py with init_db() function, schema_version table for migrations. Features table has as_of_timestamp, odds_snapshots has is_frozen. Idempotent CREATE IF NOT EXISTS pattern used.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {
        "command": "pytest tests/test_db_schema.py -v",
        "exitCode": 0,
        "observation": "6 tests passed: table creation, column constraints, idempotency, migration versioning"
      },
      {
        "command": "python -c \"from src.db import init_db; init_db('data/test.db')\"",
        "exitCode": 0,
        "observation": "Database created successfully"
      },
      {
        "command": "python -c \"import sqlite3; conn = sqlite3.connect('data/test.db'); print(conn.execute('PRAGMA table_info(features)').fetchall())\"",
        "exitCode": 0,
        "observation": "as_of_timestamp column present with NOT NULL=1"
      }
    ],
    "interactiveChecks": []
  },
  "tests": {
    "added": [
      {
        "file": "tests/test_db_schema.py",
        "cases": [
          { "name": "test_all_tables_created", "verifies": "VAL-DATA-001" },
          { "name": "test_as_of_timestamp_not_null", "verifies": "VAL-DATA-001" },
          { "name": "test_idempotent_init", "verifies": "VAL-DATA-001" }
        ]
      }
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- API endpoint or data model needed by this feature doesn't exist yet
- Configuration value needed but not defined in settings.yaml
- External API credential required but not available in environment
- Test reveals ambiguity in requirements that can't be resolved from existing context
