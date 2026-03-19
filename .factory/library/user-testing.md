# User Testing

Testing surface, validation approach, and resource considerations.

**What belongs here:** How to validate the system, testing surface classification, resource costs.

---

## Validation Surface

**Primary Surface:** CLI (command-line interface)

This is a batch processing system with no web UI. Validation is performed via:
1. **pytest suite** - Automated unit and integration tests
2. **CLI dry-run commands** - Manual verification of pipeline outputs

### Validation Tools

| Tool | Surface | Usage |
|------|---------|-------|
| pytest | Unit/Integration | Automated test execution |
| Python CLI | Pipeline | Manual dry-run verification |
| sqlite3 | Database | Inspect stored predictions |
| cat/type | Logs | Review pipeline execution |

### Key Validation Commands

```bash
# Run full test suite
pytest tests/ -v --tb=short

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Daily pipeline dry-run
python -m src.pipeline.daily --date 2025-09-15 --mode backtest --dry-run

# Backtest execution
python -m src.backtest.run --start 2022-04-01 --end 2023-09-30

# Inspect predictions
sqlite3 data/mlb.db "SELECT * FROM predictions WHERE date='2025-09-15'"

# Inspect Discord payload (from dry-run output)
# Copy JSON from console and validate structure
```

## Validation Concurrency

**Max Concurrent Validators:** 5

**Rationale:**
- pytest workers are lightweight (~100-300 MB each)
- No browser instances needed
- System has 32 GB RAM, 24 CPUs, ~12 GB free
- 5 concurrent pytest processes use ~1.5 GB
- 70% headroom rule: 12 GB × 0.7 = 8.4 GB available

**Isolation Strategy:**
- Each validator runs pytest on separate test files
- No shared state between validators
- SQLite database is read-only during validation

## Resource Cost Classification

| Activity | Memory | CPU | Duration |
|----------|--------|-----|----------|
| pytest unit tests | ~100 MB | Low | 10-30 sec |
| pytest integration tests | ~200 MB | Medium | 30-60 sec |
| Feature computation (full) | ~500 MB | Medium | 2-5 min |
| Model training | ~1 GB | High | 10-30 min |
| Backtest (1 season) | ~1 GB | High | 15-60 min |
| Daily pipeline | ~300 MB | Low | 30-60 sec |

## Testing Infrastructure

### Test Fixtures
- `tests/fixtures/` - Mock API responses, sample data
- `data/test.db` - Isolated test database
- Environment: Use `.env.test` for test credentials (mock keys)

### Test Categories

| Category | Purpose | Location |
|----------|---------|----------|
| Data Integrity | Schema, constraints, completeness | `tests/test_data_*.py` |
| Anti-Leakage | No future data in features | `tests/test_antileak.py` |
| Feature Engineering | Value ranges, formulas | `tests/test_feature_*.py` |
| Financial Logic | Edge, Kelly, settlement | `tests/test_edge_*.py`, `tests/test_bankroll*.py`, `tests/test_settlement*.py` |
| Model | Training, calibration quality | `tests/test_model*.py` |
| Pipeline | End-to-end execution | `tests/test_pipeline*.py` |

## Coverage Requirements

| Area | Target | Critical Path |
|------|--------|---------------|
| Settlement logic | 90%+ | Yes |
| Edge calculation | 90%+ | Yes |
| Anti-leakage | 90%+ | Yes |
| Bankroll management | 90%+ | Yes |
| Feature engineering | 70%+ | No |
| Model training | 70%+ | No |
| Overall | 70%+ | - |
