# Data Layout

The repo keeps important data and research outputs on GitHub, but the folder is organized by purpose.

## Main Areas

- `training/`
  - training metadata and parquet-related support files.
- `reports/`
  - versioned research and reporting outputs that are intentionally visible on GitHub.
- `models/`
  - model artifacts and metadata snapshots.
- runtime database files in the root of `data/`
  - local SQLite databases used by the app and research workflows.

## Guidelines

- Keep human explanation in `docs/` and machine outputs in `data/`.
- Prefer stable folder names over one-off root files.
- Put disposable debug output, temp images, and local-only exports outside the versioned reporting structure when possible.
