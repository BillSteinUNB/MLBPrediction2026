# Repo Hygiene

This repo keeps important research outputs on GitHub, so organization has to do the work that `.gitignore` cannot.

## Rules

- `docs/` is curated.
  - Active runbooks, handoffs, repo maps, and human-facing instructions live here.
- `archive/` is frozen.
  - Historical, superseded, or preserved-only material moves here.
- No ad hoc root markdown files.
  - If a new note is worth keeping, put it in `docs/` or `archive/` immediately.
- Keep the repo root strict.
  - Only long-lived entrypoints belong at the root: `src/`, `dashboard/`, `tests/`, `scripts/`, `docs/`, `config/`, `data/`, `archive/`, and core project files such as `README.md`, `AGENTS.md`, `pyproject.toml`, and `.gitignore`.
- Treat `docs/repo/REPO_MAP.md` as the authoritative paths document.
  - Update it whenever folders move or major repo-entrypoint paths change.

## Archive Policy

Use the existing archive buckets unless there is a strong reason not to:

- `archive/logs/`
- `archive/experiments/`
- `archive/subprojects/`
- `archive/repo_root_scratch/`

Avoid inventing new archive buckets casually. If one is necessary, add it to `archive/README.md` and `docs/repo/REPO_MAP.md`.

## Data Policy

- Versioned, intentionally visible data stays under `data/`.
- Local caches, temp state, DB sidecars, and runtime spill stay ignored.
- Prefer scoped ignores plus explicit allowlists for stable tracked artifacts.

## Integrity Check

Run the docs integrity check after repo reorganizations or link-heavy doc edits:

```bash
python scripts/check_docs_integrity.py
```
