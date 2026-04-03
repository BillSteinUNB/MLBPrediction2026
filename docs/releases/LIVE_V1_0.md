# MLBPrediction2026 V1.0

## Release Identity

- Release name: `MLBPrediction2026 V1.0`
- Release version: `1.0.0`
- User-facing model name: `Precision Engine V1.0`
- User-facing strategy name: `Bet365 Full-Game Policy v1.0`
- Technical model artifact: `ml=20260325T005859Z_4c611698:base;rl=20260325T005859Z_4c611698:base`
- Research baseline label: `workingdb_2021_2025_flat_baseline`

## What V1.0 Includes

This first stable live version is the frozen-slate and tracking release of the project. It is the first version intended to be used from the dashboard as a daily operating workflow rather than only from research scripts.

V1.0 includes:

- one-click daily slate pull from the main app
- immutable same-day capture behavior
- automatic settlement of prior tracked bets from real game results
- a Bet365-only display and execution policy for live use
- full-game `ML`, `RL`, and `Total` support in the slate view
- three separate tracking ledgers:
  - `My Bets`
  - `Machine POTD`
  - `All Machine Picks`
- manual bet tracking directly from the slate and the Today Picks screen
- frozen official picks that do not retroactively change for accuracy

## Live Policy

The current live strategy is intentionally narrower than the raw research grid.

- official picks are `full-game only`
- official picks are Bet365-only for displayed book, odds, and line
- official edge window is `15% <= edge < 22.5%`
- overall allowed odds window is `-250` to `+180`
- `Run Line` is allowed in V1.0 and is no longer treated like the broken F5 run-line path
- `F5 RL` remains excluded from live use

## Why This Version Matters

Before this release, the project had several disconnected workflows:

- research runs in the terminal
- live capture logic
- historical backfills
- dashboard views

V1.0 is the first version where those pieces are tied together cleanly enough to be used as an everyday paper-trading workflow.

The app now supports this daily loop:

1. Pull today's slate
2. Freeze the machine's official picks once
3. View all machine opinions on the slate
4. Add your own bets manually with your own odds and unit sizes
5. Return the next day and settle yesterday's results automatically

## Research Basis Used For V1.0

The current live policy is based on the fast bankroll research pass run against the repaired historical odds archive.

Important conclusions used to define V1.0:

- betting every positive-EV angle was too loose
- multiple bets per game created unrealistic volume
- the best practical zone came from stricter edge filtering
- the strongest usable threshold area was around `15%+`
- the live app was aligned to the narrower working policy:
  - `15%` to `22.5%`

This is the reason the app now treats lower-edge games as visible slate opinions but not official picks.

## Data and Book Policy

V1.0 intentionally separates research data from live execution data.

Research can still use:

- merged historical archives
- OddsPortal-derived backfills
- broader bookmaker coverage

Live dashboard execution uses:

- Bet365-only visible lines
- Bet365-only official pick selection
- local scraper refresh on recent days when `Pull Today's Slate` is pressed

This distinction exists because the live display needs to reflect the book the user can actually bet.

## Tracking Rules

### Machine POTD

- frozen once per day
- uses the official filtered policy
- cannot be rewritten later for accuracy

### All Machine Picks

- shows the broader machine view
- useful for comparing what the model liked versus what the official policy allowed

### My Bets

- user-submitted only
- can be taken from any displayed slate opinion
- uses user-entered odds and user-entered unit size
- settles later against actual game results

## Freeze Behavior

V1.0 deliberately prevents backtracking.

- if today is already frozen, pulling again does not replace official picks
- started games do not have their captured line context overwritten
- results can update later
- historical picks do not change retroactively

This is one of the main design constraints of the release and is essential to honest tracking.

## Known Limitations In V1.0

- the project still has historical/research complexity outside the live app
- F5 RL is still excluded because that path is not trusted enough yet
- some live markets may still be missing on a given slate if Bet365 data is genuinely unavailable
- manual tracking is intended for practical use, not as an execution broker
- some older tests in the repo still reflect pre-V1.0 policy and may need later cleanup

## Versioning Intent

This release should be treated as the baseline live app release.

Future versions can iterate on:

- lineup-aware gating
- confidence presentation
- alternative policy profiles
- cleaner RL/Total display logic
- better release metadata and changelog structure

But `V1.0` is the first coherent working version of:

- the model display
- the daily slate pull
- the frozen pick workflow
- the tracking workflow
