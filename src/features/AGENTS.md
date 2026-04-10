# src/features — Feature Engineering

**8 modules + adjustments/ subpackage.** Builds sabermetric feature frames for each game.

## WHERE TO LOOK

| Feature Family | File | Key Metrics |
|----------------|------|-------------|
| Offense | `offense.py` | wRC+, wOBA, OPS+, BABIP, barrel rate |
| Pitching | `pitching.py` | xFIP, SIERA, K-BB%, swinging strike |
| Bullpen | `bullpen.py` | BP ERA, FIP, fatigue index, usage patterns |
| Defense | `defense.py` | DRS, UZR, OAA, fielding runs |
| Baselines | `baselines.py` | Season-level averages, regression targets |
| Marcel blending | `marcel_blend.py` | Regression-to-mean weights (from settings.yaml) |
| Umpires | `umpires.py` | umpire zone tendency adjustments |

## adjustments/ Subpackage

| File | Purpose |
|------|---------|
| `park_factors.py` | Stadium park factor adjustments (from config/settings.yaml) |
| `weather.py` | Temperature, wind, humidity, dome adjustments |
| `abs_adjustment.py` | Automatic Ball-Strike system proxy features |

## FEATURE COMPOSITION

```
offense.py + pitching.py + bullpen.py + defense.py
       ↓           ↓           ↓            ↓
   marcel_blend.py (regression to mean)
       ↓
   adjustments/ (park, weather, ABS)
       ↓
   baselines.py (final baseline features)
       ↓
   → src/model/data_builder.py (assembles training frame)
```

## CONVENTIONS

- Multi-window rolling stats (7/14/28/56 day windows)
- Marcel blending weights defined in `config/settings.yaml`
- All features must be anti-leakage safe (only pre-game data)
- Park factors loaded from `config/settings.yaml` — never hardcoded

## ANTI-PATTERNS

- Do NOT infer event-level challenge counts from ABS proxy features
- Do NOT use future data in rolling windows (check date boundaries)
