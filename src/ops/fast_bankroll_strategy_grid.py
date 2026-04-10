from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAST_BANKROLL_DIR = Path("data/reports/run_count/fast_bankroll")
DEFAULT_FAST_BANKROLL_GRID_DIR = Path("data/reports/run_count/fast_bankroll_grid")
DEFAULT_STANDARD_EDGE_PCTS = (0.03, 0.04, 0.05, 0.06)
DEFAULT_ODDS_WINDOWS = (
    (-300, 200),
    (-250, 180),
    (-220, 160),
    (-200, 150),
    (-175, 135),
    (-150, 120),
    (-140, 110),
    (-130, 100),
)
DEFAULT_MIN_STAKE_UNITS = 0.5
DEFAULT_MAX_STAKE_UNITS = 5.0
DEFAULT_STAKE_ROUND_STEP = 0.25
DEFAULT_FULL_GAME_MARKETS = ("full_game_ml", "full_game_rl", "full_game_total")


@dataclass(frozen=True, slots=True)
class StrategyVariant:
    name: str
    min_odds: int
    max_odds: int
    standard_edge_pct: float


@dataclass(frozen=True, slots=True)
class StrategyGridResult:
    output_dir: Path
    summary_csv_path: Path
    summary_json_path: Path


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


def resolve_latest_fast_bankroll_bets_csv(root: str | Path = DEFAULT_FAST_BANKROLL_DIR) -> Path:
    resolved_root = _resolve_path(root)
    candidates = sorted(
        resolved_root.rglob("bankroll_bets.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No bankroll_bets.csv files found under {resolved_root}")
    return candidates[0]


def build_default_variants(
    *,
    standard_edge_pcts: Sequence[float] = DEFAULT_STANDARD_EDGE_PCTS,
    odds_windows: Sequence[tuple[int, int]] = DEFAULT_ODDS_WINDOWS,
) -> list[StrategyVariant]:
    variants: list[StrategyVariant] = []
    for standard_edge_pct in standard_edge_pcts:
        edge_label = int(round(float(standard_edge_pct) * 1000))
        for min_odds, max_odds in odds_windows:
            variants.append(
                StrategyVariant(
                    name=f"e{edge_label}_{min_odds}_{max_odds}",
                    min_odds=int(min_odds),
                    max_odds=int(max_odds),
                    standard_edge_pct=float(standard_edge_pct),
                )
            )
    return variants


def _edge_curve_units(edge_pct: float, standard_edge_pct: float) -> float:
    ratio = float(edge_pct) / max(float(standard_edge_pct), 1e-9)
    if ratio < 0.5:
        return 0.5
    if ratio < 0.8:
        return 0.75
    if ratio < 1.15:
        return 1.0
    if ratio < 1.4:
        return 1.5
    if ratio < 1.7:
        return 2.0
    if ratio < 2.0:
        return 3.0
    return 5.0


def _odds_risk_multiplier(american_odds: int | float) -> float:
    odds = float(american_odds)
    if odds < -250:
        return 0.55
    if odds < -200:
        return 0.65
    if odds < -150:
        return 0.8
    if odds <= 120:
        return 1.0
    if odds <= 150:
        return 0.85
    if odds <= 200:
        return 0.7
    return 0.55


def _round_to_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def size_bet_units(
    *,
    edge_pct: float,
    standard_edge_pct: float,
    american_odds: int | float,
    min_stake_units: float = DEFAULT_MIN_STAKE_UNITS,
    max_stake_units: float = DEFAULT_MAX_STAKE_UNITS,
    round_step: float = DEFAULT_STAKE_ROUND_STEP,
) -> float:
    raw_units = _edge_curve_units(edge_pct, standard_edge_pct) * _odds_risk_multiplier(american_odds)
    rounded_units = _round_to_step(raw_units, round_step)
    return float(max(min_stake_units, min(max_stake_units, rounded_units)))


def _bucket_series(values: pd.Series, bins: Sequence[float], labels: Sequence[str]) -> pd.Series:
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False)


def _write_variant_diagnostics(variant_df: pd.DataFrame, diagnostics_dir: Path, variant_name: str) -> None:
    market_summary = (
        variant_df.groupby("market_type")
        .agg(
            bet_count=("game_pk", "size"),
            total_stake_units=("stake_units", "sum"),
            net_units=("scaled_profit_units", "sum"),
            roi=("scaled_profit_units", lambda s: float(s.sum() / variant_df.loc[s.index, "stake_units"].sum())),
            avg_edge_pct=("edge_pct", "mean"),
            avg_ev=("ev", "mean"),
            avg_odds=("odds_at_bet", "mean"),
        )
        .reset_index()
    )
    market_summary.to_csv(diagnostics_dir / f"{variant_name}.market_summary.csv", index=False)

    odds_bucket_labels = ["<=-250", "-249:-201", "-200:-151", "-150:120", "121:150", "151:200"]
    odds_bucket_bins = [-10000, -250, -200, -150, 121, 151, 10000]
    odds_buckets = _bucket_series(variant_df["odds_at_bet"], odds_bucket_bins, odds_bucket_labels)
    odds_summary = (
        variant_df.assign(odds_bucket=odds_buckets)
        .groupby("odds_bucket", dropna=False)
        .agg(
            bet_count=("game_pk", "size"),
            total_stake_units=("stake_units", "sum"),
            net_units=("scaled_profit_units", "sum"),
            avg_edge_pct=("edge_pct", "mean"),
            avg_ev=("ev", "mean"),
        )
        .reset_index()
    )
    odds_summary.to_csv(diagnostics_dir / f"{variant_name}.odds_bucket_summary.csv", index=False)

    edge_bucket_labels = ["<2%", "2-4%", "4-6%", "6-8%", "8-12%", "12%+"]
    edge_bucket_bins = [-1.0, 0.02, 0.04, 0.06, 0.08, 0.12, 10.0]
    edge_buckets = _bucket_series(variant_df["edge_pct"], edge_bucket_bins, edge_bucket_labels)
    edge_summary = (
        variant_df.assign(edge_bucket=edge_buckets)
        .groupby("edge_bucket", dropna=False)
        .agg(
            bet_count=("game_pk", "size"),
            total_stake_units=("stake_units", "sum"),
            net_units=("scaled_profit_units", "sum"),
            avg_odds=("odds_at_bet", "mean"),
            win_rate=("bet_result", lambda s: float((s == "WIN").mean()) if len(s) else math.nan),
        )
        .reset_index()
    )
    edge_summary.to_csv(diagnostics_dir / f"{variant_name}.edge_bucket_summary.csv", index=False)

    haywire_cols = [
        "game_date",
        "game_pk",
        "market_type",
        "side",
        "odds_at_bet",
        "line_at_bet",
        "edge_pct",
        "ev",
        "stake_units",
        "scaled_profit_units",
    ]
    worst_losses = (
        variant_df.loc[variant_df["scaled_profit_units"] < 0]
        .sort_values(["ev", "edge_pct"], ascending=False)
        .head(50)[haywire_cols]
    )
    worst_losses.to_csv(diagnostics_dir / f"{variant_name}.worst_high_ev_losses.csv", index=False)


def _filter_base_bets(
    *,
    base_bets: pd.DataFrame,
    include_markets: Sequence[str] | None,
    rl_coinflip_min_odds: int | None,
    rl_coinflip_max_odds: int | None,
) -> pd.DataFrame:
    filtered = base_bets.copy()
    if include_markets:
        filtered = filtered.loc[filtered["market_type"].isin([str(market) for market in include_markets])].copy()
    if rl_coinflip_min_odds is not None or rl_coinflip_max_odds is not None:
        rl_mask = filtered["market_type"].eq("full_game_rl")
        rl_odds = pd.to_numeric(filtered["odds_at_bet"], errors="coerce")
        if rl_coinflip_min_odds is not None:
            rl_mask &= rl_odds >= int(rl_coinflip_min_odds)
        if rl_coinflip_max_odds is not None:
            rl_mask &= rl_odds <= int(rl_coinflip_max_odds)
        filtered = filtered.loc[~filtered["market_type"].eq("full_game_rl") | rl_mask].copy()
    return filtered


def _simulate_variant(
    *,
    base_bets: pd.DataFrame,
    variant: StrategyVariant,
    starting_bankroll_units: float,
) -> dict[str, Any]:
    filtered = base_bets.loc[
        (pd.to_numeric(base_bets["odds_at_bet"], errors="coerce") >= int(variant.min_odds))
        & (pd.to_numeric(base_bets["odds_at_bet"], errors="coerce") <= int(variant.max_odds))
    ].copy()
    filtered = filtered.sort_values(["game_date", "game_pk", "market_type", "side"]).reset_index(drop=True)
    if filtered.empty:
        return {
            "variant_name": variant.name,
            "min_odds": variant.min_odds,
            "max_odds": variant.max_odds,
            "standard_edge_pct": variant.standard_edge_pct,
            "bet_count": 0,
            "total_stake_units": 0.0,
            "avg_stake_units": None,
            "median_stake_units": None,
            "avg_edge_pct": None,
            "median_edge_pct": None,
            "avg_ev": None,
            "median_ev": None,
            "avg_odds": None,
            "net_units": 0.0,
            "roi": None,
            "ending_bankroll_units": float(starting_bankroll_units),
            "peak_bankroll_units": float(starting_bankroll_units),
            "max_drawdown_pct": 0.0,
        }, filtered

    filtered["stake_units"] = filtered.apply(
        lambda row: size_bet_units(
            edge_pct=float(row["edge_pct"]),
            standard_edge_pct=float(variant.standard_edge_pct),
            american_odds=float(row["odds_at_bet"]),
        ),
        axis=1,
    )
    filtered["scaled_profit_units"] = (
        pd.to_numeric(filtered["profit_units"], errors="coerce").fillna(0.0)
        * pd.to_numeric(filtered["stake_units"], errors="coerce").fillna(0.0)
    )

    bankroll = float(starting_bankroll_units)
    peak_bankroll = float(starting_bankroll_units)
    max_drawdown_pct = 0.0
    bankroll_after: list[float] = []
    drawdowns: list[float] = []
    for profit in filtered["scaled_profit_units"].tolist():
        bankroll = bankroll + float(profit)
        peak_bankroll = max(peak_bankroll, bankroll)
        drawdown_pct = (peak_bankroll - bankroll) / peak_bankroll if peak_bankroll > 0 else 0.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
        bankroll_after.append(bankroll)
        drawdowns.append(drawdown_pct)
    filtered["bankroll_after_units"] = bankroll_after
    filtered["bankroll_drawdown_pct"] = drawdowns

    total_stake_units = float(filtered["stake_units"].sum())
    net_units = float(filtered["scaled_profit_units"].sum())
    roi = float(net_units / total_stake_units) if total_stake_units > 0 else None
    summary = {
        "variant_name": variant.name,
        "min_odds": variant.min_odds,
        "max_odds": variant.max_odds,
        "standard_edge_pct": float(variant.standard_edge_pct),
        "bet_count": int(len(filtered)),
        "game_count": int(filtered["game_pk"].nunique()),
        "avg_bets_per_game": float(len(filtered) / max(filtered["game_pk"].nunique(), 1)),
        "total_stake_units": total_stake_units,
        "avg_stake_units": float(filtered["stake_units"].mean()),
        "median_stake_units": float(filtered["stake_units"].median()),
        "avg_edge_pct": float(filtered["edge_pct"].mean()),
        "median_edge_pct": float(filtered["edge_pct"].median()),
        "avg_ev": float(filtered["ev"].mean()),
        "median_ev": float(filtered["ev"].median()),
        "avg_odds": float(filtered["odds_at_bet"].mean()),
        "net_units": net_units,
        "roi": roi,
        "ending_bankroll_units": float(bankroll),
        "peak_bankroll_units": float(peak_bankroll),
        "max_drawdown_pct": float(max_drawdown_pct),
        "win_rate": float((filtered["bet_result"] == "WIN").mean()) if len(filtered) else None,
    }
    return summary, filtered


def run_fast_bankroll_strategy_grid(
    *,
    bets_csv_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_FAST_BANKROLL_GRID_DIR,
    starting_bankroll_units: float = 100.0,
    standard_edge_pcts: Sequence[float] = DEFAULT_STANDARD_EDGE_PCTS,
    odds_windows: Sequence[tuple[int, int]] = DEFAULT_ODDS_WINDOWS,
    include_markets: Sequence[str] | None = None,
    rl_coinflip_min_odds: int | None = None,
    rl_coinflip_max_odds: int | None = None,
) -> StrategyGridResult:
    resolved_bets_csv = (
        _resolve_path(bets_csv_path) if bets_csv_path is not None else resolve_latest_fast_bankroll_bets_csv()
    )
    base_bets = pd.read_csv(resolved_bets_csv)
    if base_bets.empty:
        raise ValueError(f"No bets found in {resolved_bets_csv}")
    base_bets = _filter_base_bets(
        base_bets=base_bets,
        include_markets=include_markets,
        rl_coinflip_min_odds=rl_coinflip_min_odds,
        rl_coinflip_max_odds=rl_coinflip_max_odds,
    )
    if base_bets.empty:
        raise ValueError("No bets remain after applying market and RL odds filters.")

    variants = build_default_variants(
        standard_edge_pcts=standard_edge_pcts,
        odds_windows=odds_windows,
    )

    resolved_output_dir = _resolve_path(output_dir)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = resolved_output_dir / f"grid_{stamp}"
    diagnostics_dir = run_dir / "diagnostics"
    run_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        summary, variant_df = _simulate_variant(
            base_bets=base_bets,
            variant=variant,
            starting_bankroll_units=float(starting_bankroll_units),
        )
        summary_rows.append(summary)
        if not variant_df.empty:
            _write_variant_diagnostics(variant_df, diagnostics_dir, variant.name)

    summary_frame = pd.DataFrame(summary_rows).sort_values(
        ["roi", "net_units", "bet_count"],
        ascending=[False, False, False],
        na_position="last",
    )
    summary_csv_path = run_dir / "strategy_grid_summary.csv"
    summary_json_path = run_dir / "strategy_grid_summary.json"
    summary_frame.to_csv(summary_csv_path, index=False)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_bets_csv": str(resolved_bets_csv),
        "starting_bankroll_units": float(starting_bankroll_units),
        "include_markets": list(include_markets) if include_markets is not None else None,
        "rl_coinflip_min_odds": rl_coinflip_min_odds,
        "rl_coinflip_max_odds": rl_coinflip_max_odds,
        "variants": [asdict(variant) for variant in variants],
        "top_by_roi": summary_frame.head(10).to_dict(orient="records"),
        "top_by_net_units": summary_frame.sort_values("net_units", ascending=False).head(10).to_dict(orient="records"),
        "summary_rows": summary_frame.to_dict(orient="records"),
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return StrategyGridResult(
        output_dir=run_dir,
        summary_csv_path=summary_csv_path,
        summary_json_path=summary_json_path,
    )


__all__ = [
    "DEFAULT_FAST_BANKROLL_GRID_DIR",
    "DEFAULT_FULL_GAME_MARKETS",
    "DEFAULT_ODDS_WINDOWS",
    "DEFAULT_STANDARD_EDGE_PCTS",
    "StrategyGridResult",
    "build_default_variants",
    "resolve_latest_fast_bankroll_bets_csv",
    "run_fast_bankroll_strategy_grid",
    "size_bet_units",
]
