from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from src.dashboard.schemas import (
    BinItem,
    CompareResult,
    FeatureImportanceItem,
    Lane,
    OverviewResponse,
    Promotion,
    RunDetail,
    RunSummary,
)
from src.ops.experiment_report import build_experiment_metrics_dataframe


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/")


def _clean_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except Exception:
        pass
    return value


def _timestamp_key(value: str) -> tuple[int, str | float]:
    try:
        return (0, float(value))
    except Exception:
        return (1, str(value))


class ExperimentDataAdapter:
    def __init__(
        self, models_dir: str | Path | None = None, experiments_dir: str | Path | None = None
    ) -> None:
        self.models_dir = Path(models_dir) if models_dir is not None else Path("data") / "models"
        self.experiments_dir = (
            Path(experiments_dir) if experiments_dir is not None else Path("data") / "experiments"
        )
        self._runs_cache: dict[str, tuple[float | None, list[RunSummary]]] = {}
        self._catalog_cache: dict[str, tuple[float | None, dict[str, Any]]] = {}

    def _resolve_models_dir(self, models_dir: str | Path | None) -> Path:
        return Path(models_dir) if models_dir is not None else self.models_dir

    def _resolve_experiments_dir(
        self,
        experiments_dir: str | Path | None,
        models_dir: str | Path | None = None,
    ) -> Path:
        if experiments_dir is not None:
            return Path(experiments_dir)

        if models_dir is not None:
            resolved_models_dir = Path(models_dir)
            inferred_experiments_dir = resolved_models_dir.parent / "experiments"
            if self.experiments_dir == Path("data") / "experiments" and inferred_experiments_dir.exists():
                return inferred_experiments_dir

        return self.experiments_dir

    def _load_dashboard_catalog(self, experiments_dir: Path) -> dict[str, Any]:
        catalog_path = experiments_dir / "dashboard_catalog.json"
        cache_key = _normalize_path(str(catalog_path.resolve()))
        try:
            current_mtime = catalog_path.stat().st_mtime
        except OSError:
            current_mtime = None

        cached = self._catalog_cache.get(cache_key)
        if cached and cached[0] == current_mtime:
            return dict(cached[1])

        if not catalog_path.exists():
            payload: dict[str, Any] = {}
        else:
            raw_payload = self._safe_json_load(catalog_path)
            payload = raw_payload if isinstance(raw_payload, dict) else {}

        self._catalog_cache[cache_key] = (current_mtime, dict(payload))
        return payload

    def _safe_json_load(self, path: Path) -> dict[str, Any] | list[Any] | None:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed reading JSON file %s: %s", path, exc)
            return None

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping malformed JSON file %s: %s", path, exc)
            return None

    def _resolve_summary_path(self, models_dir: Path, summary_path: str | Path) -> Path | None:
        raw_value = unquote(str(summary_path)).strip()
        if not raw_value:
            return None

        requested_path = Path(raw_value)
        candidates: list[Path] = []

        if requested_path.is_absolute():
            candidates.append(requested_path)
        else:
            candidates.append(REPO_ROOT / requested_path)
            candidates.append(models_dir / requested_path)

            normalized_parts = [part for part in requested_path.parts if part not in (".", "")]
            if len(normalized_parts) >= 2 and normalized_parts[0] == "data" and normalized_parts[1] == "models":
                candidates.append(models_dir / Path(*normalized_parts[2:]))

        seen: set[str] = set()
        for candidate in candidates:
            normalized = _normalize_path(str(candidate))
            if normalized in seen:
                continue
            seen.add(normalized)
            if candidate.exists():
                return candidate

        return None

    def _candidate_summary_keys(self, models_dir: Path, requested_path: Path, raw_summary_path: str) -> set[str]:
        keys = {_normalize_path(raw_summary_path), _normalize_path(str(requested_path))}
        try:
            keys.add(_normalize_path(str(requested_path.resolve())))
        except OSError:
            pass

        try:
            keys.add(_normalize_path(str(requested_path.relative_to(REPO_ROOT))))
        except ValueError:
            pass

        try:
            keys.add(_normalize_path(str(requested_path.relative_to(models_dir))))
        except ValueError:
            pass

        return keys

    def get_all_runs(self, models_dir: str | Path | None = None) -> list[RunSummary]:
        resolved_models_dir = self._resolve_models_dir(models_dir)
        resolved_experiments_dir = self._resolve_experiments_dir(None, resolved_models_dir)
        if not resolved_models_dir.exists():
            return []

        cache_key = _normalize_path(str(resolved_models_dir.resolve()))
        try:
            current_mtime = resolved_models_dir.stat().st_mtime
        except OSError:
            current_mtime = None

        cached = self._runs_cache.get(cache_key)
        if cached and cached[0] == current_mtime:
            return list(cached[1])

        try:
            frame = build_experiment_metrics_dataframe(resolved_models_dir)
        except Exception as exc:
            logger.warning(
                "Unable to build experiment metrics for %s: %s", resolved_models_dir, exc
            )
            return []

        catalog = self._load_dashboard_catalog(resolved_experiments_dir)
        include_folders = {
            str(value)
            for value in (catalog.get("include_folders") or [])
            if isinstance(value, str) and value.strip()
        }
        aliases = catalog.get("aliases") if isinstance(catalog.get("aliases"), dict) else {}

        runs: list[RunSummary] = []
        for row in frame.to_dict("records"):
            payload = {
                field_name: _clean_value(row.get(field_name))
                for field_name in RunSummary.model_fields
            }
            summary_path = payload.get("summary_path")
            if isinstance(summary_path, str):
                normalized_summary = _normalize_path(summary_path)
                payload["summary_path"] = normalized_summary
                folder_name = Path(normalized_summary).parent.name
                if include_folders and folder_name not in include_folders:
                    continue
                alias = aliases.get(folder_name)
                if isinstance(alias, str) and alias.strip():
                    payload["experiment_name"] = alias.strip()
            try:
                runs.append(RunSummary.model_validate(payload))
            except Exception as exc:
                logger.warning("Skipping invalid run summary row: %s", exc)

        self._runs_cache[cache_key] = (current_mtime, list(runs))
        return runs

    def _select_model_payload(
        self,
        payload: dict[str, Any],
        run_summary: RunSummary | None,
    ) -> dict[str, Any] | None:
        models = payload.get("models") or {}
        if not isinstance(models, dict):
            return None
        if run_summary and run_summary.model_name in models:
            model_payload = models.get(run_summary.model_name)
            if isinstance(model_payload, dict):
                return model_payload
        if run_summary:
            for model_payload in models.values():
                if (
                    isinstance(model_payload, dict)
                    and model_payload.get("target_column") == run_summary.target_column
                ):
                    return model_payload
        for model_payload in models.values():
            if isinstance(model_payload, dict):
                return model_payload
        return None

    def _build_detail_bins(self, holdout_metrics: dict[str, Any]) -> list[BinItem] | None:
        raw_bins = holdout_metrics.get("reliability_diagram")
        if not isinstance(raw_bins, list):
            return None
        bins: list[BinItem] = []
        for raw_bin in raw_bins:
            if not isinstance(raw_bin, dict):
                continue
            predicted_mean = _clean_value(
                raw_bin.get("predicted_mean", raw_bin.get("mean_predicted_probability"))
            )
            true_fraction = _clean_value(
                raw_bin.get("true_fraction", raw_bin.get("empirical_positive_rate"))
            )
            count = _clean_value(raw_bin.get("count"))
            if predicted_mean is None or true_fraction is None or count is None:
                continue
            bins.append(
                BinItem(
                    bin_index=int(raw_bin.get("bin_index", len(bins))),
                    predicted_mean=float(predicted_mean),
                    true_fraction=float(true_fraction),
                    count=int(count),
                )
            )
        return bins or None

    def get_run_detail(
        self, models_dir: str | Path | None, summary_path: str | Path
    ) -> RunDetail | None:
        resolved_models_dir = self._resolve_models_dir(models_dir)
        requested_path = self._resolve_summary_path(resolved_models_dir, summary_path)
        if requested_path is None:
            return None

        runs = self.get_all_runs(resolved_models_dir)
        requested_keys = self._candidate_summary_keys(
            resolved_models_dir,
            requested_path,
            unquote(str(summary_path)),
        )
        run_summary = next(
            (run for run in runs if _normalize_path(run.summary_path) in requested_keys),
            None,
        )

        payload = self._safe_json_load(requested_path)
        if not isinstance(payload, dict):
            return None
        model_payload = self._select_model_payload(payload, run_summary)
        if not isinstance(model_payload, dict):
            return None

        if run_summary is None:
            return None

        holdout_metrics = model_payload.get("holdout_metrics")
        if not isinstance(holdout_metrics, dict):
            holdout_metrics = {}

        raw_importance = model_payload.get("feature_importance_rankings")
        feature_importance: list[FeatureImportanceItem] | None = None
        if isinstance(raw_importance, list):
            feature_importance = []
            for item in raw_importance:
                if not isinstance(item, dict):
                    continue
                feature = item.get("feature")
                importance = _clean_value(item.get("importance"))
                if feature is None or importance is None:
                    continue
                feature_importance.append(
                    FeatureImportanceItem(feature=str(feature), importance=float(importance))
                )
            if not feature_importance:
                feature_importance = None

        meta_feature_columns = model_payload.get("meta_feature_columns")
        if not isinstance(meta_feature_columns, list):
            meta_feature_columns = payload.get("raw_meta_feature_columns")
        if not isinstance(meta_feature_columns, list):
            meta_feature_columns = None

        detail_data = run_summary.model_dump()
        detail_data["summary_path"] = _normalize_path(detail_data["summary_path"])

        detail = RunDetail(
            **detail_data,
            feature_importance=feature_importance,
            best_params=model_payload.get("best_params")
            if isinstance(model_payload.get("best_params"), dict)
            else None,
            reliability_diagram=self._build_detail_bins(holdout_metrics),
            quality_gates=holdout_metrics.get("quality_gates")
            if isinstance(holdout_metrics.get("quality_gates"), dict)
            else None,
            meta_feature_columns=meta_feature_columns,
            calibration_method=str(
                model_payload.get("calibration_method") or payload.get("calibration_method") or ""
            )
            or None,
            train_row_count=_clean_value(
                model_payload.get("train_row_count", payload.get("model_training_row_count"))
            ),
            holdout_row_count=_clean_value(
                model_payload.get("holdout_row_count", payload.get("holdout_row_count"))
            ),
            stacking_metrics=holdout_metrics or None,
        )
        return detail

    def get_lane_runs(
        self,
        runs: list[RunSummary],
        lane_id: str,
        skip: int = 0,
        limit: int = 100,
    ) -> list[RunSummary]:
        try:
            holdout_raw, target_column, variant = lane_id.split(":", 2)
            holdout_season = int(holdout_raw)
        except ValueError:
            return []

        matched = [
            run
            for run in runs
            if run.holdout_season == holdout_season
            and run.target_column == target_column
            and run.variant == variant
        ]
        matched.sort(key=lambda run: _timestamp_key(run.run_timestamp), reverse=True)
        return matched[skip : skip + limit]

    def _pick_best_run(self, runs: list[RunSummary]) -> RunSummary | None:
        if not runs:
            return None

        for metric_name, reverse in (
            ("roc_auc", True),
            ("accuracy", True),
            ("log_loss", False),
            ("brier", False),
        ):
            candidates = [run for run in runs if getattr(run, metric_name) is not None]
            if not candidates:
                continue
            return sorted(
                candidates,
                key=lambda run: (
                    float(getattr(run, metric_name) or 0.0),
                    _timestamp_key(run.run_timestamp),
                ),
                reverse=reverse,
            )[0]

        return sorted(runs, key=lambda run: _timestamp_key(run.run_timestamp), reverse=True)[0]

    def get_lanes(self, runs: list[RunSummary]) -> list[Lane]:
        if not runs:
            return []

        grouped: dict[tuple[int, str, str], list[RunSummary]] = {}
        for run in runs:
            lane_key = (run.holdout_season, run.target_column, run.variant)
            grouped.setdefault(lane_key, []).append(run)

        lanes: list[Lane] = []
        for (holdout_season, target_column, variant), lane_runs in grouped.items():
            latest_run = sorted(
                lane_runs,
                key=lambda run: _timestamp_key(run.run_timestamp),
                reverse=True,
            )[0]
            best_run = self._pick_best_run(lane_runs)
            lanes.append(
                Lane(
                    lane_id=f"{holdout_season}:{target_column}:{variant}",
                    model_name=(
                        latest_run.model_name
                        if latest_run
                        else (best_run.model_name if best_run else "")
                    ),
                    variant=variant,
                    best_run=best_run,
                    latest_run=latest_run,
                )
            )

        return sorted(lanes, key=lambda lane: lane.lane_id)

    def get_overview(self, runs: list[RunSummary]) -> OverviewResponse:
        if not runs:
            return OverviewResponse(total_runs=0, active_lanes=0, recent_runs=[])

        lanes = self.get_lanes(runs)
        latest_run = sorted(runs, key=lambda run: _timestamp_key(run.run_timestamp), reverse=True)[
            0
        ]
        best_run = self._pick_best_run(runs)

        improvements = sorted(
            [run for run in runs if (run.delta_vs_prev_roc_auc or 0.0) > 0.0],
            key=lambda run: float(run.delta_vs_prev_roc_auc or 0.0),
            reverse=True,
        )[:5]
        regressions = sorted(
            [run for run in runs if (run.delta_vs_prev_roc_auc or 0.0) < 0.0],
            key=lambda run: float(run.delta_vs_prev_roc_auc or 0.0),
        )[:5]

        recent_runs: list[RunSummary] = []
        seen: set[str] = set()
        for run in improvements + regressions:
            if run.summary_path in seen:
                continue
            seen.add(run.summary_path)
            recent_runs.append(run)

        return OverviewResponse(
            total_runs=len(runs),
            active_lanes=len(lanes),
            best_run=best_run,
            latest_run=latest_run,
            recent_runs=recent_runs,
        )

    def compare_runs(self, run_a: RunSummary, run_b: RunSummary) -> CompareResult:
        metric_deltas: dict[str, float | None] = {}
        for metric_name in ("accuracy", "log_loss", "roc_auc", "brier", "ece", "reliability_gap"):
            left = getattr(run_a, metric_name)
            right = getattr(run_b, metric_name)
            if left is None or right is None:
                metric_deltas[metric_name] = None
            else:
                metric_deltas[metric_name] = float(right) - float(left)

        score_a = 0
        score_b = 0
        for metric_name, higher_is_better in (
            ("roc_auc", True),
            ("accuracy", True),
            ("log_loss", False),
            ("brier", False),
            ("ece", False),
            ("reliability_gap", False),
        ):
            left = getattr(run_a, metric_name)
            right = getattr(run_b, metric_name)
            if left is None or right is None:
                continue
            if left == right:
                continue
            if higher_is_better:
                if left > right:
                    score_a += 1
                else:
                    score_b += 1
            else:
                if left < right:
                    score_a += 1
                else:
                    score_b += 1

        winner = "tie"
        if score_a > score_b:
            winner = "a"
        elif score_b > score_a:
            winner = "b"

        return CompareResult(
            run_a_id=_normalize_path(run_a.summary_path),
            run_b_id=_normalize_path(run_b.summary_path),
            run_a=run_a,
            run_b=run_b,
            metric_deltas=metric_deltas,
            winner=winner,
        )

    def read_promotions(self, experiments_dir: str | Path | None = None) -> list[Promotion]:
        resolved_experiments_dir = self._resolve_experiments_dir(experiments_dir)
        promotions_path = resolved_experiments_dir / "promotions.json"

        if not promotions_path.exists():
            return []

        payload = self._safe_json_load(promotions_path)
        if not isinstance(payload, list):
            return []

        promotions: list[Promotion] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("run_id"), str):
                item = {**item, "run_id": _normalize_path(item["run_id"])}
            try:
                promotions.append(Promotion.model_validate(item))
            except Exception as exc:
                logger.warning("Skipping invalid promotion entry in %s: %s", promotions_path, exc)
        return promotions

    def write_promotion(
        self, experiments_dir: str | Path | None, promotion: Promotion
    ) -> Promotion:
        resolved_experiments_dir = self._resolve_experiments_dir(experiments_dir)
        resolved_experiments_dir.mkdir(parents=True, exist_ok=True)
        promotions_path = resolved_experiments_dir / "promotions.json"

        if not promotions_path.exists():
            promotions_path.write_text("[]", encoding="utf-8")

        existing = self.read_promotions(resolved_experiments_dir)
        normalized = Promotion.model_validate(
            {
                **promotion.model_dump(),
                "run_id": _normalize_path(promotion.run_id),
            }
        )
        existing.append(normalized)
        promotions_path.write_text(
            json.dumps([item.model_dump() for item in existing], indent=2),
            encoding="utf-8",
        )
        return normalized
