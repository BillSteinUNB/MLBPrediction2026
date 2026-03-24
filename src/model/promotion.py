from __future__ import annotations

from typing import Literal, Mapping


PromotionVariant = Literal["base", "stacking", "calibrated"]


def select_promoted_variant(
    metrics_by_variant: Mapping[PromotionVariant, Mapping[str, float | None]],
) -> PromotionVariant:
    available = [
        variant
        for variant, metrics in metrics_by_variant.items()
        if metrics.get("log_loss") is not None
    ]
    if not available:
        raise ValueError("At least one candidate variant with log_loss is required")

    return min(
        available,
        key=lambda variant: (
            float(metrics_by_variant[variant]["log_loss"]),
            float("inf")
            if metrics_by_variant[variant].get("brier") is None
            else float(metrics_by_variant[variant]["brier"]),
            -(
                float(metrics_by_variant[variant]["roc_auc"])
                if metrics_by_variant[variant].get("roc_auc") is not None
                else float("-inf")
            ),
            -(
                float(metrics_by_variant[variant]["accuracy"])
                if metrics_by_variant[variant].get("accuracy") is not None
                else float("-inf")
            ),
            _promotion_variant_priority(variant),
        ),
    )


def build_promotion_reason(
    *,
    promoted_variant: PromotionVariant,
    metrics_by_variant: Mapping[PromotionVariant, Mapping[str, float | None]],
) -> str:
    promoted_metrics = metrics_by_variant[promoted_variant]
    return (
        f"Promoted {promoted_variant} on holdout log_loss="
        f"{float(promoted_metrics['log_loss']):.6f}"
    )


def _promotion_variant_priority(variant: PromotionVariant) -> int:
    if variant == "base":
        return 0
    if variant == "calibrated":
        return 1
    return 2
