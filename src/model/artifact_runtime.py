from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import sklearn
import xgboost


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VersionCompatibilityReport:
    compatible: bool
    warnings: list[str]
    errors: list[str]


def collect_runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "xgboost": xgboost.__version__,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }


def validate_runtime_versions(
    metadata_payload: Mapping[str, Any],
    *,
    artifact_path: str | Path,
    strict: bool = True,
) -> VersionCompatibilityReport:
    stored_versions_raw = metadata_payload.get("runtime_versions")
    if not isinstance(stored_versions_raw, Mapping):
        warning = (
            f"Artifact {artifact_path} is missing runtime_versions metadata; "
            "loading in compatibility fallback mode."
        )
        logger.warning(warning)
        return VersionCompatibilityReport(compatible=True, warnings=[warning], errors=[])

    stored_versions = {
        str(key): str(value)
        for key, value in stored_versions_raw.items()
        if value is not None
    }
    current_versions = collect_runtime_versions()

    warnings: list[str] = []
    errors: list[str] = []

    _compare_version_family(
        stored_versions,
        current_versions,
        package="python",
        compare_parts=2,
        errors=errors,
        warnings=warnings,
    )
    _compare_version_family(
        stored_versions,
        current_versions,
        package="xgboost",
        compare_parts=2,
        errors=errors,
        warnings=warnings,
    )
    _compare_version_family(
        stored_versions,
        current_versions,
        package="sklearn",
        compare_parts=2,
        errors=errors,
        warnings=warnings,
    )
    _compare_version_family(
        stored_versions,
        current_versions,
        package="pandas",
        compare_parts=1,
        errors=errors,
        warnings=warnings,
    )
    _compare_version_family(
        stored_versions,
        current_versions,
        package="numpy",
        compare_parts=1,
        errors=errors,
        warnings=warnings,
    )

    compatible = not errors
    if errors and strict:
        raise RuntimeError(
            f"Artifact {artifact_path} has incompatible runtime versions: {'; '.join(errors)}"
        )

    for message in warnings:
        logger.warning("Artifact %s compatibility warning: %s", artifact_path, message)
    for message in errors:
        logger.error("Artifact %s compatibility error: %s", artifact_path, message)

    return VersionCompatibilityReport(
        compatible=compatible,
        warnings=warnings,
        errors=errors,
    )


def _compare_version_family(
    stored_versions: Mapping[str, str],
    current_versions: Mapping[str, str],
    *,
    package: str,
    compare_parts: int,
    errors: list[str],
    warnings: list[str],
) -> None:
    stored = stored_versions.get(package)
    current = current_versions.get(package)
    if not stored or not current:
        warnings.append(f"{package} version information is incomplete")
        return

    stored_key = _version_key(stored, compare_parts)
    current_key = _version_key(current, compare_parts)
    if stored_key is None or current_key is None:
        warnings.append(f"{package} version could not be parsed (stored={stored}, current={current})")
        return
    if stored_key != current_key:
        errors.append(f"{package} stored={stored} current={current}")


def _version_key(version: str, compare_parts: int) -> tuple[int, ...] | None:
    parts: list[int] = []
    for raw_part in version.split("."):
        digits = "".join(character for character in raw_part if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) >= compare_parts:
            return tuple(parts[:compare_parts])
    if len(parts) >= compare_parts:
        return tuple(parts[:compare_parts])
    return None
