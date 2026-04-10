from __future__ import annotations

import base64
import logging
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests


logger = logging.getLogger(__name__)

EVA_STATCAST_CORRELATIONS_URL = "https://evanalytics.com/mlb/research/statcast-correlations"
EVA_DATATABLE_QUERY_URL = "https://evanalytics.com/admin/model/datatableQuery.php"
EVA_STATCAST_CORRELATIONS_DATATABLE_ID = 195
HTTP_TIMEOUT = 30.0
_COLUMN_RENAMES = {
    "wOBA": "woba_corr",
    "HRperc": "hr_pct_corr",
    "BABIP": "babip_corr",
    "ISO": "iso_corr",
    "BA": "ba_corr",
    "SBperc": "sb_pct_corr",
    "SBAperc": "sba_pct_corr",
}


def fetch_eva_statcast_correlations(
    *,
    timeout: float = HTTP_TIMEOUT,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch and normalize EVA Analytics' MLB Statcast correlation table."""

    encoded_parameter = base64.b64encode(
        urlencode(
            {
                "mode": "runTime",
                "dataTable_id": EVA_STATCAST_CORRELATIONS_DATATABLE_ID,
            }
        ).encode("utf-8")
    ).decode("ascii")
    post = session.post if session is not None else requests.post
    response = post(
        EVA_DATATABLE_QUERY_URL,
        data={"parameter": encoded_parameter},
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    if payload.get("error"):
        raise RuntimeError(f"EVA datatable returned an error: {payload['error']}")

    rows = payload.get("dataRows") or []
    normalized_rows: list[dict[str, object]] = []
    for row in rows:
        columns = row.get("columns") or {}
        stat_name = str(columns.get("Stat", "")).strip().upper()
        row_type = str(columns.get("Type", "")).strip().lower()
        if not stat_name or not row_type:
            continue

        normalized_row: dict[str, object] = {
            "stat": stat_name,
            "type": row_type,
            "source_page_url": EVA_STATCAST_CORRELATIONS_URL,
            "datatable_id": EVA_STATCAST_CORRELATIONS_DATATABLE_ID,
        }
        for source_name, target_name in _COLUMN_RENAMES.items():
            normalized_row[target_name] = pd.to_numeric(
                columns.get(source_name),
                errors="coerce",
            )
        normalized_rows.append(normalized_row)

    dataframe = pd.DataFrame(normalized_rows)
    if dataframe.empty:
        return pd.DataFrame(
            columns=[
                "stat",
                "type",
                *_COLUMN_RENAMES.values(),
                "source_page_url",
                "datatable_id",
            ]
        )

    ordered_columns = [
        "stat",
        "type",
        *_COLUMN_RENAMES.values(),
        "source_page_url",
        "datatable_id",
    ]
    dataframe = dataframe.loc[:, ordered_columns].copy()
    return dataframe.sort_values(["type", "stat"]).reset_index(drop=True)


def write_eva_statcast_correlations_snapshot(
    dataframe: pd.DataFrame,
    *,
    output_path: str | Path,
) -> Path:
    """Persist a normalized EVA Statcast snapshot for reproducible audits."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(path, index=False)
    logger.info("Wrote EVA Statcast correlation snapshot to %s", path)
    return path


__all__ = [
    "EVA_DATATABLE_QUERY_URL",
    "EVA_STATCAST_CORRELATIONS_DATATABLE_ID",
    "EVA_STATCAST_CORRELATIONS_URL",
    "HTTP_TIMEOUT",
    "fetch_eva_statcast_correlations",
    "write_eva_statcast_correlations_snapshot",
]
