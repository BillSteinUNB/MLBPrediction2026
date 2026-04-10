from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _resolve_season_series(dataset: pd.DataFrame) -> pd.Series:
    if "season" in dataset.columns:
        return pd.to_numeric(dataset["season"], errors="coerce")
    return pd.to_datetime(dataset["game_date"], errors="coerce").dt.year


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write a parquet copy excluding one or more seasons.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-season", dest="exclude_seasons", action="append", type=int, default=[])
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    source_table = pq.read_table(input_path)
    source_metadata = dict(source_table.schema.metadata or {})
    dataset = source_table.to_pandas()
    season_values = _resolve_season_series(dataset)
    exclude_seasons = {int(value) for value in args.exclude_seasons}

    if exclude_seasons:
        filtered = dataset.loc[~season_values.isin(exclude_seasons)].copy()
    else:
        filtered = dataset.copy()

    if source_metadata:
        pandas_payload = json.loads(source_metadata.get(b"pandas", b"{}").decode("utf-8"))
        attributes = dict(pandas_payload.get("attributes") or {})
        run_count_schema = attributes.get("run_count_training_schema")
        if run_count_schema is not None:
            filtered.attrs["run_count_training_schema"] = run_count_schema
        for key in ("schema_name", "schema_version"):
            if key in attributes:
                filtered.attrs[key] = attributes[key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_table = pa.Table.from_pandas(filtered, preserve_index=False)
    if source_metadata:
        filtered_table = filtered_table.replace_schema_metadata(source_metadata)
    pq.write_table(filtered_table, output_path)

    removed_rows = int((season_values.isin(exclude_seasons)).sum()) if exclude_seasons else 0
    print(
        f"wrote={output_path} rows={len(filtered)} "
        f"removed_rows={removed_rows} excluded_seasons={sorted(exclude_seasons)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
