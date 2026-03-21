from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_fetch_chadwick_register_and_names_cache_and_lookup(tmp_path: Path, monkeypatch) -> None:
    from src.clients import chadwick_client

    payloads = {
        f"{chadwick_client.CHADWICK_REGISTER_BASE_URL}/people-a.csv": (
            "key_person,key_uuid,key_mlbam,key_retro,key_bbref,key_fangraphs,name_last,name_first\n"
            "a0000001,uuid-a,592450,judgea001,judgeaa01,15640,Judge,Aaron\n"
        ),
        f"{chadwick_client.CHADWICK_REGISTER_BASE_URL}/names.csv": (
            "key_person,name_last,name_first,altname_type,altname_last,altname_first\n"
            "a0000001,Judge,Aaron,alias,Judge,Aaron\n"
        ),
    }
    calls: list[str] = []

    class _FakeResponse:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    def _fake_get(url: str, timeout: float):
        _ = timeout
        calls.append(url)
        return _FakeResponse(payloads[url])

    monkeypatch.setattr(chadwick_client.requests, "get", _fake_get)

    register = chadwick_client.fetch_chadwick_register(
        shards=["a"],
        raw_data_root=tmp_path,
    )
    names = chadwick_client.fetch_chadwick_names(raw_data_root=tmp_path)
    by_mlbam = chadwick_client.lookup_chadwick_register(
        mlbam_id=592450,
        raw_data_root=tmp_path,
    )
    by_retro = chadwick_client.lookup_chadwick_register(
        retrosheet_id="judgea001",
        raw_data_root=tmp_path,
    )

    assert register.loc[0, "name_last"] == "Judge"
    assert names.loc[0, "altname_type"] == "alias"
    assert by_mlbam.loc[0, "key_retro"] == "judgea001"
    assert by_retro.loc[0, "key_mlbam"] == "592450"
    assert (tmp_path / "chadwick" / "register" / "people-a.parquet").exists()
    assert (tmp_path / "chadwick" / "register" / "names.parquet").exists()

    monkeypatch.setattr(
        chadwick_client.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Chadwick client should reuse parquet cache")
        ),
    )
    cached = chadwick_client.fetch_chadwick_register(shards=["a"], raw_data_root=tmp_path)

    assert cached.equals(register)
    assert len(calls) == 2
