from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd


def _zip_bytes(member_name: str, csv_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(member_name, csv_text)
    return buffer.getvalue()


def test_fetch_retrosheet_public_datasets_cache_and_derive_views(tmp_path: Path, monkeypatch) -> None:
    from src.clients import retrosheet_client

    payloads = {
        retrosheet_client._PUBLIC_DATASETS["gameinfo"][0]: _zip_bytes(
            "gameinfo.csv",
            "gid,visteam,hometeam,date,season,umphome,ump1b\n"
            "NYA202404010,BOS,NYY,20240401,2024,bluej901,westj902\n"
        ),
        retrosheet_client._PUBLIC_DATASETS["teamstats"][0]: _zip_bytes(
            "teamstats.csv",
            "gid,team,date,season,stattype,start_l1,start_l2,start_l3,start_l4,start_l5,start_l6,start_l7,start_l8,start_l9\n"
            "NYA202404010,NYY,20240401,2024,value,judge001,soto001,stant001,volpe001,rizzo001,torre001,verdu001,wells001,gleyb001\n"
            "NYA202404010,NYY,20240401,2024,official,judge001,soto001,stant001,volpe001,rizzo001,torre001,verdu001,wells001,gleyb001\n"
        ),
        retrosheet_client._PUBLIC_DATASETS["allplayers"][0]: _zip_bytes(
            "allplayers.csv",
            "id,last,first,team,season\njudge001,Judge,Aaron,NYY,2024\n"
        ),
    }
    calls: list[str] = []

    class _FakeResponse:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    def _fake_get(url: str, timeout: float):
        _ = timeout
        calls.append(url)
        return _FakeResponse(payloads[url])

    monkeypatch.setattr(retrosheet_client.requests, "get", _fake_get)

    gameinfo = retrosheet_client.fetch_retrosheet_gameinfo(raw_data_root=tmp_path)
    teamstats = retrosheet_client.fetch_retrosheet_teamstats(raw_data_root=tmp_path)
    allplayers = retrosheet_client.fetch_retrosheet_allplayers(raw_data_root=tmp_path)
    lineups = retrosheet_client.fetch_retrosheet_starting_lineups(
        season=2024,
        raw_data_root=tmp_path,
    )
    umpires = retrosheet_client.fetch_retrosheet_umpires(season=2024, raw_data_root=tmp_path)

    assert gameinfo.loc[0, "umphome"] == "bluej901"
    assert teamstats.loc[0, "team"] == "NYY"
    assert allplayers.loc[0, "id"] == "judge001"
    assert len(lineups) == 1
    assert lineups.loc[0, "start_l1"] == "judge001"
    assert umpires.loc[0, "ump1b"] == "westj902"
    assert (tmp_path / "retrosheet" / "gameinfo.parquet").exists()
    assert (tmp_path / "retrosheet" / "teamstats.parquet").exists()
    assert (tmp_path / "retrosheet" / "allplayers.parquet").exists()

    monkeypatch.setattr(
        retrosheet_client.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Retrosheet client should reuse parquet cache")
        ),
    )
    cached = retrosheet_client.fetch_retrosheet_gameinfo(raw_data_root=tmp_path)

    assert cached.equals(gameinfo)
    assert len(calls) == 3
