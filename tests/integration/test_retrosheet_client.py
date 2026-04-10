from __future__ import annotations

import io
import sqlite3
import zipfile
from pathlib import Path


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
        retrosheet_client.RETROSHEET_GAME_LOG_URL_TEMPLATE.format(season=2024): _zip_bytes(
            "gl2024.txt",
            (
                '"20240401","0","Mon","BOS","AL",1,"NYA","AL",1,3,5,54,"N","","","","NYC01",'
                '41000,185,"000000300","20000003X",27,8,1,0,3,7,1,1,2,10,0,9,1,0,0,0,4,7,5,'
                '5,1,0,27,10,1,0,2,0,31,9,2,0,2,8,0,0,0,2,0,10,0,0,3,0,5,6,3,3,0,1,27,6,0,0,'
                '0,0,"westj902","Will Little","gibsh902","Tripp Gibson"\n'
            ),
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
    game_logs = retrosheet_client.fetch_retrosheet_game_logs(
        season=2024,
        db_path=tmp_path / "mlb.db",
    )
    umpires = retrosheet_client.fetch_retrosheet_umpires(season=2024, db_path=tmp_path / "mlb.db")

    assert gameinfo.loc[0, "umphome"] == "bluej901"
    assert teamstats.loc[0, "team"] == "NYY"
    assert allplayers.loc[0, "id"] == "judge001"
    assert len(lineups) == 1
    assert lineups.loc[0, "start_l1"] == "judge001"
    assert game_logs.loc[0, "matchup_sequence"] == 1
    assert umpires.loc[0, "umphome"] == "westj902"
    assert (tmp_path / "retrosheet" / "gameinfo.parquet").exists()
    assert (tmp_path / "retrosheet" / "teamstats.parquet").exists()
    assert (tmp_path / "retrosheet" / "allplayers.parquet").exists()
    with sqlite3.connect(tmp_path / "mlb.db") as connection:
        cached_rows = connection.execute(
            "SELECT COUNT(*) FROM retrosheet_game_logs WHERE season = 2024"
        ).fetchone()[0]
    assert cached_rows == 1

    monkeypatch.setattr(
        retrosheet_client.requests,
        "get",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Retrosheet client should reuse parquet cache")
        ),
    )
    cached = retrosheet_client.fetch_retrosheet_gameinfo(raw_data_root=tmp_path)
    cached_umpires = retrosheet_client.fetch_retrosheet_umpires(
        season=2024,
        db_path=tmp_path / "mlb.db",
    )

    assert cached.equals(gameinfo)
    assert cached_umpires.equals(umpires)
    assert len(calls) == 4
