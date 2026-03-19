from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_fetch_statcast_range_saves_daily_parquet_and_only_fetches_missing_dates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.clients import statcast_client

    calls: list[tuple[str, str]] = []

    def fake_statcast(
        start_dt: str,
        end_dt: str,
        team: str | None = None,
        verbose: bool = True,
        parallel: bool = True,
    ) -> pd.DataFrame:
        calls.append((start_dt, end_dt))
        return pd.DataFrame(
            {
                "game_date": [start_dt],
                "game_pk": [int(start_dt.replace("-", ""))],
                "launch_speed": [101.2],
                "release_speed": [95.4],
            }
        )

    monkeypatch.setattr(statcast_client, "statcast", fake_statcast)

    first = statcast_client.fetch_statcast_range(
        "2025-09-01",
        "2025-09-02",
        raw_data_root=tmp_path,
    )

    assert len(first) == 2
    assert (tmp_path / "statcast" / "statcast_2025-09-01.parquet").exists()
    assert (tmp_path / "statcast" / "statcast_2025-09-02.parquet").exists()
    assert statcast_client.pybaseball_cache.config.enabled is True
    assert statcast_client.pybaseball_cache.config.cache_type == "parquet"
    assert Path(statcast_client.pybaseball_cache.config.cache_directory) == tmp_path / "pybaseball_cache"

    second = statcast_client.fetch_statcast_range(
        "2025-09-01",
        "2025-09-03",
        raw_data_root=tmp_path,
    )

    assert len(second) == 3
    assert calls == [
        ("2025-09-01", "2025-09-01"),
        ("2025-09-02", "2025-09-02"),
        ("2025-09-03", "2025-09-03"),
    ]


def test_fetch_batting_and_pitching_stats_persist_parquet_with_advanced_metrics(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.clients import statcast_client

    batting_calls = {"count": 0}
    pitching_calls = {"count": 0}

    def fake_batting_stats(season: int, qual: int) -> pd.DataFrame:
        batting_calls["count"] += 1
        assert season == 2025
        assert qual == 50
        return pd.DataFrame({"Name": ["Juan Soto"], "wRC+": [170], "wOBA": [0.418]})

    def fake_pitching_stats(season: int, qual: int) -> pd.DataFrame:
        pitching_calls["count"] += 1
        assert season == 2025
        assert qual == 20
        return pd.DataFrame({"Name": ["Gerrit Cole"], "xFIP": [3.24], "K%": [0.29]})

    monkeypatch.setattr(statcast_client, "batting_stats", fake_batting_stats)
    monkeypatch.setattr(statcast_client, "pitching_stats", fake_pitching_stats)

    batting_df = statcast_client.fetch_batting_stats(2025, raw_data_root=tmp_path)
    pitching_df = statcast_client.fetch_pitcher_stats(2025, raw_data_root=tmp_path)

    assert list(batting_df[["wRC+", "wOBA"]].iloc[0]) == [170, 0.418]
    assert pitching_df.loc[0, "xFIP"] == 3.24
    assert (tmp_path / "fangraphs" / "batting_2025_min_pa_50.parquet").exists()
    assert (tmp_path / "fangraphs" / "pitching_2025_min_ip_20.parquet").exists()

    monkeypatch.setattr(statcast_client, "batting_stats", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("batting_stats should not refetch when parquet exists")))
    monkeypatch.setattr(statcast_client, "pitching_stats", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("pitching_stats should not refetch when parquet exists")))

    cached_batting_df = statcast_client.fetch_batting_stats(2025, raw_data_root=tmp_path)
    cached_pitching_df = statcast_client.fetch_pitcher_stats(2025, raw_data_root=tmp_path)

    assert batting_calls["count"] == 1
    assert pitching_calls["count"] == 1
    assert cached_batting_df.equals(batting_df)
    assert cached_pitching_df.equals(pitching_df)


def test_fetch_fielding_stats_merges_fangraphs_drs_with_statcast_oaa(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.clients import statcast_client

    positions_requested: list[int] = []

    def fake_fielding_stats(season: int, qual: int | None = None) -> pd.DataFrame:
        assert season == 2025
        assert qual == 0
        return pd.DataFrame(
            {
                "Name": ["Anthony Volpe", "Alex Verdugo"],
                "Team": ["NYY", "NYY"],
                "DRS": [12, 4],
            }
        )

    def fake_oaa(
        year: int,
        pos: int,
        min_att: int | str = "q",
        view: str = "Fielder",
    ) -> pd.DataFrame:
        positions_requested.append(pos)
        assert year == 2025
        assert min_att == 0
        assert view == "Fielder"
        return pd.DataFrame(
            {
                "name": ["Anthony Volpe", "Alex Verdugo"],
                "team": ["NYY", "NYY"],
                "outs_above_average": [5, 1],
            }
        )

    monkeypatch.setattr(statcast_client, "fielding_stats", fake_fielding_stats)
    monkeypatch.setattr(statcast_client, "statcast_outs_above_average", fake_oaa)

    result = statcast_client.fetch_fielding_stats(2025, raw_data_root=tmp_path)

    assert result.loc[result["Name"] == "Anthony Volpe", "OAA"].item() == 35
    assert result.loc[result["Name"] == "Alex Verdugo", "OAA"].item() == 7
    assert sorted(positions_requested) == [3, 4, 5, 6, 7, 8, 9]
    assert (tmp_path / "fielding" / "fielding_2025.parquet").exists()


def test_fetch_fielding_stats_regenerates_stale_cache_without_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.clients import statcast_client

    stale_path = tmp_path / "fielding" / "fielding_2025.parquet"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "Name": ["Qualified Only"],
            "Team": ["NYY"],
            "DRS": [1],
            "OAA": [2],
        }
    ).to_parquet(stale_path, index=False)

    calls = {"fielding": 0, "oaa": 0}

    def fake_fielding_stats(season: int, qual: int | None = None) -> pd.DataFrame:
        calls["fielding"] += 1
        assert season == 2025
        assert qual == 0
        return pd.DataFrame({"Name": ["Anthony Volpe"], "Team": ["NYY"], "DRS": [12]})

    def fake_oaa(
        year: int,
        pos: int,
        min_att: int | str = "q",
        view: str = "Fielder",
    ) -> pd.DataFrame:
        calls["oaa"] += 1
        assert year == 2025
        assert min_att == 0
        assert view == "Fielder"
        return pd.DataFrame(
            {
                "name": ["Anthony Volpe"],
                "team": ["NYY"],
                "outs_above_average": [1],
            }
        )

    monkeypatch.setattr(statcast_client, "fielding_stats", fake_fielding_stats)
    monkeypatch.setattr(statcast_client, "statcast_outs_above_average", fake_oaa)

    refreshed = statcast_client.fetch_fielding_stats(2025, raw_data_root=tmp_path)

    assert calls == {"fielding": 1, "oaa": 7}
    assert refreshed["Name"].tolist() == ["Anthony Volpe"]
    assert refreshed.loc[0, "OAA"] == 7

    metadata = json.loads((tmp_path / "fielding" / "fielding_2025.metadata.json").read_text())
    assert metadata == {
        "cache_version": statcast_client.FIELDING_CACHE_VERSION,
        "fielding_qual": 0,
        "oaa_min_att": 0,
        "oaa_positions": [3, 4, 5, 6, 7, 8, 9],
    }

    monkeypatch.setattr(
        statcast_client,
        "fielding_stats",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fielding_stats should use fresh cache metadata")
        ),
    )
    monkeypatch.setattr(
        statcast_client,
        "statcast_outs_above_average",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("statcast_outs_above_average should use fresh cache metadata")
        ),
    )

    cached = statcast_client.fetch_fielding_stats(2025, raw_data_root=tmp_path)

    assert cached.equals(refreshed)


def test_fetch_catcher_framing_and_team_game_logs_persist_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.clients import statcast_client

    requested_teams: list[str] = []

    def fake_catcher_framing(season: int, min_called_p: int | str = "q") -> pd.DataFrame:
        assert season == 2025
        assert min_called_p == 0
        return pd.DataFrame({"name": ["Jose Trevino"], "runs_extra_strikes": [6.2]})

    def fake_team_game_logs(season: int, team: str, log_type: str = "batting") -> pd.DataFrame:
        requested_teams.append(team)
        return pd.DataFrame(
            [["2025-09-01", 5, True]],
            columns=pd.MultiIndex.from_tuples(
                [
                    ("Offense", "Date"),
                    ("Offense", "R"),
                    ("Unnamed: 0_level_0", "Home"),
                ]
            ),
        )

    monkeypatch.setattr(statcast_client, "statcast_catcher_framing", fake_catcher_framing)
    monkeypatch.setattr(statcast_client, "team_game_logs", fake_team_game_logs)

    framing_df = statcast_client.fetch_catcher_framing(2025, raw_data_root=tmp_path)
    team_logs_df = statcast_client.fetch_team_game_logs(2025, "TB", raw_data_root=tmp_path)

    assert framing_df.loc[0, "runs_extra_strikes"] == 6.2
    assert requested_teams == ["TBR"]
    assert all(isinstance(column, str) for column in team_logs_df.columns)
    assert {"Offense_Date", "Offense_R", "Home"}.issubset(team_logs_df.columns)
    assert (tmp_path / "catcher_framing" / "catcher_framing_2025.parquet").exists()
    assert (tmp_path / "team_game_logs" / "TBR_2025.parquet").exists()


def test_fetch_catcher_framing_regenerates_stale_cache_when_metadata_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.clients import statcast_client

    stale_path = tmp_path / "catcher_framing" / "catcher_framing_2025.parquet"
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": ["Qualified Catcher"], "runs_extra_strikes": [2.1]}).to_parquet(
        stale_path,
        index=False,
    )
    (tmp_path / "catcher_framing" / "catcher_framing_2025.metadata.json").write_text(
        json.dumps(
            {
                "cache_version": 1,
                "min_called_p": 15,
            }
        )
    )

    calls = {"framing": 0}

    def fake_catcher_framing(season: int, min_called_p: int | str = "q") -> pd.DataFrame:
        calls["framing"] += 1
        assert season == 2025
        assert min_called_p == 0
        return pd.DataFrame({"name": ["Jose Trevino"], "runs_extra_strikes": [6.2]})

    monkeypatch.setattr(statcast_client, "statcast_catcher_framing", fake_catcher_framing)

    refreshed = statcast_client.fetch_catcher_framing(2025, raw_data_root=tmp_path)

    assert calls == {"framing": 1}
    assert refreshed["name"].tolist() == ["Jose Trevino"]
    metadata = json.loads(
        (tmp_path / "catcher_framing" / "catcher_framing_2025.metadata.json").read_text()
    )
    assert metadata == {
        "cache_version": statcast_client.CATCHER_FRAMING_CACHE_VERSION,
        "min_called_p": 0,
    }

    monkeypatch.setattr(
        statcast_client,
        "statcast_catcher_framing",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("statcast_catcher_framing should use fresh cache metadata")
        ),
    )

    cached = statcast_client.fetch_catcher_framing(2025, raw_data_root=tmp_path)

    assert cached.equals(refreshed)
