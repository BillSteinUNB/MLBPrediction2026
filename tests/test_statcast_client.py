from __future__ import annotations

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

    def fake_fielding_stats(season: int) -> pd.DataFrame:
        assert season == 2025
        return pd.DataFrame(
            {
                "Name": ["Anthony Volpe", "Alex Verdugo"],
                "Team": ["NYY", "NYY"],
                "DRS": [12, 4],
            }
        )

    def fake_oaa(year: int, pos: int, min_att: str = "q", view: str = "Fielder") -> pd.DataFrame:
        positions_requested.append(pos)
        assert year == 2025
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


def test_fetch_catcher_framing_and_team_game_logs_persist_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.clients import statcast_client

    requested_teams: list[str] = []

    def fake_catcher_framing(season: int) -> pd.DataFrame:
        assert season == 2025
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
