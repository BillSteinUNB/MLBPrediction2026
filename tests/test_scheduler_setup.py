from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from scripts import setup_env, setup_scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"


class _StubResponse:
    def __init__(self, url: str, *, status_code: int = 200, payload: object | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.request = httpx.Request("GET", url)
        self.response = httpx.Response(status_code, request=self.request)

    def json(self) -> object:
        return self._payload

    def raise_for_status(self) -> None:
        self.response.raise_for_status()


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> _StubResponse:
        self.calls.append(("GET", url, dict(kwargs)))
        if "the-odds-api.com" in url:
            return _StubResponse(url, payload=[{"key": "baseball_mlb"}])
        if "openweathermap.org" in url:
            return _StubResponse(url, payload={"weather": []})
        if "discord.com/api/webhooks" in url:
            return _StubResponse(url, payload={"id": "123456"})
        raise AssertionError(f"Unexpected URL: {url}")


def test_compute_run_time_and_windows_task_plan_use_batch_wrapper() -> None:
    run_time = setup_scheduler.compute_daily_run_time("19:05", hours_before=3)

    assert run_time == "16:05"

    plan = setup_scheduler.build_windows_task_plan(
        repo_root=REPO_ROOT,
        task_name="MLBPrediction2026 Daily Pipeline",
        run_time=run_time,
    )

    assert plan.runner_path == REPO_ROOT / "scripts" / "run_daily.bat"
    assert plan.run_time == "16:05"
    assert "Register-ScheduledTask" in plan.registration_script
    assert "-WakeToRun" in plan.registration_script
    assert "-LogonType S4U" in plan.registration_script
    assert "run_daily.bat" in plan.registration_script
    assert plan.query_command[:2] == ["schtasks.exe", "/Query"]


def test_cron_and_systemd_templates_use_shell_wrapper_and_expected_schedule() -> None:
    run_time = setup_scheduler.compute_daily_run_time("19:05", hours_before=3)

    cron_entry = setup_scheduler.build_cron_entry(repo_root=REPO_ROOT, run_time=run_time)
    service_unit = setup_scheduler.build_systemd_service(repo_root=REPO_ROOT)
    timer_unit = setup_scheduler.build_systemd_timer(run_time=run_time)

    cron_lines = cron_entry.splitlines()
    assert cron_lines[0] == setup_scheduler.CRON_MARKER
    assert cron_lines[1] == "CRON_TZ=America/New_York"
    assert cron_lines[2].startswith("5 16 * * *")
    assert "/usr/bin/env bash" in cron_lines[2]
    assert "run_daily.sh" in cron_lines[2]
    assert str(REPO_ROOT) in cron_lines[2]

    assert f"WorkingDirectory={REPO_ROOT}" in service_unit
    assert "ExecStart=/usr/bin/env bash" in service_unit
    assert "run_daily.sh" in service_unit

    assert "OnCalendar=*-*-* 16:05:00" in timer_unit
    assert "Persistent=true" in timer_unit
    assert "WakeSystem=true" in timer_unit


def test_install_windows_task_falls_back_to_schtasks_when_powershell_registration_is_denied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def _fake_run_command(
        command: list[str],
        *,
        input_text: str | None = None,
        allow_failure: bool = False,
    ) -> SimpleNamespace:
        del input_text, allow_failure
        calls.append(command)
        if command[:2] == ["powershell.exe", "-NoProfile"]:
            raise RuntimeError("Access is denied.")
        return SimpleNamespace(returncode=0, stdout="TaskName: test\n", stderr="")

    monkeypatch.setattr(setup_scheduler, "_run_command", _fake_run_command)

    result = setup_scheduler.install_windows_task(
        repo_root=tmp_path,
        task_name="MLBPrediction2026 Daily Pipeline (Test)",
        run_time="16:05",
    )

    assert result["registration_mode"] == "schtasks-fallback"
    assert calls[1][:2] == ["schtasks.exe", "/Create"]
    assert calls[2][:2] == ["schtasks.exe", "/Query"]


def test_install_cron_entry_replaces_existing_managed_block(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[list[str], str | None, bool]] = []

    def _fake_run_command(
        command: list[str],
        *,
        input_text: str | None = None,
        allow_failure: bool = False,
    ) -> SimpleNamespace:
        calls.append((command, input_text, allow_failure))
        if command == ["crontab", "-l"]:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "MAILTO=ops@example.com\n"
                    f"{setup_scheduler.CRON_MARKER}\n"
                    "CRON_TZ=America/New_York\n"
                    "5 16 * * * stale-command\n"
                    "30 8 * * * echo other-job\n"
                ),
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(setup_scheduler, "_run_command", _fake_run_command)

    result = setup_scheduler.install_cron_entry(repo_root=tmp_path, run_time="16:05")

    assert result["cron_entry"].splitlines()[0] == setup_scheduler.CRON_MARKER
    assert len(calls) == 2

    updated_content = calls[1][1]
    assert updated_content is not None
    assert updated_content.count(setup_scheduler.CRON_MARKER) == 1
    assert "stale-command" not in updated_content
    assert "30 8 * * * echo other-job" in updated_content


def test_runner_scripts_pin_repo_root_and_forward_extra_args() -> None:
    batch_script = (SCRIPTS_DIR / "run_daily.bat").read_text(encoding="utf-8")
    shell_script = (SCRIPTS_DIR / "run_daily.sh").read_text(encoding="utf-8")

    assert "%~dp0" in batch_script
    assert "-m src.pipeline.daily --date today --mode prod %*" in batch_script

    assert "BASH_SOURCE[0]" in shell_script
    assert 'src.pipeline.daily --date today --mode prod "$@"' in shell_script


def test_setup_env_validates_keys_with_test_requests_and_writes_env_file(tmp_path: Path) -> None:
    client = _RecordingClient()
    validation = setup_env.validate_env_values(
        odds_api_key="odds-key",
        openweather_api_key="weather-key",
        discord_webhook_url="https://discord.com/api/webhooks/123/token",
        client=client,
    )

    assert validation["ODDS_API_KEY"].ok is True
    assert validation["OPENWEATHER_API_KEY"].ok is True
    assert validation["DISCORD_WEBHOOK_URL"].ok is True
    assert client.calls[0][0] == "GET"
    assert client.calls[0][1] == setup_env.ODDS_API_VALIDATE_URL
    assert client.calls[1][1] == setup_env.OPENWEATHER_VALIDATE_URL
    assert client.calls[2][1] == "https://discord.com/api/webhooks/123/token"

    env_path = tmp_path / ".env"
    setup_env.write_env_file(
        env_path,
        {
            "ODDS_API_KEY": "odds-key",
            "OPENWEATHER_API_KEY": "weather-key",
            "DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/123/token",
        },
    )
    setup_env.ensure_data_directories(tmp_path)

    assert env_path.read_text(encoding="utf-8") == (
        "ODDS_API_KEY=odds-key\n"
        "OPENWEATHER_API_KEY=weather-key\n"
        "DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/123/token\n"
    )
    assert (tmp_path / "data" / "raw" / "statcast").is_dir()
    assert (tmp_path / "data" / "models").is_dir()
    assert (tmp_path / "data" / "logs").is_dir()
    assert (tmp_path / "data" / "training").is_dir()
