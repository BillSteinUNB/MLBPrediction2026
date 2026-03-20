from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Protocol, Sequence

import httpx
from dotenv import dotenv_values


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = REPO_ROOT / ".env"
DEFAULT_ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"
ODDS_API_VALIDATE_URL = "https://api.the-odds-api.com/v4/sports"
OPENWEATHER_VALIDATE_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_TEST_LAT = 40.7128
OPENWEATHER_TEST_LON = -74.0060
ENV_KEYS = ["ODDS_API_KEY", "OPENWEATHER_API_KEY", "DISCORD_WEBHOOK_URL"]


class ValidationClient(Protocol):
    def get(self, url: str, **kwargs: object) -> httpx.Response: ...


@dataclass(frozen=True, slots=True)
class ValidationResult:
    key: str
    ok: bool
    message: str


def read_template_keys(template_path: Path = DEFAULT_ENV_EXAMPLE_PATH) -> list[str]:
    """Read environment keys from `.env.example` while preserving order."""

    if not template_path.exists():
        return list(ENV_KEYS)

    keys: list[str] = []
    for line in template_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _ = stripped.split("=", maxsplit=1)
        keys.append(key.strip())
    return keys or list(ENV_KEYS)


def load_existing_env_values(env_path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    """Load any existing `.env` values without exposing them in logs."""

    if not env_path.exists():
        return {}

    return {
        key: value
        for key, value in dotenv_values(env_path).items()
        if value is not None
    }


def validate_env_values(
    *,
    odds_api_key: str,
    openweather_api_key: str,
    discord_webhook_url: str,
    client: ValidationClient | None = None,
) -> dict[str, ValidationResult]:
    """Validate all required credentials with live test requests."""

    return {
        "ODDS_API_KEY": validate_odds_api_key(odds_api_key, client=client),
        "OPENWEATHER_API_KEY": validate_openweather_api_key(
            openweather_api_key,
            client=client,
        ),
        "DISCORD_WEBHOOK_URL": validate_discord_webhook_url(
            discord_webhook_url,
            client=client,
        ),
    }


def validate_odds_api_key(
    api_key: str,
    *,
    client: ValidationClient | None = None,
) -> ValidationResult:
    """Validate The Odds API key by listing supported sports."""

    try:
        response = _http_get(
            ODDS_API_VALIDATE_URL,
            client=client,
            params={"apiKey": api_key},
        )
        payload = response.json()
    except Exception as exc:
        return ValidationResult("ODDS_API_KEY", False, f"validation failed: {exc}")

    if not isinstance(payload, list):
        return ValidationResult("ODDS_API_KEY", False, "validation failed: unexpected response")

    return ValidationResult("ODDS_API_KEY", True, "validated with sports listing request")


def validate_openweather_api_key(
    api_key: str,
    *,
    client: ValidationClient | None = None,
) -> ValidationResult:
    """Validate the OpenWeatherMap API key with a current-conditions request."""

    try:
        _http_get(
            OPENWEATHER_VALIDATE_URL,
            client=client,
            params={
                "lat": OPENWEATHER_TEST_LAT,
                "lon": OPENWEATHER_TEST_LON,
                "appid": api_key,
            },
        )
    except Exception as exc:
        return ValidationResult("OPENWEATHER_API_KEY", False, f"validation failed: {exc}")

    return ValidationResult("OPENWEATHER_API_KEY", True, "validated with current weather request")


def validate_discord_webhook_url(
    webhook_url: str,
    *,
    client: ValidationClient | None = None,
) -> ValidationResult:
    """Validate the Discord webhook URL with a read-only GET request."""

    try:
        _http_get(webhook_url, client=client)
    except Exception as exc:
        return ValidationResult("DISCORD_WEBHOOK_URL", False, f"validation failed: {exc}")

    return ValidationResult("DISCORD_WEBHOOK_URL", True, "validated with webhook metadata request")


def write_env_file(env_path: Path, values: dict[str, str]) -> None:
    """Write the ordered `.env` file contents."""

    ordered_keys = read_template_keys()
    lines = [f"{key}={values[key]}" for key in ordered_keys if key in values]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ensure_data_directories(repo_root: Path = REPO_ROOT) -> None:
    """Create the repo-local data directories needed by unattended runs."""

    for relative_path in (
        Path("data") / "raw" / "statcast",
        Path("data") / "models",
        Path("data") / "logs",
        Path("data") / "training",
    ):
        (repo_root / relative_path).mkdir(parents=True, exist_ok=True)


def prompt_for_env_values(
    *,
    env_path: Path = DEFAULT_ENV_PATH,
    template_path: Path = DEFAULT_ENV_EXAMPLE_PATH,
) -> dict[str, str]:
    """Prompt interactively for required environment values."""

    existing = load_existing_env_values(env_path)
    values: dict[str, str] = {}
    for key in read_template_keys(template_path):
        current_value = existing.get(key)
        suffix = " [press Enter to keep current value]" if current_value else ""
        entered = getpass(f"{key}{suffix}: ").strip()
        if entered:
            values[key] = entered
            continue
        if current_value:
            values[key] = current_value
            continue
        raise ValueError(f"{key} is required")
    return values


def _http_get(
    url: str,
    *,
    client: ValidationClient | None = None,
    params: dict[str, object] | None = None,
) -> httpx.Response:
    if client is not None:
        response = client.get(url, params=params, timeout=15.0)
        response.raise_for_status()
        return response

    with httpx.Client(follow_redirects=True, timeout=15.0) as http_client:
        response = http_client.get(url, params=params)
        response.raise_for_status()
        return response


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Interactively create and validate the local .env file")
    parser.add_argument("--env-path", default=str(DEFAULT_ENV_PATH))
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args(argv)

    env_path = Path(args.env_path)
    repo_root = env_path.resolve().parent

    values = prompt_for_env_values(env_path=env_path)
    validation_results = validate_env_values(
        odds_api_key=values["ODDS_API_KEY"],
        openweather_api_key=values["OPENWEATHER_API_KEY"],
        discord_webhook_url=values["DISCORD_WEBHOOK_URL"],
    )

    if not args.skip_validation:
        failures = [result for result in validation_results.values() if not result.ok]
        if failures:
            for result in validation_results.values():
                print(f"{result.key}: {result.message}")
            return 1

    write_env_file(env_path, values)
    ensure_data_directories(repo_root)

    print(
        json.dumps(
            {
                "env_path": str(env_path),
                "validated": {key: result.ok for key, result in validation_results.items()},
                "data_directories_ready": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_ENV_EXAMPLE_PATH",
    "DEFAULT_ENV_PATH",
    "ENV_KEYS",
    "ODDS_API_VALIDATE_URL",
    "OPENWEATHER_VALIDATE_URL",
    "ValidationResult",
    "ensure_data_directories",
    "load_existing_env_values",
    "main",
    "prompt_for_env_values",
    "read_template_keys",
    "validate_discord_webhook_url",
    "validate_env_values",
    "validate_odds_api_key",
    "validate_openweather_api_key",
    "write_env_file",
]
