from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

from dotenv import load_dotenv


AUTORESEARCH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = AUTORESEARCH_ROOT.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
FACTORY_SETTINGS_FILES = (
    Path.home() / ".factory" / "settings.local.json",
    Path.home() / ".factory" / "settings.json",
)
DEFAULT_DROID_EXEC_TIMEOUT_SECONDS = 600


@dataclass(frozen=True, slots=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    command: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True, slots=True)
class LLMTextResponse:
    provider: str
    model: str
    text: str
    raw_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PlannerSelfCheckResult:
    provider: str
    model: str
    command: str
    response_text: str
    session_id: str | None
    raw_payload: dict[str, Any]


def load_llm_config() -> LLMConfig | None:
    load_dotenv(DEFAULT_ENV_FILE)
    provider = os.getenv("AUTORESEARCH_PROVIDER", "droid").strip().lower()

    if provider == "heuristic":
        return None

    if provider in {"auto", "droid"}:
        model = (
            os.getenv("AUTORESEARCH_DROID_MODEL")
            or os.getenv("AUTORESEARCH_MODEL")
            or _discover_glm_zai_droid_model()
            or ""
        ).strip()
        if model:
            command = os.getenv("AUTORESEARCH_DROID_COMMAND", "droid").strip() or "droid"
            return LLMConfig(
                provider="droid",
                model=model,
                command=command,
                reasoning_effort=(os.getenv("AUTORESEARCH_DROID_REASONING") or "medium").strip(),
            )
        if provider == "droid":
            raise RuntimeError(
                "AUTORESEARCH_PROVIDER=droid but AUTORESEARCH_DROID_MODEL is not set"
            )

    if provider in {"auto", "openai"}:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return LLMConfig(
                provider="openai",
                model=(
                    os.getenv("AUTORESEARCH_MODEL")
                    or os.getenv("OPENAI_MODEL")
                    or "gpt-4o-mini"
                ),
                api_key=api_key,
                base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            )
        if provider == "openai":
            raise RuntimeError("AUTORESEARCH_PROVIDER=openai but OPENAI_API_KEY is not set")

    if provider in {"auto", "anthropic"}:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            return LLMConfig(
                provider="anthropic",
                model=(
                    os.getenv("AUTORESEARCH_MODEL")
                    or os.getenv("ANTHROPIC_MODEL")
                    or "claude-3-5-sonnet-latest"
                ),
                api_key=api_key,
            )
        if provider == "anthropic":
            raise RuntimeError("AUTORESEARCH_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set")

    if provider != "auto":
        raise RuntimeError(f"Unsupported AUTORESEARCH_PROVIDER: {provider}")
    return None


def _load_factory_settings_payload() -> dict[str, Any]:
    for path in FACTORY_SETTINGS_FILES:
        if not path.exists():
            continue
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _resolve_droid_model_from_settings_payload(payload: dict[str, Any]) -> str | None:
    custom_models = payload.get("customModels")
    if not isinstance(custom_models, list):
        return None

    preferred_candidates: list[tuple[int, str]] = []
    for index, entry in enumerate(custom_models):
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id") or "").strip()
        if not model_id.startswith("custom:"):
            continue
        haystack = " ".join(
            str(entry.get(key) or "").lower()
            for key in ("displayName", "model", "id", "baseUrl")
        )
        if "glm" not in haystack:
            continue
        if "z.ai" not in haystack and "z ai" not in haystack:
            continue

        priority = 0
        if "5.1" in haystack or "glm-5.1" in haystack or "glm 5.1" in haystack:
            priority += 10
        if "coding" in haystack:
            priority += 3
        preferred_candidates.append((priority - index, model_id))

    if not preferred_candidates:
        return None
    preferred_candidates.sort(reverse=True)
    return preferred_candidates[0][1]


def _discover_glm_zai_droid_model() -> str | None:
    return _resolve_droid_model_from_settings_payload(_load_factory_settings_payload())


def require_llm() -> bool:
    return os.getenv("AUTORESEARCH_REQUIRE_LLM", "").strip().lower() in {"1", "true", "yes", "on"}


def generate_text(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1400,
) -> LLMTextResponse:
    config = load_llm_config()
    if config is None:
        raise RuntimeError("No LLM provider is configured for autoresearch")
    if config.provider == "droid":
        return _generate_droid_exec_text(
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    if config.provider == "openai":
        return _generate_openai_text(
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    if config.provider == "anthropic":
        return _generate_anthropic_text(
            config=config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    raise RuntimeError(f"Unsupported provider: {config.provider}")


def _generate_droid_exec_text(
    *,
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
) -> LLMTextResponse:
    payload = _run_droid_exec_json(
        config=config,
        prompt=f"{system_prompt.strip()}\n\n{user_prompt.strip()}\n",
    )
    return LLMTextResponse(
        provider=config.provider,
        model=config.model,
        text=str(payload.get("result", "")).strip(),
        raw_payload=payload,
    )


def planner_self_check() -> PlannerSelfCheckResult:
    config = load_llm_config()
    if config is None:
        raise RuntimeError("No LLM provider is configured for autoresearch")
    if config.provider != "droid":
        raise RuntimeError(
            "Planner self-check is intended for AUTORESEARCH_PROVIDER=droid"
        )
    payload = _run_droid_exec_json(
        config=config,
        prompt="Reply with exactly OK and nothing else.",
    )
    response_text = str(payload.get("result", "")).strip()
    if not response_text:
        raise RuntimeError("Droid self-check returned an empty response")
    return PlannerSelfCheckResult(
        provider=config.provider,
        model=config.model,
        command=str(config.command or "droid"),
        response_text=response_text,
        session_id=None if payload.get("session_id") is None else str(payload.get("session_id")),
        raw_payload=payload,
    )


def _run_droid_exec_json(
    *,
    config: LLMConfig,
    prompt: str,
) -> dict[str, Any]:
    if not config.command:
        raise RuntimeError("Droid command is not configured")
    if not shutil.which(config.command):
        raise RuntimeError(
            f"Droid executable {config.command!r} was not found on PATH"
        )

    prompt = f"{prompt.strip()}\n"
    prompt_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".md",
            delete=False,
        ) as handle:
            handle.write(prompt)
            prompt_path = Path(handle.name)

        command = [
            config.command,
            "exec",
            "--output-format",
            "json",
            "--cwd",
            str(REPO_ROOT),
            "--model",
            config.model,
            "--reasoning-effort",
            str(config.reasoning_effort or "medium"),
            "--file",
            str(prompt_path),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=DEFAULT_DROID_EXEC_TIMEOUT_SECONDS,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "droid exec failed: "
                + (completed.stderr.strip() or completed.stdout.strip() or f"exit={completed.returncode}")
            )
        payload = json.loads(completed.stdout)
        if payload.get("is_error"):
            raise RuntimeError(f"droid exec returned an error payload: {payload}")
        return payload
    finally:
        if prompt_path is not None and prompt_path.exists():
            prompt_path.unlink()


def _generate_openai_text(
    *,
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> LLMTextResponse:
    import requests

    response = requests.post(
        f"{str(config.base_url).rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return LLMTextResponse(
        provider=config.provider,
        model=config.model,
        text=str(payload["choices"][0]["message"]["content"]).strip(),
        raw_payload=payload,
    )


def _generate_anthropic_text(
    *,
    config: LLMConfig,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_output_tokens: int,
) -> LLMTextResponse:
    import requests

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": config.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": config.model,
            "system": system_prompt,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": user_prompt}],
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    parts = []
    for block in payload.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return LLMTextResponse(
        provider=config.provider,
        model=config.model,
        text="\n".join(part for part in parts if part).strip(),
        raw_payload=payload,
    )
