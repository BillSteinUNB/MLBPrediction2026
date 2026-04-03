from __future__ import annotations

import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOTS = [PROJECT_ROOT / "README.md", PROJECT_ROOT / "docs"]

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
CODE_PATH_RE = re.compile(
    r"`((?:docs|archive|data|src|scripts|tests|dashboard|config|AutoResearch|OddsScraper)/[^`\s]+)`"
)


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in DOC_ROOTS:
        if root.is_file():
            files.append(root)
            continue
        files.extend(sorted(root.rglob("*.md")))
    return files


def normalize_target(raw: str) -> Path | None:
    target = raw.strip()
    if not target or target.startswith(("http://", "https://", "mailto:", "#")):
        return None
    if "*" in target:
        parent = target.split("*", 1)[0].rstrip("/\\")
        return PROJECT_ROOT / Path(parent) if parent else None
    if target.startswith("C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/"):
        suffix = target.removeprefix(
            "C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/"
        ).replace("%20", " ")
        return PROJECT_ROOT / Path(suffix)
    if target.startswith("/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/"):
        suffix = target.removeprefix(
            "/C:/Users/bills/Documents/Personal%20Code/MLBPrediction2026/"
        ).replace("%20", " ")
        return PROJECT_ROOT / Path(suffix)
    return PROJECT_ROOT / Path(target)


def find_missing_references() -> list[str]:
    problems: list[str] = []
    for md_file in iter_markdown_files():
        text = md_file.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        for lineno, line in enumerate(lines, start=1):
            for match in MARKDOWN_LINK_RE.finditer(line):
                target = normalize_target(match.group(1).split("#", 1)[0])
                if target is not None and not target.exists():
                    problems.append(f"{md_file}:{lineno} missing link target: {match.group(1)}")

            for match in CODE_PATH_RE.finditer(line):
                target = normalize_target(match.group(1).split("#", 1)[0])
                if target is not None and not target.exists():
                    problems.append(f"{md_file}:{lineno} missing path reference: {match.group(1)}")
    return problems


def main() -> int:
    problems = find_missing_references()
    if problems:
        print("Docs integrity check failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    print("Docs integrity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
