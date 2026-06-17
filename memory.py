"""User profile memory — reads/writes profile.md with facts shared during conversations."""

import os

HEADER = "# Profil de l'apprenant\n\n"


def _ensure_file(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(HEADER)


def get_known_facts(path: str) -> list[str]:
    _ensure_file(path)
    facts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("- "):
                facts.append(line[2:])
    return facts


def add_facts(path: str, facts: list[str]) -> None:
    _ensure_file(path)
    existing = {f.lower() for f in get_known_facts(path)}
    new_lines = [f"- {f}\n" for f in facts if f.strip() and f.lower() not in existing]
    if new_lines:
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(new_lines)
