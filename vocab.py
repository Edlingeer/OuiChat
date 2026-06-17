"""Vocabulary tracking — reads/writes vocabulary.md. Never calls the LLM."""

import os
from datetime import date

HEADER = (
    "# Vocabulaire appris\n\n"
    "| Mot (français) | Traduction (português) | Exemple | Introduit le |\n"
    "|---|---|---|---|\n"
)


def _ensure_file(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(HEADER)


def get_known_words(path: str) -> list[str]:
    _ensure_file(path)
    words: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("|") and not line.startswith("| Mot") and not line.startswith("|---"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2 and parts[1]:
                    words.append(parts[1])
    return words


def add_words(path: str, words: list[dict]) -> None:
    """Append new rows. Each dict: {mot, traduction, exemple, date}."""
    _ensure_file(path)
    existing = get_known_words(path)
    new_rows: list[str] = []
    for w in words:
        mot = w.get("mot", "").strip()
        if not mot or mot in existing:
            continue
        traduction = w.get("traduction", "").strip()
        exemple = w.get("exemple", "").strip()
        d = w.get("date", str(date.today()))
        new_rows.append(f"| {mot} | {traduction} | {exemple} | {d} |\n")
    if new_rows:
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(new_rows)
