"""Vocabulary tracking — reads/writes vocabulary.md in dictionary format."""

import os
import re
from datetime import date

_RE_WORD = re.compile(r"^\*\*(.+?)\*\*")


def _ensure_file(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Vocabulaire appris\n")


def get_known_words(path: str) -> list[str]:
    _ensure_file(path)
    words: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = _RE_WORD.match(line.strip())
            if m:
                words.append(m.group(1))
    return words


def add_words(path: str, words: list[dict]) -> None:
    """Append new entries. Each dict: {mot, traduction, exemple, date}."""
    _ensure_file(path)
    existing = set(get_known_words(path))

    by_date: dict[str, list[dict]] = {}
    for w in words:
        mot = w.get("mot", "").strip()
        if not mot or mot in existing:
            continue
        d = w.get("date", str(date.today()))
        by_date.setdefault(d, []).append(w)

    if not by_date:
        return

    with open(path, encoding="utf-8") as f:
        content = f.read()

    for d, new_words in sorted(by_date.items()):
        section_header = f"## {d}"
        entries = ""
        for w in new_words:
            mot        = w.get("mot", "").strip()
            traduction = w.get("traduction", "").strip()
            exemple    = w.get("exemple", "").strip()
            entries += f"\n**{mot}** — {traduction}\n"
            if exemple:
                entries += f"*{exemple}*\n"

        if section_header in content:
            next_sec = content.find("\n## ", content.index(section_header) + 1)
            if next_sec == -1:
                content = content.rstrip() + "\n" + entries
            else:
                content = content[:next_sec] + entries + content[next_sec:]
        else:
            content = content.rstrip() + f"\n\n{section_header}\n" + entries

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
