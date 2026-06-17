"""Append grammar corrections to corrections.md with color-coded error type badges."""

import os
import re
from datetime import datetime

HEADER = "# Corrections\n\n"

_COLORS: dict[str, tuple[str, str]] = {
    "conjugaison": ("#c0392b", "⚡ Conjugaison"),
    "accord":      ("#d35400", "⚖ Accord"),
    "article":     ("#f39c12", "📄 Article"),
    "vocabulaire": ("#2980b9", "📖 Vocabulaire"),
    "structure":   ("#8e44ad", "🔀 Structure"),
    "préposition": ("#27ae60", "🔗 Préposition"),
    "autre":       ("#7f8c8d", "💬 Autre"),
}

_KNOWN_TYPES = set(_COLORS.keys())
# Matches both "TYPE:conjugaison | ..." and "conjugaison | ..."
_RE_TYPE = re.compile(
    r"^(?:TYPE:)?(conjugaison|accord|article|vocabulaire|structure|pr[eé]position|autre)\s*\|\s*",
    re.IGNORECASE,
)


def _badge(error_type: str) -> str:
    key = error_type.lower().strip()
    color, label = _COLORS.get(key, ("#7f8c8d", f"💬 {error_type.capitalize()}"))
    return (
        f'<span style="background:{color};color:white;'
        f'padding:2px 10px;border-radius:12px;font-size:0.85em">{label}</span>'
    )


def parse_correction(correction: str) -> tuple[str, str]:
    """Return (error_type, clean_message) — strips the leading type label if present."""
    m = _RE_TYPE.match(correction)
    if m:
        return m.group(1).lower(), correction[m.end():]
    return "autre", correction


def save_correction(path: str, user_text: str, correction: str) -> None:
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(HEADER)

    error_type, message = parse_correction(correction)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = (
        f"## {timestamp} &nbsp; {_badge(error_type)}\n\n"
        f"**Vous avez dit :** *{user_text}*\n\n"
        f"{message}\n\n"
        f"---\n\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
