"""Persona management — loads famous-person profiles for the bot identity."""

import json
import os

_HERE       = os.path.dirname(os.path.abspath(__file__))
_PERSONAS_DIR = os.path.join(_HERE, "personas")

_RELATION_LABELS = {
    "née_à":      "née à",
    "née_le":     "née le",
    "habite_à":   "tu habites à",
    "profession": "tu es",
    "parle":      "tu parles",
    "aime":       "tu aimes",
    "a_un_enfant":"tu as un enfant :",
    "a_famille":  "ta famille :",
    "conjoint":   "ton partenaire :",
    "connue_pour":"tu es connue pour",
    "a_reçu":     "tu as reçu",
    "engage_pour":"tu t'engages pour",
    "grandie_à":  "tu as grandi à",
    "a_écrit":    "tu as écrit",
    "a_réalisé":  "tu as réalisé",
    "a_refusé":   "tu as refusé",
}


def list_personas() -> list[dict]:
    result = []
    for fname in sorted(os.listdir(_PERSONAS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(_PERSONAS_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
            result.append({"id": data["id"], "name": data["name"]})
    return result


def load_persona(persona_id: str) -> dict:
    path = os.path.join(_PERSONAS_DIR, f"{persona_id}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_persona(persona: dict) -> str:
    """Generate a natural-language persona block for the system prompt."""
    triples = persona.get("triples", [])
    name      = persona.get("name", "")
    character = persona.get("character", "")

    # Index persona-subject triples by relation
    idx: dict[str, list[str]] = {}
    for s, r, o in triples:
        if s == "persona":
            idx.setdefault(r, []).append(o)

    lines = [f"Tu t'appelles {name}."]
    if character:
        lines.append(f"Tu es {character}.")
    if "née_à" in idx:
        born = f"née à {idx['née_à'][0]}"
        if "née_le" in idx:
            born += f", le {idx['née_le'][0]}"
        lines.append(f"Tu es {born}.")
    if "habite_à" in idx:
        lines.append(f"Tu habites à {idx['habite_à'][0]}.")
    if "profession" in idx:
        lines.append(f"Tu es {', '.join(idx['profession'])}.")
    if "parle" in idx:
        lines.append(f"Tu parles {', '.join(idx['parle'])}.")
    if "aime" in idx:
        lines.append(f"Tu aimes : {', '.join(idx['aime'])}.")
    if "a_famille" in idx:
        lines.append(f"Ta famille : {', '.join(idx['a_famille'])}.")
    if "a_un_enfant" in idx:
        enfants = idx["a_un_enfant"]
        lines.append(f"Tu as {'un enfant' if len(enfants)==1 else str(len(enfants))+' enfants'} : {', '.join(enfants)}.")
    if "conjoint" in idx:
        lines.append(f"Ton partenaire : {idx['conjoint'][0]}.")
    if "connue_pour" in idx:
        lines.append(f"Tu es connue pour : {', '.join(idx['connue_pour'][:3])}.")
    if "a_reçu" in idx:
        lines.append(f"Tu as reçu : {', '.join(idx['a_reçu'][:2])}.")
    if "engage_pour" in idx:
        lines.append(f"Tu t'engages pour : {', '.join(idx['engage_pour'])}.")
    for rel in ("grandie_à", "a_écrit", "a_réalisé", "a_refusé"):
        if rel in idx:
            label = _RELATION_LABELS.get(rel, rel)
            lines.append(f"Tu {label.removeprefix('tu ')} : {', '.join(idx[rel])}.")

    return "\n".join(lines)
