"""Knowledge graph profile — stores typed triples, generates natural language summaries."""

import json
import os

# Relation vocabulary exposed to the LLM
RELATIONS = {
    "habite_à":       "habite à",
    "travaille_dans": "travaille dans",
    "profession":     "profession",
    "aime":           "aime",
    "n_aime_pas":     "n'aime pas",
    "a_visité":       "a visité",
    "veut_visiter":   "veut visiter",
    "a_famille":      "famille",
    "parle":          "parle",
    "type":           "type",
}

_EMPTY = {"triples": []}


def _load(path: str) -> dict:
    if not os.path.exists(path):
        _save(path, _EMPTY)
        return {"triples": []}
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        _save(path, _EMPTY)
        return {"triples": []}
    return json.loads(content)


def _save(path: str, graph: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)


def add_triples(path: str, triples: list[tuple[str, str, str]]) -> None:
    graph = _load(path)
    existing = {(s, r, o) for s, r, o in graph["triples"]}
    for triple in triples:
        if tuple(triple) not in existing:
            graph["triples"].append(list(triple))
            existing.add(tuple(triple))
    _save(path, graph)


def get_graph(path: str) -> dict:
    return _load(path)


def summarize_for_prompt(graph: dict) -> str:
    """Convert the graph to a compact natural language block for the system prompt."""
    triples = graph.get("triples", [])
    if not triples:
        return "  (rien encore)"

    # Index: subject → {relation: [objects]}
    idx: dict[str, dict[str, list[str]]] = {}
    for s, r, o in triples:
        idx.setdefault(s, {}).setdefault(r, []).append(o)

    lines: list[str] = []

    def secondary(entity: str) -> str:
        """Inline secondary facts about an entity."""
        facts = idx.get(entity, {})
        parts = []
        for rel, objs in facts.items():
            label = RELATIONS.get(rel, rel)
            parts.append(f"{label} {', '.join(objs)}")
        return f" ({'; '.join(parts)})" if parts else ""

    ap = idx.get("apprenant", {})
    for rel, label in RELATIONS.items():
        if rel not in ap:
            continue
        objs = ap[rel]
        # Attach secondary facts for each object that is itself a node
        rich = []
        for o in objs:
            rich.append(o + secondary(o))
        lines.append(f"  {label.capitalize()} : {', '.join(rich)}")

    return "\n".join(lines) if lines else "  (rien encore)"
