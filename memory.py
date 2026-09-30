"""Knowledge graph profile — stores typed triples, generates natural language summaries."""

import json
import os

# Relation vocabulary exposed to the LLM
RELATIONS = {
    "nom":            "nom",
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

LEARNER = "apprenant"
# Subjects the LLM sometimes uses for the learner instead of "apprenant".
_LEARNER_ALIASES = {"apprenant", "apprenante", "l'apprenant", "l'apprenante",
                    "je", "moi", "utilisateur", "user", "learner"}


def _learner_names(triples) -> set[str]:
    """Lower-cased names the learner goes by: objects of apprenant|nom|X, plus
    self-referencing X|nom|X triples (the LLM's "Paulo|nom|Paulo")."""
    names = set()
    for s, r, o in triples:
        if r == "nom" and (s.lower() in _LEARNER_ALIASES or s.lower() == o.lower()):
            names.add(o.lower())
    return names


def _is_learner(subject: str, names: set[str]) -> bool:
    return subject.lower() in _LEARNER_ALIASES or subject.lower() in names


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
    names = _learner_names(graph["triples"] + [list(t) for t in triples])
    for s, r, o in triples:
        triple = (LEARNER if _is_learner(s, names) else s, r, o)
        if triple not in existing:
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

    # Index: subject → {relation: [objects]}. Facts filed under the learner's
    # name (older profiles) are merged into "apprenant".
    names = _learner_names(triples)
    idx: dict[str, dict[str, list[str]]] = {}
    for s, r, o in triples:
        s = LEARNER if _is_learner(s, names) else s
        objs = idx.setdefault(s, {}).setdefault(r, [])
        if o not in objs:
            objs.append(o)

    lines: list[str] = []

    def secondary(entity: str) -> str:
        """Inline secondary facts about an entity."""
        facts = idx.get(entity, {})
        parts = []
        for rel, objs in facts.items():
            label = RELATIONS.get(rel, rel)
            parts.append(f"{label} {', '.join(objs)}")
        return f" ({'; '.join(parts)})" if parts else ""

    ap = idx.get(LEARNER, {})
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
