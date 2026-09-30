"""LLM wrapper: history management, tag parsing, thinking-token stripping.

Backend (local Ollama vs NVIDIA hosted API) is selected in providers.py.
"""

import re
from datetime import date

import providers

from vocab import add_words, get_known_words
from memory import add_triples, get_graph, summarize_for_prompt
from persona import summarize_persona

_history: list[dict] = []

MAX_HISTORY_EXCHANGES = 8   # user+assistant pairs kept; older turns are dropped

_LEVELS: dict[str, dict] = {
    "a1": {
        "name": "A1 — Débutant",
        "constraints": (
            "- Mots très simples et courants uniquement.\n"
            "- Phrases courtes (maximum 10 mots).\n"
            "- Présent de l'indicatif uniquement."
        ),
    },
    "a2": {
        "name": "A2 — Élémentaire",
        "constraints": (
            "- Vocabulaire courant, quelques expressions idiomatiques simples.\n"
            "- Phrases jusqu'à 15 mots.\n"
            "- Présent + passé composé + futur proche (aller + infinitif)."
        ),
    },
    "b1": {
        "name": "B1 — Intermédiaire",
        "constraints": (
            "- Vocabulaire varié, expressions naturelles.\n"
            "- Phrases jusqu'à 20 mots, subordonnées simples.\n"
            "- Présent + passé composé + imparfait + futur simple + conditionnel présent."
        ),
    },
}


def list_levels() -> list[dict]:
    return [{"id": k, "name": v["name"]} for k, v in _LEVELS.items()]

_RE_THINKING = re.compile(r"<think>.*?</think>|<\|think\|>.*?<\|/think\|>", re.DOTALL)
_RE_NOUVEAUX = re.compile(r"\[NOUVEAU\w*:\s*(.*?)\]", re.DOTALL)
_RE_CORRECAO = re.compile(r"\[CORRE[ÇC][ÃA]O:\s*(.*?)\]", re.DOTALL)
_RE_FAITS    = re.compile(r"\[FAITS:\s*(.*?)\]", re.DOTALL)
_RE_ANY_TAG  = re.compile(r"\[.*?\]", re.DOTALL)


def build_system_prompt(
    vocab_file: str,
    profile_file: str,
    new_words_per_turn: int,
    enable_thinking: bool,
    persona_data: dict,
    level: str = "a1",
) -> str:
    known_words = get_known_words(vocab_file)
    graph = get_graph(profile_file)
    think_token = "<|think|>\n" if enable_thinking else ""
    known_str = ", ".join(known_words) if known_words else "aucun pour l'instant"
    profile_str = summarize_for_prompt(graph)
    persona_str = summarize_persona(persona_data)
    level_info = _LEVELS.get(level, _LEVELS["a1"])
    level_name = level_info["name"]
    level_constraints = level_info["constraints"]

    return f"""{think_token}{persona_str}
Tu parles avec un(e) apprenant(e) brésilien(ne) qui apprend le français (niveau {level_name}).
Ton rôle : pratiquer la conversation naturellement, pas enseigner.

PERSONA ET STYLE :
- Parle de façon détendue, comme avec un ami. Pas de "Super !", "Fantastique !", "Génial !".
- Réagis de façon proportionnée — si c'est banal, réponds simplement.
- Si l'apprenant te pose une question sur toi, réponds brièvement et sincèrement,
  puis pose une question en retour.
- Pose toujours UNE seule question courte à la fin. Jamais deux.
- Ne répète jamais une question déjà posée dans cette session.
- Alterne naturellement entre : famille, loisirs, travail, voyages, nourriture, ville, météo, amis.

MÉMOIRE DE L'APPRENANT :
{profile_str}
- Utilise ces informations naturellement. Ne dis jamais "je me souviens" ou "tu m'as dit".
- Relie deux faits connus pour former une question plus intéressante
  (ex : sait qu'il aime la plage et habite à Rio → "Tu vas souvent à Copacabana ?").
- Approfondis un thème déjà abordé avant d'en explorer un nouveau.

ENTRÉE INCOMPRÉHENSIBLE :
- Si la phrase n'a aucune intention communicative claire (mots sans lien, bruit de fond),
  dis simplement : "Désolé, je n'ai pas bien compris — tu peux répéter ?"
  Ne corrige pas, n'invente pas de sens.
- Exemples incompréhensibles (→ désolé) : "j'ai parfait la mer", "bonjour table mange soleil".
- Exemples d'erreurs à corriger (→ correction normale) : "je suis allé à plage", "j'ai mangé le pomme".

NIVEAU DE LANGUE ({level_name}) :
{level_constraints}

VOCABULAIRE PROGRESSIF :
- Mots déjà connus : {known_str}.
- Réutilise ces mots librement. Introduis AU MAXIMUM {new_words_per_turn} mot(s) nouveau(x) par réponse.
- Quand tu introduis un nouveau mot, utilise-le naturellement dans ta réponse.
- Enregistre dans [NOUVEAUX_MOTS] TOUT mot français nouveau que tu introduis
  OU que l'apprenant te demande d'expliquer.
- Format de CHAQUE entrée : motfrançais=traducao_portugais|exemple_en_français
  · à gauche du «=» : le mot français réel
  · entre «=» et «|» : sa traduction en portugais brésilien
  · après «|» : une courte phrase d'exemple en français
  Plusieurs entrées séparées par « ; ». Si aucun mot nouveau : [NOUVEAUX_MOTS: aucun]
- Exemple concret : [NOUVEAUX_MOTS: dessert=sobremesa|J'adore le dessert au chocolat]
- N'écris JAMAIS les mots-gabarits « mot », « traducao » ou « exemple » —
  remplace-les toujours par les vraies valeurs.

CORRECTIONS :
- Si erreur grammaticale ou de vocabulaire importante :
    [CORREÇÃO: type | Ótimo esforço ! explicação EM PORTUGUÊS. Forme correcte : "..."]
- type = conjugaison | accord | article | vocabulaire | structure | préposition | autre
- L'explication doit TOUJOURS être en portugais brésilien, jamais en français.
- Inclus toujours la forme correcte française entre guillemets à la fin de la correction.
- Corrige uniquement les phrases avec une intention communicative claire mais une erreur de forme.
- Ignore les erreurs mineures d'accent. Ne parle JAMAIS la correction.
- Si aucune erreur : n'inclus pas la balise [CORREÇÃO].

---
À LA FIN DE CHAQUE RÉPONSE, colle ces balises dans cet ordre exact :

[FAITS: sujet|relation|objet ; sujet|relation|objet]        ← ou [FAITS: aucun]
[NOUVEAUX_MOTS: motfrançais=traducao|exemple ; ...]         ← ou [NOUVEAUX_MOTS: aucun]
[CORREÇÃO: type | explicação em português. Forme correcte : "..."]  ← seulement si erreur

Pour les faits sur l'apprenant, le sujet est TOUJOURS « apprenant », jamais son prénom.
Relations FAITS : nom, habite_à, travaille_dans, profession, aime, n_aime_pas, a_visité, veut_visiter, a_famille, parle
Exemples :
  "Je m'appelle Thomas"  → [FAITS: apprenant|nom|Thomas]
  "J'habite à Lyon"      → [FAITS: apprenant|habite_à|Lyon]
  "J'aime le foot"       → [FAITS: apprenant|aime|football]
  "Je suis médecin"      → [FAITS: apprenant|profession|médecin]
  "Ma sœur vit à Paris"  → [FAITS: sœur|habite_à|Paris]
  "Bonjour !"            → [FAITS: aucun]
"""


# Literal template placeholders the model sometimes copies verbatim instead of
# filling in — never store these as real vocabulary.
_NOUVEAUX_PLACEHOLDERS = {"mot", "motfrançais", "motfrancais", "tradução",
                          "traducao", "traduction", "exemple", "exemplo", "aucun"}


def _parse_nouveaux(tag_content: str, vocab_file: str) -> list[str]:
    tag_content = tag_content.strip()
    if tag_content.lower() == "aucun" or not tag_content:
        return []
    new_words: list[str] = []
    rows: list[dict] = []
    for entry in tag_content.split(";"):
        entry = entry.strip()
        if "=" not in entry:
            continue
        mot, rest = entry.split("=", 1)
        mot = mot.strip()
        traduction, exemple = (rest.split("|", 1) if "|" in rest else (rest.strip(), ""))
        traduction, exemple = traduction.strip(), exemple.strip()
        # Skip malformed entries where the model left the template placeholders in
        # (e.g. "mot=dessert|exemple") — these would pollute the vocab file.
        if (not mot or not traduction
                or mot.lower() in _NOUVEAUX_PLACEHOLDERS
                or traduction.lower() in _NOUVEAUX_PLACEHOLDERS):
            print(f"  [NOUVEAUX_MOTS ignoré — gabarit non rempli] {entry!r}")
            continue
        rows.append({"mot": mot, "traduction": traduction,
                     "exemple": exemple, "date": str(date.today())})
        new_words.append(mot)
    if rows:
        add_words(vocab_file, rows)
    return new_words


def _parse_faits(tag_content: str, profile_file: str) -> None:
    tag_content = tag_content.strip()
    if tag_content.lower() == "aucun" or not tag_content:
        return
    triples: list[tuple[str, str, str]] = []
    for entry in tag_content.split(";"):
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) == 3:
            triples.append((parts[0], parts[1], parts[2]))
    if triples:
        add_triples(profile_file, triples)


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep system messages and the last MAX_HISTORY_EXCHANGES user/assistant pairs."""
    system = [m for m in history if m["role"] == "system"]
    exchanges = [m for m in history if m["role"] != "system"]
    return system + exchanges[-(MAX_HISTORY_EXCHANGES * 2):]


def chat(
    user_text: str,
    vocab_file: str,
    profile_file: str,
    new_words_per_turn: int,
    enable_thinking: bool,
    persona_data: dict,
    level: str = "a1",
) -> tuple[str, str | None, list[str]]:
    """Returns (french_reply, correcao_or_None, new_word_list)."""
    if not _history:
        system = build_system_prompt(vocab_file, profile_file, new_words_per_turn, enable_thinking, persona_data, level)
        _history.append({"role": "system", "content": system})

    user_turns = sum(1 for m in _history if m["role"] == "user")
    if user_turns > 0 and user_turns % 4 == 0:
        bot_questions = [
            m["content"].split("?")[0].strip() + "?"
            for m in _history if m["role"] == "assistant" and "?" in m["content"]
        ]
        if bot_questions:
            _history.append({"role": "system", "content": (
                "[Rappel — ne pas afficher] Questions déjà posées : "
                + " | ".join(bot_questions[-8:])
            )})

    _history.append({"role": "user", "content": user_text})

    full_response = ""
    for chunk in providers.stream_chat(_trim_history(_history), enable_thinking):
        full_response += chunk

    return _extract_and_record(full_response, vocab_file, profile_file)


def _extract_and_record(
    full_response: str, vocab_file: str, profile_file: str
) -> tuple[str, str | None, list[str]]:
    """Strip thinking tokens, parse/record the trailing tags, and append the clean
    French reply to history. Returns (french_text, correcao_or_None, new_word_list)."""
    full_response = _RE_THINKING.sub("", full_response).strip()

    correcao: str | None = None
    m = _RE_CORRECAO.search(full_response)
    if m:
        raw_correcao = m.group(1).strip()
        if raw_correcao.lower() != "aucun":
            correcao = raw_correcao
        full_response = full_response[: m.start()].rstrip() + full_response[m.end():]

    new_words: list[str] = []
    m = _RE_NOUVEAUX.search(full_response)
    if m:
        new_words = _parse_nouveaux(m.group(1), vocab_file)
        full_response = full_response[: m.start()].rstrip() + full_response[m.end():]

    # Debug: show last 300 chars of raw response so we can see what tags the LLM emits
    print(f"  [DBG raw tail] ...{full_response[-300:]!r}")
    m = _RE_FAITS.search(full_response)
    if m:
        raw_faits = m.group(1).strip()
        print(f"  [FAITS] {raw_faits}")
        _parse_faits(raw_faits, profile_file)
        full_response = full_response[: m.start()].rstrip() + full_response[m.end():]
    else:
        print("  [FAITS] tag absent")

    french_text = _RE_ANY_TAG.sub("", full_response).strip()
    _history.append({"role": "assistant", "content": french_text})
    return french_text, correcao, new_words


def opening(
    vocab_file: str,
    profile_file: str,
    new_words_per_turn: int,
    enable_thinking: bool,
    persona_data: dict,
    level: str = "a1",
) -> tuple[str, list[str]]:
    """Have the bot open the conversation itself (greeting + one question).
    Starts a fresh session. Returns (french_greeting, new_word_list)."""
    _history.clear()
    system = build_system_prompt(vocab_file, profile_file, new_words_per_turn,
                                 enable_thinking, persona_data, level)
    _history.append({"role": "system", "content": system})

    # Ephemeral nudge — asks the model to open. Not kept in history; only its reply is.
    nudge = {"role": "user", "content": (
        "[Début de session] Ouvre toi-même la conversation : salue l'apprenant "
        "chaleureusement en une phrase, puis pose UNE seule question simple et "
        "ouverte pour lancer l'échange. Ne corrige rien (il n'a encore rien dit)."
    )}
    full_response = ""
    for chunk in providers.stream_chat(_trim_history(_history) + [nudge], enable_thinking):
        full_response += chunk

    french_text, _correcao, new_words = _extract_and_record(
        full_response, vocab_file, profile_file)
    return french_text, new_words
