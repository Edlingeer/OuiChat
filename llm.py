"""Ollama wrapper: history management, tag parsing, thinking-token stripping."""

import re
from datetime import date

import ollama

from vocab import add_words, get_known_words
from memory import add_triples, get_graph, summarize_for_prompt

_history: list[dict] = []

MAX_HISTORY_EXCHANGES = 8   # user+assistant pairs kept; older turns are dropped

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
) -> str:
    known_words = get_known_words(vocab_file)
    graph = get_graph(profile_file)
    think_token = "<|think|>\n" if enable_thinking else ""
    known_str = ", ".join(known_words) if known_words else "aucun pour l'instant"
    profile_str = summarize_for_prompt(graph)

    return f"""{think_token}Tu es un ami francophone qui aide un grand débutant à pratiquer le français (niveau A1).
Tes réponses doivent respecter ces règles STRICTEMENT :

STYLE DE CONVERSATION :
- Parle de façon naturelle et détendue, comme dans une vraie conversation entre amis.
- Ne sois pas excessivement enthousiaste : évite "Super !", "Fantastique !", "Génial !".
- Réagis de façon proportionnée à ce que dit l'apprenant.
- Pose toujours une question simple à la fin de chaque réponse. Une seule, courte et directe.
- Ne répète jamais une question déjà posée dans cette session.
- Alterne entre ces thèmes : famille, loisirs, travail, voyages, nourriture, ville, météo, amis.

MÉMOIRE DE L'APPRENANT :
{profile_str}
- Utilise ces informations naturellement, comme si tu les savais depuis toujours.
- Ne dis jamais "je me souviens que..." ou "tu m'as dit que...".
- Toutes les 3 ou 4 réponses, construis ta question à partir de ce que tu sais déjà
  (ex: si tu sais qu'il a une sœur, demande "Ta sœur travaille aussi ?").
- Sinon, explore un nouveau thème pour enrichir le profil.

EXTRACTION DE FAITS (obligatoire à chaque réponse) :
Lis attentivement ce que l'apprenant vient de dire. Extrais TOUS les faits personnels.
Format : [FAITS: sujet|relation|objet ; sujet|relation|objet]
Relations : habite_à, travaille_dans, profession, aime, n_aime_pas, a_visité, veut_visiter, a_famille, parle
Le sujet est "apprenant" ou un proche (frère, sœur, père, mère, ami, etc.)
Exemples :
  "J'habite à Lyon"              → [FAITS: apprenant|habite_à|Lyon]
  "J'aime le foot"               → [FAITS: apprenant|aime|football]
  "Je suis médecin"              → [FAITS: apprenant|profession|médecin]
  "Ma sœur vit à Paris"         → [FAITS: sœur|habite_à|Paris]
  "J'ai visité Tokyo"            → [FAITS: apprenant|a_visité|Tokyo]
  "Bonjour, ça va ?"             → [FAITS: aucun]
Si l'apprenant ne dit rien de personnel : [FAITS: aucun]

NIVEAU DE LANGUE :
- Utilise uniquement des mots très simples et courants.
- Fais des phrases courtes (maximum 10 mots).
- Utilise uniquement le présent de l'indicatif.

VOCABULAIRE PROGRESSIF :
- Mots déjà connus : {known_str}.
- Réutilise ces mots librement. Introduis AU MAXIMUM {new_words_per_turn} mot(s) nouveau(x) par réponse.
- Quand tu introduis un nouveau mot, utilise-le dans une phrase exemple simple.
- Liste les nouveaux mots en fin de réponse :
    [NOUVEAUX_MOTS: mot1=tradução1|exemple1 ; mot2=tradução2|exemple2]
  Si aucun mot nouveau : [NOUVEAUX_MOTS: aucun]

CORRECTIONS :
- Si erreur grammaticale ou de vocabulaire importante, ajoute en fin de réponse :
    [CORREÇÃO: xxx | Ótimo esforço ! explication en português + forme correcte]
- xxx = conjugaison | accord | article | vocabulaire | structure | préposition | autre
- Commence par "Ótimo esforço !". Ignore les erreurs mineures d'accent.
- Ne parle JAMAIS la correction — texte affiché uniquement.
- Si aucune correction : n'inclus pas la balise [CORREÇÃO].

FORMAT OBLIGATOIRE de fin de réponse (dans cet ordre) :
  [FAITS: ...]
  [NOUVEAUX_MOTS: ...]
  [CORREÇÃO: ...] (seulement si erreur)
"""


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
        rows.append({"mot": mot, "traduction": traduction.strip(),
                     "exemple": exemple.strip(), "date": str(date.today())})
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
    model: str,
    vocab_file: str,
    profile_file: str,
    new_words_per_turn: int,
    enable_thinking: bool,
) -> tuple[str, str | None, list[str]]:
    """Returns (french_reply, correcao_or_None, new_word_list)."""
    if not _history:
        system = build_system_prompt(vocab_file, profile_file, new_words_per_turn, enable_thinking)
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
    for chunk in ollama.chat(
        model=model,
        messages=_trim_history(_history),
        stream=True,
        think=enable_thinking,
    ):
        full_response += chunk["message"]["content"]

    full_response = _RE_THINKING.sub("", full_response).strip()

    correcao: str | None = None
    m = _RE_CORRECAO.search(full_response)
    if m:
        correcao = m.group(1).strip()
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
