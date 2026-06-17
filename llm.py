"""Ollama wrapper: history management, tag parsing, thinking-token stripping."""

import re
from datetime import date

import ollama

from vocab import add_words, get_known_words
from memory import add_facts, get_known_facts

_history: list[dict] = []

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
    known_facts = get_known_facts(profile_file)
    think_token = "<|think|>\n" if enable_thinking else ""
    known_str = ", ".join(known_words) if known_words else "aucun pour l'instant"
    facts_str = "\n".join(f"  - {f}" for f in known_facts) if known_facts else "  (rien encore)"

    return f"""{think_token}Tu es un ami francophone qui aide un grand débutant à pratiquer le français (niveau A1).
Tes réponses doivent respecter ces règles STRICTEMENT :

STYLE DE CONVERSATION :
- Parle de façon naturelle et détendue, comme dans une vraie conversation entre amis.
- Ne sois pas excessivement enthousiaste : évite les exclamations inutiles comme "Super !", "Fantastique !", "Génial !".
- Réagis de façon proportionnée : une réponse ordinaire mérite une réaction ordinaire.
- Pose toujours une question simple à la fin de chaque réponse pour continuer la conversation.
- Ne pose qu'une seule question par réponse. La question doit être courte et directe.
- MÉMOIRE DES QUESTIONS : lis l'historique complet de la conversation avant de choisir ta question.
  Ne répète jamais une question déjà posée dans cette session.
  Alterne entre ces thèmes : famille, goûts, journée, travail/études, nourriture, ville,
  animaux, voyages, loisirs, météo, amis. Choisis un thème non encore abordé.

MÉMOIRE DE L'APPRENANT :
- Ce que tu sais déjà sur lui :
{facts_str}
- Toutes les 3 ou 4 réponses, pose une question qui s'appuie sur ce que tu sais déjà de lui
  (ex: si tu sais qu'il habite à Rio, demande "Tu aimes aller à la plage à Rio ?").
- Sinon, explore un nouveau thème.
- À la fin de chaque réponse, si l'apprenant a partagé un fait personnel nouveau
  (ville, travail, famille, goûts, habitudes), note-le dans ce format exact :
    [FAITS: fait en français court ; autre fait]
  Si aucun fait nouveau : [FAITS: aucun]

NIVEAU DE LANGUE :
- Utilise uniquement des mots très simples et courants.
- Fais des phrases courtes (maximum 10 mots).
- Utilise uniquement le présent de l'indicatif. Évite le subjonctif et le conditionnel.

VOCABULAIRE PROGRESSIF :
- Mots déjà connus de l'apprenant : {known_str}.
- Réutilise librement ces mots connus.
- Introduis AU MAXIMUM {new_words_per_turn} mot(s) nouveau(x) par réponse.
- Quand tu introduis un nouveau mot, utilise-le dans une phrase exemple simple.
- À la fin de chaque réponse, liste les nouveaux mots dans ce format exact :
    [NOUVEAUX_MOTS: mot1=tradução1|exemple1 ; mot2=tradução2|exemple2]
  Si aucun nouveau mot : [NOUVEAUX_MOTS: aucun]

CORRECTIONS D'ERREURS :
- Si l'apprenant fait une erreur grammaticale ou de vocabulaire importante, ajoute
  une correction à la fin de ta réponse en portugais brésilien écrit UNIQUEMENT,
  dans ce format exact :
    [CORREÇÃO: xxx | Ótimo esforço ! explication en português + forme correcte en français]
- Remplace xxx par exactement un de ces mots (sans guillemets, sans préfixe) :
    conjugaison   — mauvaise forme verbale
    accord        — erreur de genre ou de nombre
    article       — article manquant ou incorrect
    vocabulaire   — mauvais choix de mot ou faux ami
    structure     — ordre des mots incorrect
    préposition   — préposition manquante ou incorrecte
    autre         — autre type d'erreur
- Commence chaque correction par "Ótimo esforço !"
- Ne corrige pas les erreurs mineures d'accent ou de prononciation.
- Ne parle JAMAIS la correction — texte affiché uniquement.
- Si aucune correction : n'inclus pas la balise [CORREÇÃO].

FORMAT DE RÉPONSE :
Réponds d'abord en français simple, puis ajoute [FAITS:...], [NOUVEAUX_MOTS:...],
et si besoin [CORREÇÃO:...]. Ne mélange jamais le français et le portugais dans le corps.
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
        if "|" in rest:
            traduction, exemple = rest.split("|", 1)
        else:
            traduction, exemple = rest.strip(), ""
        rows.append({"mot": mot, "traduction": traduction.strip(),
                     "exemple": exemple.strip(), "date": str(date.today())})
        new_words.append(mot)
    if rows:
        add_words(vocab_file, rows)
    return new_words


def _parse_faits(tag_content: str, profile_file: str) -> list[str]:
    tag_content = tag_content.strip()
    if tag_content.lower() == "aucun" or not tag_content:
        return []
    facts = [f.strip() for f in tag_content.split(";") if f.strip()]
    if facts:
        add_facts(profile_file, facts)
    return facts


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

    # Every 4 user turns inject a reminder of questions already asked
    user_turns = sum(1 for m in _history if m["role"] == "user")
    if user_turns > 0 and user_turns % 4 == 0:
        bot_questions = [
            m["content"].split("?")[0].strip() + "?"
            for m in _history if m["role"] == "assistant" and "?" in m["content"]
        ]
        if bot_questions:
            reminder = (
                "[Rappel interne — ne pas afficher] "
                "Questions déjà posées : "
                + " | ".join(bot_questions[-8:])
                + " — choisis un thème différent."
            )
            _history.append({"role": "system", "content": reminder})

    _history.append({"role": "user", "content": user_text})

    full_response = ""
    for chunk in ollama.chat(model=model, messages=_history, stream=True,
                             think=enable_thinking):
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

    m = _RE_FAITS.search(full_response)
    if m:
        _parse_faits(m.group(1), profile_file)
        full_response = full_response[: m.start()].rstrip() + full_response[m.end():]

    french_text = _RE_ANY_TAG.sub("", full_response).strip()
    _history.append({"role": "assistant", "content": french_text})
    return french_text, correcao, new_words
