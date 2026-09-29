"""LLM backend abstraction — dispatch between local Ollama and NVIDIA's
build.nvidia.com hosted API (OpenAI-compatible).

Configure via environment variables (or edit the defaults below):

    LLM_BACKEND      "ollama" (default) | "nvidia"
    OLLAMA_MODEL     local Ollama tag           (default "qwen3.6:27b")
    NVIDIA_MODEL     NVIDIA catalog id          (default "meta/llama-3.3-70b-instruct")
    NVIDIA_API_KEY   required when LLM_BACKEND="nvidia"
    NVIDIA_BASE_URL  (default "https://integrate.api.nvidia.com/v1")

Thinking (chain-of-thought) is toggled by ENABLE_THINKING in main.py/api.py,
threaded down as `enable_thinking`:
  - ollama: the `think=` argument to ollama.chat, plus the `<|think|>` token at
            the start of the system prompt (added in llm.build_system_prompt).
  - nvidia: the `chat_template_kwargs={"enable_thinking": ...}` request field,
            per NVIDIA's Gemma 4 API sample.
Either way, the reasoning is stripped from the reply by llm._RE_THINKING.
"""

import os
from typing import Iterator

LLM_BACKEND     = os.environ.get("LLM_BACKEND", "ollama").strip().lower()
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "qwen3.6:27b")
# Default to llama-3.3-70b: google/gemma-4-31b-it is listed but currently
# unresponsive on NVIDIA's hosted API (times out). Override via NVIDIA_MODEL.
NVIDIA_MODEL    = os.environ.get("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")


def active_model() -> str:
    """Model id for the current backend (used for banners/logging)."""
    return NVIDIA_MODEL if LLM_BACKEND == "nvidia" else OLLAMA_MODEL


def backend_label() -> str:
    return f"{LLM_BACKEND}:{active_model()}"


# ── NVIDIA (OpenAI-compatible) client, created lazily ────────────────────────
_nvidia_client = None


def _nvidia() -> "object":
    global _nvidia_client
    if _nvidia_client is None:
        from openai import OpenAI  # imported lazily so ollama-only setups need no openai
        if not NVIDIA_API_KEY:
            raise RuntimeError(
                "LLM_BACKEND=nvidia but NVIDIA_API_KEY is not set. "
                "Export it, e.g.  export NVIDIA_API_KEY=nvapi-..."
            )
        # Generous timeout: NVIDIA's free tier can be slow to first token
        # (llama-3.3-70b was observed at 20-80s under load).
        _nvidia_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=120)
    return _nvidia_client


# Spoken to the user when NVIDIA is too slow / errors, instead of crashing.
_NVIDIA_FALLBACK = "Désolé, la connexion est lente en ce moment. Tu peux répéter ?"


def _prepare_messages(messages: list[dict]) -> list[dict]:
    """Gemma models on NVIDIA reject the `system` role (500 "System role not
    supported"). For them, fold any system message(s) into the first user turn."""
    if "gemma" not in NVIDIA_MODEL.lower():
        return messages
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    if not system_parts:
        return messages
    prefix = "\n\n".join(system_parts) + "\n\n"
    out, injected = [], False
    for m in messages:
        if m["role"] == "system":
            continue
        if m["role"] == "user" and not injected:
            out.append({"role": "user", "content": prefix + m["content"]})
            injected = True
        else:
            out.append(m)
    if not injected:  # no user message at all — send the system text as the user turn
        out.insert(0, {"role": "user", "content": prefix.strip()})
    return out


# ── Public API ───────────────────────────────────────────────────────────────
def stream_chat(messages: list[dict], enable_thinking: bool) -> Iterator[str]:
    """Yield reply text chunks for a chat completion."""
    if LLM_BACKEND == "nvidia":
        # chat_template_kwargs is Gemma-specific; only send it when thinking is on
        # so it doesn't trip up models that don't accept the field (e.g. llama).
        extra = {"chat_template_kwargs": {"enable_thinking": True}} if enable_thinking else {}
        try:
            stream = _nvidia().chat.completions.create(
                model=NVIDIA_MODEL,
                messages=_prepare_messages(messages),
                stream=True,
                extra_body=extra,
            )
            for chunk in stream:
                if not chunk.choices:  # NVIDIA sends a final usage-only chunk with no choices
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            print(f"  [NVIDIA error] {type(e).__name__}: {str(e)[:200]}")
            yield _NVIDIA_FALLBACK
    else:
        import ollama
        for chunk in ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
            think=enable_thinking,
        ):
            yield chunk["message"]["content"]


def generate(prompt: str) -> str:
    """One-shot, non-streaming completion (used for word lookups)."""
    if LLM_BACKEND == "nvidia":
        try:
            resp = _nvidia().chat.completions.create(
                model=NVIDIA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"  [NVIDIA error] {type(e).__name__}: {str(e)[:200]}")
            return ""
    else:
        import ollama
        return ollama.generate(model=OLLAMA_MODEL, prompt=prompt, think=False)["response"]
