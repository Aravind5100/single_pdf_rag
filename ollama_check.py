"""Friendly pre-flight check that Ollama is reachable and the needed models are pulled.

Used by the Streamlit app and the eval CLI so a missing server or model shows
a clear message instead of a raw requests traceback.
"""
from typing import Iterable

import requests

from embedding_strategy import EMBED_MODEL, OLLAMA_URL
from generation_strategy import LLM_MODEL

# OLLAMA_URL is ".../api/embeddings"; reuse its host instead of repeating it.
OLLAMA_BASE_URL = OLLAMA_URL.rsplit("/api/", 1)[0]


class OllamaNotReady(RuntimeError):
    """Raised when Ollama is unreachable or a required model is not pulled."""


def _is_pulled(model: str, pulled: set) -> bool:
    # "nomic-embed-text" is listed by Ollama as "nomic-embed-text:latest".
    return model in pulled or (":" not in model and f"{model}:latest" in pulled)


def check_ollama(required_models: Iterable[str] = (EMBED_MODEL, LLM_MODEL), timeout: float = 3.0) -> None:
    """Return None if Ollama is up and every required model is pulled, else raise OllamaNotReady."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OllamaNotReady(
            f"Cannot reach Ollama at {OLLAMA_BASE_URL}. "
            "Start it with `ollama serve` (or open the Ollama app) and try again."
        ) from exc

    pulled = {m.get("name", "") for m in resp.json().get("models", [])}
    missing = [m for m in required_models if not _is_pulled(m, pulled)]
    if missing:
        commands = "\n".join(f"  ollama pull {m}" for m in missing)
        raise OllamaNotReady(
            f"Ollama is running but these models are not pulled: {', '.join(missing)}.\n"
            f"Pull them with:\n{commands}"
        )
