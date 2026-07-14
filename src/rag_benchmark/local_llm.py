"""Local LLM client: Ollama chat + sentence-transformers embeddings."""

from __future__ import annotations

from typing import Any

import httpx
from sentence_transformers import SentenceTransformer

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.token_tracker import TokenLedger

_EMBEDDER_CACHE: dict[str, SentenceTransformer] = {}


def _get_embedder(model_name: str) -> SentenceTransformer:
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


class TrackedLocalLLM:
    """Runs chat via Ollama and embeddings via sentence-transformers (fully local)."""

    def __init__(self, config: BenchmarkConfig, ledger: TokenLedger):
        self.config = config
        self.ledger = ledger
        self.default_model = config.chat_model
        self._ollama_url = config.ollama_base_url.rstrip("/")
        self._embedder = _get_embedder(config.embedding_model)

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        phase: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        model = model or self.default_model
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if kwargs.get("response_format", {}).get("type") == "json_object":
            payload["format"] = "json"

        response = httpx.post(
            f"{self._ollama_url}/api/chat",
            json=payload,
            timeout=600.0,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]

        prompt_text = "\n".join(message["content"] for message in messages)
        self.ledger.record(phase=phase, model=model, text=prompt_text, role="prompt")
        self.ledger.record(phase=phase, model=model, text=content, role="completion")
        return content

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        phase: str,
    ) -> list[list[float]]:
        model = model or self.config.embedding_model
        vectors = self._embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        for text in texts:
            self.ledger.record(
                phase=phase,
                model=model,
                text=text,
                role="prompt",
            )
        return vectors.tolist()

    @staticmethod
    def ensure_ollama_ready(base_url: str, chat_model: str) -> None:
        """Verify Ollama is running and the chat model is available."""
        url = base_url.rstrip("/")
        try:
            response = httpx.get(f"{url}/api/tags", timeout=10.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {url}. Start it with: brew services start ollama"
            ) from exc

        models = {item["name"] for item in response.json().get("models", [])}
        aliases = {chat_model, f"{chat_model}:latest"}
        if not models.intersection(aliases) and not any(
            name.startswith(f"{chat_model}:") for name in models
        ):
            raise RuntimeError(
                f"Ollama model '{chat_model}' not found. Run: ollama pull {chat_model}"
            )
