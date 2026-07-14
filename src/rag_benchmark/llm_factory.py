"""Factory for OpenAI or local LLM clients."""

from __future__ import annotations

from typing import Any, Protocol

from openai import OpenAI

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.local_llm import TrackedLocalLLM
from rag_benchmark.token_tracker import TokenLedger, TrackedOpenAI


class TrackedLLMClient(Protocol):
    default_model: str

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        model: str | None = None,
        phase: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str: ...

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        phase: str,
    ) -> list[list[float]]: ...


def create_tracked_client(
    config: BenchmarkConfig,
    ledger: TokenLedger | None = None,
) -> TrackedLLMClient:
    ledger = ledger or TokenLedger()
    if config.llm_backend == "local":
        TrackedLocalLLM.ensure_ollama_ready(config.ollama_base_url, config.chat_model)
        return TrackedLocalLLM(config, ledger)

    api_key = config.openai_api_key
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required when llm.backend=openai. "
            "Set llm.backend=local in config/benchmark.yaml for fully local inference."
        )
    return TrackedOpenAI(OpenAI(api_key=api_key), ledger, config.chat_model)


def clone_client_for_ledger(
    client: TrackedLLMClient,
    config: BenchmarkConfig,
    ledger: TokenLedger,
) -> TrackedLLMClient:
    if isinstance(client, TrackedLocalLLM):
        return TrackedLocalLLM(config, ledger)
    if isinstance(client, TrackedOpenAI):
        return TrackedOpenAI(client.client, ledger, config.chat_model)
    raise TypeError(f"Unsupported client type: {type(client)}")
