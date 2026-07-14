from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tiktoken


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            calls=self.calls + other.calls,
        )

    def add_from_response(self, usage: Any) -> None:
        if usage is None:
            return
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", prompt + completion) or (prompt + completion)
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total
        self.calls += 1

    def to_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@dataclass
class TokenLedger:
    """Accumulates token usage by phase and model."""

    by_phase: dict[str, TokenUsage] = field(default_factory=dict)
    by_model: dict[str, TokenUsage] = field(default_factory=dict)

    def record(
        self,
        *,
        phase: str,
        model: str,
        usage: Any | None = None,
        text: str | None = None,
        role: str = "prompt",
    ) -> None:
        phase_usage = self.by_phase.setdefault(phase, TokenUsage())
        model_usage = self.by_model.setdefault(model, TokenUsage())

        if usage is not None:
            phase_usage.add_from_response(usage)
            model_usage.add_from_response(usage)
            return

        if text:
            tokens = count_tokens(text, model)
            if role == "completion":
                phase_usage.completion_tokens += tokens
                model_usage.completion_tokens += tokens
            else:
                phase_usage.prompt_tokens += tokens
                model_usage.prompt_tokens += tokens
            phase_usage.total_tokens += tokens
            model_usage.total_tokens += tokens
            phase_usage.calls += 1
            model_usage.calls += 1

    def phase(self, name: str) -> TokenUsage:
        return self.by_phase.setdefault(name, TokenUsage())

    def total(self) -> TokenUsage:
        total = TokenUsage()
        for usage in self.by_phase.values():
            total = total.add(usage)
        return total

    def estimate_cost_usd(self, pricing: dict[str, dict[str, float]]) -> float:
        cost = 0.0
        for model, usage in self.by_model.items():
            rates = pricing.get(model, {})
            input_rate = rates.get("input", 0.0) / 1_000_000
            output_rate = rates.get("output", 0.0) / 1_000_000
            cost += usage.prompt_tokens * input_rate
            cost += usage.completion_tokens * output_rate
        return cost

    def to_frame_rows(self, method: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for phase, usage in self.by_phase.items():
            rows.append(
                {
                    "method": method,
                    "phase": phase,
                    **usage.to_dict(),
                }
            )
        return rows


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


class TrackedOpenAI:
    """Thin wrapper around the OpenAI client that records token usage."""

    def __init__(self, client: Any, ledger: TokenLedger, default_model: str):
        self.client = client
        self.ledger = ledger
        self.default_model = default_model

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
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        self.ledger.record(phase=phase, model=model, usage=response.usage)
        return response.choices[0].message.content or ""

    def embed_texts(
        self,
        texts: list[str],
        *,
        model: str | None = None,
        phase: str,
    ) -> list[list[float]]:
        model = model or "text-embedding-3-small"
        response = self.client.embeddings.create(model=model, input=texts)
        self.ledger.record(phase=phase, model=model, usage=response.usage)
        return [item.embedding for item in response.data]
