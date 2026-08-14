from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from rag_benchmark.llm_factory import TrackedLLMClient


@dataclass
class EvalQuestion:
    id: str
    question: str
    expected_answer: str
    query_type: str
    source_doc: str | None = None
    best_method: str | None = None
    rationale: str | None = None
    supporting_titles: list[str] | None = None


@dataclass
class AccuracyResult:
    question_id: str
    method: str
    query_type: str = "local"
    llm_judge_score: float | None = None
    token_f1: float | None = None
    exact_match: bool | None = None
    contains_answer: bool | None = None
    judge_rationale: str | None = None
    retrieval_recall: float | None = None
    retrieval_precision: float | None = None
    gold_in_context: bool | None = None
    evidence_override: bool | None = None

    def composite_score(self) -> float:
        parts: list[float] = []
        if self.llm_judge_score is not None:
            parts.append(self.llm_judge_score)
        if self.token_f1 is not None:
            parts.append(self.token_f1)
        if self.exact_match is not None:
            parts.append(1.0 if self.exact_match else 0.0)
        if self.contains_answer is not None:
            parts.append(1.0 if self.contains_answer else 0.0)
        return sum(parts) / len(parts) if parts else 0.0

    def generative_score(self) -> float:
        """Judge + contains — fair for generative GraphRAG prose."""
        parts: list[float] = []
        if self.llm_judge_score is not None:
            parts.append(self.llm_judge_score)
        if self.contains_answer is not None:
            parts.append(1.0 if self.contains_answer else 0.0)
        return sum(parts) / len(parts) if parts else 0.0

    def extractive_score(self) -> float:
        """EM + token F1 — Hotpot-style short-span scoring."""
        parts: list[float] = []
        if self.token_f1 is not None:
            parts.append(self.token_f1)
        if self.exact_match is not None:
            parts.append(1.0 if self.exact_match else 0.0)
        return sum(parts) / len(parts) if parts else 0.0


def load_eval_questions(path: Any) -> list[EvalQuestion]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        EvalQuestion(
            id=item["id"],
            question=item["question"],
            expected_answer=item["expected_answer"],
            query_type=item.get("query_type", "local"),
            source_doc=item.get("source_doc"),
            best_method=item.get("best_method"),
            rationale=item.get("rationale"),
            supporting_titles=item.get("supporting_titles")
            or item.get("gold_titles")
            or item.get("evidence_titles"),
        )
        for item in payload
    ]


def exact_match(prediction: str, reference: str) -> bool:
    """HotpotQA-style normalized exact match."""
    return _normalize_answer(prediction) == _normalize_answer(reference)


def _normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _normalize_code(text: str) -> str:
    """Whitespace-insensitive code normalization for contains/EM on programs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip markdown fences if the model wrapped the answer
    text = re.sub(r"^```(?:python)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text.strip())
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def token_f1(prediction: str, reference: str, *, code: bool = False) -> float:
    pred_tokens = set(_normalize_tokens(prediction, code=code))
    ref_tokens = set(_normalize_tokens(reference, code=code))
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def contains_answer(prediction: str, reference: str, threshold: int = 80, *, code: bool = False) -> bool:
    if code:
        pred = _normalize_code(prediction)
        ref = _normalize_code(reference)
        if not ref:
            return False
        # Exact normalized match or high partial ratio on code body
        if ref in pred or pred in ref:
            return True
        return fuzz.partial_ratio(ref, pred) >= max(70, threshold - 10)
    return fuzz.partial_ratio(reference.lower(), prediction.lower()) >= threshold


def _normalize_tokens(text: str, *, code: bool = False) -> list[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    if code:
        # Keep short keywords (def/if/for) — they matter for programs
        return [t for t in tokens if t]
    return [token for token in tokens if len(token) > 2]


def _first_line_answer(text: str) -> str:
    """Use the first non-empty line so EM/F1 match Hotpot short spans."""
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•* ").strip()
        if cleaned:
            return cleaned
    return text.strip()


def _looks_like_code(question: EvalQuestion) -> bool:
    qid = str(question.id)
    if qid.startswith(("HumanEval", "MBPP", "HumanEval-", "MBPP-", "DS1000-", "ODEX-")):
        return True
    expected = question.expected_answer or ""
    return "def " in expected or "return " in expected or expected.lstrip().startswith("from ")


class AccuracyEvaluator:
    def __init__(self, tracked_client: TrackedLLMClient, judge_model: str):
        self.client = tracked_client
        self.judge_model = judge_model

    def evaluate(
        self,
        *,
        method: str,
        question: EvalQuestion,
        prediction: str,
        use_llm_judge: bool = True,
    ) -> AccuracyResult:
        code = _looks_like_code(question)
        # Code: score the full program, not the first line (signatures alone inflate F1 noise).
        pred_for_lexical = prediction if code else _first_line_answer(prediction)
        result = AccuracyResult(
            question_id=question.id,
            method=method,
            query_type=question.query_type,
            token_f1=token_f1(pred_for_lexical, question.expected_answer, code=code),
            exact_match=(
                _normalize_code(prediction) == _normalize_code(question.expected_answer)
                if code
                else exact_match(pred_for_lexical, question.expected_answer)
            ),
            contains_answer=contains_answer(
                prediction, question.expected_answer, code=code
            ),
        )

        if use_llm_judge:
            if result.exact_match:
                result.llm_judge_score = 1.0
                result.judge_rationale = "skipped: exact match"
            else:
                score, rationale = self._llm_judge(
                    question=question.question,
                    expected=question.expected_answer,
                    prediction=prediction,
                    code=code,
                )
                result.llm_judge_score = score
                result.judge_rationale = rationale

        return result

    def _llm_judge(
        self,
        *,
        question: str,
        expected: str,
        prediction: str,
        code: bool = False,
    ) -> tuple[float, str]:
        if code:
            prompt = f"""You are grading a code-generation answer for a RAG coding benchmark.

Problem:
{question}

Reference solution:
{expected}

Model solution:
{prediction}

Score 0.0–1.0 for functional correctness / algorithmic equivalence (not string identity).
Ignore minor naming/whitespace differences. Penalize wrong logic or missing edge cases.
Return JSON only: {{"score": <float>, "rationale": "<short reason>"}}"""
        else:
            prompt = f"""You are grading a RAG answer.

Question: {question}
Reference answer: {expected}
Model answer: {prediction}

Score from 0.0 to 1.0 based on factual overlap and completeness.
Return JSON only: {{"score": <float>, "rationale": "<short reason>"}}"""

        raw = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model=self.judge_model,
            phase="evaluation",
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        try:
            payload = json.loads(raw)
            score = float(payload.get("score", 0.0))
            rationale = str(payload.get("rationale", ""))
            return max(0.0, min(1.0, score)), rationale
        except (json.JSONDecodeError, TypeError, ValueError):
            return 0.0, f"Could not parse judge response: {raw[:200]}"
