from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag_benchmark.config import BenchmarkConfig
from rag_benchmark.token_tracker import TokenLedger


_LOG_PREFIXES = ("info", "debug", "warning", "error", "usage:", "litellm:")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass
class GraphRAGResult:
    answer: str
    raw_output: str
    search_method: str


class GraphRAGRunner:
    """Runs Microsoft GraphRAG via CLI in an isolated workspace."""

    def __init__(
        self,
        config: BenchmarkConfig,
        ledger: TokenLedger,
        *,
        workspace_dir: Path,
        indexing_method: str = "standard",
        search_method: str = "global",
        relevance_budget: int | None = None,
    ):
        self.config = config
        self.ledger = ledger
        self.workspace_dir = workspace_dir
        self.indexing_method = _normalize_index_method(indexing_method)
        self.search_method = _normalize_search_method(search_method)
        self.relevance_budget = relevance_budget

    def prepare_workspace(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        input_dir = self.workspace_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)

        if self.config.reuse_indexes and self._index_exists():
            return

        for stale in ("output", "cache", ".graphrag"):
            path = self.workspace_dir / stale
            if path.exists():
                shutil.rmtree(path)

        for source in sorted(self.config.corpus_dir.glob("*.txt"))[: self.config.max_documents]:
            shutil.copy2(source, input_dir / source.name)

        self._ensure_settings()
        self._ensure_env()

    def build_index(self) -> None:
        self.prepare_workspace()
        if self.config.reuse_indexes and self._index_exists():
            return
        command = [
            "graphrag",
            "index",
            "--root",
            str(self.workspace_dir),
            "--method",
            self.indexing_method,
        ]
        self._run_cli(command, phase="graph_index")

    def query(self, question: str) -> GraphRAGResult:
        command = [
            "graphrag",
            "query",
            "--root",
            str(self.workspace_dir),
            "--method",
            self.search_method,
            "--community-level",
            str(self.config.community_level),
            "--response-type",
            "Short Answer",
            question,
        ]
        output = self._run_cli(command, phase="graph_query")
        answer = self._extract_answer(output)
        self._estimate_query_tokens(question, answer)
        return GraphRAGResult(answer=answer, raw_output=output, search_method=self.search_method)

    def _run_cli(self, command: list[str], *, phase: str) -> str:
        env = os.environ.copy()
        env.setdefault("GRAPHRAG_API_KEY", env.get("OPENAI_API_KEY", ""))
        completed = subprocess.run(
            command,
            cwd=self.workspace_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            combined = "\n".join(part for part in (stdout, stderr) if part)
            raise RuntimeError(
                f"GraphRAG command failed ({' '.join(command)}):\n{combined}"
            )

        usage = self._parse_usage_from_logs(f"{stdout}\n{stderr}")
        if usage:
            self.ledger.record(phase=phase, model=self.config.chat_model, usage=usage)
        return stdout if stdout else stderr

    def _index_exists(self) -> bool:
        output = self.workspace_dir / "output"
        # community_reports required for global/drift; entities for local
        return (output / "entities.parquet").exists() and (
            (output / "community_reports.parquet").exists()
            or self.search_method in {"local", "basic"}
        )

    def _ensure_env(self) -> None:
        env_path = self.workspace_dir / ".env"
        if env_path.exists():
            return
        if self.config.llm_backend == "local":
            env_path.write_text("GRAPHRAG_API_KEY=ollama\n", encoding="utf-8")
            return
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GRAPHRAG_API_KEY", "")
        env_path.write_text(f"GRAPHRAG_API_KEY={api_key}\n", encoding="utf-8")

    def _ensure_settings(self) -> None:
        settings_path = self.workspace_dir / "settings.yaml"
        if settings_path.exists() and self.config.llm_backend != "local":
            return

        if settings_path.exists() and self.config.llm_backend == "local":
            settings_path.unlink()

        init = subprocess.run(
            [
                "graphrag",
                "init",
                "--root",
                str(self.workspace_dir),
                "--force",
                "--model",
                self.config.graphrag_chat_model,
                "--embedding",
                self.config.graphrag_embedding_model,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if init.returncode != 0 and not settings_path.exists():
            raise RuntimeError(
                f"GraphRAG init failed:\n{init.stdout}\n{init.stderr}"
            )

        if self.config.llm_backend == "local":
            self._patch_settings_for_ollama(settings_path)

    def _patch_settings_for_ollama(self, settings_path: Path) -> None:
        with settings_path.open("r", encoding="utf-8") as handle:
            settings = yaml.safe_load(handle)

        base_url = self.config.ollama_base_url.rstrip("/")
        for section in ("completion_models", "embedding_models"):
            models = settings.get(section, {})
            for model_cfg in models.values():
                model_cfg["model_provider"] = "ollama"
                model_cfg["api_base"] = base_url
                model_cfg["api_key"] = "ollama"
                model_cfg["auth_method"] = "api_key"

        if self.indexing_method == "fast":
            nlp = settings.setdefault("extract_graph_nlp", {})
            analyzer = nlp.setdefault("text_analyzer", {})
            analyzer["extractor_type"] = "syntactic_parser"
            analyzer["model_name"] = "en_core_web_sm"

        with settings_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(settings, handle, sort_keys=False)

    @staticmethod
    def _extract_answer(output: str) -> str:
        text = _ANSI_RE.sub("", output)
        paragraphs: list[str] = []
        current: list[str] = []

        for line in text.splitlines():
            cleaned = line.strip()
            if not cleaned:
                if current:
                    paragraphs.append(" ".join(current))
                    current = []
                continue
            lower = cleaned.lower()
            if lower.startswith(_LOG_PREFIXES):
                continue
            if cleaned.startswith("{") and "usage" in cleaned:
                continue
            current.append(cleaned)

        if current:
            paragraphs.append(" ".join(current))

        answer = "\n\n".join(paragraphs).strip()
        return answer or text.strip()

    @staticmethod
    def _parse_usage_from_logs(output: str) -> Any | None:
        for line in output.splitlines():
            if "usage" not in line.lower():
                continue
            try:
                payload = json.loads(line)
                if isinstance(payload, dict) and "usage" in payload:
                    return payload["usage"]
            except json.JSONDecodeError:
                continue
        return None

    def _estimate_query_tokens(self, question: str, answer: str) -> None:
        self.ledger.record(
            phase="graph_query_estimate",
            model=self.config.chat_model,
            text=question,
            role="prompt",
        )
        self.ledger.record(
            phase="graph_query_estimate",
            model=self.config.chat_model,
            text=answer,
            role="completion",
        )


def _normalize_index_method(method: str) -> str:
    aliases = {"lazy": "fast"}
    return aliases.get(method, method)


def _normalize_search_method(method: str) -> str:
    aliases = {"lazy": "basic"}
    return aliases.get(method, method)


class LazyGraphRAGRunner(GraphRAGRunner):
    """Cheap GraphRAG preset — NOT Microsoft LazyGraphRAG.

    Microsoft LazyGraphRAG (Nov 2024 blog) defers LLM work to query time with
    iterative-deepening relevance tests + a relevance-test budget. As of 2026-07
    that full system is in Microsoft Discovery / Azure Local preview, not in the
    open-source `graphrag` CLI (query methods: local|global|drift|basic only).

    What we run here is GraphRAG's shipped *FastGraphRAG*-style path:
    NLP noun-phrase indexing (`fast`) + `basic` search. Same cost *shape* as the
    LazyGraphRAG index story, different query algorithm. `relevance_budget` is
    retained for future wiring and currently unused.
    """

    def __init__(self, config: BenchmarkConfig, ledger: TokenLedger):
        super().__init__(
            config,
            ledger,
            workspace_dir=config.lazy_workspace,
            indexing_method="fast",
            search_method="basic",
            relevance_budget=config.lazy_relevance_budget,
        )
