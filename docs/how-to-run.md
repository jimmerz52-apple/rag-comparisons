# How to run this bake-off on the source data

Copy-paste commands. All paths are from the **repo root**.

Live dashboard: <https://jimmerz52-apple.github.io/rag-comparisons/>

---

## 0. What “source data” means here

Nothing is scraped at score time. Scripts **download public datasets once**, write a **local corpus of `.txt` files** + a **QA JSON**, then index/score those files.

| Bench | Upstream source | Local corpus | Local questions |
|-------|-----------------|--------------|-----------------|
| HotpotQA distractor val | HuggingFace `hotpotqa/hotpot_qa` split=`validation` (hard only) | `data/corpus_hotpot/*.txt` | `data/qa/hotpot_eval.json` |
| CodeRAG-Bench | HF `code-rag-bench/{humaneval,mbpp,ds1000,odex,programming-solutions,library-documentation}` | `data/corpus_code_rag/*.txt` | `data/qa/code_rag_eval.json` |
| GraphRAG-Bench Novel | GraphRAG-Bench Novel-4128 | `data/corpus_graphrag_bench/` | `data/qa/graphrag_bench_eval.json` |
| MultiHop-RAG | MultiHop-RAG news | `data/corpus_multihop/` | `data/qa/multihop_eval.json` |
| Tiny wiki smoke | Wikipedia titles in `scripts/run_benchmark.py` | `data/corpus/` | `data/qa/eval_questions.json` |

Your own docs: drop `.txt` files in a folder, point `config.corpus_dir` + a QA JSON (see §6).

---

## 1. Machine prerequisites

```bash
# Python 3.12 + venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

# Local LLM (required for generate + judge)
brew install ollama          # or https://ollama.com
ollama serve                 # leave running
ollama pull llama3.2:3b
ollama pull nomic-embed-text # GraphRAG / LightRAG embeddings
```

Confirm:

```bash
curl -s http://127.0.0.1:11434/api/tags
```

Models and URLs live in `config/benchmark.yaml`:

- `llm.chat_model` / `judge_model` → `llama3.2:3b`
- `llm.embedding_model` → `all-MiniLM-L6-v2` (sentence-transformers, **not** Ollama)
- `llm.graphrag_embedding_model` → `nomic-embed-text`

Optional `.env` (never commit):

```
HF_TOKEN=hf_...          # faster/less-throttled HuggingFace downloads
OLLAMA_HOST=http://127.0.0.1:11434
```

---

## 2. HotpotQA — 7,405 Q / ~66k Wikipedia paragraphs

**Materialize from HF (idempotent if already on disk):**

```bash
export PYTHONPATH=src
python -c "
from pathlib import Path
from rag_benchmark.hotpotqa import build_hotpot_subset
print(build_hotpot_subset(project_root=Path('.').resolve(), n_questions=None)['meta'])
"
```

Writes:

- `data/corpus_hotpot/<slug>.txt` — one file per unique distractor paragraph
- `data/qa/hotpot_eval.json` — 7405 questions
- `data/qa/hotpot_meta.json`

**Score (vector methods, reuse Chroma index):**

```bash
PYTHONPATH=src python scripts/run_hotpot_benchmark.py all
# equivalent:
PYTHONPATH=src python scripts/run_hotpot_benchmark.py all semantic_rag,rerank_semantic,hybrid_dense_sparse
```

Smoke (minutes, not days):

```bash
PYTHONPATH=src python scripts/run_hotpot_benchmark.py 50 semantic_rag,rerank_semantic
```

GraphRAG on the **full** 66k corpus is days on 3B — opt-in only:

```bash
PYTHONPATH=src python scripts/run_hotpot_benchmark.py 100 semantic_rag,lazygraph_rag
```

Index files: `.chroma/sdk_lineage/hotpot_semantic_full__all_minilm_l6_v2/`  
Scores: `results/accuracy_results.csv`, `results/summary.csv`, `results/latency_results.csv`, `results/token_results.csv` (prompt / completion / phase totals; live scoring also writes per-question token columns).  
Progress: every 25 Q → `results/_checkpoint_semantic_rag_accuracy.csv`  
Log (detached overnight): `/tmp/hotpot_thousands.log`

Detached full run:

```bash
./scripts/run_thousands_detached.sh
# or: caffeinate -dims ./scripts/run_thousands_detached.sh
```

---

## 3. CodeRAG-Bench — ~2,103 Q / ~34k library docs

Paper protocol: **leave gold HumanEval/MBPP solutions out of the datastore.**

```bash
PYTHONPATH=src python scripts/run_code_rag_benchmark.py
# methods:
PYTHONPATH=src python scripts/run_code_rag_benchmark.py semantic_rag,rerank_semantic
# optional StackOverflow (~20k extra posts):
PYTHONPATH=src python scripts/run_code_rag_benchmark.py semantic_rag --stackoverflow
```

Materialize only:

```bash
PYTHONPATH=src python -c "
from pathlib import Path
from rag_benchmark.code_rag_bench import build_code_rag_bench_subset
print(build_code_rag_bench_subset(project_root=Path('.').resolve())['meta'])
"
```

Results → `results_code_rag/`.

---

## 4. GraphRAG-Bench Novel (all 72) + MultiHop-RAG (all 150)

Defaults are the **full indexed sets**, not the old 2–3 questions per type.

```bash
PYTHONPATH=src python scripts/run_graphrag_bench.py
# n_per_type=0 → every Novel question that passed the coherence filter (~72)
PYTHONPATH=src python scripts/run_multihop_benchmark.py
# 50 per type × 3 = 150
```

After the detached Hotpot→CodeRAG job (`scripts/run_thousands_detached.sh`) finishes, a follow-on scores these two so they do not steal Ollama from the 7,405-Q run:

```bash
# already started if you used the agent follow-on; otherwise:
caffeinate -dims ./scripts/run_full_set_followon.sh
# log: /tmp/full_set_followon.log
```

Token / type notebooks (full catalog, not a 24-Q slice):

- `notebooks/hotpot_tokens.ipynb`
- `notebooks/code_rag_tokens.ipynb`
- `notebooks/graphrag_bench_tokens.ipynb`
- `notebooks/multihop_tokens.ipynb`

Generator stays `llama3.2:3b` (SETN 2026: 3B ≈ 8B when retrieval is strong; 1B–3B are retrieval-bound). Speed-ups are resume, skip-EM judge, and `num_predict=96` — not swapping the generator mid-run.

---

## 5. Rebuild the dashboard (GitHub Pages)

```bash
PYTHONPATH=src python scripts/build_dashboard.py
# → docs/index.html  (Pages)
# → results/dashboard.html
git add docs/ results/*.csv results/dashboard.html
```

CI: `.github/workflows/pages.yml` rebuilds and deploys `docs/` on push to `main`.

---

## 6. Run on **your** documents

1. Put one document per file:

```text
data/corpus_custom/handbook.txt
data/corpus_custom/api.md.txt
```

Plain UTF-8 text. First line can be a `# Title`.

2. Write questions (`data/qa/custom_eval.json`):

```json
[
  {
    "id": "q1",
    "question": "What is the refund window?",
    "expected_answer": "30 days",
    "query_type": "local"
  }
]
```

3. Score:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from rag_benchmark import BenchmarkConfig, BenchmarkRunner, create_tracked_client

ROOT = Path('.').resolve()
cfg = BenchmarkConfig.from_yaml(ROOT)
cfg.project_root = ROOT
cfg.corpus_dir = ROOT / 'data' / 'corpus_custom'
cfg.qa_path = ROOT / 'data' / 'qa' / 'custom_eval.json'
cfg.semantic_collection = 'custom_semantic'
cfg.max_documents = 100_000
cfg.reuse_indexes = True
cfg.results_dir = lambda: ROOT / 'results_custom'

runner = BenchmarkRunner(cfg, create_tracked_client(cfg))
results = runner.run_all(methods=['semantic_rag', 'rerank_semantic', 'hybrid_dense_sparse'])
print(runner.save_results(results))
PY
```

4. Point `scripts/build_dashboard.py` `BENCHES` at `results_custom/` **or** open `results_custom/summary.csv` directly.

---

## 7. Method IDs you can pass on the CLI

`semantic_rag` · `rerank_semantic` · `hybrid_dense_sparse` · `hybrid_rag` · `lazygraph_rag` · `graph_rag` · `graph_local_rag` · `adaptive_rag` · `frontier_rag` · `light_rag` · `hippo_rag` · `raptor_rag` · `parent_child_rag` · `proposition_rag`

---

## 8. What not to do

- Do **not** set `reuse_indexes=False` on the 66k Hotpot index unless you intend a full re-embed (slow; can hang Chroma).
- Do **not** default GraphRAG global on full Hotpot.
- Do **not** treat live `n_scored` as the full validation until `n_scored == n_questions` in `hotpot_meta.json`.
- HuggingFace downloads need network; unauthenticated requests throttle — set `HF_TOKEN`.
