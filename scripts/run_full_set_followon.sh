#!/bin/bash
# After the Hotpot+CodeRAG thousands job finishes, score GraphRAG-Bench (all 72)
# and MultiHop-RAG (all 150) on the same local 3B generator.
# Does not touch the in-flight Hotpot process.
set -uo pipefail
cd /Users/jimmyscray/Code/rag-benchmark || exit 1
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/mplconfig
mkdir -p /tmp/mplconfig
LOG=/tmp/full_set_followon.log
exec >>"$LOG" 2>&1

WAIT_PID="${1:-20726}"
echo "==== $(date) follow-on start pid=$$ waiting_for=$WAIT_PID ===="
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "Waiting for thousands job $WAIT_PID (Hotpot 7405 → CodeRAG 2103) to finish so we do not steal Ollama."
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
  echo "Thousands job $WAIT_PID exited at $(date)"
else
  echo "Thousands pid $WAIT_PID not running — scoring remaining benches now."
fi

echo "[GraphRAG-Bench] all Novel questions (n_per=0)"
.venv/bin/python scripts/run_graphrag_bench.py semantic_rag,rerank_semantic,hybrid_rag,frontier_rag,lazygraph_rag 0
echo "[GraphRAG-Bench] finished exit=$?"
.venv/bin/python scripts/build_dashboard.py || true

echo "[MultiHop] all 150 (50 per type)"
.venv/bin/python scripts/run_multihop_benchmark.py semantic_rag,rerank_semantic,hybrid_rag,frontier_rag,lazygraph_rag 50
echo "[MultiHop] finished exit=$?"
.venv/bin/python scripts/build_dashboard.py || true

echo "==== $(date) FOLLOW-ON DONE ===="
