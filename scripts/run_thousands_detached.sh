#!/bin/bash
# Detached thousand-scale Hotpot → CodeRAG scoring (survives Cursor agent exit).
set -uo pipefail
cd /Users/jimmyscray/Code/rag-benchmark || exit 1
export PATH="$PWD/.venv/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export MPLCONFIGDIR=/tmp/mplconfig
mkdir -p /tmp/mplconfig
LOG=/tmp/hotpot_thousands.log
exec > >(tee -a "$LOG") 2>&1

echo "==== $(date) START pid=$$ ===="
# Prevent idle sleep while scoring
caffeinate -dims &
CAFFEINE_PID=$!
trap 'kill $CAFFEINE_PID 2>/dev/null || true' EXIT

echo "[Hotpot] scoring 7405 Q × semantic,rerank,hybrid (reuse 66k index)"
.venv/bin/python scripts/run_hotpot_benchmark.py all semantic_rag,rerank_semantic,hybrid_dense_sparse
echo "[Hotpot] finished exit=$?"

.venv/bin/python scripts/build_dashboard.py || true

echo "[CodeRAG] scoring 2103 Q"
.venv/bin/python scripts/run_code_rag_benchmark.py semantic_rag,rerank_semantic
echo "[CodeRAG] finished exit=$?"

.venv/bin/python scripts/build_dashboard.py || true
echo "==== $(date) ALL DONE ===="
