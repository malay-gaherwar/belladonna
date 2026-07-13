#!/bin/bash
#SBATCH --job-name=belladonna_rag_chroma
#SBATCH --partition=gpu            # no GPU requested below; switch if a cpu partition exists
#SBATCH --time=04:00:00            # ~20min cold index load + slow RAG; raise if killed mid-run
#SBATCH --mem=80G                  # chroma HNSW indexes are ~47G resident (Elsevier 25G + EPMC 20G)
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=your-email@example.com
# No --gres=gpu on purpose: embeddings/rerank/answer-LLM all run on the pluto
# gateway, so this node needs CPU + RAM + network only.
#
# Usage: sbatch run_rag_chroma.slurm <MODEL> [REASONING_EFFORT]
#   sbatch run_rag_chroma.slurm GLM-5.2-FP8 high     # GLM needs effort=high (runaway reasoning)
#   sbatch run_rag_chroma.slurm GPT-OSS-120B         # others: default effort, matches their baseline

MODEL="${1:-GLM-5.2-FP8}"
EFFORT="${2:-}"                          # empty => no reasoning_effort override
BULK="${BULK:?export BULK = path to the belladonna project root (contains belladonna_rag_serve and benchmarking_belladonna)}"
PORT=$((9000 + SLURM_JOB_ID % 1000))     # unique per job so two runs can share a node

# Gateway creds + chroma config, exported BEFORE the server starts so it inherits them.
export BASE_URL="${BASE_URL:?export BASE_URL, e.g. http://your-gateway/v1/ (see .env.example)}"
export VIRTUAL_API_KEY="${VIRTUAL_API_KEY:?export VIRTUAL_API_KEY (see .env.example)}"
export BELLADONNA_VECTOR_BACKEND=chroma
export BELLADONNA_ANSWER_MAX_TOKENS=32768
export ANONYMIZED_TELEMETRY=False        # chroma telemetry hangs with no internet egress
export CHROMA_TELEMETRY_IMPL=none
[ -n "$EFFORT" ] && export BELLADONNA_REASONING_EFFORT="$EFFORT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate belladonna

# Start the chroma RAG server in the background; kill it whenever the job exits.
cd "$BULK/belladonna_rag_serve"
python -m uvicorn app:app --host 127.0.0.1 --port "$PORT" &
trap 'kill $! 2>/dev/null' EXIT

# /api comes up in seconds; the 47G indexes load lazily on the first /query.
for _ in $(seq 1 60); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/api" >/dev/null && break
  sleep 5
done
curl -sf -m 3 "http://127.0.0.1:$PORT/api" >/dev/null || { echo "RAG server never came up — aborting"; exit 1; }

# Result file: rag__<MODEL>-chroma-noev__... (live llm.py is the unconstrained/noev variant).
cd "$BULK/benchmarking_belladonna/bench"
export RAG_RESULT_SUFFIX=-chroma-noev
python run_rag.py \
  --models "$MODEL" \
  --rag-url "http://127.0.0.1:$PORT/query" \
  --concurrency 4 --top-k 15 --fresh
