#!/bin/bash
#SBATCH --job-name=belladonna_rag_qdrant
#SBATCH --partition=gpu            # no GPU requested below
#SBATCH --time=02:00:00
#SBATCH --mem=16G                  # qdrant is remote (on saturn); RAG server is light — no 47G chroma load
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=all
#SBATCH --mail-user=your-email@example.com
# No --gres=gpu: embeddings/rerank/answer-LLM all run on the pluto gateway.
# Same serve code as the chroma job — only the backend differs — so this is the
# clean chroma-vs-qdrant comparison on identical current code.
#
# Usage: sbatch run_rag_qdrant.slurm <MODEL> [REASONING_EFFORT]
#   sbatch run_rag_qdrant.slurm GPT-OSS-120B
#   sbatch run_rag_qdrant.slurm GLM-5.2-FP8 max     # max thinking on the answer LLM

MODEL="${1:-GPT-OSS-120B}"
EFFORT="${2:-}"                          # empty => no reasoning_effort override
CONC="${3:-4}"                           # parallel questions (gateway load); default 4
OPTIONWISE="${4:-0}"                      # 1 => option-wise evidence adjudication before commit
BULK="${BULK:?export BULK = path to the belladonna project root (contains belladonna_rag_serve and benchmarking_belladonna)}"
PORT=$((9000 + SLURM_JOB_ID % 1000))

export BASE_URL="${BASE_URL:?export BASE_URL, e.g. http://your-gateway/v1/ (see .env.example)}"
export VIRTUAL_API_KEY="${VIRTUAL_API_KEY:?export VIRTUAL_API_KEY (see .env.example)}"
export BELLADONNA_VECTOR_BACKEND=qdrant
export BELLADONNA_QDRANT_URL="${BELLADONNA_QDRANT_URL:?export BELLADONNA_QDRANT_URL, e.g. http://your-qdrant-host:6333}"
export BELLADONNA_ANSWER_MAX_TOKENS=32768
export ANONYMIZED_TELEMETRY=False
export CHROMA_TELEMETRY_IMPL=none
# Reasoning effort applies to the grounded-answer call only (router/critic unaffected).
# Also enable the reasoning_content fallback: max thinking can spend the whole 32k
# budget on the hidden trace and return empty content — recover the letter from it.
if [ -n "$EFFORT" ]; then
  export BELLADONNA_REASONING_EFFORT="$EFFORT"
  export BELLADONNA_USE_REASONING_CONTENT=1
fi
# Option-wise evidence check: label each option supported/contradicted/partial/
# not-addressed against the retrieved evidence before committing.
[ "$OPTIONWISE" = "1" ] && export BELLADONNA_OPTIONWISE=1

# Fail fast if this node can't reach saturn's qdrant (else 200 garbage errors).
curl -sf -m 5 "$BELLADONNA_QDRANT_URL/collections" >/dev/null || { echo "qdrant unreachable at $BELLADONNA_QDRANT_URL — aborting"; exit 1; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate belladonna

cd "$BULK/belladonna_rag_serve"
python -m uvicorn app:app --host 127.0.0.1 --port "$PORT" &
trap 'kill $! 2>/dev/null' EXIT

for _ in $(seq 1 40); do
  curl -sf -m 3 "http://127.0.0.1:$PORT/api" >/dev/null && break
  sleep 3
done
curl -sf -m 3 "http://127.0.0.1:$PORT/api" >/dev/null || { echo "RAG server never came up — aborting"; exit 1; }

cd "$BULK/benchmarking_belladonna/bench"
# Tag the result file so variants never collide with the plain -qdrant-noev
# baseline: e.g. -qdrant-noev-effmax, -qdrant-noev-optionwise.
SUFFIX="-qdrant-noev"
[ -n "$EFFORT" ] && SUFFIX="${SUFFIX}-eff${EFFORT}"
[ "$OPTIONWISE" = "1" ] && SUFFIX="${SUFFIX}-optionwise"
export RAG_RESULT_SUFFIX="$SUFFIX"
# No --fresh: resume mode keeps good answers and retries only errored/unanswered
# ones (so a flaky-gateway rerun fills gaps instead of restarting). Add --fresh to redo all.
python run_rag.py \
  --models "$MODEL" \
  --rag-url "http://127.0.0.1:$PORT/query" \
  --concurrency "$CONC" --top-k 15
