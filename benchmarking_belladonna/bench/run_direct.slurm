#!/bin/bash
#SBATCH --job-name=belladonna_direct
#SBATCH --partition=gpu            # no GPU requested below; gateway does all inference
#SBATCH --time=02:00:00
#SBATCH --mem=8G                   # direct = plain HTTP calls to the gateway; tiny footprint
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=all
#SBATCH --mail-user=your-email@example.com
# No --gres, no chroma/qdrant, no local server: run_direct.py POSTs each MCQ
# straight to the pluto gateway. Zero-shot, NO retrieval.
#
# Usage: sbatch run_direct.slurm [MODEL] [REASONING_EFFORT] [CONCURRENCY]
#   sbatch run_direct.slurm GPT-OSS-120B                 # edgecases prompt, conc 4
#   sbatch run_direct.slurm GLM-5.2-FP8 high 20          # high effort + rc rescue, conc 20
# Result suffix is auto-tagged (-edgecase or -edgecase-eff<effort>) so no run
# clobbers the plain baseline direct__<MODEL>__expert200v2.json.

MODEL="${1:-GPT-OSS-120B}"
EFFORT="${2:-}"                          # empty => no reasoning_effort override
CONC="${3:-4}"                           # parallel questions
BULK="${BULK:?export BULK = path to the belladonna project root (contains belladonna_rag_serve and benchmarking_belladonna)}"

export BASE_URL="${BASE_URL:?export BASE_URL, e.g. http://your-gateway/v1/ (see .env.example)}"
export VIRTUAL_API_KEY="${VIRTUAL_API_KEY:?export VIRTUAL_API_KEY (see .env.example)}"
# Reasoning models (GLM-5.2) can spend the whole budget on the hidden trace and
# return empty content — enable the reasoning_content rescue so a letter is still
# recovered. (This is the '-rc' that lifted GLM direct 74.5% -> 84.5%.)
if [ -n "$EFFORT" ]; then
  export DIRECT_REASONING_EFFORT="$EFFORT"
  export DIRECT_USE_REASONING_CONTENT=1
fi
export DIRECT_RESULT_SUFFIX="-edgecase${EFFORT:+-eff$EFFORT}"

# Fail fast if this node can't reach the gateway.
curl -sf -m 5 "${BASE_URL}models" -H "Authorization: Bearer $VIRTUAL_API_KEY" >/dev/null \
  || { echo "gateway unreachable at $BASE_URL — aborting"; exit 1; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate belladonna

cd "$BULK/benchmarking_belladonna/bench"
# No --fresh: resume mode keeps good answers and retries only errors on a rerun
# (MAX_RETRIES=1 on this copy, so a flaky gateway can leave gaps a rerun fills).
python run_direct.py --models "$MODEL" --concurrency "$CONC"
