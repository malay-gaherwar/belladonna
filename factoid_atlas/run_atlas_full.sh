#!/usr/bin/env bash
# Drives the full-corpus factoid-atlas build AFTER the export is launched.
# Steps (one by one, with gates): wait export -> stop qdrant -> classify(resume)
# + UMAP in parallel -> pack. All milestones logged to $LOG.
set -u
BASE=/mnt/bulk-saturn/malaygaherwar/belladonna
AT=$BASE/factoid_atlas
LOG=/tmp/atlas_pipeline.log
cd "$AT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate belladonna
# Import creds straight from the user's bashrc export lines (bypasses the
# interactive-shell early-return; nothing secret is ever stored in this file).
eval "$(grep -hE '^[[:space:]]*export[[:space:]]+(VIRTUAL_API_KEY|BASE_URL)=' ~/.bashrc 2>/dev/null)"
: "${BASE_URL:=http://192.168.33.27/v1/}"; export BASE_URL
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
if [ -z "${VIRTUAL_API_KEY:-}" ]; then echo "VIRTUAL_API_KEY not found in env — abort" | tee -a "$LOG"; exit 1; fi

say "==== atlas pipeline driver start ===="

# 1) wait for the running export to finish
say "STEP export: waiting…"
while pgrep -f "export_from_qdrant.py" >/dev/null 2>&1; do sleep 20; done
if ! grep -q "\[export\] wrote" export_full.log; then say "EXPORT FAILED (no 'wrote' line) — abort"; exit 1; fi
N=$(python3 -c "import json;print(json.load(open('full.shape.json'))['n'])" 2>/dev/null || echo 0)
say "STEP export: DONE  N=$N  $(ls -lh full.f32 | awk '{print $5}') f32"
if [ "$N" -lt 4000000 ]; then say "N too small ($N) — abort"; exit 1; fi

# 2) stop qdrant to free RAM for UMAP (it was down before we started it)
say "STEP qdrant: stopping to free RAM…"
pkill -f "qdrant_bin/qdrant" 2>/dev/null; sleep 4
say "STEP qdrant: stopped"

# 3) launch classify (resume — skips ids already in full.labels.jsonl) in background
DONE0=$(wc -l < full.labels.jsonl 2>/dev/null || echo 0)
say "STEP classify: launching (resume base=$DONE0 labels)…"
nohup setsid python classify_factoids.py --input full.jsonl --output full.labels.jsonl \
    --concurrency 200 > classify_full2.log 2>&1 </dev/null & disown
sleep 8
say "STEP classify: $(grep -m1 'factoids ·' classify_full2.log || echo 'starting')"

# 4) UMAP layout on the full corpus (foreground within driver; runs alongside classify)
say "STEP umap: running…"
python build_layout.py --stem full --pca-dim 50 > layout_full.log 2>&1
RC=$?
say "STEP umap: exit=$RC  $(ls -lh full.xy.npy 2>/dev/null | awk '{print $5,$6,$7,$8}')"
if [ "$RC" -ne 0 ] || [ ! -s full.xy.npy ]; then say "UMAP FAILED — see layout_full.log (continuing to wait on classify)"; fi

# 5) wait for classify to finish
say "STEP classify: waiting for completion…"
while pgrep -f "classify_factoids.py" >/dev/null 2>&1; do sleep 30; done
LBL=$(wc -l < full.labels.jsonl 2>/dev/null || echo 0)
say "STEP classify: DONE  labels=$LBL"

# 6) pack the web bundle
if [ -s full.xy.npy ]; then
  say "STEP pack: building web/data…"
  python build_atlas_data.py --stem full --meta full.jsonl --labels full.labels.jsonl --out web/data >> "$LOG" 2>&1
  MN=$(python3 -c "import json;print(json.load(open('web/data/manifest.json'))['n'])" 2>/dev/null || echo 0)
  say "STEP pack: DONE  manifest n=$MN  $(du -sh web | awk '{print $1}')"
  say "==== PIPELINE COMPLETE ===="
else
  say "skipping pack — full.xy.npy missing; ==== PIPELINE INCOMPLETE ===="
fi
