#!/usr/bin/env bash
# Re-run ONLY the layout with init='pca' (kills the random-init disk artifact),
# then re-pack. Vectors + labels are untouched, so this is fast & reversible.
set -u
AT=/mnt/bulk-saturn/malaygaherwar/belladonna/factoid_atlas
LOG=/tmp/atlas_relayout.log
cd "$AT"
source ~/miniconda3/etc/profile.d/conda.sh; conda activate belladonna
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "==== relayout init=pca start ===="
cp -f full.xy.npy full.xy.npy.randinit.bak 2>/dev/null && say "backed up old layout -> full.xy.npy.randinit.bak"

say "STEP umap: running init=pca…"
python build_layout.py --stem full --pca-dim 50 --init pca --n-neighbors 30 --min-dist 0.1 > layout_pca.log 2>&1
RC=$?
say "STEP umap: exit=$RC  $(ls -lh full.xy.npy 2>/dev/null | awk '{print $5,$6,$7,$8}')"
if [ "$RC" -ne 0 ] || [ ! -s full.xy.npy ]; then
  say "UMAP FAILED — restoring previous layout"; cp -f full.xy.npy.randinit.bak full.xy.npy; exit 1
fi

say "STEP pack: rebuilding web/data…"
python build_atlas_data.py --stem full --meta full.jsonl --labels full.labels.jsonl --out web/data >> "$LOG" 2>&1
MN=$(python3 -c "import json;print(json.load(open('web/data/manifest.json'))['n'])" 2>/dev/null || echo 0)
say "STEP pack: DONE  manifest n=$MN"
say "==== relayout COMPLETE — re-rsync points.bin (docs unchanged) ===="
