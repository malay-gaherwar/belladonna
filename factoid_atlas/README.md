# BELLADONNA — Factoid Atlas

A per-**factoid** embedding scatter (à la the Apollo concept-atlas, Mahmood lab)
where the same cloud of dots can be **recoloured by clinical meaning** instead of
just by source. Each dot is one factoid; GPT-OSS-120B labels it along four axes
and you toggle which axis drives the colour.

```
 source · drug class · biomarker/molecular · disease setting · evidence type · topic · year
```

## Pipeline (uses the EXISTING embeddings — no re-embedding)

```
Qdrant @ localhost:6333  (belladonna_<source>, 1024-d Qwen3-Embedding-8B, already built)
        │
        │ export_from_qdrant.py     scroll vectors + payload together (requests, stdlib)
        ▼
   proto.f32 (raw float32 vectors) + proto.ids.json + proto.jsonl {id,source,year,doc_id,text}
        │
        ├─► classify_factoids.py ─► proto.labels.jsonl     (GPT-OSS-120B, 4 dims, batched)
        ├─► build_layout.py      ─► proto.xy.npy (+ topics) (UMAP 2D + k-means; GPU cuml or CPU)
        ▼
   build_atlas_data.py  (joins vectors-meta + labels + xy)
        ▼
   web/data/{points.bin, attrs.bin, docs.json, manifest.json, clusters.json}
   web/  (index.html + app.js: ImageData scatter, 7 colour modes)
```

| File | Role |
|---|---|
| `taxonomy.py` / `taxonomy.json` | The 4-dimension colour scheme. `taxonomy_preview.html` shows the swatches. |
| `export_from_qdrant.py` | **Vector source.** Scrolls the existing Qdrant collections → raw `.f32` vectors + aligned `.jsonl` (text+meta). No re-embedding. |
| `classify_factoids.py` | Batched, resumable GPT-OSS-120B classifier → 4 labels/factoid. |
| `build_layout.py` | UMAP→2D (GPU cuml if present, else umap-learn) + optional k-means topics. Reads `.f32.npy` or raw `.f32`+`.shape.json`. |
| `build_atlas_data.py` | Packs the binary atlas files + manifest for the web renderer. |
| `analyze_labels.py` | QA: label distributions, per-source coverage, drug-dictionary agreement. |
| `sample_factoids.py` | Stratified sampler from factoid JSONs — used for the taxonomy validation set (not needed once we read from qdrant). |
| `embed_factoids.py` | **Unused fallback.** Re-embeds via the gateway; only if qdrant is ever unavailable. The default path reads existing vectors from qdrant. |
| `web/` | The viewer (fork of `/knowledge-graph/`, modes data-driven from manifest). |

## Where things run

* **Laptop** hosts qdrant (`localhost:6333`, all 7 collections) + the factoid data → run `export_from_qdrant.py` here.
* **mars** has the GPU + the LLM/embedding gateway (`BASE_URL`, serves `GPT-OSS-120B`) →
  run `classify_factoids.py` and `build_layout.py` here.

```bash
# mars env
eval "$(grep -E '^export (VIRTUAL_API_KEY|BASE_URL)=' ~/.bashrc)"
source ~/miniconda3/etc/profile.d/conda.sh && conda activate belladonna
# build_layout.py needs UMAP:  pip install umap-learn   (CPU) or a cuml GPU env
```

## Prototype run (~40k factoids)

```bash
# 1) export vectors + text from the existing qdrant (LAPTOP, stdlib+requests)
python export_from_qdrant.py --out proto --total 40000
#    scp proto.f32 proto.shape.json proto.ids.json proto.jsonl  mars:.../factoid_atlas/

# 2) on mars — classify (gateway) + layout (UMAP), independent
python classify_factoids.py --input proto.jsonl --output proto.labels.jsonl
python build_layout.py      --stem proto --texts proto.jsonl --k 28

# 3) pack
python build_atlas_data.py --stem proto --meta proto.jsonl \
    --labels proto.labels.jsonl --topics proto.topics.json --out web/data

# 4) view (serve statically; fetch() needs http, not file://)
#    scp web/data back to the laptop, then:
cd web && python3 -m http.server 8000     # open http://localhost:8000
```

## Scaling to the full corpus (~4.5M factoids)

* `export_from_qdrant.py --all` exports every point (1024-d × 4.5M ≈ 18 GB float32).
* Classification ≈ 4.5M / ~25 factoids·s⁻¹ per 50-way batch → run sharded/resumable.
* `build_layout.py` should use **GPU UMAP (cuml)** at this scale; CPU umap-learn is hours.
* `docs.json` (hover text) must be **sharded** for the browser at full scale — the
  prototype loads it whole; production should lazy-load by index range.
