#!/usr/bin/env python3
"""
BELLADONNA Factoid Atlas — 2D layout.

Projects the 1024-d factoid vectors to 2D for the scatter, and (optionally)
clusters them into k topics with readable TF-IDF labels — same recipe as the
existing /knowledge-graph/ build, but per-factoid.

Input:
    <stem>.f32.npy   float32 [N, 1024]  (from embed_factoids.py)
    <stem>.ids.json  aligned ids
Output:
    <stem>.xy.npy        float32 [N, 2]   UMAP coordinates
    <stem>.clusters.npy  uint16 [N]       k-means topic id per factoid (if --k>0)
    <stem>.topics.json   [{id, count, terms:[...]}]  (needs --texts for terms)

UMAP backend: prefers GPU cuml.UMAP if available (mars has RTX GPUs), else CPU
umap-learn. For 4.5M points, GPU is strongly preferred.

Example:
    python build_layout.py --stem proto --texts proto.jsonl --k 28 \
        --n-neighbors 30 --min-dist 0.1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(f"[layout] {msg}", flush=True)


def load_vectors(stem: Path) -> np.ndarray:
    """Load vectors from <stem>.f32.npy, or raw <stem>.f32 + <stem>.shape.json
    (the format export_from_qdrant.py writes without numpy)."""
    npy = stem.with_suffix(".f32.npy")
    if npy.exists():
        return np.load(npy)
    raw = stem.with_suffix(".f32")
    shape_path = stem.with_suffix(".shape.json")
    if raw.exists() and shape_path.exists():
        shp = json.loads(shape_path.read_text())
        return np.fromfile(raw, dtype="<f4").reshape(shp["n"], shp["dim"])
    sys.exit(f"no vectors found at {npy} or {raw}+{shape_path}")


def svd_reduce(vecs: np.ndarray, k: int) -> np.ndarray:
    """Randomized TruncatedSVD high-dim -> k-dim before UMAP. Drops the working
    set from ~12 GB (3M x 1024) to ~0.6 GB (3M x 50), which avoids the OOM that
    full-dim UMAP hits on a shared box, and speeds up the kNN graph build."""
    import gc
    from sklearn.decomposition import TruncatedSVD
    log(f"SVD {vecs.shape[1]}d -> {k}d on {len(vecs):,} rows…")
    t = time.time()
    svd = TruncatedSVD(n_components=k, algorithm="randomized", n_iter=5, random_state=42)
    red = np.asarray(svd.fit_transform(vecs), dtype=np.float32)
    log(f"SVD done in {(time.time()-t)/60:.1f} min — kept "
        f"{svd.explained_variance_ratio_.sum():.0%} variance, shape {red.shape}")
    del svd
    gc.collect()
    return red


def pca2d(vecs: np.ndarray, seed: int) -> np.ndarray:
    """Numpy-only 2D PCA. SMOKE-TEST FALLBACK ONLY — install umap-learn for the
    real layout; PCA gives a crude linear projection, fine to sanity-check the
    pipeline but not the production atlas."""
    log("FALLBACK: PCA-2D (numpy only) — install umap-learn/cuml for the real layout")
    X = vecs - vecs.mean(axis=0, keepdims=True)
    # top-2 right singular vectors via SVD on a (capped) sample for speed
    cap = min(len(X), 20000)
    _, _, Vt = np.linalg.svd(X[:cap], full_matrices=False)
    return np.asarray(X @ Vt[:2].T, dtype=np.float32)


def run_umap(vecs: np.ndarray, n_neighbors: int, min_dist: float, seed: int,
             init: str = "random") -> np.ndarray:
    """2D layout. Try GPU (cuml) → CPU (umap-learn) → numpy PCA fallback."""
    try:
        from cuml.manifold import UMAP as cuUMAP  # type: ignore
        log(f"using GPU cuml.UMAP (n={len(vecs)})")
        reducer = cuUMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric="cosine", random_state=seed)
        return np.asarray(reducer.fit_transform(vecs), dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        log(f"cuml unavailable ({type(e).__name__}); trying CPU umap-learn")
    try:
        import umap  # umap-learn
        n_neighbors = min(n_neighbors, max(2, len(vecs) - 1))
        # IMPORTANT: do NOT pass random_state — umap-learn forces single-threaded
        # (n_jobs=1) when a seed is set, which is untenable at millions of points.
        # Omitting it lets UMAP parallelise across all cores (n_jobs=-1).
        # init choice (see --init):
        #   'spectral' (umap default) eigendecomposes an N-node graph Laplacian —
        #     pathologically slow / stalls at millions of points. AVOID at scale.
        #   'random' is fast but has NO global anchoring: semantically diffuse points
        #     stay near their random start → a meaningless uniform disk/ring artifact.
        #   'pca' seeds from the data's top-2 PCs — fast (no graph eigendecomp) AND
        #     globally anchored, so the layout reflects real structure everywhere.
        log(f"umap-learn init={init!r}")
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                            metric="cosine", init=init, low_memory=True,
                            verbose=True, n_jobs=-1)
        return np.asarray(reducer.fit_transform(vecs), dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        log(f"umap-learn unavailable ({type(e).__name__})")
        return pca2d(vecs, seed)


# ---- TF-IDF topic labels (cheap, lexical; only for legend text) ----
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-/+]{2,}")
_STOP = set("""the and for with that this from are was were has have had not but you your they
their them which who whom whose when where what why how into onto over under than then thus
also may can will would could should about above below between within without across per via
patients patient treatment therapy study trial breast cancer disease clinical results result
data show shown using used based associated compared significant significantly versus group
groups arm arms phase year years month months day days dose doses given received including""".split())


def cluster_and_label(vecs, texts, k, seed):
    from sklearn.cluster import MiniBatchKMeans
    log(f"k-means k={k} on {len(vecs)} vectors")
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3)
    labels = km.fit_predict(vecs).astype(np.uint16)

    topics = []
    if texts is not None:
        from collections import Counter, defaultdict
        import math
        df = Counter()
        per = defaultdict(Counter)
        doc_tok = []
        for t in texts:
            toks = {w.lower() for w in _TOKEN.findall(t or "") if w.lower() not in _STOP}
            doc_tok.append(toks)
            for w in toks:
                df[w] += 1
        N = len(texts)
        for ci, toks in zip(labels, doc_tok):
            for w in toks:
                per[int(ci)][w] += 1
        for c in range(k):
            cnt = per[c]
            size = int((labels == c).sum())
            scored = sorted(
                ((w, f * math.log(N / (1 + df[w]))) for w, f in cnt.items()),
                key=lambda kv: kv[1], reverse=True,
            )
            terms = [w for w, _ in scored[:5]]
            topics.append({"id": c, "count": size, "terms": terms})
    else:
        for c in range(k):
            topics.append({"id": c, "count": int((labels == c).sum()), "terms": []})
    return labels, topics


def main() -> None:
    ap = argparse.ArgumentParser(description="UMAP 2D layout (+ optional k-means topics).")
    ap.add_argument("--stem", required=True, help="input/output stem (expects <stem>.f32.npy)")
    ap.add_argument("--texts", default="", help="JSONL with {id,text} for topic labels (optional)")
    ap.add_argument("--k", type=int, default=0, help="k-means topics (0 = skip)")
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", default="random", choices=["random", "pca", "spectral"],
                    help="UMAP init. 'pca' = globally-anchored & fast (recommended at "
                         "scale); 'random' = fast but produces a uniform-disk artifact; "
                         "'spectral' = umap default, stalls at millions of points.")
    ap.add_argument("--pca-dim", type=int, default=50,
                    help="SVD pre-reduce to this many dims before UMAP (0=off). "
                         "Cuts memory/time massively at scale; same recipe as the "
                         "production knowledge-graph atlas.")
    args = ap.parse_args()

    stem = Path(args.stem)
    vecs = load_vectors(stem)
    log(f"loaded {vecs.shape} vectors")

    if args.pca_dim and vecs.shape[1] > args.pca_dim:
        vecs = svd_reduce(vecs, args.pca_dim)

    t0 = time.time()
    xy = run_umap(vecs, args.n_neighbors, args.min_dist, args.seed, init=args.init)
    np.save(stem.with_suffix(".xy.npy"), xy)
    log(f"UMAP done in {(time.time()-t0)/60:.1f} min -> {stem.with_suffix('.xy.npy')}")

    if args.k > 0:
        texts = None
        if args.texts:
            ids = json.loads(stem.with_suffix(".ids.json").read_text())
            tmap = {}
            with open(args.texts, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        tmap[r["id"]] = r.get("text", "")
            texts = [tmap.get(i, "") for i in ids]
        labels, topics = cluster_and_label(vecs, texts, args.k, args.seed)
        np.save(stem.with_suffix(".clusters.npy"), labels)
        stem.with_suffix(".topics.json").write_text(json.dumps(topics, indent=2), encoding="utf-8")
        log(f"clusters -> {stem.with_suffix('.clusters.npy')}, topics -> {stem.with_suffix('.topics.json')}")


if __name__ == "__main__":
    main()
