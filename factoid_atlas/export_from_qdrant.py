#!/usr/bin/env python3
"""
BELLADONNA Factoid Atlas — export vectors + text from the existing Qdrant.

The factoids are ALREADY embedded (Qwen3-Embedding-8B, 1024-d, L2-normalised)
and live in the running Qdrant at $QDRANT_URL (default http://localhost:6333),
collections belladonna_<source>. We DO NOT re-embed — we scroll the collections
and pull each point's vector + payload (factoid_text + metadata) together, so the
layout vector, the LLM-classification text and the hover text are guaranteed to
be the same row.

Pure stdlib + `requests` (no numpy / qdrant-client needed) so it runs on the box
that hosts qdrant. Vectors are written as raw little-endian float32; the shape
sidecar lets build_layout.py load them with np.fromfile on the GPU box.

Point ids are UUID5(seed) → uniformly distributed, so scrolling the first N
points of a collection is an unbiased random sample. For the full corpus pass,
set a large --per-collection (or --total 0 with --all).

Outputs (stem = --out):
    <stem>.f32         raw float32 [N*dim]      (row-major, N rows of `dim`)
    <stem>.shape.json  {"n": N, "dim": 1024}
    <stem>.ids.json    [qdrant point id, ...]   aligned with rows
    <stem>.jsonl       {"id","source","year","doc_id","doc_title","text"} per row

Example:
    python export_from_qdrant.py --out proto --total 40000
    python export_from_qdrant.py --out full  --all          # everything
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

# collection -> canonical source name (matches build_atlas_data.SOURCES)
COLLECTIONS = {
    "belladonna_ago": "AGO", "belladonna_ctg": "CTG", "belladonna_elsevier": "Elsevier",
    "belladonna_ema": "EMA", "belladonna_epmc": "EPMC", "belladonna_esmo": "ESMO",
    "belladonna_fda": "FDA",
}
DIM = 1024
PAGE = 1000


def _year(p: dict):
    for k in ("document_year", "doc_YEAR", "YEAR", "year"):
        v = p.get(k)
        if v in (None, ""):
            continue
        s = "".join(ch for ch in str(v) if ch.isdigit())[:4]
        if len(s) == 4:
            y = int(s)
            if 1900 <= y <= 2035:
                return y
    return 0


def _doc_id(p: dict):
    for k in ("doc_PMCID", "PMCID", "doc_DOI", "doi", "doc_ID", "factoid_id", "file_name"):
        v = p.get(k)
        if v:
            return str(v)
    return ""


def count(col: str) -> int:
    r = requests.get(f"{QDRANT_URL}/collections/{col}", timeout=30)
    r.raise_for_status()
    return int(r.json()["result"].get("points_count") or 0)


def scroll(col: str, limit: int):
    """Yield (id, vector, payload) for up to `limit` points of a collection."""
    offset = None
    got = 0
    while got < limit:
        body = {"limit": min(PAGE, limit - got), "with_payload": True, "with_vector": True}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{QDRANT_URL}/collections/{col}/points/scroll",
                          json=body, timeout=120)
        r.raise_for_status()
        res = r.json()["result"]
        pts = res.get("points", [])
        if not pts:
            break
        for pt in pts:
            yield pt["id"], pt.get("vector"), pt.get("payload") or {}
            got += 1
            if got >= limit:
                break
        offset = res.get("next_page_offset")
        if offset is None:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Export vectors + text from Qdrant (no re-embed).")
    ap.add_argument("--out", required=True, help="output stem")
    ap.add_argument("--total", type=int, default=40000, help="approx total rows (sqrt-weighted across sources)")
    ap.add_argument("--per-collection", type=int, default=0, help="exact cap per collection (overrides --total)")
    ap.add_argument("--all", action="store_true", help="export ALL points (ignores --total)")
    ap.add_argument("--collections", default="", help="comma list of source names to include (default all)")
    args = ap.parse_args()

    want = set(s.strip() for s in args.collections.split(",") if s.strip())
    cols = {c: s for c, s in COLLECTIONS.items() if (not want or s in want)}

    counts = {c: count(c) for c in cols}
    print("[export] collection counts:", {cols[c]: n for c, n in counts.items()}, flush=True)

    if args.all:
        quota = dict(counts)
    elif args.per_collection:
        quota = {c: min(args.per_collection, counts[c]) for c in cols}
    else:
        weights = {c: math.sqrt(counts[c]) for c in cols if counts[c]}
        wsum = sum(weights.values()) or 1.0
        quota = {c: min(counts[c], max(1, round(args.total * w / wsum))) for c, w in weights.items()}
    print("[export] quota:", {cols[c]: q for c, q in quota.items()}, flush=True)

    out = Path(args.out)
    ids = []
    # STREAM vectors straight to disk so memory stays O(1) even for ~3M points
    # (accumulating in RAM would need ~12 GB at full scale).
    fbin = open(out.with_suffix(".f32"), "wb")
    fout = open(out.with_suffix(".jsonl"), "w", encoding="utf-8")
    t0 = time.time()
    n = 0
    bad_dim = 0
    for col, src in cols.items():
        k = quota.get(col, 0)
        if k <= 0:
            continue
        cn = 0
        for pid, v, p in scroll(col, k):
            if not v or len(v) != DIM:
                bad_dim += 1
                continue
            fbin.write(array.array("f", v).tobytes())
            ids.append(pid)
            fout.write(json.dumps({
                "id": pid, "source": src, "year": _year(p),
                "doc_id": _doc_id(p), "doc_title": p.get("document_title", ""),
                "text": (p.get("factoid_text") or p.get("text") or "").strip(),
            }, ensure_ascii=False) + "\n")
            n += 1
            cn += 1
        print(f"[export] {src}: {cn} ({n} total, {n/max(time.time()-t0,1e-6):.0f}/s)", flush=True)
    fout.close()
    fbin.close()

    out.with_suffix(".shape.json").write_text(json.dumps({"n": n, "dim": DIM}), encoding="utf-8")
    out.with_suffix(".ids.json").write_text(json.dumps(ids), encoding="utf-8")
    print(f"[export] wrote {n} vectors ({bad_dim} skipped bad dim) "
          f"-> {out.with_suffix('.f32')} ({out.with_suffix('.f32').stat().st_size/1e6:.0f} MB), "
          f"{out.with_suffix('.jsonl')}", flush=True)


if __name__ == "__main__":
    main()
