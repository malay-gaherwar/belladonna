#!/usr/bin/env python3
"""
BELLADONNA Factoid Atlas — stratified factoid sampler.

Reads the per-source factoid JSONs under a data root and writes a flat JSONL of
factoids, sampled proportionally (with a per-source floor) across the eight
sources. Used to (a) build a quick validation set for the LLM classifier and
(b) build a prototype atlas before committing to the full 4.5M pass.

No qdrant / embeddings needed — pure file read.

Output JSONL line:
    {"id": "<factoid_id>", "source": "EPMC", "year": 2023,
     "doc_id": "PMC10062385", "text": "<factoid_text>"}

The id matches the qdrant point_key scheme used by the embedding scripts
(e.g. "PMC10062385_factoids_23", "FDA_114", "<eid>_<n>") so labels can later be
joined to the Qwen vectors pulled from qdrant.

Examples
--------
    # 200-per-source validation set
    python sample_factoids.py --data-root /home/malay/Documents/belladonna_v1_2 \
        --out sample.jsonl --per-source 200

    # ~50k stratified prototype set
    python sample_factoids.py --data-root $BELLADONNA_DATA_ROOT \
        --out proto.jsonl --total 50000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator, Optional

SOURCES = ["AGO", "ASCO", "CTG", "EMA", "EPMC", "ESMO", "Elsevier", "FDA"]


def _year(meta: dict) -> Optional[int]:
    for k in ("document_year", "YEAR", "year"):
        v = meta.get(k)
        if v is None:
            continue
        s = "".join(ch for ch in str(v) if ch.isdigit())[:4]
        if len(s) == 4:
            y = int(s)
            if 1900 <= y <= 2035:
                return y
    return None


def _doc_id(meta: dict, path: Path) -> str:
    for k in ("PMCID", "pmcid", "eid", "scopus_id", "file_name"):
        v = meta.get(k)
        if v:
            return str(v)
    return path.stem.replace("_factoids", "")


def _point_key(source: str, doc_id: str, fid, path: Path) -> str:
    """Best-effort reconstruction of the qdrant point_key for a factoid.
    Exact scheme varies per source; build_atlas_data.py ultimately trusts the
    payload ids pulled from qdrant, so this id is primarily a stable handle."""
    stem = path.stem  # e.g. "PMC10062385_factoids"
    if source == "FDA":
        return f"FDA_{fid}"
    return f"{stem}_{fid}"


def iter_source(data_root: Path, source: str) -> Iterator[dict]:
    fac_dir = data_root / source / "factoids"
    if not fac_dir.exists():
        return
    for path in sorted(fac_dir.glob("*_factoids.json")) or sorted(fac_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        meta = data.get("metadata", {}) or {}
        year = _year(meta)
        doc_id = _doc_id(meta, path)
        for fac in data.get("factoids", []) or []:
            if not isinstance(fac, dict):
                continue
            text = (fac.get("factoid_text") or "").strip()
            if not text:
                continue
            yield {
                "id": _point_key(source, doc_id, fac.get("id"), path),
                "source": source,
                "year": year,
                "doc_id": doc_id,
                "text": text,
            }


def reservoir(stream: Iterator[dict], k: int, rng: random.Random) -> list:
    """Uniform sample of up to k items from a stream of unknown length."""
    sample: list = []
    for n, item in enumerate(stream):
        if len(sample) < k:
            sample.append(item)
        else:
            j = rng.randint(0, n)
            if j < k:
                sample[j] = item
    return sample


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified factoid sampler.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=0,
                    help="Exact (max) factoids per source. Overrides --total.")
    ap.add_argument("--total", type=int, default=0,
                    help="Target total, split across sources by sqrt(size) weighting.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)

    # Pass 1: count factoids per source (cheap-ish; reads files once).
    if args.per_source:
        quota = {s: args.per_source for s in SOURCES}
    else:
        counts = {}
        for s in SOURCES:
            counts[s] = sum(1 for _ in iter_source(data_root, s))
        # sqrt weighting so tiny guideline sources still appear meaningfully.
        import math
        weights = {s: math.sqrt(c) for s, c in counts.items() if c}
        wsum = sum(weights.values()) or 1.0
        quota = {s: max(1, round(args.total * w / wsum)) for s, w in weights.items()}
        print("[sample] per-source quota:", quota, flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for s in SOURCES:
            k = quota.get(s, 0)
            if k <= 0:
                continue
            picked = reservoir(iter_source(data_root, s), k, rng)
            for rec in picked:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += len(picked)
            print(f"[sample] {s}: {len(picked)}", flush=True)
    print(f"[sample] wrote {written} factoids -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
