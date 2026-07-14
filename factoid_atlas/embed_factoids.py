#!/usr/bin/env python3
"""
BELLADONNA Factoid Atlas — embedder.

  ⚠️ UNUSED FALLBACK. The factoids are ALREADY embedded and stored in Qdrant
  (belladonna_v1_3/qdrant_storage). The default pipeline reads those existing
  vectors via export_from_qdrant.py and does NOT re-embed. This script is kept
  only in case the qdrant store is ever unavailable and vectors must be rebuilt.


Embeds factoid text with the team's hosted Qwen3-Embedding-8B (the same model
already used to build the qdrant collections), then Matryoshka-truncates to
1024 dims and L2-normalises — identical to scripts/qdrant_ingest.py — so the
atlas layout lives in the same vector space as the RAG index.

We re-embed (rather than scroll qdrant) because the gateway serves the model and
the 20 GB qdrant_storage is not co-located with the GPU box. The embedding text
is the factoid_text alone (optionally prefixed with the document title) — the
atlas dot IS a factoid, so its position should reflect the factoid's meaning.

Input  (--input):  JSONL with {"id", "text", ...} (e.g. from sample_factoids.py)
Output:
    <out>.f32.npy   float32 [N, 1024], L2-normalised   (or raw .f32 if --raw-bin)
    <out>.ids.json  list of ids aligned row-for-row with the matrix

Env: VIRTUAL_API_KEY, BASE_URL  (server ~/.bashrc)

Example:
    python embed_factoids.py --input proto.jsonl --out proto --batch-size 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

EMBED_MODEL = os.getenv("EMBED_MODEL", "Qwen3-Embedding-8B")
ORIG_DIM = 4096
TARGET_DIM = 1024
MAX_TEXT_CHARS = 1200


def truncate_normalize(vec: np.ndarray, dim: int = TARGET_DIM) -> np.ndarray:
    v = np.asarray(vec[:dim], dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def make_client():
    from openai import AsyncOpenAI
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        sys.exit("ERROR: VIRTUAL_API_KEY / BASE_URL not set (source server ~/.bashrc).")
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=2)


def embed_text(rec: dict, with_title: bool) -> str:
    text = (rec.get("text") or "").strip()
    if with_title and rec.get("doc_title"):
        text = f"{rec['doc_title']}. {text}"
    return text[:MAX_TEXT_CHARS]


async def embed_batch(client, texts: List[str]) -> List[np.ndarray]:
    """Embed a batch; on failure, fall back to one-at-a-time so a single bad
    item can't sink the batch (mirrors qdrant_ingest._embed_batch)."""
    try:
        resp = await client.embeddings.create(model=EMBED_MODEL, input=texts)
        return [truncate_normalize(np.asarray(d.embedding, dtype=np.float32)) for d in resp.data]
    except Exception as exc:  # noqa: BLE001
        if len(texts) <= 1:
            print(f"  [embed] single-item failure: {type(exc).__name__}: {exc}", flush=True)
            return [np.zeros(TARGET_DIM, dtype=np.float32)]
        out: List[np.ndarray] = []
        for t in texts:
            out.extend(await embed_batch(client, [t]))
        return out


async def run(args) -> None:
    in_path = Path(args.input)
    records = [r for r in read_jsonl(in_path) if (r.get("text") or "").strip()]
    if args.limit:
        records = records[: args.limit]
    N = len(records)
    if N == 0:
        sys.exit("nothing to embed")
    print(f"[embed] {N} factoids · model={EMBED_MODEL} · batch={args.batch_size} "
          f"· concurrency={args.concurrency} -> {TARGET_DIM}d", flush=True)

    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)
    mat = np.zeros((N, TARGET_DIM), dtype=np.float32)
    ids = [r["id"] for r in records]
    texts = [embed_text(r, args.with_title) for r in records]

    t0 = time.time()
    done = 0
    lock = asyncio.Lock()

    async def worker(start: int):
        nonlocal done
        chunk = texts[start:start + args.batch_size]
        async with sem:
            vecs = await embed_batch(client, chunk)
        for k, v in enumerate(vecs):
            mat[start + k] = v
        async with lock:
            done += len(chunk)
            if done % (args.batch_size * 20) < args.batch_size:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done}/{N} ({rate:.0f}/s, ETA {(N-done)/max(rate,1e-6)/60:.1f} min)", flush=True)

    starts = list(range(0, N, args.batch_size))
    inflight = args.concurrency * 4
    pending: set = set()
    for s in starts:
        pending.add(asyncio.create_task(worker(s)))
        if len(pending) >= inflight:
            _, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    if pending:
        await asyncio.wait(pending)

    out = Path(args.out)
    np.save(out.with_suffix(".f32.npy"), mat)
    out.with_suffix(".ids.json").write_text(json.dumps(ids), encoding="utf-8")
    zero = int((np.abs(mat).sum(axis=1) == 0).sum())
    print(f"[embed] done — {N} vectors in {(time.time()-t0)/60:.1f} min "
          f"({zero} failed/zero) -> {out.with_suffix('.f32.npy')}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed factoids via Qwen3-Embedding-8B gateway.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="output stem (writes <out>.f32.npy + <out>.ids.json)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--with-title", action="store_true", help="prefix factoid with doc_title if present")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
