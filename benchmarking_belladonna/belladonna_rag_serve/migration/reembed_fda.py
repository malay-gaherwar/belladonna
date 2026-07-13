"""Re-embed FDA factoids and ingest directly into Qdrant.

Why this is a separate script instead of chroma_to_qdrant.py:
- FDA's chroma persistence is broken: data_level0.bin only holds 100 of
  the 263 vectors, the index_metadata.pickle is missing entirely, and the
  embeddings_queue contains only delete operations. The 4096-dim vectors
  cannot be recovered from this state.
- FDA is small (263 factoids) and the source JSON is clean, so the
  cheapest fix is to re-embed from `fda_factoids.json` against the same
  inference endpoint we use at query time and write straight to Qdrant.

Embeddings are truncated to 1024 dims (Matryoshka) and L2-renormalized to
match the rest of the corpus.

Usage:
    export VIRTUAL_API_KEY=...
    export BASE_URL=http://.../v1/
    python migration/reembed_fda.py
    python migration/reembed_fda.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


FDA_JSON   = Path("/home/malay/Documents/belladonna_v1_2/FDA/factoids/fda_factoids.json")
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "belladonna_fda"
TARGET_DIM = 1024
MODEL_NAME = "Qwen3-Embedding-8B"
EMBED_BATCH = 4   # small batches: the inference endpoint times out on 16
EMBED_TIMEOUT = 180.0    # seconds per embedding call
EMBED_MAX_RETRIES = 5

# Same namespace as chroma_to_qdrant.py so re-ingesting upserts deterministically.
NAMESPACE_BELLADONNA = uuid.UUID("8d2f8b1a-2b3c-4f5e-9d6a-7f8a9b0c1d2e")


def _openai_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("VIRTUAL_API_KEY or BASE_URL not set")
    # Long timeout + retries so transient slowness on the shared inference
    # endpoint doesn't kill the run halfway through.
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=EMBED_TIMEOUT,
        max_retries=EMBED_MAX_RETRIES,
    )


def _year_from_label_date(label_date: Any) -> str:
    """FDA label_date is "YYYYMMDD" as a string. Extract YYYY."""
    s = str(label_date or "").strip()
    if len(s) >= 4 and s[:4].isdigit():
        return s[:4]
    return ""


def _build_payload(factoid: Dict[str, Any]) -> Dict[str, Any]:
    """Shape the FDA factoid record into the payload the retriever expects.

    Mirrors what `chroma_to_qdrant.build_payload` produces: a flat dict
    with `factoid_text` and `source` at top level plus all the original
    fields so downstream code (evidence tier classifier, citation label
    builder) can reach them."""
    text = (factoid.get("factoid_text") or "").strip()
    year = _year_from_label_date(factoid.get("label_date"))
    brand = (factoid.get("brand_name") or "").strip()
    generic = (factoid.get("generic_name") or "").strip()
    title = f"FDA label: {brand or generic}".strip(": ") or "FDA label"
    payload: Dict[str, Any] = dict(factoid)  # keep everything
    payload["factoid_text"] = text
    payload["source"] = "FDA"
    payload["source_family"] = "FDA Labels"
    payload["document_title"] = title
    payload["document_year"] = year
    payload["document_type"] = "Regulatory label"
    payload["factoid_id"] = f"FDA_{factoid.get('id')}"
    return payload


def ensure_collection(client: QdrantClient) -> None:
    """Create with the same config as chroma_to_qdrant.py so the index is
    consistent across sources (INT8 quantized in RAM, full vectors on disk)."""
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(
            size=TARGET_DIM,
            distance=qm.Distance.COSINE,
            on_disk=True,
        ),
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8,
                quantile=0.99,
                always_ram=True,
            ),
        ),
        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100, on_disk=False),
    )


def embed_batch(openai: OpenAI, texts: List[str]) -> List[np.ndarray]:
    """Embed a batch, truncate each to TARGET_DIM, L2-renormalize.

    Falls back to one-at-a-time on batch failure so a single slow item
    doesn't take down the whole batch."""
    try:
        resp = openai.embeddings.create(model=MODEL_NAME, input=texts)
    except Exception as exc:
        if len(texts) <= 1:
            raise
        print(f"  batch of {len(texts)} failed ({type(exc).__name__}); retrying one at a time")
        out: List[np.ndarray] = []
        for t in texts:
            r = openai.embeddings.create(model=MODEL_NAME, input=[t])
            v = np.asarray(r.data[0].embedding, dtype=np.float32)[:TARGET_DIM]
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            out.append(v)
        return out

    out = []
    for item in resp.data:
        v = np.asarray(item.embedding, dtype=np.float32)[:TARGET_DIM]
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        out.append(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="report only — don't embed or write")
    p.add_argument("--qdrant-url", default=QDRANT_URL)
    args = p.parse_args()

    if not FDA_JSON.exists():
        print(f"missing FDA factoid file: {FDA_JSON}", file=sys.stderr)
        sys.exit(2)
    raw = json.loads(FDA_JSON.read_text())
    factoids = raw.get("factoids") or []
    print(f"[FDA] {len(factoids)} factoids in source JSON")

    if args.dry_run:
        return

    openai = _openai_client()
    client = QdrantClient(url=args.qdrant_url, timeout=60.0)
    ensure_collection(client)

    points: List[qm.PointStruct] = []
    t0 = time.time()
    n_embedded = 0

    for i in range(0, len(factoids), EMBED_BATCH):
        batch = factoids[i : i + EMBED_BATCH]
        texts = [(f.get("factoid_text") or "").strip() for f in batch]
        keep = [(b, t) for b, t in zip(batch, texts) if t]
        if not keep:
            continue
        vecs = embed_batch(openai, [t for _, t in keep])
        for (factoid, _), vec in zip(keep, vecs):
            payload = _build_payload(factoid)
            point_id = str(uuid.uuid5(NAMESPACE_BELLADONNA, payload["factoid_id"]))
            points.append(qm.PointStruct(id=point_id, vector=vec.tolist(), payload=payload))
        # Flush periodically so a mid-run crash doesn't lose everything.
        if len(points) >= 64:
            client.upsert(collection_name=COLLECTION, points=points, wait=False)
            n_embedded += len(points)
            points = []
            print(f"  {n_embedded}/{len(factoids)} ingested  ({n_embedded / max(time.time() - t0, 1e-6):.1f}/s)")

    if points:
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        n_embedded += len(points)

    elapsed = time.time() - t0
    final = client.count(COLLECTION, exact=True).count
    print(f"[FDA] done: embedded {n_embedded}, qdrant count={final}, took {elapsed:.1f}s")


if __name__ == "__main__":
    main()
