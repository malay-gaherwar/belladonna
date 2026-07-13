"""Cross-encoder reranking over the merged retrieval candidate pool.

The retriever fans a query out to several independently-built per-source
ChromaDB collections and merges them by `1/(1+L2)`. Those L2 distances are
not comparable across collections, so the merged order is only roughly
right. This module re-scores the merged pool with Qwen3-Reranker-8B (served
Cohere-style at `<BASE_URL>/rerank`) to get one consistent query-relevance
signal across all sources.

Failure here must never break answering: any error (endpoint down, bad
response, timeout) falls back to the existing vector order.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import httpx

from config import (
    RERANKER_MODEL_NAME,
    RERANK_MAX_DOC_CHARS,
)


def _rerank_url() -> str:
    base = os.getenv("BASE_URL")
    if not base:
        raise RuntimeError("BASE_URL not found in environment.")
    # BASE_URL is e.g. http://host/v1/ ; the rerank route lives next to it.
    return base.rstrip("/") + "/rerank"


def rerank(query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Reorder `hits` by reranker relevance to `query`, return the top_k.

    On any failure, returns the original (vector-ordered) hits truncated to
    top_k so the caller's behaviour is unchanged.
    """
    if not hits:
        return hits

    api_key = os.getenv("VIRTUAL_API_KEY")
    documents = [
        (h.get("factoid_text") or "")[:RERANK_MAX_DOC_CHARS] for h in hits
    ]

    try:
        resp = httpx.post(
            _rerank_url(),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANKER_MODEL_NAME,
                "query": query,
                "documents": documents,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
    except Exception as exc:  # noqa: BLE001 - degrade to vector order
        print(
            f"[reranker] falling back to vector order: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return hits[:top_k]

    if not results:
        return hits[:top_k]

    ranked: List[Dict[str, Any]] = []
    for r in results:
        idx = r.get("index")
        if idx is None or idx < 0 or idx >= len(hits):
            continue
        item = dict(hits[idx])
        item["rerank_score"] = float(r.get("relevance_score", 0.0))
        ranked.append(item)

    if not ranked:
        return hits[:top_k]

    # The server returns results sorted by relevance, but don't rely on it.
    ranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
    return ranked[:top_k]
