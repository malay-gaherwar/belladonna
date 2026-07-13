"""Qdrant-backed retriever, parallel to retriever.py.

Same `SourceRetriever` interface as the Chroma version so callers (the
module-level `retrieve()` orchestrator, the answerer, the evaluation
harness) don't have to know which backend is in use. The choice is made
in config.py via `VECTOR_BACKEND`.

Differences vs the Chroma retriever:
- Queries are sent as 1024-dim (Qwen3 MRL truncation of the 4096-dim
  embedding from the inference server). We truncate client-side after the
  embed call so the script works whether or not the embedding endpoint
  honours the OpenAI `dimensions` parameter.
- Qdrant returns cosine similarity in [-1, 1] (vectors are L2-normalised
  during migration). We map it to the same `score = 1/(1+distance)` form
  the Chroma retriever produces, so the merge/rerank/RRF code downstream
  sees comparable scores across backends.
- Per-source isolation: one `QdrantSourceRetriever` per source, each
  pinned to one Qdrant collection. Mirrors the per-source `SourceRetriever`
  in retriever.py — the deliberate "agentic seam" stays intact.
"""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import (
    QDRANT_URL,
    QDRANT_COLLECTION_NAMES,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DIM,
    MAX_HITS_PER_SOURCE,
    EVIDENCE_TIER_RULES,
    SOURCE_TIER_FALLBACK,
    DEFAULT_TIER,
)


# ============================================================
# QUERY EMBEDDING (shared with Chroma retriever in intent, duplicated here
# so this module can stand alone in case the Chroma path is removed)
# ============================================================

@lru_cache(maxsize=1)
def _embedding_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("VIRTUAL_API_KEY or BASE_URL not found in environment.")
    return OpenAI(api_key=api_key, base_url=base_url)


def embed_query(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """Embed `text` and return a `dim`-dimensional L2-normalised vector.

    The on-disk vectors in Qdrant are Matryoshka-truncated Qwen3 4096-dim
    embeddings sliced to the first 1024 floats and L2-normalised. To match,
    we do the same client-side: take the full embedding, slice, renormalise.
    """
    client = _embedding_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=[text])
    full = np.asarray(response.data[0].embedding, dtype=np.float32)
    truncated = full[:dim]
    norm = float(np.linalg.norm(truncated))
    if norm > 0:
        truncated = truncated / norm
    return truncated.tolist()


# ============================================================
# EVIDENCE TIER (duplicate of the Chroma side; kept here so this module
# doesn't depend on retriever.py)
# ============================================================

def classify_tier(metadata: Dict[str, Any], source: str) -> tuple[int, str]:
    """Same rules as retriever.classify_tier: substring match on
    document_type in tier order, fall back to the source's default tier."""
    doc_type = str(metadata.get("document_type", "") or "").strip().lower()
    if doc_type:
        for tier_num, label, keywords in EVIDENCE_TIER_RULES:
            if any(kw in doc_type for kw in keywords):
                return tier_num, label

    tier_num = SOURCE_TIER_FALLBACK.get(source, DEFAULT_TIER)
    for t, label, _ in EVIDENCE_TIER_RULES:
        if t == tier_num:
            return tier_num, f"{label} (by source)"
    return DEFAULT_TIER, "Unranked"


# ============================================================
# CITATION LABEL (same rules as retriever.build_citation_label)
# ============================================================

_PAPER_SOURCES = {"EPMC", "Elsevier"}


def _first_author_lastname(authors: Any) -> str:
    """Pull a single surname out of a metadata `authors` field.

    Comes in three shapes depending on the source / how it got into the
    store:
      - Python list, e.g. ["Nasrazadani Y", "Smith J"]      (Elsevier raw)
      - JSON-encoded string, e.g. '["Nasrazadani Y", ...]'  (Elsevier via
        Chroma's flat metadata table -> Qdrant payload)
      - Plain string, e.g. "Sammarco A, Gomiero C, ..."     (EPMC raw)
    We unwrap the first two into a list and then take the first surname."""
    if not authors:
        return ""

    items: List[str] = []
    if isinstance(authors, list):
        items = [str(a) for a in authors]
    else:
        s = str(authors).strip()
        # JSON-encoded list -> parse it. Fall back to the raw string if not JSON.
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    items = [str(a) for a in parsed]
            except (json.JSONDecodeError, ValueError):
                pass
        if not items:
            items = [p.strip() for p in s.split(",") if p.strip()]

    if not items:
        return ""

    first = items[0].strip().strip('"').strip("'")
    # "Last, First" -> "Last"
    if "," in first:
        first = first.split(",", 1)[0].strip()
    # "Last First" -> "Last" (heuristic)
    if " " in first:
        first = first.split()[0]
    return first.strip()


def _nct_id(meta: Dict[str, Any], file_name: str) -> str:
    for key in ("nct_id", "NCTId", "nct"):
        val = str(meta.get(key, "") or "").strip()
        if val.upper().startswith("NCT"):
            return val.upper()
    for token in str(file_name).replace(".", "_").split("_"):
        if token.upper().startswith("NCT") and token[3:].isdigit():
            return token.upper()
    return ""


def _first_authors_from_meta(meta: Dict[str, Any]) -> Any:
    """Chroma flattens nested metadata with a `doc_` prefix for some
    sources (notably EPMC). Try all the known author keys in order."""
    for key in ("authors", "AUTHORS", "doc_AUTHORS", "doc_authors", "author"):
        if key in meta and meta[key]:
            return meta[key]
    return None


def build_citation_label(source: str, meta: Dict[str, Any], file_name: str, year: str) -> str:
    year = (year or "").strip()
    if source in _PAPER_SOURCES:
        last = _first_author_lastname(_first_authors_from_meta(meta))
        if last and year:
            return f"{last} et al. {year}"
        if last:
            return f"{last} et al."
        return f"{source} {year}".strip() or source
    if source == "CTG":
        nct = _nct_id(meta, file_name)
        if nct:
            return nct
        return f"CTG {year}".strip() or "CTG"
    return f"{source} {year}".strip() or source


# ============================================================
# PER-SOURCE QDRANT RETRIEVER (mirrors SourceRetriever shape)
# ============================================================

@lru_cache(maxsize=1)
def _qdrant_client() -> QdrantClient:
    """One process-wide Qdrant client. Qdrant python client is thread-safe."""
    return QdrantClient(url=QDRANT_URL, timeout=30.0)


class QdrantSourceRetriever:
    """Vector retrieval scoped to a single Qdrant collection."""

    def __init__(self, source: str):
        self.source = source
        self._collection_name = QDRANT_COLLECTION_NAMES.get(source)
        self._lock = threading.Lock()
        self._available: Optional[bool] = None
        self._unavailable_reason: Optional[str] = None

    def _probe(self) -> bool:
        """One-time check whether the collection exists on the server."""
        if self._available is not None:
            return self._available
        with self._lock:
            if self._available is not None:
                return self._available
            if not self._collection_name:
                self._unavailable_reason = f"No Qdrant collection mapped for {self.source}"
                self._available = False
                return False
            try:
                client = _qdrant_client()
                if client.collection_exists(self._collection_name):
                    self._available = True
                else:
                    self._unavailable_reason = (
                        f"Collection {self._collection_name} not found on Qdrant"
                    )
                    self._available = False
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                self._unavailable_reason = f"{type(exc).__name__}: {exc}"
                self._available = False
            return self._available

    def health(self) -> Dict[str, Any]:
        """Used by /status. Returns {ok, detail, count}."""
        ok = self._probe()
        out: Dict[str, Any] = {"ok": ok, "detail": self._unavailable_reason}
        if ok:
            try:
                out["count"] = int(
                    _qdrant_client().count(self._collection_name, exact=False).count
                )
            except Exception as exc:  # noqa: BLE001
                out["ok"] = False
                out["detail"] = f"count failed: {type(exc).__name__}: {exc}"
        return out

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        if not self._probe():
            return []
        try:
            hits = _qdrant_client().query_points(
                collection_name=self._collection_name,
                query=query_embedding,
                limit=max(1, top_k),
                with_payload=True,
                # INT8 quantization is on by default; ask the server to
                # rescore with the full-precision (on-disk) vectors so the
                # final top-k matches what an un-quantized search would
                # have returned. Small latency cost, large quality win.
                search_params=qm.SearchParams(
                    quantization=qm.QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=2.0,
                    ),
                ),
            ).points
        except Exception as exc:  # noqa: BLE001 - record so it isn't silent
            self._unavailable_reason = f"query failed: {type(exc).__name__}: {exc}"
            return []

        rows: List[Dict[str, Any]] = []
        for p in hits:
            payload = p.payload or {}
            factoid_text = (payload.get("factoid_text") or "").strip()
            if not factoid_text:
                continue
            document_title = (
                str(payload.get("document_title")
                    or payload.get("payload_document_title")
                    or "").strip()
            )
            document_year = (
                str(payload.get("document_year")
                    or payload.get("payload_document_year")
                    or "").strip()
            )
            document_type = (
                str(payload.get("document_type")
                    or payload.get("payload_document_type")
                    or "").strip()
            )
            file_name = str(payload.get("file_name", "") or "").strip()
            display_title = document_title or file_name or self.source
            tier_num, tier_label = classify_tier(payload, self.source)
            citation_label = build_citation_label(
                self.source, payload, file_name, document_year
            )

            # Cosine similarity (Qdrant's score) -> the same bounded
            # similarity shape the Chroma retriever uses, so the merge
            # step in retrieve() stays meaningful across backends.
            cosine = float(p.score)
            distance = max(0.0, 1.0 - cosine)
            score = 1.0 / (1.0 + distance)

            rows.append(
                {
                    "source": self.source,
                    "file_name": file_name,
                    "factoid_id": payload.get("factoid_id") or payload.get("chroma_id"),
                    "factoid_text": factoid_text,
                    "score": score,
                    "distance": distance,
                    "display_title": display_title,
                    "doi": str(payload.get("doi", "") or "").strip(),
                    "document_year": document_year,
                    "source_family": str(payload.get("source_family", "") or "").strip(),
                    "document_type": document_type,
                    "source_pdf_name": str(payload.get("source_pdf_name", "") or "").strip(),
                    "evidence_tier": tier_num,
                    "evidence_tier_label": tier_label,
                    "citation_label": citation_label,
                    "metadata": dict(payload),
                }
            )
        return rows


@lru_cache(maxsize=None)
def get_qdrant_source_retriever(source: str) -> QdrantSourceRetriever:
    return QdrantSourceRetriever(source)


def qdrant_source_status() -> Dict[str, Dict[str, Any]]:
    """Per-source health for /status."""
    return {
        source: get_qdrant_source_retriever(source).health()
        for source in QDRANT_COLLECTION_NAMES
    }
