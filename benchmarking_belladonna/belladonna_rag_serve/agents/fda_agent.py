"""FDA sub-agent — drug-aware retrieval for FDA labels.

Why this beats plain vector search on FDA:
- FDA payloads carry `generic_name` and `brand_name` as exact strings
  (e.g. "Imlunestrant" / "INLURIYO"). For a question like "what is the
  recommended dose of tamoxifen?", a Qdrant payload filter scoped to
  the tamoxifen label is much more precise than a free-text similarity
  search across all 263 factoids — every hit is guaranteed to be from
  the right drug's label.
- When the question mentions multiple drugs (e.g. "compare tamoxifen and
  raloxifene contraindications"), we fan out — one filtered search per
  drug, then combine — so neither drug crowds the other out of the top-k.
- When no specific drug is named ("what CDK4/6 inhibitors are approved?")
  we fall through to the plain SourceRetriever — the agent never makes
  things worse.

Drug names come out of the question via a tiny LLM extractor. Output
shape is a pure JSON array of strings. Any failure falls back to plain
search so a critic that goes wrong cannot break answers."""

from __future__ import annotations

import json
import os
import re
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import QDRANT_COLLECTION_NAMES, QDRANT_URL, MAX_HITS_PER_SOURCE, MODEL_NAME
from llm import get_local_client
from qdrant_retriever import (
    embed_query,
    get_qdrant_source_retriever,
    build_citation_label,
    classify_tier,
)


_SOURCE = "FDA"


# ============================================================
# DRUG EXTRACTION (LLM, structured output)
# ============================================================

_DRUG_EXTRACT_PROMPT = """You extract drug names from a clinician's question for filtering FDA labels.

Output ONLY a JSON array of strings. Each string is one drug name (generic or brand). No prose, no markdown.

Rules:
1. Include both generic and brand names mentioned (verbatim).
2. Do NOT include drug classes (e.g. "CDK4/6 inhibitor") or indications.
3. Do NOT invent drugs that aren't in the question.
4. If no specific drug is named, return [].

Examples:
Q: "What is the recommended dose of palbociclib?"
A: ["palbociclib"]

Q: "Compare tamoxifen vs raloxifene contraindications"
A: ["tamoxifen", "raloxifene"]

Q: "What CDK4/6 inhibitors are approved for HR+ breast cancer?"
A: []

Q: "Is IBRANCE approved for adjuvant therapy?"
A: ["IBRANCE"]"""


_JSON_ARRAY_RE = re.compile(r"\[[^\[\]]*\]", re.DOTALL)


def _extract_drugs_with_llm(question: str) -> List[str]:
    """Return a list of drug-name strings as the model identified them.
    On any error returns an empty list (agent falls through to plain search)."""
    q = (question or "").strip()
    if not q:
        return []
    try:
        client = get_local_client()
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _DRUG_EXTRACT_PROMPT},
                {"role": "user", "content": q},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        text = (resp.choices[0].message.content or "").strip().strip("`")
    except Exception:  # noqa: BLE001 - never block answering
        return []

    if not text.startswith("["):
        m = _JSON_ARRAY_RE.search(text)
        if not m:
            return []
        text = m.group(0)
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(arr, list):
        return []
    return [str(x).strip() for x in arr if str(x).strip()]


# ============================================================
# DRUG-NAME INDEX (cached: lower-case -> {actual-case forms})
# ============================================================

_drug_index_lock = threading.Lock()
_drug_index_cache: Optional[Dict[str, Tuple[Set[str], Set[str]]]] = None


def _build_drug_index(client: QdrantClient, collection: str) -> Dict[str, Tuple[Set[str], Set[str]]]:
    """Scroll the FDA collection once and build:
        lowercase token -> ({generic_name actual-case}, {brand_name actual-case})

    Per-drug filtering then becomes case-insensitive: lowercase the user's
    drug, look up which actual-case names to match in Qdrant."""
    out: Dict[str, Tuple[Set[str], Set[str]]] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=512,
            offset=offset,
            with_payload=["generic_name", "brand_name"],
        )
        if not points:
            break
        for p in points:
            md = p.payload or {}
            gen = (md.get("generic_name") or "").strip()
            brand = (md.get("brand_name") or "").strip()
            if gen:
                slot = out.setdefault(gen.lower(), (set(), set()))
                slot[0].add(gen)
            if brand:
                slot = out.setdefault(brand.lower(), (set(), set()))
                slot[1].add(brand)
        if offset is None:
            break
    return out


def _get_drug_index(client: QdrantClient, collection: str) -> Dict[str, Tuple[Set[str], Set[str]]]:
    global _drug_index_cache
    with _drug_index_lock:
        if _drug_index_cache is None:
            _drug_index_cache = _build_drug_index(client, collection)
        return _drug_index_cache


def _match_drug_to_filter_values(
    drug: str, index: Dict[str, Tuple[Set[str], Set[str]]]
) -> Tuple[Set[str], Set[str]]:
    """Map a user-typed drug name to the actual-case generic/brand values
    stored in Qdrant. Returns ({generic_name values}, {brand_name values})."""
    drug_lc = drug.strip().lower()
    if drug_lc in index:
        return index[drug_lc]
    return (set(), set())


# ============================================================
# AGENT
# ============================================================

class FDAAgent:
    """Drug-aware FDA retrieval."""

    source = _SOURCE

    def __init__(self) -> None:
        self._collection = QDRANT_COLLECTION_NAMES.get(_SOURCE)
        self._client = QdrantClient(url=QDRANT_URL, timeout=30.0)
        self._fallback = get_qdrant_source_retriever(_SOURCE)

    def search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        if not self._collection or not self._fallback._probe():
            return []

        drugs = _extract_drugs_with_llm(question)
        if not drugs:
            # No drug named -> plain semantic search over all FDA labels.
            return self._plain_search(question, top_k)

        try:
            index = _get_drug_index(self._client, self._collection)
        except Exception:  # noqa: BLE001
            return self._plain_search(question, top_k)

        # Build per-drug filters. Use MatchAny on the actual-case generic
        # and brand fields so the search is case-insensitive end-to-end.
        per_drug_filters: List[qm.Filter] = []
        for drug in drugs:
            gen_vals, brand_vals = _match_drug_to_filter_values(drug, index)
            if not gen_vals and not brand_vals:
                continue
            shoulds: List[qm.FieldCondition] = []
            if gen_vals:
                shoulds.append(qm.FieldCondition(
                    key="generic_name",
                    match=qm.MatchAny(any=sorted(gen_vals)),
                ))
            if brand_vals:
                shoulds.append(qm.FieldCondition(
                    key="brand_name",
                    match=qm.MatchAny(any=sorted(brand_vals)),
                ))
            per_drug_filters.append(qm.Filter(should=shoulds))

        if not per_drug_filters:
            # Drug named but no matching label exists in our store. Fall
            # back to plain search so we don't return zero hits.
            return self._plain_search(question, top_k)

        embedding = embed_query(question)
        per_drug_k = max(3, top_k // max(1, len(per_drug_filters)))

        merged: Dict[str, Dict[str, Any]] = {}
        for filt in per_drug_filters:
            hits = self._search_with_filter(embedding, filt, per_drug_k)
            for h in hits:
                fid = h.get("factoid_id") or h.get("metadata", {}).get("chroma_id")
                if fid is None:
                    continue
                # Keep the higher-scored copy if the same factoid surfaces twice.
                if fid not in merged or h["score"] > merged[fid]["score"]:
                    merged[fid] = h

        # Order final pool by relevance and trim to the caller's top_k.
        rows = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return rows[:top_k]

    # ---- internals ----

    def _plain_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        embedding = embed_query(question)
        return self._fallback.search(embedding, top_k)

    def _search_with_filter(
        self, embedding: List[float], qfilter: qm.Filter, top_k: int,
    ) -> List[Dict[str, Any]]:
        """Same shape as SourceRetriever.search, but with payload filter."""
        try:
            hits = self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=max(1, top_k),
                with_payload=True,
                query_filter=qfilter,
                search_params=qm.SearchParams(
                    quantization=qm.QuantizationSearchParams(
                        ignore=False, rescore=True, oversampling=2.0,
                    ),
                ),
            ).points
        except Exception:  # noqa: BLE001
            return []

        rows: List[Dict[str, Any]] = []
        for p in hits:
            payload = p.payload or {}
            factoid_text = (payload.get("factoid_text") or "").strip()
            if not factoid_text:
                continue
            document_title = str(payload.get("document_title") or "").strip()
            document_year = str(payload.get("document_year") or "").strip()
            document_type = str(payload.get("document_type") or "").strip()
            file_name = str(payload.get("file_name") or "").strip()
            display_title = document_title or file_name or _SOURCE
            tier_num, tier_label = classify_tier(payload, _SOURCE)
            citation_label = build_citation_label(_SOURCE, payload, file_name, document_year)

            cosine = float(p.score)
            distance = max(0.0, 1.0 - cosine)
            score = 1.0 / (1.0 + distance)

            rows.append({
                "source": _SOURCE,
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
            })
        return rows
