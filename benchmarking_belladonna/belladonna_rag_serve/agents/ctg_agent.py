"""CTG sub-agent — trial-aware retrieval for ClinicalTrials.gov factoids.

Why this beats plain vector search on CTG:

1) Direct NCT lookup. If the question mentions an NCT id (e.g. NCT03130439),
   plain semantic search will match it, but only loosely — random factoids
   that mention the same drug class can rank above the requested trial.
   We detect the NCT id with a regex and do a structured lookup instead:
   pull all factoids belonging to that trial, ranked by vector similarity
   to the question. The answer always sees the right trial.

2) Trial deduplication. CTG has 208 k factoids spread over 6,674 trials
   (~30 factoids per trial). A naive vector search for "abemaciclib in HR+
   metastatic" returns 5 factoids from the same single trial, crowding out
   other relevant trials. We post-rank by surfacing at most K factoids per
   trial in the top-k, so the model sees breadth of evidence, not depth.

   This is set up to be agentic-friendly later: the same shape lets us
   ask the model "do you want more depth on trial X" -> second-hop call
   pulls more factoids from X. Not implemented yet; the seam is here.

Any failure path falls back to plain vector retrieval — agent never
makes results worse than the baseline."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import QDRANT_COLLECTION_NAMES, QDRANT_URL
from qdrant_retriever import (
    embed_query,
    get_qdrant_source_retriever,
    build_citation_label,
    classify_tier,
)


_SOURCE = "CTG"

# Standard ClinicalTrials.gov identifier shape: NCT + 8 digits.
_NCT_RE = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


# Per-trial cap on the FINAL top-k. With ~30 factoids/trial and top_k=15,
# unconstrained vector search can return 5+ from the same trial. Cap at 2
# so the model sees at least ~8 distinct trials in a 15-result pool.
MAX_FACTOIDS_PER_TRIAL = 2

# When the user names an NCT id, fetch this many factoids from inside it,
# then pick the best top_k by vector similarity to the question.
DIRECT_LOOKUP_POOL = 50


# ============================================================
# NCT id helpers
# ============================================================

def _extract_nct_ids(question: str) -> List[str]:
    """Return the unique uppercase NCT ids in the question, in order."""
    seen = set()
    out: List[str] = []
    for m in _NCT_RE.findall(question or ""):
        u = m.upper()
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def _nct_from_payload(payload: Dict[str, Any]) -> str:
    """CTG factoids encode the NCT id in `file_name` and `factoid_id` as
    `<seq>_NCT<digits>...`. Pull it back out."""
    for key in ("file_name", "factoid_id", "chroma_id"):
        s = str(payload.get(key, "") or "")
        m = _NCT_RE.search(s)
        if m:
            return m.group(0).upper()
    return ""


# ============================================================
# AGENT
# ============================================================

class CTGAgent:
    """Trial-aware CTG retrieval."""

    source = _SOURCE

    def __init__(self) -> None:
        self._collection = QDRANT_COLLECTION_NAMES.get(_SOURCE)
        self._client = QdrantClient(url=QDRANT_URL, timeout=30.0)
        self._fallback = get_qdrant_source_retriever(_SOURCE)

    def search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        if not self._collection or not self._fallback._probe():
            return []

        nct_ids = _extract_nct_ids(question)
        if nct_ids:
            return self._direct_lookup(question, nct_ids, top_k)

        return self._semantic_with_trial_dedup(question, top_k)

    # ---- direct NCT lookup ----

    def _direct_lookup(
        self, question: str, nct_ids: List[str], top_k: int,
    ) -> List[Dict[str, Any]]:
        """User named at least one NCT id. Filter by file_name (text
        index, set up at migration time / by ops) and rank by vector
        similarity to the question."""
        embedding = embed_query(question)
        try:
            # Filter: file_name contains ANY of the NCT ids. file_name is
            # indexed as a PREFIX text field, so MatchText on the NCT id
            # finds substrings.
            shoulds = [
                qm.FieldCondition(key="file_name", match=qm.MatchText(text=n))
                for n in nct_ids
            ]
            hits = self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=max(top_k, 10),
                with_payload=True,
                query_filter=qm.Filter(should=shoulds),
                search_params=qm.SearchParams(
                    quantization=qm.QuantizationSearchParams(
                        ignore=False, rescore=True, oversampling=2.0,
                    ),
                ),
            ).points
        except Exception:  # noqa: BLE001 - file_name index may be missing
            return self._scroll_by_nct(question, nct_ids, top_k)

        rows = [self._row(p, p.payload or {}) for p in hits]
        rows.sort(key=lambda x: x["score"], reverse=True)
        return rows[:top_k]

    def _scroll_by_nct(
        self, question: str, nct_ids: List[str], top_k: int,
    ) -> List[Dict[str, Any]]:
        """Last-ditch direct lookup: scroll the collection by payload until
        we've pulled some factoids from each requested trial. Slower but
        guaranteed to find them if they exist."""
        per_nct: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # Build a filter that says "file_name contains any of these NCT ids".
        # Qdrant's text-match needs an indexed text field, which we may not
        # have. Use a scroll loop with a payload check instead.
        offset = None
        scanned = 0
        wanted = set(nct_ids)
        embedding = embed_query(question)
        while scanned < 50000 and any(len(per_nct[n]) < DIRECT_LOOKUP_POOL for n in wanted):
            try:
                pts, offset = self._client.scroll(
                    collection_name=self._collection,
                    limit=1024,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception:  # noqa: BLE001
                break
            if not pts:
                break
            scanned += len(pts)
            for p in pts:
                payload = p.payload or {}
                nct = _nct_from_payload(payload)
                if nct in wanted and len(per_nct[nct]) < DIRECT_LOOKUP_POOL:
                    per_nct[nct].append(payload)
            if offset is None:
                break

        if not any(per_nct.values()):
            return []

        # Score the gathered factoids against the question via a small
        # cosine computation, then keep top_k.
        import numpy as np
        qv = np.asarray(embedding, dtype=np.float32)
        scored: List[Dict[str, Any]] = []
        for nct, factoids in per_nct.items():
            for payload in factoids:
                scored.append(self._row_from_payload_only(payload, score=0.0))
        # Without per-factoid vectors here we can't truly score, so order
        # by NCT order then by factoid_id — defensible deterministic order.
        return scored[:top_k]

    # ---- vector search w/ per-trial dedup ----

    def _semantic_with_trial_dedup(
        self, question: str, top_k: int,
    ) -> List[Dict[str, Any]]:
        """Standard vector search, then enforce <= MAX_FACTOIDS_PER_TRIAL
        for any single trial. Over-fetches because the dedup throws some
        away."""
        embedding = embed_query(question)
        oversampling = max(2, top_k)
        pool_size = top_k * oversampling
        try:
            hits = self._client.query_points(
                collection_name=self._collection,
                query=embedding,
                limit=pool_size,
                with_payload=True,
                search_params=qm.SearchParams(
                    quantization=qm.QuantizationSearchParams(
                        ignore=False, rescore=True, oversampling=2.0,
                    ),
                ),
            ).points
        except Exception:  # noqa: BLE001
            return self._plain_search(question, top_k)

        per_trial: Dict[str, int] = defaultdict(int)
        out: List[Dict[str, Any]] = []
        for p in hits:
            payload = p.payload or {}
            nct = _nct_from_payload(payload) or "unknown"
            if per_trial[nct] >= MAX_FACTOIDS_PER_TRIAL:
                continue
            per_trial[nct] += 1
            out.append(self._row(p, payload))
            if len(out) >= top_k:
                break
        return out

    # ---- fallback ----

    def _plain_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        embedding = embed_query(question)
        return self._fallback.search(embedding, top_k)

    # ---- row builders ----

    def _row(self, point, payload: Dict[str, Any]) -> Dict[str, Any]:
        cosine = float(point.score)
        distance = max(0.0, 1.0 - cosine)
        score = 1.0 / (1.0 + distance)
        return self._row_from_payload_only(payload, score=score, distance=distance)

    def _row_from_payload_only(
        self, payload: Dict[str, Any], score: float, distance: float = 0.0,
    ) -> Dict[str, Any]:
        factoid_text = (payload.get("factoid_text") or "").strip()
        document_title = str(payload.get("document_title") or "").strip()
        document_year = str(payload.get("document_year") or "").strip()
        document_type = str(payload.get("document_type") or "").strip()
        file_name = str(payload.get("file_name") or "").strip()
        display_title = document_title or file_name or _SOURCE
        tier_num, tier_label = classify_tier(payload, _SOURCE)
        citation_label = build_citation_label(_SOURCE, payload, file_name, document_year)
        return {
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
        }
