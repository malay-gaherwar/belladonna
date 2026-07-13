"""Embedding-based retrieval over the per-source ChromaDB vector stores.

Each data source (AGO, ESMO, CTG, ...) is an independent ChromaDB collection.
`SourceRetriever` wraps exactly one source: it owns that source's collection
and knows nothing about the others. The module-level `retrieve()` is a thin
orchestrator that fans a query out to several `SourceRetriever`s and merges
the results.

This per-source isolation is deliberate: a future agentic RAG can wrap each
`SourceRetriever` in its own agent without touching this retrieval code.
"""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional

import chromadb
from openai import OpenAI

from config import (
    SOURCE_COLLECTION_NAMES,
    SOURCE_EMBEDDING_DIRS,
    EMBEDDING_MODEL_NAME,
    MAX_HITS_PER_SOURCE,
    RERANK_ENABLED,
    RERANK_CANDIDATE_POOL,
    EVIDENCE_TIER_RULES,
    SOURCE_TIER_FALLBACK,
    DEFAULT_TIER,
    RECENCY_APPLICABLE_TIERS,
    RRF_K,
    RRF_USE_RERANK,
    RRF_USE_TIER,
    RRF_USE_RECENCY,
    VECTOR_BACKEND,
)
from reranker import rerank

# Sources ordered by the evidence hierarchy (tier 1 first). Replaces the
# old PRIORITY/SECONDARY split — single source of truth lives in config.
TIER_ORDERED_SOURCES = sorted(
    SOURCE_TIER_FALLBACK,
    key=lambda s: SOURCE_TIER_FALLBACK[s],
)

SOURCE_ALIASES = {
    "AGO": ["ago", "arbeitsgemeinschaft gynakologische onkologie"],
    "ESMO": ["esmo"],
    "FDA": ["fda", "drugsfda"],
    "EMA": ["ema"],
    "CTG": ["ctg", "clinicaltrials.gov", "clinicaltrials", "trial", "trials"],
    "EPMC": ["epmc", "europe pmc", "pubmed", "pmc"],
    "Elsevier": ["elsevier"],
}


# ============================================================
# QUERY EMBEDDING
# ============================================================

@lru_cache(maxsize=1)
def _embedding_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("VIRTUAL_API_KEY or BASE_URL not found in environment.")
    return OpenAI(api_key=api_key, base_url=base_url)


def embed_query(text: str) -> List[float]:
    """Embed a single query string with the same model used to build the DBs."""
    client = _embedding_client()
    response = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=[text])
    return response.data[0].embedding


# ============================================================
# EVIDENCE TIER CLASSIFICATION
# ============================================================

def classify_tier(metadata: Dict[str, Any], source: str) -> tuple[int, str]:
    """Classify a hit into the evidence hierarchy using its document_type
    metadata, falling back to the source's default tier when the type is
    missing or unrecognised. Returns (tier_number, human_label).

    Substring + case-insensitive match in tier order, so tags like
    "Systematic review and meta-analysis" land in tier 3 before tier 4."""
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


def _parse_year(year_str: Any) -> Optional[int]:
    """Best-effort parse of a 4-digit year from possibly-dirty metadata."""
    try:
        return int(str(year_str).strip()[:4])
    except (ValueError, TypeError):
        return None


# ============================================================
# CITATION LABEL
# ============================================================
# One short, human-readable string per evidence item. This is what the
# answerer asks the model to put inside [[...]] in the answer, and what
# the frontend renders as the inline citation chip. Per-source rules:
#   AGO / ESMO / ASCO  -> "AGO 2026"             (guideline name + year)
#   EMA / FDA          -> "FDA 2024"             (regulator + year)
#   CTG                -> "NCT01432223"          (registry id)
#   EPMC / Elsevier    -> "Sammarco et al. 2023" (first author + year)

_PAPER_SOURCES = {"EPMC", "Elsevier"}


def _first_author_lastname(authors: Any) -> str:
    """Pull a single surname out of a metadata `authors` field.

    Three shapes show up depending on the source / pipeline:
      - Python list: ["Nasrazadani Y", "Smith J"]            (Elsevier raw)
      - JSON-encoded string: '["Nasrazadani Y", ...]'        (Elsevier via
        Chroma's flat metadata table; sqlite has no list type)
      - Plain string: "Sammarco A, Gomiero C, ..."           (EPMC raw)
    """
    if not authors:
        return ""

    items: List[str] = []
    if isinstance(authors, list):
        items = [str(a) for a in authors]
    else:
        import json as _json
        s = str(authors).strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = _json.loads(s)
                if isinstance(parsed, list):
                    items = [str(a) for a in parsed]
            except (ValueError, _json.JSONDecodeError):
                pass
        if not items:
            items = [p.strip() for p in s.split(",") if p.strip()]

    if not items:
        return ""

    first = items[0].strip().strip('"').strip("'")
    if "," in first:
        first = first.split(",", 1)[0].strip()
    if " " in first:
        first = first.split()[0]
    return first.strip()


def _nct_id(meta: Dict[str, Any], file_name: str) -> str:
    """CTG factoids encode the NCT id in the source filename (and sometimes
    in metadata)."""
    for key in ("nct_id", "NCTId", "nct"):
        val = str(meta.get(key, "") or "").strip()
        if val.upper().startswith("NCT"):
            return val.upper()
    # filename like "0011941_NCT01925170_factoids.json"
    for token in str(file_name).replace(".", "_").split("_"):
        if token.upper().startswith("NCT") and token[3:].isdigit():
            return token.upper()
    return ""


def _first_authors_from_meta(meta: Dict[str, Any]) -> Any:
    """Chroma flattens nested metadata with a `doc_` prefix for some
    sources (notably EPMC). Try the known author keys in order."""
    for key in ("authors", "AUTHORS", "doc_AUTHORS", "doc_authors", "author"):
        if key in meta and meta[key]:
            return meta[key]
    return None


def build_citation_label(source: str, meta: Dict[str, Any], file_name: str, year: str) -> str:
    """Compose the short citation label for one evidence row."""
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
    # Guidelines (AGO/ESMO/ASCO) and regulators (EMA/FDA) cite as "<Source> <year>".
    return f"{source} {year}".strip() or source


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================
# Final ranking: fuse cross-encoder relevance, evidence tier, and recency
# via RRF (Cormack et al., SIGIR 2009). Rank-based fusion means we never
# tune relative score magnitudes — the only constant is K, set to 60 per
# the original paper. See config.py for the supporting-literature block.

def _ranks_from_key(items: List[Dict[str, Any]], key) -> List[int]:
    """Return 1-indexed dense ranks for `items`, sorted ascending by `key`.
    Ties get the same rank ("standard competition" 1-2-2-4 style)."""
    n = len(items)
    order = sorted(range(n), key=lambda i: key(items[i]))
    ranks = [0] * n
    prev_key = None
    prev_rank = 0
    for position, idx in enumerate(order, start=1):
        k_now = key(items[idx])
        if prev_key is not None and k_now == prev_key:
            ranks[idx] = prev_rank
        else:
            ranks[idx] = position
            prev_rank = position
            prev_key = k_now
    return ranks


def _rerank_key(d: Dict[str, Any]) -> float:
    score = d.get("rerank_score")
    if score is None:
        score = d.get("score", 0.0)
    return -float(score)  # higher score => lower (better) rank


def _tier_key(d: Dict[str, Any]) -> int:
    return int(d.get("evidence_tier", DEFAULT_TIER))


def _recency_key(d: Dict[str, Any]) -> tuple:
    """Sort key for the recency ranking.

    - Tier 1 and tier 2 (guidelines, regulatory) are treated as effectively
      current and tied at the top, so RRF doesn't penalise them for an
      old publication year on continuously-maintained documents.
    - For paper-like tiers, newer year ranks first; unparseable years go
      to the bottom of the list.
    """
    tier = _tier_key(d)
    if tier not in RECENCY_APPLICABLE_TIERS:
        return (0, 0)  # effectively current, tied at top
    year = _parse_year(d.get("document_year"))
    if year is None:
        return (2, 0)  # unknown year => bottom
    return (1, -year)  # newer first


def reciprocal_rank_fusion(
    items: List[Dict[str, Any]],
    k: int = RRF_K,
    use_rerank: bool = RRF_USE_RERANK,
    use_tier: bool = RRF_USE_TIER,
    use_recency: bool = RRF_USE_RECENCY,
) -> List[Dict[str, Any]]:
    """Annotate each item with per-signal ranks and a fused RRF score.

    RRF_score(d) = sum_i 1 / (k + rank_i(d))

    Returns the same list (mutated) sorted by descending fused score."""
    if not items:
        return items

    rerank_ranks = _ranks_from_key(items, _rerank_key) if use_rerank else None
    tier_ranks = _ranks_from_key(items, _tier_key) if use_tier else None
    recency_ranks = _ranks_from_key(items, _recency_key) if use_recency else None

    for i, item in enumerate(items):
        contributions: Dict[str, float] = {}
        rank_breakdown: Dict[str, int] = {}
        if rerank_ranks is not None:
            r = rerank_ranks[i]
            rank_breakdown["rerank"] = r
            contributions["rerank"] = 1.0 / (k + r)
        if tier_ranks is not None:
            r = tier_ranks[i]
            rank_breakdown["tier"] = r
            contributions["tier"] = 1.0 / (k + r)
        if recency_ranks is not None:
            r = recency_ranks[i]
            rank_breakdown["recency"] = r
            contributions["recency"] = 1.0 / (k + r)

        item["rrf_ranks"] = rank_breakdown
        item["rrf_contributions"] = contributions
        item["final_score"] = sum(contributions.values())

    items.sort(key=lambda x: x["final_score"], reverse=True)
    return items


# ============================================================
# PER-SOURCE RETRIEVER  (the agentic seam)
# ============================================================

class SourceRetriever:
    """Vector retrieval scoped to a single Belladonna source collection."""

    def __init__(self, source: str):
        self.source = source
        self._collection = None
        self._lock = threading.Lock()
        self._unavailable_reason: Optional[str] = None

    def _get_collection(self):
        if self._collection is not None or self._unavailable_reason is not None:
            return self._collection

        with self._lock:
            if self._collection is not None or self._unavailable_reason is not None:
                return self._collection

            db_dir = SOURCE_EMBEDDING_DIRS.get(self.source)
            coll_name = SOURCE_COLLECTION_NAMES.get(self.source)
            if db_dir is None or coll_name is None:
                self._unavailable_reason = f"No config for source {self.source}"
                return None
            if not db_dir.exists():
                self._unavailable_reason = f"Embeddings dir missing: {db_dir}"
                return None

            try:
                client = chromadb.PersistentClient(path=str(db_dir))
                # embedding_function=None: we always supply query embeddings
                # explicitly, matching how the DB was written.
                self._collection = client.get_collection(name=coll_name)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                self._unavailable_reason = f"{type(exc).__name__}: {exc}"
                self._collection = None

            return self._collection

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        if collection is None:
            return []

        try:
            res = collection.query(
                query_embeddings=[query_embedding],
                n_results=max(1, top_k),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001 - record so it isn't silent
            self._unavailable_reason = f"query failed: {type(exc).__name__}: {exc}"
            return []

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        rows: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            meta = meta or {}
            factoid_text = (doc or "").strip()
            if not factoid_text:
                continue

            # FDA's chroma metadata stores title/year/type under payload_* keys,
            # not at the top level. Fall back so citations don't show filenames.
            document_title = (
                str(meta.get("document_title") or meta.get("payload_document_title") or "").strip()
            )
            document_year = (
                str(meta.get("document_year") or meta.get("payload_document_year") or "").strip()
            )
            document_type = (
                str(meta.get("document_type") or meta.get("payload_document_type") or "").strip()
            )
            file_name = str(meta.get("file_name", "") or "").strip()
            display_title = document_title or file_name or self.source
            tier_num, tier_label = classify_tier(meta, self.source)
            citation_label = build_citation_label(self.source, meta, file_name, document_year)

            rows.append(
                {
                    "source": self.source,
                    "file_name": file_name,
                    "factoid_id": meta.get("factoid_id"),
                    "factoid_text": factoid_text,
                    # L2 distance -> bounded similarity (higher = better).
                    "score": 1.0 / (1.0 + float(dist)),
                    "distance": float(dist),
                    "display_title": display_title,
                    "doi": str(meta.get("doi", "") or "").strip(),
                    "document_year": document_year,
                    "source_family": str(meta.get("source_family", "") or "").strip(),
                    "document_type": document_type,
                    "source_pdf_name": str(meta.get("source_pdf_name", "") or "").strip(),
                    "evidence_tier": tier_num,
                    "evidence_tier_label": tier_label,
                    "citation_label": citation_label,
                    "metadata": dict(meta),
                }
            )
        return rows


@lru_cache(maxsize=None)
def get_source_retriever(source: str) -> SourceRetriever:
    """Cached per-source retriever (one collection handle per process)."""
    return SourceRetriever(source)


def source_status() -> Dict[str, Dict[str, Any]]:
    """Probe every configured source's collection. Used by /status so broken
    vector stores (e.g. corrupt HNSW indexes) are visible, not silent.
    Backend-aware: dispatches to the Qdrant probe when VECTOR_BACKEND='qdrant'."""
    if VECTOR_BACKEND == "qdrant":
        from qdrant_retriever import qdrant_source_status
        return qdrant_source_status()

    status: Dict[str, Dict[str, Any]] = {}
    for source in SOURCE_COLLECTION_NAMES:
        retriever = get_source_retriever(source)
        collection = retriever._get_collection()
        ok = False
        detail = retriever._unavailable_reason
        if collection is not None:
            try:
                collection.count()
                ok = True
                detail = None
            except Exception as exc:  # noqa: BLE001
                detail = f"count failed: {type(exc).__name__}: {exc}"
        status[source] = {"ok": ok, "detail": detail}
    return status


# ============================================================
# SOURCE SELECTION
# ============================================================

@lru_cache(maxsize=1)
def _alias_patterns() -> Dict[str, re.Pattern]:
    """Compile a word-boundary regex per source. Naive substring matching
    used to misfire: e.g. the EMA alias "ema" matched "ab*ema*ciclib" and
    silently rerouted unrelated questions to the EMA collection."""
    compiled: Dict[str, re.Pattern] = {}
    for source, aliases in SOURCE_ALIASES.items():
        # Sort longer aliases first so multi-word matches win over short ones.
        parts = sorted({a.strip() for a in aliases if a and a.strip()}, key=len, reverse=True)
        if not parts:
            continue
        joined = "|".join(re.escape(p) for p in parts)
        compiled[source] = re.compile(rf"(?<!\w)(?:{joined})(?!\w)", re.IGNORECASE)
    return compiled


def find_requested_sources(question: str) -> set[str]:
    requested: set[str] = set()
    for source, pattern in _alias_patterns().items():
        if pattern.search(question or ""):
            requested.add(source)
    return requested


def build_source_order(sources: Optional[List[str]], question: str) -> List[str]:
    """Pick + order sources to query. Order follows the evidence hierarchy
    (tier 1 first); selection respects explicit mentions in the question and
    any explicit `sources` filter."""
    requested = find_requested_sources(question)
    if requested:
        return [s for s in TIER_ORDERED_SOURCES if s in requested]

    if sources:
        allowed = set(sources)
        return [s for s in TIER_ORDERED_SOURCES if s in allowed] + [
            s for s in sources if s not in SOURCE_TIER_FALLBACK
        ]

    return list(TIER_ORDERED_SOURCES)


def deduplicate_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in hits:
        key = (
            item.get("source", ""),
            item.get("display_title", ""),
            item.get("factoid_text", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


# ============================================================
# ORCHESTRATOR
# ============================================================

def retrieve(
    question: str,
    sources: Optional[List[str]] = None,
    top_k: int = 15,
) -> List[Dict[str, Any]]:
    """Embed the question once, fan out to the selected source collections,
    merge by similarity, and return the global top-k factoids. Dispatches
    to the Qdrant backend when VECTOR_BACKEND='qdrant'.

    Per-source sub-agents (see agents/) are consulted before plain vector
    retrieval. If a source has an agent registered, the agent's
    `search(question, top_k)` runs in place of the embed-and-search path;
    agents do their own embedding + payload filtering inside. Sources
    without an agent fall through to the plain SourceRetriever."""
    selected = build_source_order(sources, question)
    if not selected:
        return []

    # Lazy: only embed if at least one selected source uses plain retrieval.
    query_embedding: Optional[List[float]] = None

    if VECTOR_BACKEND == "qdrant":
        from qdrant_retriever import embed_query as _qdrant_embed, get_qdrant_source_retriever
        from agents import get_agent_for
        get_retriever = get_qdrant_source_retriever
        _embed = _qdrant_embed
    else:
        from agents import get_agent_for   # noqa: F401 - safe no-op for chroma
        get_retriever = get_source_retriever
        _embed = embed_query

    per_source_k = min(MAX_HITS_PER_SOURCE, max(top_k, 1))
    all_hits: List[Dict[str, Any]] = []
    for source in selected:
        agent = get_agent_for(source)
        if agent is not None:
            all_hits.extend(agent.search(question, per_source_k))
            continue
        # No agent for this source -> plain vector retrieval. Embed once.
        if query_embedding is None:
            query_embedding = _embed(question)
        retriever = get_retriever(source)
        all_hits.extend(retriever.search(query_embedding, per_source_k))

    all_hits = deduplicate_hits(all_hits)
    all_hits.sort(key=lambda x: x["score"], reverse=True)

    if RERANK_ENABLED:
        # Cross-encoder rescoring for cross-source consistency. Falls back to
        # vector order inside rerank() on any failure.
        candidate_pool = all_hits[: max(RERANK_CANDIDATE_POOL, top_k)]
        ranked = rerank(question, candidate_pool, max(RERANK_CANDIDATE_POOL, top_k))
    else:
        ranked = all_hits[: max(RERANK_CANDIDATE_POOL, top_k)]

    # Fuse the three signals (cross-encoder relevance, evidence tier,
    # recency) via Reciprocal Rank Fusion. See config.py for the supporting
    # literature and rationale for each signal.
    ranked = reciprocal_rank_fusion(ranked)
    return ranked[:top_k]
