"""Per-source overview used by the website sidebar.

Cheap to compute and cached in-process: counts come from the filesystem
(one `*_factoids.json` per document), years for the small sources come
from reading those JSONs directly, and chroma health is probed once.

We deliberately avoid touching the EPMC/Elsevier HNSW indexes here —
they may fail to load, and the sidebar must still render. Article
counts for those sources come from disk."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import SOURCE_EMBEDDING_DIRS, DATA_ROOT, VECTOR_BACKEND
from retriever import get_source_retriever, source_status


# Each source has a "kind" that drives how the sidebar renders it.
SOURCE_KIND = {
    "AGO": "guideline",
    "ESMO": "guideline",
    "ASCO": "guideline",
    "EMA": "regulator",
    "FDA": "regulator",
    "CTG": "registry",
    "EPMC": "paper",
    "Elsevier": "paper",
}

# Human-readable name used in the sidebar header.
SOURCE_LONG_NAME = {
    "AGO": "AGO — German breast cancer guidelines",
    "ASCO": "ASCO — American Society of Clinical Oncology",
    "ESMO": "ESMO — European Society for Medical Oncology",
    "EMA": "EMA — European Medicines Agency",
    "FDA": "FDA — US Food & Drug Administration",
    "CTG": "ClinicalTrials.gov",
    "EPMC": "Europe PMC",
    "Elsevier": "Elsevier journals",
}


def _factoids_dir(source: str) -> Path:
    """The `<DATA_ROOT>/<source>/factoids/` directory."""
    return DATA_ROOT / source / "factoids"


def _count_factoid_files(source: str) -> int:
    """Number of `*_factoids.json` files (one per document/trial/article)."""
    d = _factoids_dir(source)
    if not d.exists():
        return 0
    # Fast: glob without recursion, stop at JSONs only.
    return sum(1 for _ in d.glob("*_factoids.json"))


def _peek_years_from_files(source: str, max_files: int = 200) -> List[int]:
    """Open a small number of factoid JSONs and pull `document_year` /
    `metadata.document_year` / `metadata.YEAR`. Used for the guidelines and
    regulators only (their factoid files are small in number)."""
    d = _factoids_dir(source)
    if not d.exists():
        return []
    years: set[int] = set()
    for i, path in enumerate(sorted(d.glob("*_factoids.json"))):
        if i >= max_files:
            break
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:  # noqa: BLE001 - skip unreadable files
            continue
        years |= _extract_years(obj)
    return sorted(years)


def _extract_years(obj: Any) -> set[int]:
    """Pull every 4-digit year we can find in a factoid file's metadata."""
    found: set[int] = set()

    def add(v: Any) -> None:
        try:
            n = int(str(v).strip()[:4])
        except (ValueError, TypeError):
            return
        if 1900 <= n <= 2100:
            found.add(n)

    if isinstance(obj, dict):
        meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else None
        if meta:
            for k in ("document_year", "YEAR", "payload_document_year", "year"):
                if k in meta:
                    add(meta[k])
        # Top-level fields (e.g. AGO factoid lists embed metadata per-factoid).
        for k in ("document_year", "year"):
            if k in obj:
                add(obj[k])
        # Walk one level into list-of-factoids structures.
        factoids = obj.get("factoids")
        if isinstance(factoids, list):
            for it in factoids[:5]:  # one file rarely covers many years
                found |= _extract_years(it)
    elif isinstance(obj, list):
        for it in obj[:5]:
            found |= _extract_years(it)

    return found


def _factoid_count_from_vector_db(source: str) -> Optional[int]:
    """Best-effort vector-store row count. Returns None when the backend
    can't open the collection (e.g. broken Chroma HNSW for EPMC/Elsevier,
    or Qdrant collection not yet created)."""
    if VECTOR_BACKEND == "qdrant":
        from qdrant_retriever import get_qdrant_source_retriever
        health = get_qdrant_source_retriever(source).health()
        return health.get("count") if health.get("ok") else None

    retriever = get_source_retriever(source)
    col = retriever._get_collection()
    if col is None:
        return None
    try:
        return int(col.count())
    except Exception:  # noqa: BLE001 - HNSW load failure on large collections
        return None


def _build_one(source: str, status: Dict[str, Any]) -> Dict[str, Any]:
    kind = SOURCE_KIND.get(source, "other")
    info: Dict[str, Any] = {
        "name": SOURCE_LONG_NAME.get(source, source),
        "kind": kind,
        "ok": bool(status.get("ok")),
        "status_detail": status.get("detail"),
        "n_documents": _count_factoid_files(source),
        "n_factoids": _factoid_count_from_vector_db(source),
    }
    if kind == "guideline":
        info["years"] = _peek_years_from_files(source)
    elif kind == "regulator":
        years = _peek_years_from_files(source)
        info["years"] = years
        info["latest_year"] = max(years) if years else None
    elif kind == "registry":
        info["n_trials"] = info["n_documents"]
    elif kind == "paper":
        info["n_articles"] = info["n_documents"]
    return info


_cache_lock = threading.Lock()
_cache: Optional[Dict[str, Dict[str, Any]]] = None


def get_source_info(force: bool = False) -> Dict[str, Dict[str, Any]]:
    """Per-source summary for the sidebar. Cached after first call."""
    global _cache
    with _cache_lock:
        if _cache is not None and not force:
            return _cache
        status = source_status()
        info = {src: _build_one(src, status.get(src, {})) for src in SOURCE_EMBEDDING_DIRS}
        _cache = info
        return info
