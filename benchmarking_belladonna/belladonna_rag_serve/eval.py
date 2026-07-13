"""Retrieval evaluation harness for Belladonna.

Runs the production `retrieve()` pipeline over a labelled query set and
reports standard IR metrics so ranking choices (RRF signal toggles, k,
candidate-pool size, reranker on/off) can be tuned defensibly instead of
by intuition.

Input format
------------
JSONL, one query per line:

    {"query": "How often should mammography be repeated?",
     "relevant_factoid_ids": ["AGO_2025E_03_..._12", "ESMO_2024_..._47"],
     "graded_relevance": {"AGO_..._12": 3, "ESMO_..._47": 2}}

`relevant_factoid_ids` is required (used for MAP/MRR and as a fallback
for nDCG when graded relevance is absent). `graded_relevance` is optional
and overrides binary relevance when present, enabling true nDCG with
multi-level judgements (0=irrelevant, 1=marginal, 2=relevant, 3=highly
relevant — a typical TREC-style grading).

Metrics
-------
- nDCG@k     — Normalized Discounted Cumulative Gain (Jarvelin & Kekalainen,
               ACM TOIS 2002).
- MAP        — Mean Average Precision (Manning, Raghavan, Schutze 2008).
- MRR        — Mean Reciprocal Rank.
- Recall@k   — fraction of known-relevant docs retrieved in top k.

Usage
-----
    python eval.py path/to/queries.jsonl
    python eval.py path/to/queries.jsonl --k 10 --top-k 20
    python eval.py path/to/queries.jsonl --ablate     # toggle RRF signals
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import config  # for RRF toggle ablation
from retriever import retrieve


# ============================================================
# METRICS
# ============================================================

def dcg(grades: List[float]) -> float:
    """Discounted Cumulative Gain with the standard (2^rel - 1) formulation."""
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(retrieved_grades: List[float], all_grades: List[float], k: int) -> float:
    """nDCG@k. `retrieved_grades` is the graded relevance of the top-k
    retrieved docs in order. `all_grades` is the full set of graded
    relevances for the query (used to build the ideal ranking)."""
    actual = dcg(retrieved_grades[:k])
    ideal = dcg(sorted(all_grades, reverse=True)[:k])
    return actual / ideal if ideal > 0 else 0.0


def average_precision(binary_rel: List[int], total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    hits = 0
    ap = 0.0
    for i, r in enumerate(binary_rel, start=1):
        if r:
            hits += 1
            ap += hits / i
    return ap / total_relevant


def reciprocal_rank(binary_rel: List[int]) -> float:
    for i, r in enumerate(binary_rel, start=1):
        if r:
            return 1.0 / i
    return 0.0


def recall_at_k(binary_rel: List[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(binary_rel[:k]) / total_relevant


# ============================================================
# EVAL LOOP
# ============================================================

@dataclass
class QueryRecord:
    query: str
    relevant_ids: List[str]
    graded_relevance: Dict[str, float]


def _load_queries(path: Path) -> List[QueryRecord]:
    records: List[QueryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        records.append(
            QueryRecord(
                query=obj["query"],
                relevant_ids=list(obj.get("relevant_factoid_ids", [])),
                graded_relevance={
                    str(fid): float(g) for fid, g in (obj.get("graded_relevance") or {}).items()
                },
            )
        )
    return records


def _grade_for(hit: Dict[str, Any], rec: QueryRecord) -> float:
    fid = str(hit.get("factoid_id") or "")
    if rec.graded_relevance:
        return rec.graded_relevance.get(fid, 0.0)
    return 1.0 if fid in rec.relevant_ids else 0.0


def evaluate(
    queries_path: Path,
    k: int = 10,
    top_k: int = 20,
    sources: Optional[List[str]] = None,
) -> Dict[str, Any]:
    queries = _load_queries(queries_path)
    if not queries:
        return {"n_queries": 0}

    per_query = []
    for rec in queries:
        hits = retrieve(question=rec.query, sources=sources, top_k=top_k)
        retrieved_grades = [_grade_for(h, rec) for h in hits]
        binary_rel = [1 if g > 0 else 0 for g in retrieved_grades]
        all_grades = (
            list(rec.graded_relevance.values())
            if rec.graded_relevance
            else [1.0] * len(rec.relevant_ids)
        )
        total_relevant = sum(1 for g in all_grades if g > 0)
        per_query.append(
            {
                "query": rec.query,
                f"nDCG@{k}": ndcg_at_k(retrieved_grades, all_grades, k),
                "MAP": average_precision(binary_rel, total_relevant),
                "MRR": reciprocal_rank(binary_rel),
                f"Recall@{k}": recall_at_k(binary_rel, total_relevant, k),
            }
        )

    def _agg(name: str) -> float:
        return float(mean(q[name] for q in per_query)) if per_query else 0.0

    return {
        "n_queries": len(per_query),
        f"nDCG@{k}": _agg(f"nDCG@{k}"),
        "MAP": _agg("MAP"),
        "MRR": _agg("MRR"),
        f"Recall@{k}": _agg(f"Recall@{k}"),
        "per_query": per_query,
    }


# ============================================================
# RRF ABLATION
# ============================================================
# Toggle each RRF signal independently to measure its marginal contribution.
# Useful for the methods section: "removing the recency signal reduced
# nDCG@10 from X to Y on N held-out queries."

_RRF_FLAGS = ("RRF_USE_RERANK", "RRF_USE_TIER", "RRF_USE_RECENCY")


def ablate(queries_path: Path, k: int = 10, top_k: int = 20) -> Dict[str, Any]:
    original = {f: getattr(config, f) for f in _RRF_FLAGS}
    results = {}
    try:
        # baseline: all signals on (as configured)
        results["all"] = evaluate(queries_path, k=k, top_k=top_k)
        # leave-one-out
        for off in _RRF_FLAGS:
            for f in _RRF_FLAGS:
                setattr(config, f, f != off)
            # retriever reads these at call time via `from config import ...`
            # but the module-level imports bound them. Reload retriever so the
            # new values are picked up.
            import importlib
            import retriever as _r
            importlib.reload(_r)
            from retriever import retrieve as _retr  # noqa: F401
            results[f"without_{off}"] = evaluate(queries_path, k=k, top_k=top_k)
        # rerank-only / tier-only / recency-only
        for solo in _RRF_FLAGS:
            for f in _RRF_FLAGS:
                setattr(config, f, f == solo)
            import importlib
            import retriever as _r
            importlib.reload(_r)
            results[f"only_{solo}"] = evaluate(queries_path, k=k, top_k=top_k)
    finally:
        for f, v in original.items():
            setattr(config, f, v)
        import importlib
        import retriever as _r
        importlib.reload(_r)
    # Strip per-query lists from the ablation report to keep it compact.
    for name in list(results):
        results[name].pop("per_query", None)
    return results


# ============================================================
# CLI
# ============================================================

def _main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("queries", type=Path, help="JSONL file of labelled queries")
    p.add_argument("--k", type=int, default=10, help="cutoff for nDCG/Recall (default 10)")
    p.add_argument("--top-k", type=int, default=20, help="retrieval depth (default 20)")
    p.add_argument("--sources", nargs="*", default=None, help="restrict to these sources")
    p.add_argument("--ablate", action="store_true", help="run RRF signal ablation")
    args = p.parse_args()

    if args.ablate:
        report = ablate(args.queries, k=args.k, top_k=args.top_k)
    else:
        report = evaluate(args.queries, k=args.k, top_k=args.top_k, sources=args.sources)
        report.pop("per_query", None)  # CLI summary mode
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    _main()
