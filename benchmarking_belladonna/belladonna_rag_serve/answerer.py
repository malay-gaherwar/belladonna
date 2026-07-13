"""Top-level Q&A orchestration.

Flow (the actual agentic loop lives here):

    pick sources (router)
        |
        v
    initial retrieve
        |
        v
    critic.grade_evidence
        |
        +-- sufficient --> answer with evidence
        |
        v
    insufficient + has gaps + hops < MAX_REFORMULATION_HOPS
        |
        v
    for each gap: retrieve(gap.follow_up_query) -> merge & dedupe
        |
        v
    answer, with the verdict surfaced both in the prompt (so the model
    can openly say "evidence does not address X") and in the API
    response (so a UI can show an "evidence partial" indicator).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from retriever import retrieve, TIER_ORDERED_SOURCES
from llm import generate_grounded_answer, condense_question, route_sources
from conversation import STORE
from config import SOURCE_TIER_FALLBACK
from critic import grade_evidence, EvidenceVerdict


# Cap how aggressive the second-hop loop is. One reformulation round is
# enough to handle the typical "compare X vs Y; only X retrieved" case;
# additional rounds cost latency without much marginal gain.
MAX_REFORMULATION_HOPS = 1
PER_GAP_TOP_K = 5            # focused, smaller pool per follow-up query


def build_default_source_order(user_sources: list[str] | None = None) -> list[str]:
    """Order sources by the evidence hierarchy (tier 1 first), optionally
    restricted to a caller-supplied whitelist."""
    if user_sources:
        allowed = set(user_sources)
        return [s for s in TIER_ORDERED_SOURCES if s in allowed] + [
            s for s in user_sources if s not in SOURCE_TIER_FALLBACK
        ]
    return list(TIER_ORDERED_SOURCES)


def _pick_sources(question: str, sources: list[str] | None, model: str | None = None) -> tuple[list[str], str]:
    """Decide which sources to query.

    Order of preference:
      1. Caller passed an explicit `sources` list (API intent wins).
      2. LLM router returns a usable list.
      3. Default to ALL sources (full tier order).

    The previous regex fallback was removed in 2026-06 — it did dumb
    substring matching ("ago" in "18 months ago" force-routed to AGO),
    which silently capped retrieval on ~7 of 200 benchmark questions.
    Defaulting to all sources is safer: over-retrieves rather than under.
    Returns (sources, routing_method) so callers can surface it."""
    if sources:
        return build_default_source_order(sources), "caller"

    routed = route_sources(question, model=model)
    if routed:
        return build_default_source_order(routed), "llm"

    return build_default_source_order(None), "default"


# ============================================================
# CRITIC LOOP
# ============================================================

def _dedupe_evidence(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Same (source, display_title, factoid_text) tuple appearing twice
    after a second-hop merge is wasted context. Keep the higher-scored copy."""
    seen: Dict[tuple, Dict[str, Any]] = {}
    for it in items:
        key = (
            it.get("source", ""),
            it.get("display_title", ""),
            it.get("factoid_text", ""),
        )
        prior = seen.get(key)
        if prior is None or it.get("final_score", 0.0) > prior.get("final_score", 0.0):
            seen[key] = it
    return list(seen.values())


def _retrieve_with_critic(
    question: str,
    sources: List[str],
    top_k: int,
    model: str | None = None,
) -> tuple[List[Dict[str, Any]], EvidenceVerdict]:
    """Run the agentic retrieval loop: initial retrieve, grade, optional
    second-hop on the critic-suggested follow-up queries, then re-grade.
    If the verdict is STILL insufficient and the routed sources are a
    strict subset of all sources, do one final broad-fallback retrieval
    against every source — recovers the case where the router under-routed.

    Returns (final_evidence, final_verdict). The verdict reflects the
    state of `final_evidence` (not the initial pool)."""
    evidence = retrieve(question=question, sources=sources, top_k=top_k)
    verdict = grade_evidence(question, evidence, model=model)

    if verdict.sufficient or not verdict.gaps:
        return evidence, verdict

    # One reformulation round: hit each gap with a small focused retrieve
    # against the SAME (routed) sources.
    for hop in range(MAX_REFORMULATION_HOPS):
        merged_new: List[Dict[str, Any]] = []
        for gap in verdict.gaps:
            sub = retrieve(
                question=gap.follow_up_query,
                sources=sources,
                top_k=PER_GAP_TOP_K,
            )
            merged_new.extend(sub)
        if not merged_new:
            break

        combined = _dedupe_evidence(evidence + merged_new)
        # Re-rank the combined pool by RRF-style final_score where present,
        # falling back to similarity score.
        combined.sort(
            key=lambda it: (it.get("final_score") or 0.0, it.get("score") or 0.0),
            reverse=True,
        )
        evidence = combined[: max(top_k, len(verdict.gaps) * PER_GAP_TOP_K)]
        verdict = grade_evidence(question, evidence, model=model)
        if verdict.sufficient:
            break

    # Broad-fallback: if we're STILL insufficient and the LLM router picked a
    # strict subset of sources, the most likely failure is that the right
    # evidence lives in a source we didn't query. Retry once against every
    # source with the original question, merge, re-grade. We tolerate a
    # bigger pool here (2x top_k) because the broad pass casts a wider net
    # and we want the rerank to pick the best across sources, not throw
    # away the new candidates because the pool size cap fired first.
    routed_set = set(sources)
    all_set = set(TIER_ORDERED_SOURCES)
    if not verdict.sufficient and routed_set < all_set:
        broad = retrieve(question=question, sources=TIER_ORDERED_SOURCES, top_k=top_k)
        combined = _dedupe_evidence(evidence + broad)
        combined.sort(
            key=lambda it: (it.get("final_score") or 0.0, it.get("score") or 0.0),
            reverse=True,
        )
        evidence = combined[: top_k * 2]
        verdict = grade_evidence(question, evidence, model=model)

    return evidence, verdict


# ============================================================
# PUBLIC ENTRY POINTS
# ============================================================

def answer_question(question: str, sources: list[str] | None = None, top_k: int = 15, model: str | None = None):
    retrieval_sources, routing = _pick_sources(question, sources, model=model)

    evidence, verdict = _retrieve_with_critic(
        question=question, sources=retrieval_sources, top_k=top_k, model=model,
    )

    if not evidence:
        return {
            "question": question,
            "answer": "I could not find relevant evidence in the Belladonna factoid store for this question.",
            "evidence": [],
            "routed_sources": retrieval_sources,
            "routing_method": routing,
            "evidence_verdict": verdict.to_dict(),
        }

    meta = {}
    answer = generate_grounded_answer(question, evidence, verdict=verdict, model=model, meta=meta)

    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "routed_sources": retrieval_sources,
        "routing_method": routing,
        "evidence_verdict": verdict.to_dict(),
        "completion_tokens": meta.get("completion_tokens"),
        "finish_reason": meta.get("finish_reason"),
    }


def chat_answer(
    message: str,
    session_id: str | None = None,
    sources: list[str] | None = None,
    top_k: int = 15,
):
    """Multi-turn variant of answer_question.

    Resolves the session, rewrites the message into a standalone query
    using prior turns, retrieves, answers with conversation context, then
    persists both the user message and the assistant answer."""
    session_id = STORE.get_or_create(session_id)
    history = STORE.history(session_id)

    # History-aware query rewrite (no-op when there is no prior context).
    search_query = condense_question(history, message)

    retrieval_sources, routing = _pick_sources(search_query, sources)

    evidence, verdict = _retrieve_with_critic(
        question=search_query, sources=retrieval_sources, top_k=top_k,
    )

    if not evidence:
        answer = (
            "I could not find relevant evidence in the Belladonna factoid "
            "store for this question."
        )
    else:
        answer = generate_grounded_answer(
            search_query, evidence, history=history, verdict=verdict,
        )

    STORE.append(session_id, "user", message)
    STORE.append(session_id, "assistant", answer)

    return {
        "session_id": session_id,
        "answer": answer,
        "search_query": search_query,
        "evidence": evidence,
        "history": STORE.history(session_id),
        "routed_sources": retrieval_sources,
        "routing_method": routing,
        "evidence_verdict": verdict.to_dict(),
    }
