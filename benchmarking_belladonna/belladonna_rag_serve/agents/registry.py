"""Dispatch: which source has a smart agent, who delivers it.

Kept tiny on purpose — sources without an agent fall through to plain
vector retrieval in `retriever.retrieve()`, so the orchestrator just
asks `has_agent(source)` before deciding which path to take.

Agents are only used in the Qdrant backend; the Chroma fallback uses
plain SourceRetriever because adding payload filters meaningfully would
require a chroma-specific code path we don't need long-term."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from config import VECTOR_BACKEND


@lru_cache(maxsize=None)
def get_agent_for(source: str):
    """Return the registered agent for `source`, or None if there isn't one.
    Cached so the per-agent state (compiled regexes, LLM client) is reused."""
    if VECTOR_BACKEND != "qdrant":
        # Agents rely on Qdrant payload filtering; the chroma path skips
        # them and goes straight to the plain retriever.
        return None

    if source == "FDA":
        from .fda_agent import FDAAgent
        return FDAAgent()
    if source == "CTG":
        from .ctg_agent import CTGAgent
        return CTGAgent()
    return None


def has_agent(source: str) -> bool:
    return get_agent_for(source) is not None
