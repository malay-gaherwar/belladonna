"""Per-source sub-agents.

retriever.py's `SourceRetriever` is the agentic seam — vanilla vector
search scoped to one source. The agents in this package wrap the same
seam with source-specific intelligence:

- FDA: extract drug name(s) from the question, payload-filter on
       generic/brand name, return drug-scoped hits.
- CTG: detect NCT ids in the question for direct lookup, dedupe results
       by trial so 1 trial doesn't take 5 of the top-k slots.

Each agent has the same .search(question, top_k) shape as the plain
retriever, so the orchestrator in retriever.py can drop it in without
caring about source-specific details. Agents that fail (LLM down,
filter mismatch, ...) MUST fall back to plain vector retrieval — they
can only ever IMPROVE results, never degrade them.
"""

from typing import Any, Dict, List, Protocol


class SearchAgent(Protocol):
    """Contract every source agent satisfies. Mirrors the
    `SourceRetriever.search()` interface but takes the raw text question
    so the agent can do its own preprocessing (entity extraction, query
    rewriting) before hitting the vector store."""

    source: str

    def search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        ...


from .registry import get_agent_for, has_agent  # noqa: E402, F401  re-export
