"""Evidence-sufficiency critic.

After the initial retrieval, an LLM critic reads the question alongside
the retrieved factoids and decides whether the evidence actually covers
the question. If not, it names the missing aspects and proposes a
follow-up retrieval query for each.

This is the core agentic move: instead of a fixed
`retrieve -> answer` pipeline, the system inspects its own state and
decides whether another retrieval pass is warranted.

The critic is a small structured LLM call. It returns a verdict that:
- the answerer uses to drive a single second-hop retrieval if needed
- the final answer prompt sees as context (so the model can openly say
  "this evidence does not address X" instead of bluffing past the gap)
- the API response surfaces (`evidence_verdict`) so a UI can render an
  "evidence partial" banner

Failure isolation: ANY critic failure (LLM down, bad JSON, etc.) returns
a "sufficient" verdict so the existing pipeline keeps working. We never
want this layer to make answers worse than the baseline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from config import MODEL_NAME
from llm import get_local_client


# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class EvidenceGap:
    """One missing aspect of the question + a focused query to fill it."""
    aspect: str
    follow_up_query: str


@dataclass
class EvidenceVerdict:
    """Result of one critic pass."""
    sufficient: bool
    confidence: float                       # 0.0 .. 1.0
    gaps: List[EvidenceGap] = field(default_factory=list)
    notes: str = ""
    # `source` tells callers whether this came from the LLM critic or the
    # safe-fallback path (no LLM call / parse failure).
    source: str = "llm"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


# ============================================================
# PROMPT
# ============================================================

_CRITIC_SYSTEM_PROMPT = """You are a clinical evidence quality critic for a breast-oncology RAG.

Read the clinician's question and the retrieved evidence factoids. Decide whether the retrieved evidence is sufficient to answer the question well.

A factoid is "sufficient" if the answer can be written from the retrieved evidence alone, without important caveats or guessed claims.

Common reasons evidence is INSUFFICIENT:
- The question asks about a specific drug / indication / population that the evidence does not directly address.
- The question is a comparison (X vs Y), but only one arm is covered.
- The question asks for guideline / regulatory / trial / paper content but only one type was retrieved.
- The question asks for a numeric outcome (HR, OS, 5-year survival, etc.) and the retrieved factoids are qualitative only.
- The question is multi-part and only one part has support.

When evidence is insufficient, identify up to 2 specific MISSING aspects and propose one focused follow-up retrieval query per missing aspect. The follow-up query is a search string sent to a vector database — concise, specific, and self-contained (no pronouns or "the patient"). Use the same drug/disease terminology the question uses.

Output ONLY a JSON object with this exact shape:
{
  "sufficient": <true|false>,
  "confidence": <number 0.0 to 1.0>,
  "gaps": [
    {"aspect": "<short label, e.g. 'dosing in elderly'>",
     "follow_up_query": "<a single search-engine-style query>"}
  ],
  "notes": "<one short sentence explaining the verdict; max 200 chars>"
}

If sufficient is true, "gaps" must be an empty list.
Never propose more than 2 gaps. No prose outside the JSON. No markdown fences."""


_EVIDENCE_LINE_LIMIT = 12          # cap factoids fed to the critic
_FACTOID_TEXT_LIMIT  = 400         # chars per factoid (the body is what counts)


def _format_evidence_for_critic(evidence: List[Dict[str, Any]]) -> str:
    """Compact rendering — the critic doesn't need full metadata, just
    enough to judge coverage."""
    lines: List[str] = []
    for i, item in enumerate(evidence[:_EVIDENCE_LINE_LIMIT], 1):
        src = (item.get("source") or "").strip()
        year = str(item.get("document_year") or "").strip()
        title = (item.get("display_title") or item.get("file_name") or "").strip()[:100]
        text = (item.get("factoid_text") or "").strip()[:_FACTOID_TEXT_LIMIT]
        lines.append(f"[{i}] ({src} {year}) {title}\n    {text}")
    return "\n".join(lines)


# ============================================================
# RESPONSE PARSING
# ============================================================

# Greedy `{ ... }` match — captures the JSON object even when the LLM
# wraps it in reasoning text, code fences, or markdown.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[str]:
    """Pull a balanced top-level `{...}` out of the LLM response, tolerating
    code fences, reasoning preambles, and trailing chatter.

    Greedy `\{.*\}` matches the WIDEST {...} window, so it handles the
    common case of "<reasoning> ```json {...} ``` <closer>" — the inner
    JSON object is captured cleanly because everything between the first
    `{` and last `}` is the object body."""
    if not text:
        return None
    s = text.strip()
    # Strip standard fences (```json ... ``` or ``` ... ```).
    if s.startswith("```"):
        s = s[3:]
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.lstrip("\n :")
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    m = _JSON_OBJECT_RE.search(s)
    if not m:
        return None
    return m.group(0)


def _coerce_float(v: Any, default: float = 0.5) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f < 0:
        return 0.0
    if f > 1:
        return 1.0
    return f


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "y", "sufficient")
    return False


def _parse_verdict(raw: str) -> Optional[EvidenceVerdict]:
    """Pull the JSON object out of the LLM response and validate fields."""
    candidate = _extract_json_object(raw or "")
    if candidate is None:
        return None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    sufficient = _coerce_bool(obj.get("sufficient", False))
    confidence = _coerce_float(obj.get("confidence", 0.5))

    raw_gaps = obj.get("gaps") or []
    gaps: List[EvidenceGap] = []
    if isinstance(raw_gaps, list):
        for g in raw_gaps[:2]:
            if not isinstance(g, dict):
                continue
            aspect = str(g.get("aspect") or "").strip()
            fq = str(g.get("follow_up_query") or "").strip()
            if aspect and fq:
                gaps.append(EvidenceGap(aspect=aspect, follow_up_query=fq))

    # If model contradicts itself (sufficient=true with gaps), trust the
    # gaps — the explicit content beats the boolean.
    if gaps:
        sufficient = False

    notes = str(obj.get("notes") or "").strip()[:240]
    return EvidenceVerdict(
        sufficient=sufficient, confidence=confidence,
        gaps=gaps, notes=notes, source="llm",
    )


# ============================================================
# PUBLIC API
# ============================================================

def grade_evidence(question: str, evidence: List[Dict[str, Any]], model: str | None = None) -> EvidenceVerdict:
    """Ask the critic whether `evidence` covers `question` well enough.

    Safe-default behaviour: any failure (no evidence, LLM error, bad
    JSON) returns `sufficient=True` so the caller proceeds with the
    existing fixed pipeline — the critic is never load-bearing."""
    if not evidence:
        # No evidence is a different failure mode handled by the answerer
        # directly; flag as insufficient with no gaps so the loop won't
        # try to second-hop (which would also return nothing).
        return EvidenceVerdict(
            sufficient=False, confidence=1.0,
            gaps=[], notes="no evidence retrieved",
            source="empty",
        )

    try:
        client = get_local_client(model)
        rendered = _format_evidence_for_critic(evidence)
        user = (
            f"Clinician question:\n{question.strip()}\n\n"
            f"Retrieved evidence ({len(evidence)} factoids, top {min(len(evidence), _EVIDENCE_LINE_LIMIT)} shown):\n{rendered}"
        )
        resp = client.chat.completions.create(
            model=model or MODEL_NAME,
            messages=[
                {"role": "system", "content": _CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_tokens=8192,       # high: reasoning models need room for the trace + JSON verdict
        )
        text = resp.choices[0].message.content or ""
    except Exception:  # noqa: BLE001 - critic must never break answering
        return EvidenceVerdict(
            sufficient=True, confidence=0.0,
            gaps=[], notes="critic unavailable", source="error",
        )

    verdict = _parse_verdict(text)
    if verdict is None:
        # Surface a short snippet of the raw output so the next person can
        # see what the model emitted instead of valid JSON.
        snippet = (text or "").strip().replace("\n", " ")[:200]
        import sys
        print(f"[critic] parse-error; raw snippet: {snippet!r}", file=sys.stderr)
        return EvidenceVerdict(
            sufficient=True, confidence=0.0,
            gaps=[], notes="critic output unparseable", source="parse-error",
        )
    return verdict
