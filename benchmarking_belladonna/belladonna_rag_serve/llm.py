import json
import os
import re

import httpx
from openai import OpenAI

from config import MODEL_NAME, SOURCE_TIER_FALLBACK


# ── Local Ollama backend (thinking disabled) ────────────────────────────────
# A few benchmark models run on a local Ollama server instead of the gateway.
# Ollama's OpenAI-compatible /v1 endpoint cannot turn off "thinking" for the
# Qwen3.5 reasoning model — it always emits a reasoning trace that eats the
# token budget of the short router/critic calls (leaving content empty). The
# *native* /api/chat endpoint can disable it (think=false), so we route these
# models there through a tiny shim that mimics just the slice of the OpenAI
# response object the rest of the RAG reads (resp.choices[0].message.content).
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Token ceiling for the grounded-answer step, applied to every model. Kept the
# SAME as the non-RAG (direct) benchmark's default (8192) so RAG vs direct is an
# apples-to-apples comparison — identical generation budget on both sides.
# Override via env if needed.
ANSWER_MAX_TOKENS = int(os.getenv("BELLADONNA_ANSWER_MAX_TOKENS", "32768"))

# Friendly benchmark/label name -> {model: actual Ollama tag, think: bool}.
# The friendly name is what appears in MODELS and result filenames. Two
# variants of the same weights let the benchmark compare the model with its
# reasoning on vs off (the reasoning peers effectively run with thinking on).
#
# `think` here means "let the model reason on the FINAL grounded-answer step".
# The router and critic always run fast (no reasoning) regardless — they never
# request thinking — so the variants differ only in the answer step, which is
# the part that actually drives MCQ accuracy. This keeps the on/off comparison
# clean and the run tractable (~3h vs ~10h if every step reasoned).
OLLAMA_MODELS = {
    "Qwen3.5-35B-A3B-ollama":       {"model": "qwen3.5:35b-a3b", "think": False},
    "Qwen3.5-35B-A3B-ollama-think": {"model": "qwen3.5:35b-a3b", "think": True},
}

# Generation caps (num_predict). Ollama's own default can be small and would
# truncate the answer + FINAL ANSWER marker, so we set explicit floors.
#   - thinking OFF: a caller max_tokens, else this default.
#   - thinking ON: a high floor so the reasoning trace doesn't eat the budget
#     before the actual answer (num_predict caps thinking+content combined).
# num_predict is a ceiling, not a target — generation still stops at EOS — so a
# high floor is harmless for short router/critic calls.
_OLLAMA_DEFAULT_NUM_PREDICT = 4096
# A reasoning answer needs room for a long thinking trace AND the full grounded
# answer + FINAL ANSWER marker; ~6.5K thinking + ~1K answer was observed, so
# give generous headroom or the trace eats the budget and content comes back empty.
_OLLAMA_THINK_MIN_NUM_PREDICT = 16384

# Context window per request. The model's default (256K) reserves a huge KV
# cache that prevents running parallel requests; this benchmark's prompts
# (system + ~15 evidence factoids + an 8K reasoning budget) fit in 16K, which
# — together with the answer's 16K reasoning budget — fits the thinking trace
# plus the grounded answer, while still leaving room for parallel slots.
_OLLAMA_NUM_CTX = 24576


class _Msg:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Msg(content)


class _OllamaResponse:
    def __init__(self, content): self.choices = [_Choice(content)]


class _OllamaCompletions:
    def create(self, model, messages, temperature=0.0, max_tokens=None,
               think=False, **_ignored):
        cfg = OLLAMA_MODELS.get(model, {})
        real_model = cfg.get("model", model)
        # Reason only when the caller asks (the answer step passes think=True)
        # AND this variant is configured to reason. Router/critic never ask, so
        # they always run fast; the plain variant never reasons at all.
        effective_think = bool(think and cfg.get("think", False))
        if effective_think:
            num_predict = max(max_tokens or 0, _OLLAMA_THINK_MIN_NUM_PREDICT)
        else:
            num_predict = max_tokens or _OLLAMA_DEFAULT_NUM_PREDICT
        payload = {
            "model": real_model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "stream": False,
            "think": effective_think,   # native-only flag; toggles reasoning
            "options": {"temperature": temperature, "num_predict": num_predict,
                        "num_ctx": _OLLAMA_NUM_CTX},
        }
        resp = httpx.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=900.0)
        resp.raise_for_status()
        # message.content holds the final answer; message.thinking (if any) is
        # the separate reasoning trace, which we intentionally drop.
        content = (resp.json().get("message") or {}).get("content", "") or ""
        return _OllamaResponse(content)


class _OllamaChat:
    def __init__(self): self.completions = _OllamaCompletions()


class _OllamaClient:
    """Minimal stand-in for an OpenAI client that talks to a local Ollama
    server's native /api/chat with thinking disabled. Implements only the
    chat.completions.create surface the RAG actually calls."""
    def __init__(self): self.chat = _OllamaChat()


def get_local_client(model: str | None = None):
    """Return a chat client for `model`. Models registered in OLLAMA_MODELS are
    served by a local Ollama backend (thinking disabled via the native API);
    everything else uses the configured OpenAI-compatible gateway. Embeddings
    and reranking always use the gateway regardless of `model`."""
    if model and model in OLLAMA_MODELS:
        return _OllamaClient()

    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key or not base_url:
        raise RuntimeError("VIRTUAL_API_KEY or BASE_URL not found in environment.")

    return OpenAI(api_key=api_key, base_url=base_url)


def _format_history(history: list[dict], max_messages: int = 6) -> str:
    """Render the last few turns as a compact transcript for prompting."""
    recent = [m for m in history if m.get("content", "").strip()][-max_messages:]
    lines = []
    for m in recent:
        who = "Clinician" if m.get("role") == "user" else "Belladonna"
        lines.append(f"{who}: {m['content'].strip()}")
    return "\n".join(lines)


VALID_SOURCES = set(SOURCE_TIER_FALLBACK)

ROUTER_SYSTEM_PROMPT = """You route a clinician's question to the right knowledge sources in the Belladonna breast-oncology database.

Sources (evidence tier in parentheses, lower = stronger):
- AGO       German Working Group for Gynaecological Oncology — breast cancer guidelines (1)
- ESMO      European Society for Medical Oncology — clinical practice guidelines (1)
- EMA       European Medicines Agency — drug regulatory product information (2)
- FDA       US Food and Drug Administration — drug labels and approvals (2)
- CTG       ClinicalTrials.gov — trial registry entries (6)
- EPMC      Europe PMC — biomedical literature: systematic reviews, meta-analyses, RCTs, observational, case reports (3-7)
- Elsevier  Elsevier journals — biomedical literature: same mix as EPMC (3-7)

Routing rules:
1. The user names a source or unambiguous synonym (e.g. "AGO", "german guideline" -> AGO; "european authority/regulator" -> EMA; "ClinicalTrials.gov" or NCT id -> CTG; "europe pmc"/"pubmed" -> EPMC) -> return ONLY that source.
2. Clinical recommendation / standard of care / "what should I do" -> AGO, ESMO.
3. Drug approval, indication, label, contraindications, dosing per regulator -> EMA, FDA.
4. Ongoing or completed trials, eligibility, NCT ids, trial design -> CTG.
5. Specific study data, primary literature, meta-analysis evidence -> EPMC, Elsevier (you may also add guidelines if relevant).
6. Broad or ambiguous question with no clear category -> return all 7.

Output ONLY a JSON array of source names from {AGO, CTG, EMA, EPMC, ESMO, FDA, Elsevier}. No prose, no markdown fences.

Do not write any reasoning, preamble, planning, or explanation. The JSON array MUST be the very first content you emit. A typical correct reply is one line, ~15-40 characters, e.g.  ["AGO","ESMO"]   or   ["AGO","ESMO","EPMC","Elsevier"]"""


def _parse_router_output(text: str) -> list[str] | None:
    """Extract a JSON list of source names from the model's reply, tolerating
    code fences or stray prose."""
    if not text:
        return None
    text = text.strip().strip("`").strip()
    # Strip a leading "json" language tag from fenced output.
    if text.lower().startswith("json"):
        text = text[4:].lstrip(": \n")
    # Fall back to the first [...] block if the model wrapped it in prose.
    if not text.startswith("["):
        match = re.search(r"\[[^\[\]]*\]", text)
        if not match:
            return None
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    picked = [s for s in data if isinstance(s, str) and s in VALID_SOURCES]
    return picked or None


def route_sources(question: str, model: str | None = None) -> list[str] | None:
    """LLM source router: pick which sources to consult for this question.
    Returns None on any failure so callers can fall back."""
    question = (question or "").strip()
    if not question:
        return None
    try:
        client = get_local_client(model)
        response = client.chat.completions.create(
            model=model or MODEL_NAME,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.0,
            # Bumped from 120 -> 600 so reasoning-capable backends (GPT-OSS,
            # think-mode Qwen) have headroom for an internal trace AND still
            # emit the JSON list. The prompt also explicitly forbids reasoning,
            # but a few endpoints ignore that; the generous ceiling makes the
            # empty-string failure mode much less common either way.
            max_tokens=600,
        )
    except Exception:  # noqa: BLE001 - routing must never block answering
        return None
    return _parse_router_output(response.choices[0].message.content or "")


def condense_question(history: list[dict], message: str) -> str:
    """Rewrite a possibly context-dependent follow-up into a standalone
    search query, using the conversation so far to resolve references
    ("it", "that drug", "what about ESMO?", "in older patients?").

    Returns the original message unchanged when there is no history.
    """
    history = history or []
    if not history:
        return message.strip()

    client = get_local_client()
    transcript = _format_history(history)

    system_prompt = (
        "You rewrite a clinician's latest message into a single standalone "
        "search query for a breast-oncology evidence database. Use the "
        "conversation to resolve pronouns and ellipsis (it, that drug, this "
        "regimen, what about ..., in older patients). Preserve any explicitly "
        "named source (AGO, ESMO, FDA, EMA, CTG, EPMC, Elsevier). "
        "Output ONLY the rewritten query as one line, with no preamble, "
        "quotes, or explanation. If the message is already standalone, "
        "return it unchanged."
    )
    user_prompt = (
        f"Conversation so far:\n{transcript}\n\n"
        f"Latest clinician message:\n{message.strip()}\n\n"
        "Standalone search query:"
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        # Guard against the model returning empty / a refusal.
        return rewritten if rewritten else message.strip()
    except Exception:  # noqa: BLE001 - retrieval must still proceed
        return message.strip()


def generate_grounded_answer(
    question: str,
    evidence_blocks: list[dict],
    history: list[dict] | None = None,
    verdict=None,
    model: str | None = None,
    meta: dict | None = None,
) -> str:
    client = get_local_client(model)

    evidence_lines = []
    for i, item in enumerate(evidence_blocks, start=1):
        title = item.get("display_title", item.get("file_name", "")).strip()
        source = item.get("source", "").strip()
        year = str(item.get("document_year", "")).strip()
        doi = item.get("doi", "").strip()
        citation = (item.get("citation_label") or "").strip() or f"{source} {year}".strip()
        text = item.get("factoid_text", "").strip()

        evidence_lines.append(
            f"[Evidence {i} | Cite: {citation} | Title: {title} | Source: {source} | Year: {year} | DOI: {doi}]\n{text}"
        )

    evidence_text = "\n\n".join(evidence_lines)

    system_prompt = """
You are Belladonna, a breast oncology assistant.

Rules:
1. The retrieved evidence below is provided as helpful context — use it when relevant.
2. You MAY and SHOULD also use your own expert medical knowledge to answer.
3. You MUST select the single best option. Never refuse, never answer "none of the above" or "insufficient evidence" — always commit to the most likely correct option using your full clinical judgment, even when the evidence is incomplete, absent, or does not directly address the question.
4. (intentionally blank)
5. If the evidence directly answers the question, answer plainly and specifically.
6. For comparison questions, reason about each option.
7. Do not invent citations, titles, years, study names, or source names.
8. Do not use markdown bold.
9. Do not use markdown headings.
10. Do not use asterisks for emphasis.
11. Do not use the heading 'Direct answer:'.
12. Start immediately with the answer itself.
13. Do not write citations like [Evidence 4] or [Evidence 5].
14. Every citation must use exactly this format:
   [[<Cite>]]
   where <Cite> is the string from the matching evidence block's "Cite:"
   field, copied verbatim. Examples:
     - Guidelines and regulators: [[AGO 2026]], [[ESMO 2024]], [[FDA 2024]]
     - Trial registry: [[NCT01432223]]
     - Papers: [[Sammarco et al. 2023]]
15. Never invent a citation. Never put the document title, the factoid
    sentence, or any extra text inside [[...]] — only the Cite string.
    Never use the old "[[Title | Source | Year]]" form.
16. NEVER use a bare evidence index as a citation. Citations of the form
    [[1]], [[2]], [[5]], etc. are forbidden. Even when multiple evidence
    items share the same Cite string (e.g. several factoids from the same
    trial all cite as [[NCT03130439]]), repeat the Cite string verbatim —
    do not differentiate them by index.
17. Do not output a "Supporting evidence:" section. Cite inline instead.
17. A conversation so far may be provided for continuity. Use it only to
    understand what the clinician is referring to. Every clinical or factual
    claim in your answer must still be supported by the retrieved evidence
    below, never by the conversation alone.
""".strip()

    # Closing instruction. With BELLADONNA_OPTIONWISE=1 the model must first
    # adjudicate EVERY option against the retrieved evidence (supported /
    # contradicted / partially supported / not addressed) before committing —
    # an option-wise evidence check. Otherwise the usual brief-reasoning commit.
    if os.getenv("BELLADONNA_OPTIONWISE") == "1":
        closing = (
            "Before choosing, adjudicate EVERY answer option against the retrieved "
            "evidence. For each option letter output exactly one line:\n"
            "  <letter>) <LABEL> - <brief reason, citing evidence as [[Cite]] when relevant>\n"
            "where <LABEL> is exactly one of: SUPPORTED, CONTRADICTED, PARTIALLY SUPPORTED, "
            "NOT ADDRESSED. Definitions: SUPPORTED = the evidence directly affirms the option; "
            "CONTRADICTED = the evidence directly refutes it; PARTIALLY SUPPORTED = the evidence "
            "gives partial or indirect support; NOT ADDRESSED = the evidence is silent on it. "
            "After adjudicating all options, select the single best option - prefer SUPPORTED, then "
            "PARTIALLY SUPPORTED, then NOT ADDRESSED, and avoid a CONTRADICTED option unless every "
            "option is contradicted. When the evidence is silent, fall back to your own expert "
            "clinical judgment. You MUST still commit to exactly one option."
        )
    else:
        closing = "Reason briefly over the options, then commit to the single best option."
    system_prompt = system_prompt + "\n\n" + closing

    conversation_block = ""
    if history:
        conversation_block = f"""Conversation so far (context only):
{_format_history(history)}

"""

    # Pass the critic's verdict to the model — when the retrieval missed
    # specific aspects, the model can openly name them in its "Uncertainty
    # / limitations" section instead of bluffing past the gap.
    verdict_block = ""
    if verdict is not None and not getattr(verdict, "sufficient", True):
        gap_aspects = [g.aspect for g in getattr(verdict, "gaps", []) if g.aspect]
        if gap_aspects:
            verdict_block = (
                "Evidence-quality note (from the upstream critic):\n"
                "Note: the retrieved evidence may be incomplete for this "
                f"question (aspects possibly not covered: {', '.join(gap_aspects)}). "
                "Use your own medical knowledge to still commit to the single "
                "best option.\n\n"
            )

    user_prompt = f"""
{conversation_block}{verdict_block}Question:
{question}

Retrieved evidence:
{evidence_text}
""".strip()

    # The grounded-answer step is the one place we let a reasoning model think
    # (when its variant opts in). Only the local Ollama shim understands the
    # `think` kwarg; the gateway OpenAI client must not receive it.
    #
    # Cap the answer generation at a very high token budget for EVERY model, so
    # reasoning models (Qwen3.5-397B, GPT-OSS, etc.) can emit their full
    # reasoning trace AND the final answer without being truncated mid-thought.
    # It's a ceiling — models still stop at EOS — so it's free for short answers.
    create_kwargs = {"max_tokens": ANSWER_MAX_TOKENS}
    if isinstance(client, _OllamaClient):
        create_kwargs["think"] = True
    _eff = os.getenv("BELLADONNA_REASONING_EFFORT")  # e.g. "high" (gateway models only)
    if _eff and not isinstance(client, _OllamaClient):
        create_kwargs["reasoning_effort"] = _eff

    response = client.chat.completions.create(
        model=model or MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        **create_kwargs,
    )

    if meta is not None:  # surface token usage + why generation stopped
        meta["completion_tokens"] = getattr(getattr(response, "usage", None), "completion_tokens", None)
        meta["finish_reason"] = response.choices[0].finish_reason

    _msg = response.choices[0].message
    content = _msg.content
    # Reasoning models (GLM-5.2) may spend the whole budget on the hidden trace
    # and return empty content with the actual answer stranded in reasoning_content.
    # When enabled, recover the answer from the reasoning trace so the parser can
    # still extract a committed letter (deterministic no-commit recovery).
    if os.getenv("BELLADONNA_USE_REASONING_CONTENT") == "1":
        _r = getattr(_msg, "reasoning_content", None)
        if _r is None and getattr(_msg, "model_extra", None):
            _r = _msg.model_extra.get("reasoning_content")
        _r = _r or ""
        if _r and (not content or "FINAL ANSWER" not in (content or "").upper()):
            content = (content or "") + "\n[REASONING_FALLBACK]\n" + _r
    if not content or not content.strip():
        return "FINAL ANSWER: (model returned no content)"
    return content