#!/usr/bin/env python3
"""
BELLADONNA Factoid Atlas — LLM classifier.

Labels every factoid along the four clinical dimensions defined in taxonomy.py
(drug_class, biomarker, setting, evidence) using the team's hosted
GPT-OSS-120B (OpenAI-compatible gateway), the same client pattern as
scripts/add_source_hierarchy.py and scripts/FDA/factoids_fda.py.

Key design choices
------------------
* **Batched prompts**: N factoids per request (default 20). Cuts ~4.5M factoids
  to ~225k requests. GPT-OSS-120B has ample context for this.
* **Resumable**: results stream to a JSONL keyed by factoid id; a re-run skips
  ids already present, so an interrupted full pass picks up where it stopped.
* **Strict vocab**: the model may only return codes from taxonomy.py; anything
  else is coerced to the dimension's default (index 0 / "none" / background).
* **enable_thinking: False**: gpt-oss reasoning off for throughput, matching the
  existing belladonna scripts.

Input  (--input):  JSONL, one factoid per line: {"id": str, "text": str, ...}
Output (--output): JSONL, one line per factoid:
    {"id": str, "drug_class": "cdk46", "biomarker": "hr_pos",
     "setting": "metastatic", "evidence": "rct"}

Env (same as every other belladonna LLM script):
    VIRTUAL_API_KEY, BASE_URL   (read from the server ~/.bashrc)

Examples
--------
    # validation: 200 factoids, look at the result by eye
    python classify_factoids.py --input sample.jsonl --output sample.labels.jsonl

    # full corpus
    python classify_factoids.py --input factoids.jsonl --output labels.jsonl \
        --batch-size 20 --concurrency 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import taxonomy as tax

MODEL_NAME = os.getenv("MODEL_NAME", "GPT-OSS-120B")
LLM_MAX_RETRIES = 4
LLM_TIMEOUT_SECONDS = 180
# GPT-OSS always emits a hidden reasoning channel (even with reasoning_effort
# "low" it spends ~600-1000 tokens BEFORE any answer). The completion-token
# budget must cover that reasoning PLUS the JSON answer (~30 tokens/factoid),
# or the answer is truncated to empty. Budget = overhead + per-factoid output.
REASONING_OVERHEAD_TOKENS = 1800
OUTPUT_TOKENS_PER_FACTOID = 200   # v2 schema: 7 fields incl. an agent LIST (was 3 fields @90)
MAX_TEXT_CHARS = 600                      # truncate very long factoids for the prompt


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _dimension_block(dim: str) -> str:
    lines = [f"{dim}:"]
    for e in tax.TAXONOMY[dim]:
        lines.append(f"  - {e['code']}: {e['desc']}")
    return "\n".join(lines)


def _name_of(dim: str, code: str) -> str:
    for e in tax.TAXONOMY[dim]:
        if e["code"] == code:
            return e["name"]
    return code


def _agent_block() -> str:
    """drug_agent LIST, grouped by parent class/subclass (mirrors the spec)."""
    groups: list = []
    order: dict = {}
    for code, name, cls, sub in tax.AGENT_META:
        label = _name_of("drug_class", cls) + (f" — {_name_of('drug_subclass', sub)}" if sub else "")
        if label not in order:
            order[label] = len(groups)
            groups.append((label, []))
        groups[order[label]][1].append(f"  - {code}: {name}")
    out = ["drug_agent (LIST — include EVERY agent listed below that the factoid refers to,",
           "even if it is not the main focus; leave empty [] if none applies):"]
    for label, items in groups:
        out.append("")
        out.append(f"  {label}:")
        out.extend(items)
    return "\n".join(out)


SYSTEM_PROMPT = (
    "You are a breast-oncology expert annotator. You label short factual "
    "statements ('factoids') extracted from breast-cancer guidelines, trials, "
    "regulatory labels and the literature.\n\n"
    "For EACH factoid, assign one code from each dimension below. drug_agent takes "
    "a LIST (may be empty, may contain several codes). drug_subclass and "
    "drug_agent_primary may be null. Choose the single best fit. If a dimension does "
    "not clearly apply, use its default ('none' for drug_class / biomarker / setting, "
    "'background_def' for evidence, [] for drug_agent, null for drug_subclass and "
    "drug_agent_primary). Pick the code that reflects the factoid's main clinical "
    "focus, not every entity it mentions.\n\n"
    "The focus of this annotation is clinical. Preclinical and mechanistic statements "
    "are captured through the evidence dimension, but drug_class / biomarker / setting "
    "still describe the clinical concept the factoid is about.\n\n"
    "DIMENSIONS AND ALLOWED CODES:\n\n"
    + _dimension_block("drug_class") + "\n\n"
    + "drug_subclass:\n"
      "  The therapeutic subgroup the factoid refers to, or null if none applies.\n"
      "  Assign it also when the factoid refers to the subgroup generically without\n"
      "  naming a substance (e.g. \"oral SERDs are an option after progression\",\n"
      "  \"Trop-2-directed ADCs\").\n"
    + "\n".join(f"  - {e['code']}: {e['desc']}"
                for e in tax.TAXONOMY["drug_subclass"] if e["code"] != "none")
    + "\n\n" + _agent_block() + "\n\n"
      "  Rules for drug_agent:\n"
      "  - Use ONLY the agent codes listed above. Any other drug receives a\n"
      "    drug_class code and an empty drug_agent list.\n"
      "  - Match brand names, INN and development codes to the same agent code\n"
      "    (e.g. Enhertu/DS-8201 -> t_dxd; Ibrance/PD-0332991 -> palbociclib).\n"
      "  - CRITICAL: 'trastuzumab' refers ONLY to the naked antibody. Do NOT assign\n"
      "    trastuzumab when the factoid refers to trastuzumab deruxtecan or\n"
      "    trastuzumab emtansine — use t_dxd / t_dm1 instead. Assign both trastuzumab\n"
      "    and pertuzumab for the fixed-dose combination (Phesgo).\n"
      "  - Include an agent when the factoid negates it or advises against it (the\n"
      "    drug is still the subject). Do NOT include agents that are merely named as\n"
      "    unrelated background.\n\n"
      "drug_agent_primary:\n"
      "  The single agent from drug_agent the factoid is mainly about, or null if\n"
      "  drug_agent is empty. If several agents are compared, choose the one whose\n"
      "  effect or use the factoid is primarily reporting (e.g. for 'T-DXd improved\n"
      "  PFS versus T-DM1', the primary agent is t_dxd). Must be one of the codes in\n"
      "  drug_agent.\n\n"
      "  Consistency: if drug_agent is non-empty, drug_class MUST be the class that\n"
      "  drug_agent_primary belongs to, and drug_subclass MUST be the subgroup that\n"
      "  drug_agent_primary belongs to (where a subgroup exists for that class).\n\n"
    + _dimension_block("biomarker") + "\n"
      "\n  HER2 category priority: her2_pos > her2_ultralow > her2_low > her2_neg.\n"
      "  Use the most specific HER2 category the factoid supports; fall back to\n"
      "  her2_neg only when no finer category applies. Note that HER2-low and\n"
      "  HER2-ultralow are subsets of HER2-negative disease — do NOT use her2_neg\n"
      "  for factoids that specify HER2-low or HER2-ultralow.\n"
      "  If a factoid centers on triple-negative disease, use tnbc rather than\n"
      "  her2_neg. If it centers on HR+/luminal disease without a specific HER2\n"
      "  focus, use hr_pos.\n"
      "  Apply these codes both when the factoid states IHC/ISH criteria explicitly\n"
      "  and when it uses the corresponding terminology without criteria.\n"
      "  Classify retrospectively: apply current definitions regardless of\n"
      "  publication year. A pre-2022 factoid describing IHC 1+ is her2_low, even\n"
      "  though the term did not exist at the time.\n\n"
    + _dimension_block("setting") + "\n\n"
    + _dimension_block("evidence") + "\n"
      "\n  If a factoid reports both preclinical and clinical findings, use clinical.\n\n"
      "OUTPUT FORMAT: Return ONLY a JSON array, no prose, no markdown fences. One\n"
      "object per factoid, IN THE SAME ORDER as given, each of the form:\n"
      '{"i": <index>, "drug_class": "<code>", "drug_subclass": "<code>" | null,\n'
      ' "drug_agent": ["<code>", ...], "drug_agent_primary": "<code>" | null,\n'
      ' "biomarker": "<code>", "setting": "<code>", "evidence": "<code>"}\n'
      "Use only the codes listed above (the part before the colon)."
)


def build_user_prompt(batch: List[dict]) -> str:
    lines = ["Classify these factoids:\n"]
    for i, f in enumerate(batch):
        text = (f.get("text") or "").strip().replace("\n", " ")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "…"
        lines.append(f"[{i}] {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.S)
# default = index-0 code of each active dimension (the grey "not specific")
_DEFAULT = {d: tax.default_code(d) for d in tax.DIMENSIONS}
_VALID = {d: tax.valid_codes(d) for d in tax.DIMENSIONS}


def _coerce(obj: dict) -> Dict[str, object]:
    """Validate one label object. Unknown/null codes fall back to the index-0
    default. drug_agent is a LIST; drug_class/drug_subclass are then forced to
    agree with drug_agent_primary (the spec's consistency rule)."""
    out: Dict[str, object] = {}
    for d in tax.DIMENSIONS:
        code = obj.get(d)
        out[d] = code if (isinstance(code, str) and code in _VALID[d]) else _DEFAULT[d]

    raw = obj.get("drug_agent")
    agents: List[str] = []
    if isinstance(raw, list):
        for a in raw:
            if isinstance(a, str) and a in tax.AGENT_CODES and a not in agents:
                agents.append(a)

    prim = out["drug_agent_primary"]
    if not agents:
        prim = "none"
    elif prim == "none" or prim not in agents:
        prim = agents[0]          # model omitted/mismatched primary -> first agent
    if prim != "none":            # consistency: class + subclass follow the agent
        out["drug_class"] = tax.AGENT_TO_CLASS[prim]
        out["drug_subclass"] = tax.AGENT_TO_SUBCLASS[prim]

    out["drug_agent"] = agents
    out["drug_agent_primary"] = prim
    return out


def parse_response(text: str, batch_size: int) -> Optional[List[Dict[str, object]]]:
    """Parse the model's JSON array into batch_size label dicts, aligned by 'i'.
    Returns None if nothing parseable (caller retries)."""
    m = _JSON_ARRAY_RE.search(text or "")
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(arr, list):
        return None

    by_idx: Dict[int, dict] = {}
    for k, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        idx = item.get("i", k)
        try:
            idx = int(idx)
        except Exception:
            idx = k
        by_idx[idx] = item

    out: List[Dict[str, object]] = []
    for i in range(batch_size):
        out.append(_coerce(by_idx.get(i, {})))
    return out


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def classify_batch(client, sem, batch: List[dict]) -> List[Dict[str, object]]:
    user = build_user_prompt(batch)
    max_tokens = REASONING_OVERHEAD_TOKENS + OUTPUT_TOKENS_PER_FACTOID * len(batch)
    last_err: Optional[Exception] = None

    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            async with sem:
                async def _do():
                    resp = await client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user},
                        ],
                        max_completion_tokens=max_tokens,
                        temperature=0,
                        extra_body={"reasoning_effort": "low"},
                    )
                    ch = resp.choices[0]
                    return (ch.message.content or ""), ch.finish_reason

                text, finish = await asyncio.wait_for(_do(), timeout=LLM_TIMEOUT_SECONDS)
            parsed = parse_response(text, len(batch))
            if parsed is not None:
                return parsed
            last_err = ValueError(f"unparseable (finish={finish}): {text[:160]!r}")
        except Exception as e:  # noqa: BLE001
            last_err = e
        await asyncio.sleep(0.5 * attempt)

    # Retries exhausted. Rather than default the whole batch to grey, split it
    # and retry the halves (a truncated/garbled response usually only affects a
    # big batch). Only a genuinely-unparseable SINGLE factoid falls back to default.
    if len(batch) > 1:
        mid = len(batch) // 2
        left = await classify_batch(client, sem, batch[:mid])
        right = await classify_batch(client, sem, batch[mid:])
        return left + right
    print(f"[classify] 1 factoid unparseable after splitting ({last_err}); defaulting", flush=True)
    return [dict(_DEFAULT)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        for rec in read_jsonl(path):
            if "id" in rec:
                done.add(rec["id"])
    return done


def make_client():
    from openai import AsyncOpenAI
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        sys.exit("ERROR: VIRTUAL_API_KEY / BASE_URL not set (source the server ~/.bashrc).")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def run(args) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_ids(out_path) if args.resume else set()
    if done:
        print(f"[classify] resuming — {len(done)} ids already labelled", flush=True)

    factoids = [f for f in read_jsonl(in_path)
                if f.get("id") not in done and (f.get("text") or "").strip()]
    if args.limit:
        factoids = factoids[: args.limit]
    total = len(factoids)
    if total == 0:
        print("[classify] nothing to do.", flush=True)
        return
    print(f"[classify] {total} factoids · batch={args.batch_size} · "
          f"concurrency={args.concurrency} · model={MODEL_NAME}", flush=True)

    client = make_client()
    sem = asyncio.Semaphore(args.concurrency)
    batches = [factoids[i:i + args.batch_size] for i in range(0, total, args.batch_size)]

    t0 = time.time()
    done_n = 0
    write_lock = asyncio.Lock()
    fout = open(out_path, "a", encoding="utf-8")

    async def worker(batch: List[dict]):
        nonlocal done_n
        labels = await classify_batch(client, sem, batch)
        async with write_lock:
            for f, lab in zip(batch, labels):
                rec = {"id": f["id"], **lab}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done_n += len(batch)
            if done_n % (args.batch_size * 10) < args.batch_size:
                rate = done_n / max(time.time() - t0, 1e-6)
                eta = (total - done_n) / max(rate, 1e-6)
                print(f"  {done_n}/{total} ({rate:.0f}/s, ETA {eta/60:.1f} min)", flush=True)

    # Bound the number of in-flight batch coroutines a bit above the semaphore
    # so we don't build 200k coroutines at once for the full corpus.
    inflight_cap = args.concurrency * 4
    pending: set = set()
    for batch in batches:
        pending.add(asyncio.create_task(worker(batch)))
        if len(pending) >= inflight_cap:
            done_set, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
    if pending:
        await asyncio.wait(pending)

    fout.close()
    print(f"[classify] done — {done_n} factoids in {(time.time()-t0)/60:.1f} min "
          f"-> {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-label factoids along 4 clinical dimensions.")
    ap.add_argument("--input", required=True, help="JSONL of factoids: {id, text, ...}")
    ap.add_argument("--output", required=True, help="JSONL of labels (append/resume).")
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=200,
                    help="parallel in-flight requests to the local LLM API "
                         "(vLLM batches these; raise for throughput, lower to be "
                         "gentler on the shared gateway)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.set_defaults(resume=True)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
