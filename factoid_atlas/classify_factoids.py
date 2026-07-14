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
OUTPUT_TOKENS_PER_FACTOID = 90
MAX_TEXT_CHARS = 600                      # truncate very long factoids for the prompt


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _dimension_block(dim: str) -> str:
    lines = [f"{dim}:"]
    for e in tax.TAXONOMY[dim]:
        lines.append(f"  - {e['code']}: {e['desc']}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    "You are a breast-oncology expert annotator. You label short factual "
    "statements ('factoids') extracted from breast-cancer guidelines, trials, "
    "regulatory labels and the literature.\n\n"
    "For EACH factoid, assign exactly ONE code from EACH of the four dimensions "
    "below. Choose the single best fit. If a dimension does not clearly apply, "
    "use the dimension's default code ('none' for drug_class/biomarker/setting, "
    "'background_def' for evidence). Pick the code that reflects the factoid's "
    "main clinical focus, not every entity it mentions.\n\n"
    "DIMENSIONS AND ALLOWED CODES:\n\n"
    + "\n\n".join(_dimension_block(d) for d in tax.DIMENSIONS)
    + "\n\n"
    "OUTPUT FORMAT: Return ONLY a JSON array, no prose, no markdown fences. "
    "One object per factoid, IN THE SAME ORDER as given, each of the form:\n"
    + '{"i": <index>, '
    + ", ".join(f'"{d}": "<code>"' for d in tax.DIMENSIONS)
    + "}\n"
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


def _coerce(obj: dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in tax.DIMENSIONS:
        code = obj.get(d)
        out[d] = code if (isinstance(code, str) and code in _VALID[d]) else _DEFAULT[d]
    return out


def parse_response(text: str, batch_size: int) -> Optional[List[Dict[str, str]]]:
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

    out: List[Dict[str, str]] = []
    for i in range(batch_size):
        out.append(_coerce(by_idx.get(i, {})))
    return out


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def classify_batch(client, sem, batch: List[dict]) -> List[Dict[str, str]]:
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
