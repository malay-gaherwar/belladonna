#!/usr/bin/env python3
"""Direct (no-RAG) zero-shot MCQ benchmark of the LOCAL models.

This is the edgecases-main / OpenRouter-style benchmark, but pointed at your
own infrastructure instead of OpenRouter and with NO retrieval: each question
is sent straight to the model, which answers from its own knowledge. It is the
clean baseline to compare against the RAG runs (bench/run_rag.py).

Two backends, mirroring the RAG's llm.py so the same model ids behave the same:
  - Gateway models (GPT-OSS-120B, DeepSeek-V4-Flash, …) → the OpenAI-compatible
    endpoint at BASE_URL with VIRTUAL_API_KEY.
  - Ollama models (Qwen3.5-35B-A3B-ollama[-think]) → the local Ollama native
    /api/chat at OLLAMA_BASE_URL, with reasoning toggled via the `think` flag.

NOT OpenRouter. No network calls leave your machines except to those endpoints.

Anti-memorization (same as the RAG runner): the gold answer is never shown to
the model; pure zero-shot; options in original order (--shuffle to opt in);
fixed seed.

Usage:
    python3 bench/run_direct.py                       # all models, all 200 Qs
    python3 bench/run_direct.py --models GPT-OSS-120B  # a subset
    python3 bench/run_direct.py --limit 20             # smoke test
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rag import (  # noqa: E402
    load_questions, present_question, parse_answer,
    OPTION_LABELS, SEED, ROOT,
)

try:
    from dotenv import load_dotenv
    for cand in (ROOT / ".env",
                 ROOT.parent / "belladonna" / "belladonnawebsite-main" / "belladonna_rag" / ".env"):
        if cand.exists():
            load_dotenv(cand)
except Exception:
    pass

# ── Config ──────────────────────────────────────────────────────────────────

# The new (v2.0) 200-question set lives in this folder.
DEFAULT_QUESTIONS = ROOT / "08062026_expert_question.2.0.json"
RESULTS_DIR = ROOT / "results_direct"
DATASET_ID = "expert200v2"
DATASET_NAME = "Belladonna Expert 200 (v2.0)"

# Only models actually DEPLOYED on the gateway right now (probed via /models +
# a liveness check on 2026-06-11). Most of the 84 registered ids return 503
# "ServiceUnavailableError" because they aren't loaded — DeepSeek-V4-* and the
# gateway qwen3.5-35b-a3b among them — so they're left out. Re-probe and update
# this list when the deployed set changes. The two Ollama Qwen entries were
# dropped (the box wasn't reachable and the gateway 35B isn't loaded either).
MODELS = [
    "medgemma-1.5-4b-it",      # small medical Gemma (emits a <unused94>thought trace) — new
    "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",  # reasoning-distilled — done: 88.0%
    "medgemma-27b-it",         # medical-tuned Gemma — done: 76.5%
    "GPT-OSS-120B",            # reasoning — needs the big max_tokens (done: 82%)
    "gemma-4-31B-it-h200",     # done: 89.5%
    "gemma-4-31B-it",          # new
    "Qwen3.5-397B-A17B-FP8",   # reasoning — rerun with the raised token budget
    "Qwen3-235B-A22B-FP8",     # new
]

# Edgecases sets max_tokens = 4096 if reasoning else 32. We keep that split but
# raise the reasoning ceiling to 32k (per request). A model listed here emits a
# reasoning trace, so a 32-token cap would truncate it before the answer letter;
# everything else answers with a bare letter and 32 tokens is plenty.
# NOTE: if a model you expect to answer instantly comes back all "no-answer",
# it's probably a hidden thinker — move it into this set.
REASONING_MODELS = {
    "Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled",
    "GPT-OSS-120B",
    "Qwen3.5-397B-A17B-FP8",
    "Qwen3-235B-A22B-FP8",
    "medgemma-1.5-4b-it",   # emits a <unused94> thought trace
}
REASONING_MAX_TOKENS = 32768
BARE_LETTER_MAX_TOKENS = 32

# Ollama-served models (mirror of belladonna_rag/llm.py OLLAMA_MODELS): friendly
# name -> real Ollama tag + whether to reason on the answer call.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODELS = {
    "Qwen3.5-35B-A3B-ollama":       {"model": "qwen3.5:35b-a3b", "think": False},
    "Qwen3.5-35B-A3B-ollama-think": {"model": "qwen3.5:35b-a3b", "think": True},
}
# num_predict caps thinking+content combined; give reasoning runs headroom so
# the trace doesn't eat the budget before the FINAL ANSWER line.
_OLLAMA_NUM_PREDICT = 32           # bare-letter answer (edgecases-style)
_OLLAMA_THINK_NUM_PREDICT = 32768  # reasoning trace + answer
_OLLAMA_NUM_CTX = 24576
# Ollama is a single GPU box — keep it gentle so requests don't 500 / OOM.
OLLAMA_CONCURRENCY = 2

MAX_RETRIES = 3
RETRY_DELAY = 5

# edgecases-main/bench/run.py SYSTEM_PROMPT, verbatim (bare-letter, no explanation).
SYSTEM_PROMPT = (
    "You are an expert answering multiple-choice questions. "
    "Reply with ONLY the letter of the correct answer (e.g. A). "
    "Do not include any explanation."
)


def gateway_endpoint():
    base = os.getenv("BASE_URL")
    key = os.getenv("VIRTUAL_API_KEY")
    if not base or not key:
        raise RuntimeError(
            "BASE_URL / VIRTUAL_API_KEY not set — export the same gateway creds "
            "your RAG uses. (This benchmark does NOT use OpenRouter.)"
        )
    return base.rstrip("/") + "/chat/completions", key


def build_direct_prompt(q: dict, pres: dict) -> str:
    """Zero-shot MCQ user message — question + options, no evidence, no RAG."""
    options = q["options"]
    lines = [f"{disp}) {options[orig]}" for disp, orig in pres["label_map"].items()]
    return f"{q['question']}\n\n" + "\n".join(lines)


# ── Backends ────────────────────────────────────────────────────────────────

async def call_gateway(session, url, key, model, user_msg, sem, temperature, max_tokens):
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                timeout = aiohttp.ClientTimeout(total=180)
                async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
                    body = await resp.json()
                    if resp.status != 200:
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1)); continue
                        return None, None, body.get("error", {}).get("message", f"HTTP {resp.status}")
                    choices = body.get("choices") or []
                    if not choices:
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1)); continue
                        return None, None, "no choices in response"
                    content = choices[0]["message"]["content"]
                    return content, body.get("usage", {}), None
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1)); continue
                return None, None, str(e)
    return None, None, "max retries exceeded"


async def call_ollama(session, cfg, user_msg, sem, temperature):
    """Local Ollama native /api/chat. `think` toggles the reasoning trace
    (returned separately as message.thinking, which we drop)."""
    think = cfg["think"]
    num_predict = _OLLAMA_THINK_NUM_PREDICT if think else _OLLAMA_NUM_PREDICT
    payload = {
        "model": cfg["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "think": think,
        "options": {"temperature": temperature, "num_predict": num_predict,
                    "num_ctx": _OLLAMA_NUM_CTX},
    }
    url = OLLAMA_BASE_URL.rstrip("/") + "/api/chat"
    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                timeout = aiohttp.ClientTimeout(total=900)
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        txt = await resp.text()
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1)); continue
                        return None, None, f"HTTP {resp.status}: {txt[:160]}"
                    body = await resp.json()
                    content = (body.get("message") or {}).get("content", "") or ""
                    usage = {"completion_tokens": body.get("eval_count", 0)}
                    return content, usage, None
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1)); continue
                return None, None, str(e)
    return None, None, "max retries exceeded"


# ── Per-model worker ────────────────────────────────────────────────────────

async def benchmark_model(session, gw_url, gw_key, model, questions, presentations, args):
    is_ollama = model in OLLAMA_MODELS
    conc = min(args.concurrency, OLLAMA_CONCURRENCY) if is_ollama else args.concurrency
    sem = asyncio.Semaphore(conc)

    safe = model.replace("/", "_")
    result_file = RESULTS_DIR / f"direct__{safe}__{DATASET_ID}.json"
    completed = {}
    if result_file.exists() and not args.fresh:
        with open(result_file, encoding="utf-8") as f:
            for a in json.load(f).get("answers", []):
                # Only cache GOOD answers. Errored or no-answer (e.g. a model
                # truncated before "FINAL ANSWER", or the endpoint was down)
                # questions are left out so a plain rerun retries exactly the
                # failures once the underlying issue is fixed.
                if not a.get("error") and a.get("predicted") is not None:
                    completed[a["id"]] = a
    remaining = [q for q in questions if q["id"] not in completed]
    answers = list(completed.values())
    total = len(questions)
    done = len(completed)
    lock = asyncio.Lock()

    print(f"\n{'─' * 60}")
    print(f"▶ {model}  [{'ollama' if is_ollama else 'gateway'}]  conc={conc}")
    print(f"  {total} questions ({len(remaining)} to run, {len(completed)} cached) -> {result_file.name}")
    sys.stdout.flush()

    async def process(q):
        nonlocal done
        pres = presentations[q["id"]]
        user_msg = build_direct_prompt(q, pres)
        t0 = time.monotonic()
        if is_ollama:
            content, usage, err = await call_ollama(session, OLLAMA_MODELS[model], user_msg, sem, args.temperature)
        else:
            mt = args.max_tokens or (REASONING_MAX_TOKENS if model in REASONING_MODELS
                                     else BARE_LETTER_MAX_TOKENS)
            content, usage, err = await call_gateway(session, gw_url, gw_key, model, user_msg, sem,
                                                      args.temperature, mt)
        latency = round(time.monotonic() - t0, 2)
        ans = {"id": q["id"], "category": q.get("category", ""),
               "gold": pres["gold"], "gold_original": pres["gold_original"],
               "shuffled": pres["shuffled"], "label_map": pres["label_map"],
               "latency_s": latency}
        if err is not None:
            ans.update(predicted=None, correct=False, error=err, answer_text="")
        else:
            cleaned = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL)
            predicted = parse_answer(cleaned, pres["num_options"])
            ans.update(predicted=predicted,
                       correct=(predicted is not None and predicted == pres["gold"]),
                       answer_text=(content or "").strip(),
                       output_tokens=(usage or {}).get("completion_tokens", 0),
                       no_answer=(predicted is None))
        async with lock:
            answers.append(ans); done += 1
            if done % 10 == 0 or done == total:
                acc = sum(1 for a in answers if a.get("correct"))
                print(f"  [{model}] {done}/{total}  running acc {round(acc / max(done,1) * 100, 1)}%")
                sys.stdout.flush()
                write_result(result_file, model, questions, answers, is_ollama, args)

    batch = conc * 3
    for i in range(0, len(remaining), batch):
        await asyncio.gather(*(process(q) for q in remaining[i:i + batch]))
    write_result(result_file, model, questions, answers, is_ollama, args)

    correct = sum(1 for a in answers if a.get("correct"))
    errors = sum(1 for a in answers if a.get("error"))
    noa = sum(1 for a in answers if a.get("no_answer"))
    print(f"  ✓ {model}: {round(correct / 200 * 100, 1)}% over 200  "
          f"({correct} correct, {errors} errors, {noa} no-answer)")
    sys.stdout.flush()


def write_result(path, model, questions, answers, is_ollama, args):
    by_id = {q["id"]: q for q in questions}
    correct = sum(1 for a in answers if a.get("correct"))
    errors = sum(1 for a in answers if a.get("error"))
    graded = len(answers) - errors
    enriched = []
    for a in sorted(answers, key=lambda x: x["id"]):
        q = by_id.get(a["id"], {})
        lm = a.get("label_map", {})
        enriched.append({**a, "question": q.get("question", ""),
                         "options": {d: q.get("options", {}).get(o, "") for d, o in lm.items()}})
    result = {
        "model": model, "dataset_id": DATASET_ID, "dataset_name": DATASET_NAME,
        "engine": f"direct, no RAG ({'ollama' if is_ollama else 'gateway'})",
        "total": len(questions), "answered": len(answers),
        "correct": correct, "errors": errors,
        "accuracy_over_200": round(correct / max(len(questions), 1) * 100, 2),
        "accuracy_graded": round(correct / max(graded, 1) * 100, 2),
        "output_tokens": sum(a.get("output_tokens", 0) for a in answers),
        "config": {"zero_shot": True, "shuffle_options": args.shuffle, "seed": SEED,
                   "temperature": args.temperature, "gold_seen_by_model": False},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "answers": enriched,
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


# ── Comparison build (direct dashboard) ─────────────────────────────────────

def build_compare():
    models, all_categories = [], set()
    for f in sorted(RESULTS_DIR.glob(f"direct__*__{DATASET_ID}.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        answers = data["answers"]
        graded = [a for a in answers if not a.get("error")]
        cats = {}
        for a in answers:
            c = a.get("category", "uncategorized") or "uncategorized"
            all_categories.add(c)
            d = cats.setdefault(c, {"total": 0, "correct": 0, "errors": 0})
            d["total"] += 1
            if a.get("error"):
                d["errors"] += 1
            elif a.get("correct"):
                d["correct"] += 1
        lat = [a["latency_s"] for a in answers if a.get("latency_s")]
        # Score over ALL 200 (errors + no-answer count as wrong) — the honest,
        # consistent denominator for a head-to-head.
        models.append({
            "model": data["model"], "engine": data["engine"],
            "total": data["total"], "answered": data["answered"],
            "correct": sum(1 for a in graded if a.get("correct")),
            "errors": data["errors"],
            "no_answer": sum(1 for a in graded if a.get("no_answer")),
            "accuracy": data["accuracy_over_200"],
            "metric": data.get("output_tokens", 0),
            "mean_latency_s": round(sum(lat) / max(len(lat), 1), 2),
            "timestamp": data["timestamp"],
            "categories": {c: round(v["correct"] / max(v["total"], 1) * 100, 2)
                           for c, v in cats.items()},
            "answers": answers,
        })
    models.sort(key=lambda m: m["accuracy"], reverse=True)
    payload = {
        "dataset_name": DATASET_NAME,
        "engine": "Direct — local models, no RAG",
        "subject_label": "Model", "subject_plural": "local model(s)",
        "metric_label": "Out tok",
        "leaderboard_title": "Leaderboard — direct (no RAG)",
        "categories": sorted(all_categories),
        "models": models,
    }
    out_dir = ROOT / "compare_direct"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        "// Auto-generated by run_direct.py — do not edit.\n"
        "window.COMPARE_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8")
    (RESULTS_DIR / "compare_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  wrote {out_dir / 'data.js'}  ({len(models)} model(s))")


# ── Main ────────────────────────────────────────────────────────────────────

async def main_async(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gw_url, gw_key = gateway_endpoint()
    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]
    models = args.models or MODELS

    rng = random.Random(SEED)
    presentations = {q["id"]: present_question(q, args.shuffle, rng) for q in questions}

    print("Direct (no-RAG) benchmark of local models")
    print(f"  questions : {len(questions)}  ({Path(args.questions).name})")
    print(f"  gateway   : {gw_url}")
    print(f"  ollama    : {OLLAMA_BASE_URL}")
    print(f"  models    : {', '.join(models)}")
    print(f"  shuffle   : {'ON' if args.shuffle else 'OFF (original order)'}")
    sys.stdout.flush()

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        for model in models:        # sequential — one model at a time
            await benchmark_model(session, gw_url, gw_key, model, questions, presentations, args)

    build_compare()
    print("\n" + "=" * 60)
    print("ALL MODELS DONE — open compare_direct/index.html")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="Direct (no-RAG) MCQ benchmark of local models (not OpenRouter)")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    p.add_argument("--models", nargs="*", default=None, help="subset of model ids to run")
    p.add_argument("--concurrency", type=int, default=4, help="parallel questions (gateway; ollama capped lower)")
    p.add_argument("--temperature", type=float, default=0.7)   # edgecases uses 0.7
    p.add_argument("--max-tokens", type=int, default=0, dest="max_tokens",
                   help="gateway max_tokens override; 0 = auto (32 for bare-letter "
                        "models, 32768 for models in REASONING_MODELS).")
    p.add_argument("--shuffle", action="store_true", help="opt in to option-label shuffling (OFF by default)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
