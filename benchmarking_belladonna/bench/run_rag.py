#!/usr/bin/env python3
"""Belladonna RAG zero-shot MCQ benchmark.

This is a self-contained re-implementation of the EdgeCase zero-shot runner
that answers each multiple-choice question with the *Belladonna RAG system*
instead of OpenRouter. It never talks to OpenRouter — it POSTs each question
to the local Belladonna RAG FastAPI server (the `/query` endpoint), which
runs the real retrieval + critic + grounded-answer pipeline, and then grades
the letter the RAG committed to against the gold answer.

────────────────────────────────────────────────────────────────────────────
ANTI-MEMORIZATION GUARANTEES (why this benchmark can't "cheat")
────────────────────────────────────────────────────────────────────────────
1. The gold answer letter is loaded into a SEPARATE structure and is used
   ONLY for grading after the RAG has already replied. It is never placed in
   the prompt, the retrieval query, or anything the model can read.
2. The prompt contains ONLY the question text and the options in their
   original order — no category, no answer, no few-shot examples (true
   zero-shot). Nothing in the prompt reveals which option is correct.
3. Optional option-label SHUFFLING is available (opt in with --shuffle) for an
   extra guard against answer-key memorization, but it is OFF by default so
   options are presented exactly as authored.
4. Runs are deterministic (fixed seed) so a reviewer can reproduce exactly
   what the model saw and confirm no leakage.

It tests the RAG once per backing LLM, all in a single run: the model id is
sent as a per-request override to /query, so no server restart is needed. The
list of models lives in MODELS below; results land in one comparison dashboard.

Usage:
    # 1. Start the Belladonna RAG server once (separate terminal):
    #       cd .../belladonna_rag && ./start.sh        # serves on :8001
    # 2. Then run the benchmark — it loops over every model in MODELS:
    python3 bench/run_rag.py
    python3 bench/run_rag.py --models GPT-OSS-120B DeepSeek-V4-Flash  # subset
    python3 bench/run_rag.py --models auto        # just the RAG's current model
    python3 bench/run_rag.py --limit 20           # smoke-test on first 20 Qs
    python3 bench/run_rag.py --concurrency 2      # gentler on the LLM backend
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import aiohttp

# ── Paths & config ──────────────────────────────────────────────────────────

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO_ROOT = ROOT.parent  # /home/malay/Documents/Belladonna

# The v2.0 (2026-06-08) 200-question expert set. Same file the direct
# (no-RAG) benchmark uses, so RAG vs direct is apples-to-apples on one dataset.
DEFAULT_QUESTIONS = ROOT / "08062026_expert_question.2.0.json"
RESULTS_DIR = ROOT / "results"

# Local Belladonna RAG server. NOT OpenRouter. start.sh serves on :8001.
DEFAULT_RAG_URL = "http://127.0.0.1:8001/query"

# Bumped to v2 so these runs don't mix with the older expert200 RAG results
# (rag__*__expert200.json) that were on the previous question file.
DATASET_ID = "expert200v2"
DATASET_NAME = "Belladonna Expert 200 (v2.0)"

# The RAG is benchmarked once per backing LLM, all in one run, by sending the
# model id as a per-request override to /query (no server restart needed). Edit
# this list to add/remove models, or pass --models on the command line. Use the
# literal "auto" to test whatever model the RAG server is currently configured
# with (auto-detected from /api). Result files: rag__<model>__expert200.json.
MODELS = [
    "GPT-OSS-120B",
    "DeepSeek-V4-Flash",
    "gemma-4-31B-it-h200",
    "Qwen3.5-397B-A17B-FP8",
    "Qwen3.5-35B-A3B-ollama",        # local Ollama (saturn), thinking OFF
    "Qwen3.5-35B-A3B-ollama-think",  # local Ollama (saturn), thinking ON (natural reasoning mode)
]

DEFAULT_LABEL = "belladonna-rag"

# The clinician chatbot posts all 7 sources on every request (no ASCO). Passing
# these via --all-sources makes the benchmark "unconstrained by source": it
# replicates the real chatbot (caller path, clinical-mindset retrieval over
# every source) instead of sending sources=None (which lets the LLM router pick
# a subset). ASCO is intentionally excluded — it has no collection.
ALL_SOURCES = ["AGO", "ESMO", "FDA", "EMA", "CTG", "EPMC", "Elsevier"]

OPTION_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
SEED = 1337       # deterministic shuffles → reproducible / auditable

# We instruct the RAG to end with this exact marker so we can parse a clean
# letter out of its free-text grounded answer.
ANSWER_MARKER = "FINAL ANSWER:"

INSTRUCTION = (
    "Using only the retrieved Belladonna evidence and your clinical reasoning, "
    "choose the single best option above. Briefly justify your choice, then end "
    f"your reply with a final line in exactly this format:\n{ANSWER_MARKER} <letter>"
)

# Options that reference other options can't be safely reshuffled.
SELF_REF_RE = re.compile(
    r"\b(all of the above|none of the above|both\s+[a-e]\b|"
    r"[a-e]\s+and\s+[a-e]\b|options?\s+[a-e]|answers?\s+[a-e])",
    re.IGNORECASE,
)


# ── Dataset loading ─────────────────────────────────────────────────────────

def load_questions(path: Path) -> list[dict]:
    """Load questions_final.json. Each item: {id, question, options{A:..},
    answer:'C', category}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    qs = raw["questions"] if isinstance(raw, dict) else raw
    if not isinstance(qs, list):
        raise ValueError(f"Unexpected dataset format in {path}")
    return qs


# ── Prompt building (with anti-memorization shuffle) ────────────────────────

def present_question(q: dict, shuffle: bool, rng: random.Random) -> dict:
    """Return a presentation of the question with (optionally) reshuffled
    option labels.

    Returns a dict with:
      prompt        : the user-facing MCQ text sent to the RAG
      gold          : the correct letter AS PRESENTED (what we grade against)
      gold_original : the correct letter in the source file
      label_map     : {presented_label: original_label}  (audit trail)
      shuffled      : whether labels were actually shuffled
    """
    options = q["options"]                      # {"A": text, ...}
    orig_labels = sorted(options.keys())        # ['A','B','C','D','E']
    gold_original = str(q["answer"]).strip().upper()

    has_self_ref = any(SELF_REF_RE.search(str(t)) for t in options.values())
    do_shuffle = shuffle and not has_self_ref and len(orig_labels) > 1

    if do_shuffle:
        order = orig_labels[:]
        rng.shuffle(order)                      # shuffled source labels
    else:
        order = orig_labels[:]                  # keep original order

    display_labels = OPTION_LABELS[: len(order)]
    label_map = {}                              # presented -> original
    gold = None
    lines = []
    for disp, orig in zip(display_labels, order):
        label_map[disp] = orig
        lines.append(f"{disp}) {options[orig]}")
        if orig == gold_original:
            gold = disp

    prompt = f"{q['question']}\n\n" + "\n".join(lines) + f"\n\n{INSTRUCTION}"
    return {
        "prompt": prompt,
        "gold": gold,
        "gold_original": gold_original,
        "label_map": label_map,
        "shuffled": do_shuffle,
        "num_options": len(order),
    }


def parse_answer(text: str, num_options: int) -> str | None:
    """Extract the chosen letter from the RAG's free-text answer.
    Returns the uppercase letter, or None if no valid choice was found."""
    if not text:
        return None
    valid = set(OPTION_LABELS[:num_options])

    # 1. Preferred: the explicit "FINAL ANSWER: X" marker (last occurrence).
    matches = re.findall(rf"{ANSWER_MARKER}\s*\(?\s*([A-Za-z])", text, re.IGNORECASE)
    if matches:
        cand = matches[-1].upper()
        if cand in valid:
            return cand

    # 2. Common phrasings: "the answer is X", "option X", "best option is X".
    m = re.search(
        r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?\s*([A-Za-z])\b",
        text, re.IGNORECASE,
    )
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()

    # 3. Last resort: the last standalone capital letter in the reply.
    standalone = re.findall(r"\b([A-Z])\b", text)
    for cand in reversed(standalone):
        if cand in valid:
            return cand

    return None


# ── RAG API call ────────────────────────────────────────────────────────────

async def call_rag(session, rag_url, prompt, sem, top_k, model=None, sources=None):
    """POST one question to the local Belladonna RAG /query endpoint.
    `model` overrides the RAG's backing answer-LLM for this request.
    `sources` (when given) pins the source set — passing ALL_SOURCES makes the
    run match the clinician chatbot; None lets the RAG's LLM router pick.
    Returns (response_dict, None) on success or (None, error_str) on failure."""
    payload = {"question": prompt, "top_k": top_k}
    if sources:
        payload["sources"] = sources
    if model:
        payload["model"] = model

    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                # The RAG pipeline (retrieve + critic + grounded answer) is
                # slow; give it a generous timeout.
                timeout = aiohttp.ClientTimeout(total=300)
                async with session.post(rag_url, json=payload, timeout=timeout) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                            continue
                        return None, f"HTTP {resp.status}: {body[:200]}"
                    data = await resp.json()
                    return data, None
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None, str(e)

    return None, "max retries exceeded"


# ── Worker ──────────────────────────────────────────────────────────────────

async def resolve_label(rag_url: str, override: str | None) -> str:
    """Figure out which backing LLM the RAG is using, for labelling this run.
    --label always wins; otherwise ask the RAG's /api endpoint (it now reports
    config.MODEL_NAME); fall back to DEFAULT_LABEL."""
    if override:
        return override
    api_url = rag_url.rsplit("/", 1)[0] + "/api"
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession() as s:
            async with s.get(api_url, timeout=timeout) as resp:
                if resp.status == 200:
                    model = (await resp.json()).get("model")
                    if model:
                        return model
    except Exception:
        pass
    return DEFAULT_LABEL


async def benchmark_one_model(session, args, label, questions, presentations):
    """Benchmark the RAG once, with `label` as its backing answer-LLM (sent as
    the per-request `model` override). Writes rag__<label>__expert200.json."""
    # NOTE: `label` is passed explicitly everywhere (never stored on the shared
    # `args`) so models can run concurrently without clobbering each other's id.
    safe = label.replace("/", "_")
    result_file = RESULTS_DIR / f"rag__{safe}__{DATASET_ID}.json"

    # Resume: cache only GOOD answers, so a rerun re-attempts the failures
    # (errored or no-answer — e.g. a truncated/looping backing LLM) once the
    # underlying issue is fixed, instead of treating them as done.
    completed = {}
    if result_file.exists() and not args.fresh:
        with open(result_file, encoding="utf-8") as f:
            for a in json.load(f).get("answers", []):
                if not a.get("error") and a.get("predicted") is not None:
                    completed[a["id"]] = a

    remaining = [q for q in questions if q["id"] not in completed]
    answers = list(completed.values())
    total = len(questions)

    print(f"\n{'─' * 60}")
    print(f"▶ RAG backed by: {label}")
    print(f"  {total} questions  ({len(remaining)} to run, {len(completed)} cached)  -> {result_file.name}")
    sys.stdout.flush()

    if not remaining:
        print("  already complete — rebuilding its summary.")
        write_result(result_file, label, questions, answers, presentations, args)
        build_summary(result_file)
        return result_file

    sem = asyncio.Semaphore(args.concurrency)
    done = len(completed)
    lock = asyncio.Lock()

    async def process(q):
        nonlocal done
        pres = presentations[q["id"]]
        t0 = time.monotonic()
        data, err = await call_rag(
            session, args.rag_url, pres["prompt"], sem, args.top_k, model=label,
            sources=(ALL_SOURCES if args.all_sources else None),
        )
        latency = round(time.monotonic() - t0, 2)

        ans = {
            "id": q["id"],
            "category": q.get("category", ""),
            "gold": pres["gold"],
            "gold_original": pres["gold_original"],
            "shuffled": pres["shuffled"],
            "label_map": pres["label_map"],
            "latency_s": latency,
        }

        if err is not None:
            ans.update(predicted=None, correct=False, error=err, answer_text="")
        else:
            answer_text = (data.get("answer") or "").strip()
            predicted = parse_answer(answer_text, pres["num_options"])
            ans.update(
                predicted=predicted,
                correct=(predicted is not None and predicted == pres["gold"]),
                answer_text=answer_text,
                routed_sources=data.get("routed_sources", []),
                routing_method=data.get("routing_method", ""),
                evidence_count=len(data.get("evidence", []) or []),
                evidence_sufficient=(data.get("evidence_verdict") or {}).get("sufficient"),
                no_answer=(predicted is None),
            )

        async with lock:
            answers.append(ans)
            done += 1
            if done % 10 == 0 or done == total:
                acc = sum(1 for a in answers if a.get("correct"))
                print(f"  [{label}] {done}/{total}  running acc {round(acc / max(done, 1) * 100, 1)}%")
                sys.stdout.flush()
                write_result(result_file, label, questions, answers, presentations, args)

    # Batch so periodic saves happen and we don't queue all 200 at once.
    batch = args.concurrency * 3
    for i in range(0, len(remaining), batch):
        await asyncio.gather(*(process(q) for q in remaining[i:i + batch]))

    write_result(result_file, label, questions, answers, presentations, args)
    build_summary(result_file)

    correct = sum(1 for a in answers if a.get("correct"))
    errors = sum(1 for a in answers if a.get("error"))
    graded = total - errors
    print(f"  ✓ {label}: {round(correct / max(graded, 1) * 100, 2)}%  "
          f"({correct}/{graded} graded, {errors} errors)")
    sys.stdout.flush()
    return result_file


async def run_benchmark(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    # Which backing LLMs to test. --models overrides the built-in MODELS list.
    # A single "auto" entry means: don't override — use whatever model the RAG
    # server is configured with (auto-detected from /api).
    models = args.models if args.models else list(MODELS)
    resolved = []
    for m in models:
        resolved.append(await resolve_label(args.rag_url, None) if m == "auto" else m)

    rng = random.Random(SEED)
    # Identical presentation for every model -> a fair, apples-to-apples
    # comparison (same questions, same option order, same shuffle decision).
    presentations = {q["id"]: present_question(q, args.shuffle, rng) for q in questions}

    print("Belladonna RAG benchmark — comparing the RAG across backing LLMs")
    print(f"  RAG endpoint : {args.rag_url}  (local — NOT OpenRouter)")
    print(f"  models       : {', '.join(resolved)}")
    print(f"  questions    : {len(questions)}")
    print(f"  shuffle      : {'ON (anti-memorization)' if args.shuffle else 'OFF (original order)'}")
    print(f"  concurrency  : {args.concurrency} per model × {len(resolved)} models in PARALLEL")
    sys.stdout.flush()

    # Each model is a separate gateway deployment, so run them all at once.
    # The connector must allow concurrency × models in-flight requests.
    connector = aiohttp.TCPConnector(limit=args.concurrency * len(resolved) + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Parallel: all models run together (label is passed explicitly, never
        # via shared args, so there's no cross-model race on the result label).
        await asyncio.gather(*(
            benchmark_one_model(session, args, label, questions, presentations)
            for label in resolved
        ))

    build_compare()

    print("\n" + "=" * 60)
    print("ALL MODELS DONE")
    print("=" * 60)
    print(f"  Compare  : open compare/index.html    (leaderboard across backing LLMs)")
    print(f"  Detail   : open dashboard/index.html  (per-question, last model run)")


# ── Persistence ─────────────────────────────────────────────────────────────

def write_result(path, label, questions, answers, presentations, args):
    """Write the full per-question result file (raw, for the dashboard)."""
    by_id = {q["id"]: q for q in questions}
    correct = sum(1 for a in answers if a.get("correct"))
    errors = sum(1 for a in answers if a.get("error"))
    graded = len(answers) - errors

    enriched = []
    for a in sorted(answers, key=lambda x: x["id"]):
        q = by_id.get(a["id"], {})
        # Embed the question + presented options so the dashboard can show the
        # full item without re-loading the source file. Options are shown in
        # PRESENTED (shuffled) lettering to match what the RAG actually saw.
        lm = a.get("label_map", {})
        presented_options = {
            disp: q.get("options", {}).get(orig, "") for disp, orig in lm.items()
        }
        enriched.append({**a, "question": q.get("question", ""),
                         "options": presented_options})

    result = {
        "model": label,
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "engine": "Belladonna RAG (local /query)",
        "total": len(questions),
        "answered": len(answers),
        "correct": correct,
        "errors": errors,
        "accuracy": round(correct / max(graded, 1) * 100, 2),
        "config": {
            "shuffle_options": args.shuffle,
            "seed": SEED,
            "top_k": args.top_k,
            "zero_shot": True,
            "gold_seen_by_model": False,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "answers": enriched,
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def build_summary(result_file: Path):
    """Aggregate the raw result file into summary.json + dashboard/data.js
    for the dashboard. Importable from build_dashboard.py too."""
    with open(result_file, encoding="utf-8") as f:
        data = json.load(f)

    answers = data["answers"]
    graded = [a for a in answers if not a.get("error")]
    correct = sum(1 for a in graded if a.get("correct"))

    # Per-category accuracy.
    cats: dict[str, dict] = {}
    for a in answers:
        c = a.get("category", "uncategorized") or "uncategorized"
        d = cats.setdefault(c, {"total": 0, "correct": 0, "errors": 0})
        d["total"] += 1
        if a.get("error"):
            d["errors"] += 1
        elif a.get("correct"):
            d["correct"] += 1
    categories = [
        {
            "category": c,
            "total": v["total"],
            "correct": v["correct"],
            "errors": v["errors"],
            "accuracy": round(v["correct"] / max(v["total"] - v["errors"], 1) * 100, 2),
        }
        for c, v in sorted(cats.items())
    ]

    latencies = [a["latency_s"] for a in answers if a.get("latency_s")]
    summary = {
        "model": data["model"],
        "dataset_name": data["dataset_name"],
        "engine": data["engine"],
        "config": data["config"],
        "timestamp": data["timestamp"],
        "total": data["total"],
        "answered": data["answered"],
        "correct": correct,
        "errors": data["errors"],
        "no_answer": sum(1 for a in graded if a.get("no_answer")),
        "accuracy": round(correct / max(len(graded), 1) * 100, 2),
        "mean_latency_s": round(sum(latencies) / max(len(latencies), 1), 2),
        "categories": categories,
        "answers": answers,
    }

    out = result_file.parent / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also emit dashboard/data.js so the dashboard works by simply opening
    # index.html (no web server / fetch / CORS needed).
    dash_data = ROOT / "dashboard" / "data.js"
    dash_data.write_text(
        "// Auto-generated by run_rag.py / build_dashboard.py — do not edit.\n"
        "window.BENCH_DATA = " + json.dumps(summary, indent=2) + ";\n",
        encoding="utf-8",
    )
    print(f"  wrote {out}")
    print(f"  wrote {dash_data}")


def build_compare():
    """Aggregate every per-backing-LLM RAG result file (rag__*__expert200.json)
    into compare/data.js for the side-by-side comparison dashboard. Each file is
    the RAG run with one backing LLM; this is what 'compare these models' means
    for the RAG (the model varies, the retrieval pipeline is identical)."""
    models = []
    all_categories = set()
    for f in sorted(RESULTS_DIR.glob(f"rag__*__{DATASET_ID}.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        answers = data["answers"]
        graded = [a for a in answers if not a.get("error")]
        cats: dict[str, dict] = {}
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
        ev = [a["evidence_count"] for a in answers if a.get("evidence_count") is not None]
        models.append({
            "model": data["model"],
            "engine": data["engine"],
            "total": data["total"],
            "answered": data["answered"],
            "correct": sum(1 for a in graded if a.get("correct")),
            "errors": data["errors"],
            "no_answer": sum(1 for a in graded if a.get("no_answer")),
            "accuracy": data["accuracy"],
            "mean_evidence": round(sum(ev) / max(len(ev), 1), 1),
            "mean_latency_s": round(sum(lat) / max(len(lat), 1), 2),
            "timestamp": data["timestamp"],
            "categories": {c: round(v["correct"] / max(v["total"] - v["errors"], 1) * 100, 2)
                           for c, v in cats.items()},
            "answers": answers,
        })
    models.sort(key=lambda m: m["accuracy"], reverse=True)
    payload = {
        "dataset_name": DATASET_NAME,
        "engine": "Belladonna RAG — same pipeline, different backing LLM",
        "categories": sorted(all_categories),
        "models": models,
    }
    out_dir = ROOT / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        "// Auto-generated by run_rag.py / build_dashboard.py — do not edit.\n"
        "window.COMPARE_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "compare_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir / 'data.js'}  ({len(models)} backing-LLM run(s))")


def build_compare():
    """Aggregate every rag__<label>__expert200.json into compare/data.js — the
    side-by-side comparison of the RAG running on different backing LLMs.

    Each result file is one backing LLM (one run of the full RAG pipeline). The
    comparison dashboard reads window.COMPARE_DATA.
    """
    models = []
    all_categories: set[str] = set()
    for f in sorted(RESULTS_DIR.glob(f"rag__*__{DATASET_ID}.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        answers = data["answers"]
        graded = [a for a in answers if not a.get("error")]

        cats: dict[str, dict] = {}
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
        ev = [a.get("evidence_count", 0) for a in graded]
        models.append({
            "model": data["model"],
            "engine": data.get("engine", "Belladonna RAG"),
            "total": data["total"],
            "answered": data["answered"],
            "correct": sum(1 for a in graded if a.get("correct")),
            "errors": data["errors"],
            "no_answer": sum(1 for a in graded if a.get("no_answer")),
            "accuracy": data["accuracy"],
            "mean_latency_s": round(sum(lat) / max(len(lat), 1), 2),
            "mean_evidence": round(sum(ev) / max(len(ev), 1), 1),
            "timestamp": data["timestamp"],
            "categories": {
                c: round(v["correct"] / max(v["total"] - v["errors"], 1) * 100, 2)
                for c, v in cats.items()
            },
            "answers": answers,
        })

    models.sort(key=lambda m: m["accuracy"], reverse=True)
    payload = {
        "dataset_name": DATASET_NAME,
        "engine": "Belladonna RAG (retrieval + critic + grounded answer)",
        "categories": sorted(all_categories),
        "models": models,
    }
    out_dir = ROOT / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        "// Auto-generated by run_rag.py / build_dashboard.py — do not edit.\n"
        "window.COMPARE_DATA = " + json.dumps(payload, indent=2) + ";\n",
        encoding="utf-8",
    )
    (RESULTS_DIR / "compare_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir / 'data.js'}  ({len(models)} backing LLM(s))")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Belladonna RAG zero-shot MCQ benchmark")
    p.add_argument("--questions", default=str(DEFAULT_QUESTIONS),
                   help="path to questions_final.json")
    p.add_argument("--rag-url", default=DEFAULT_RAG_URL, dest="rag_url",
                   help="Belladonna RAG /query endpoint (local, not OpenRouter)")
    p.add_argument("--models", nargs="*", default=None,
                   help="backing LLM ids to test the RAG with (default: the "
                        "MODELS list in this file). Use 'auto' to test the RAG's "
                        "currently-configured model. Each becomes one comparison row.")
    p.add_argument("--concurrency", type=int, default=4,
                   help="parallel in-flight questions (keep low for a local LLM)")
    p.add_argument("--top-k", type=int, default=15, dest="top_k",
                   help="evidence chunks the RAG retrieves per question")
    p.add_argument("--all-sources", action="store_true", dest="all_sources",
                   help="send all 7 sources on every request (the clinician "
                        "chatbot's behaviour: clinical-mindset over every source) "
                        "instead of sources=None (LLM router picks a subset). "
                        "This is the 'unconstrained by source' run.")
    p.add_argument("--shuffle", action="store_true",
                   help="opt in to deterministic option-label shuffling "
                        "(OFF by default — options are presented in original order)")
    p.add_argument("--limit", type=int, default=0,
                   help="only run the first N questions (smoke test)")
    p.add_argument("--fresh", action="store_true",
                   help="ignore any cached results and re-run everything")
    args = p.parse_args()

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
