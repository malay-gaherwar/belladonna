# Belladonna RAG Benchmark

Zero-shot multiple-choice benchmark that scores the **Belladonna RAG system**
on the 200 expert breast-oncology questions in
`../edgecases-main/questions_final.json`.

This is the EdgeCase zero-shot harness, rebuilt to answer each question with
**your local Belladonna RAG pipeline instead of OpenRouter**. It never contacts
OpenRouter. Each question is sent to the RAG's `/query` endpoint, which runs the
real retrieval → critic → grounded-answer loop, and the letter the RAG commits
to is graded against the gold answer.

```
benchmarking_belladonna/
├── bench/run_rag.py        # the benchmark runner (this is the whole engine)
├── build_dashboard.py      # rebuild dashboard data from an existing run
├── results/                # generated: raw results + summary.json
├── dashboard/              # static dashboard (open index.html)
│   ├── index.html
│   ├── data.js             # generated: window.BENCH_DATA
│   ├── css/style.css
│   └── js/app.js
└── .env.example
```

## How to run

**1. Start the Belladonna RAG server** (separate terminal). The benchmark
talks to it over HTTP, so it must be up first:

```bash
cd ../belladonna/belladonnawebsite-main/belladonna_rag
./start.sh          # brings up Qdrant + FastAPI on http://127.0.0.1:8001
```

**2. Run the benchmark:**

```bash
cd benchmarking_belladonna
python3 bench/run_rag.py
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--limit 20` | only the first 20 questions (smoke test) |
| `--concurrency 2` | fewer parallel questions (gentler on the local LLM) |
| `--shuffle` | opt in to option-label shuffling (OFF by default) |
| `--top-k 15` | evidence chunks the RAG retrieves per question |
| `--rag-url URL` | point at a different RAG endpoint |
| `--fresh` | ignore cached results and re-run everything |

The run is **resumable**: results are saved every 10 questions, and re-running
skips questions already answered (unless `--fresh`).

**3. View the dashboard** — just open `dashboard/index.html` in a browser
(it loads `dashboard/data.js`, no server needed). To refresh it from an
existing run without re-querying the RAG: `python3 build_dashboard.py`.

## How the benchmarking works

For each of the 200 questions:

1. **Present the question.** The question text and its options are formatted as
   a zero-shot MCQ, with the options in their original order, and the RAG is
   told to finish its reply with `FINAL ANSWER: <letter>`.
2. **Ask the RAG.** The prompt is POSTed to the local `/query` endpoint. The RAG
   routes sources, retrieves evidence, runs its critic loop, and returns a
   grounded free-text answer — exactly as it would for a real clinician.
3. **Parse the choice.** The runner extracts the chosen letter from the RAG's
   reply (preferring the `FINAL ANSWER:` marker, with sensible fallbacks). If
   the RAG commits to no option, it is recorded as *no-answer* (counts as wrong,
   tracked separately).
4. **Grade.** The chosen letter is compared to the gold letter. Per-question and
   per-category accuracy are recorded, along with latency, routed sources,
   evidence count, and the critic's sufficiency verdict.

Concurrency is handled with `asyncio` + `aiohttp` (default 4 in-flight
questions — keep it low so you don't overwhelm the local LLM backend). Requests
retry up to 3× with backoff and a 300 s timeout (the RAG pipeline is slow).

## Anti-memorization — how we keep it honest

The point of this benchmark is to measure whether the **RAG retrieves and
reasons correctly**, not whether the underlying LLM memorized an answer key.
Safeguards:

1. **The gold answer is never shown to the model.** It is loaded into a separate
   structure and used *only* for grading after the RAG has already replied. It
   never appears in the prompt or the retrieval query. Nothing the model sees
   reveals which option is correct.
2. **Pure zero-shot.** No few-shot examples are ever included, so no answers can
   leak through demonstrations.
3. **Deterministic & auditable.** A fixed seed (`1337`) makes the run
   reproducible, so anyone can re-derive exactly what the model saw.
4. **Optional option-label shuffling (off by default, `--shuffle`).** Available
   as an extra guard if you ever want it: the option texts are re-lettered per
   question with the fixed seed, so even answer-key/positional memorization
   ("question N → C") can't help. The exact mapping (`label_map`,
   presented→original) is stored in the results. Questions whose options
   cross-reference each other ("both A and B", "none of the above", …) are
   detected and left untouched. **By default this is disabled and options are
   presented in their original order.**

> Note: these guarantees cover the benchmark harness. They assume the RAG's
> own evidence store (the Qdrant factoid DB) does **not** contain this quiz's
> questions-with-answers. If you ever ingest the quiz into the corpus, the RAG
> could retrieve the answer directly — keep the quiz out of the indexed sources.

## Output files

- `results/rag__<backing-llm>__expert200.json` — one file per backing LLM:
  per-question predicted/gold/correct, the RAG's answer text, presented options,
  routing, evidence count, latency, label map, and run config.
- `results/summary.json` — the most recent run, aggregated (overall +
  per-category), feeding the single-run detail dashboard.
- `results/compare_summary.json` — all backing-LLM runs aggregated for the
  comparison leaderboard.
- `dashboard/data.js` (`window.BENCH_DATA`) — single-run detail view.
- `compare/data.js` (`window.COMPARE_DATA`) — multi-LLM comparison view.

## Comparing the RAG across backing LLMs (one run, all models)

You always test **the RAG** (full retrieve → critic → grounded answer). To
compare candidate answer-LLMs, the runner loops over a list of models **in a
single run** — no server restarts. For each model it sends that model id as a
per-request `model` override to `/query`, so the RAG runs its whole pipeline on
that LLM. Retrieval inputs are identical across models (same questions, same
option order), so the comparison is apples-to-apples; only the LLM differs.

The model list lives in `MODELS` near the top of
[`bench/run_rag.py`](bench/run_rag.py):

```python
MODELS = [
    "GPT-OSS-120B",
    "DeepSeek-V4-Flash",
    "gemma-4-31B-it-h200",
    "Qwen3.5-397B-A17B-FP8",
]
```

Start the RAG **once**, then run the benchmark — it does every model
sequentially, one after another:

```bash
# terminal A — start the RAG once (any backing model; it gets overridden per request)
cd ../belladonna/belladonnawebsite-main/belladonna_rag && ./start.sh

# terminal B — benchmark the RAG across all models in MODELS
cd benchmarking_belladonna
python3 bench/run_rag.py
python3 bench/run_rag.py --models GPT-OSS-120B DeepSeek-V4-Flash   # a subset
python3 bench/run_rag.py --models auto                              # RAG's own model
```

This works because of two tiny, backward-compatible additions to the RAG:
- `/query` now accepts an optional `model` field, threaded through the router,
  critic, and answer generation (`config.py` / `models.py` / `app.py` /
  `answerer.py` / `llm.py` / `critic.py`). When omitted, the server's configured
  `MODEL_NAME` is used, so normal RAG behaviour is unchanged.
- `MODEL_NAME` is also env-overridable (`BELLADONNA_MODEL_NAME`) and reported on
  `/api`, used by the `auto` option.

Each model writes `results/rag__<model>__expert200.json` (resumable per model).

**View the comparison:** open [`compare/index.html`](compare/index.html) — a
leaderboard (accuracy, correct, errors, no-answer, mean evidence retrieved,
latency) ranked by accuracy, an accuracy bar chart, and a category × backing-LLM
accuracy heatmap. Open [`dashboard/index.html`](dashboard/index.html) for the
per-question detail of the last model run.

To rebuild both dashboards from existing result files without re-querying:
`python3 build_dashboard.py`.

> Note: every model's `/query` call here overrides the **whole** RAG (router +
> critic + answer LLM), so the model also drives source routing. If you'd rather
> hold retrieval fixed (same evidence for all models, varying only the final
> answer LLM), say so — it's a one-line change to only override
> `generate_grounded_answer`.

## What this is NOT

- It does **not** use OpenRouter or any external model API.
- It does **not** test raw models — every run goes through the full RAG
  pipeline; only the RAG's backing LLM varies between runs.
- It does **not** do self-consistency or multi-agent — it's single-shot
  zero-shot through the RAG.
