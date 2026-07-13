# Belladonna RAG

Embedding-based retrieval-augmented QA over the Belladonna breast-oncology
knowledge base.

## How it works

```
question
   │
   ▼
embed_query()            Qwen3-Embedding-8B  (via OpenAI-compatible BASE_URL)
   │
   ▼
SourceRetriever × N      one ChromaDB collection per source
   │   (AGO, CTG, Elsevier, EMA, EPMC, ESMO, FDA)
   ▼
merge + dedupe           candidate pool by vector (L2) similarity
   │
   ▼
rerank                   Qwen3-Reranker-8B (<BASE_URL>/rerank) re-scores the
   │                     pool with one cross-source relevance signal;
   │                     falls back to vector order on any failure
   ▼
RRF fusion → top-k       Reciprocal Rank Fusion of three rankings:
   │                       1. cross-encoder relevance
   │                       2. evidence tier (OCEBM/GRADE)
   │                       3. recency (paper-like tiers only)
   ▼
generate_grounded_answer GPT-OSS-120B, answer strictly from retrieved evidence
```

Each source is an **independent** `SourceRetriever` (see `retriever.py`). The
module-level `retrieve()` only orchestrates fan-out/merge — this isolation is
intentional so each source can later be wrapped in its own agent.

## Ranking

The final top-k is produced by **Reciprocal Rank Fusion** (RRF; Cormack,
Clarke & Buettcher, *SIGIR* 2009) over three independently-derived rankings:

```
RRF_score(d) = Σ_i  1 / (k + rank_i(d))      with k = 60
```

| Signal | Source | Justification |
|---|---|---|
| **1. Cross-encoder relevance** | Qwen3-Reranker-8B over the merged vector candidate pool | Late-stage re-ranking of dense-retrieval results is standard practice; cf. Nogueira & Cho, *Passage Re-ranking with BERT*, arXiv:1901.04085 (2019). |
| **2. Evidence tier** | `document_type` metadata mapped to a 7-tier hierarchy (`config.EVIDENCE_TIER_RULES`) | OCEBM Levels of Evidence (Oxford Centre for Evidence-Based Medicine, 2011); GRADE working group (Guyatt et al., *BMJ* 2008;336:924–26). |
| **3. Recency** | `document_year`; applied only to paper-like tiers (3–7) | ~50% of systematic reviews need updating within 5.5 years (Shojania et al., *Ann Intern Med* 2007;147(4):224–33). Guidelines (tier 1) and regulatory docs (tier 2) are versioned and continuously maintained, so they are treated as effectively current. |

RRF is **parameter-free** apart from `k`, which is set to 60 — the value
Cormack et al. found robust across TREC tracks. There are no hand-tuned
relative weights between the three signals.

The retriever surfaces per-signal ranks (`rrf_ranks`) and contributions
(`rrf_contributions`) on every evidence item for auditability.

## Evaluation

`eval.py` runs the production `retrieve()` pipeline against a labelled
JSONL of queries and reports nDCG@k, MAP, MRR, and Recall@k (Jarvelin &
Kekalainen, *ACM TOIS* 2002; Manning, Raghavan & Schutze, *IIR* 2008):

```bash
# headline metrics
python eval.py path/to/queries.jsonl --k 10 --top-k 20

# leave-one-out + only-one ablation across the three RRF signals
python eval.py path/to/queries.jsonl --ablate
```

Input format (one JSON per line):

```json
{"query": "How often should mammography be repeated for average-risk women?",
 "relevant_factoid_ids": ["AGO_2025E_03_..._12", "ESMO_2024_..._47"],
 "graded_relevance": {"AGO_2025E_03_..._12": 3, "ESMO_2024_..._47": 2}}
```

`graded_relevance` is optional and enables true graded nDCG (0–3 = TREC-
style judgements); otherwise binary relevance from `relevant_factoid_ids`
is used.

## Data

Per-source persisted ChromaDB vector stores live at
`/home/malay/Documents/belladonna_v1_1/<SOURCE>/embeddings/`
(configured in `config.py`). Collections are named
`<source>_factoids_qwen_embeddings`, except FDA which is
`fda_seed_qwen_embeddings`. Vectors are 4096-dim Qwen3 embeddings; the query
must be embedded with the same model.

## Setup

Use the `belladonna` conda env (Python 3.11, already has the deps), or:

```bash
pip install -r requirements.txt
```

Required environment variables (OpenAI-compatible inference endpoint used for
both query embeddings and answer generation):

```bash
export VIRTUAL_API_KEY=...
export BASE_URL=...
```

## Run

```bash
conda run -n belladonna uvicorn app:app --host 0.0.0.0 --port 8000
# from inside belladonna_rag/  (modules import by bare name)
```

## API

- `GET /health` → `{"status": "ok"}`
- `GET /sources` → configured source list
- `GET /status` → per-source vector-store health
- `POST /query` — single-shot QA (stateless)
  ```json
  { "question": "How often is screening mammography recommended?",
    "sources": ["AGO", "ESMO"],
    "top_k": 15 }
  ```
  `sources` is optional (defaults to priority order; an explicit source named
  in the question, e.g. "according to AGO", overrides it). Returns the answer
  plus the retrieved evidence with similarity scores.

- `POST /chat` — multi-turn chat with memory
  ```json
  { "message": "What about in older patients?",
    "session_id": "<from previous response, omit on first message>",
    "sources": ["AGO", "ESMO"],
    "top_k": 15 }
  ```
  Returns `{ session_id, answer, search_query, evidence, history }`. Pass the
  returned `session_id` back on the next message to keep context. Follow-ups
  are rewritten into a standalone retrieval query using prior turns
  (`search_query` shows what was actually searched). Conversation memory is
  in-process and ephemeral (bounded to the last 16 messages, 6-hour TTL); an
  unknown `session_id` (e.g. after a restart) silently starts a new session.

- `POST /chat/reset` → `{ "session_id": "..." }` forgets that conversation.

### Conversation flow

```
message + session_id
   │
   ▼
condense_question()      rewrite follow-up → standalone query (uses history)
   │
   ▼
retrieve()               same per-source fan-out as /query
   │
   ▼
generate_grounded_answer history passed as context-only; claims still must
   │                     be grounded in retrieved evidence
   ▼
ConversationStore        append user + assistant turns, return session_id
```
