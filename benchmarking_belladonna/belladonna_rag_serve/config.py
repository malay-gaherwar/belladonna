import os
from pathlib import Path


# ============================================================
# VECTOR BACKEND
# ============================================================
# "chroma" — the original PersistentClient setup; reads from per-source
#            ChromaDB collections at <DATA_ROOT>/<SOURCE>/embeddings/.
#            Embeddings are full 4096-dim Qwen3 vectors.
#
# "qdrant" — the production-targeted backend; one Qdrant collection per
#            source on a single server. Vectors are Matryoshka-truncated
#            to 1024-dim and INT8-quantized so the whole corpus fits the
#            8 GB RAM cap on the deployment box. The chroma data is left
#            untouched so the two backends can be A/B compared.
#
# Toggle at runtime with the BELLADONNA_VECTOR_BACKEND env var.
VECTOR_BACKEND = os.getenv("BELLADONNA_VECTOR_BACKEND", "chroma").strip().lower()
if VECTOR_BACKEND not in ("chroma", "qdrant"):
    raise RuntimeError(
        f"BELLADONNA_VECTOR_BACKEND must be 'chroma' or 'qdrant', got {VECTOR_BACKEND!r}"
    )


# Root of the embedded Belladonna knowledge base (v1_2).
# Each source has its own ChromaDB at <DATA_ROOT>/<SOURCE>/embeddings/.
DATA_ROOT = Path(os.getenv("BELLADONNA_DATA_ROOT", "artifacts"))

# Persisted ChromaDB directory per source.
SOURCE_EMBEDDING_DIRS = {
    "AGO": DATA_ROOT / "AGO" / "embeddings",
    "CTG": DATA_ROOT / "CTG" / "embeddings",
    "Elsevier": DATA_ROOT / "Elsevier" / "embeddings",
    "EMA": DATA_ROOT / "EMA" / "embeddings",
    "EPMC": DATA_ROOT / "EPMC" / "embeddings",
    "ESMO": DATA_ROOT / "ESMO" / "embeddings",
    "FDA": DATA_ROOT / "FDA" / "embeddings",
}

# Chroma collection name per source. FDA was built under a different name in v1_2.
SOURCE_COLLECTION_NAMES = {
    "AGO": "ago_factoids_qwen_embeddings",
    "CTG": "ctg_factoids_qwen_embeddings",
    "Elsevier": "elsevier_factoids_qwen_embeddings",
    "EMA": "ema_factoids_qwen_embeddings",
    "EPMC": "epmc_factoids_qwen_embeddings",
    "ESMO": "esmo_factoids_qwen_embeddings",
    "FDA": "fda_factoid_qwen_embeddings",
}

# Kept for backwards compatibility with any code still importing the old name.
SOURCE_FACTOID_DIRS = SOURCE_EMBEDDING_DIRS

# Query-side embedding model. Must match the model used to build the DBs
# (see scripts/ago_embedding.py), served via the OpenAI-compatible BASE_URL.
EMBEDDING_MODEL_NAME = "Qwen3-Embedding-8B"

# Query dimension. For Chroma we use the native 4096 dim of Qwen3; for
# Qdrant we truncate to the first 1024 dims (Matryoshka), matching how the
# vectors were ingested in migration/chroma_to_qdrant.py. The retriever
# truncates the OpenAI embedding response client-side so this works whether
# or not the inference endpoint honours `dimensions=`.
EMBEDDING_DIM = 4096 if VECTOR_BACKEND == "chroma" else 1024


# ============================================================
# QDRANT
# ============================================================
QDRANT_URL = os.getenv("BELLADONNA_QDRANT_URL", "http://127.0.0.1:6333")

# Collection names in Qdrant (set by the migration script). Distinct from
# the Chroma names so a misconfigured backend fails loudly rather than
# pretending to hit the right index.
QDRANT_COLLECTION_NAMES = {
    "AGO":      "belladonna_ago",
    "CTG":      "belladonna_ctg",
    "Elsevier": "belladonna_elsevier",
    "EMA":      "belladonna_ema",
    "EPMC":     "belladonna_epmc",
    "ESMO":     "belladonna_esmo",
    "FDA":      "belladonna_fda",
}

MAX_HITS_PER_SOURCE = 15
MAX_EVIDENCE_ITEMS = 15

# Answer-generation LLM (OpenAI-compatible endpoint).
# Overridable via env so the same code can be benchmarked against different
# backing LLMs without editing this file:  BELLADONNA_MODEL_NAME=... ./start.sh
MODEL_NAME = os.environ.get("BELLADONNA_MODEL_NAME", "GPT-OSS-120B")

# Cross-encoder reranker (Cohere-style /rerank on the same BASE_URL).
# Vector scores are merged from independently-built per-source collections,
# so their L2 distances aren't directly comparable; the reranker re-scores
# the merged candidate pool with one consistent query-relevance signal.
RERANKER_MODEL_NAME = "Qwen3-Reranker-8B"
RERANK_ENABLED = True
# How many top vector-ranked candidates to send to the reranker before
# truncating to the caller's top_k.
RERANK_CANDIDATE_POOL = 50
# Cap each document's length (chars) sent to the reranker to bound tokens.
RERANK_MAX_DOC_CHARS = 1500


# ============================================================
# EVIDENCE HIERARCHY
# ============================================================
# Belladonna's evidence pyramid. Lower tier number = stronger evidence.
# Tiers bias the FINAL ranking (after rerank) so a higher-tier hit wins
# ties against a similarly-relevant lower-tier hit. We do not exclude lower
# tiers — they still appear when nothing better is found.
#
# document_type matching is substring, case-insensitive, in tier order, so a
# doc tagged "systematic review and meta-analysis" hits tier 3 before tier 4.
EVIDENCE_TIER_RULES = [
    (1, "Guideline",                  ["guideline"]),
    (2, "Regulatory",                 ["regulatory", "label", "product information", "spc", "summary of product"]),
    (3, "Systematic review / meta",   ["systematic review", "meta-analysis", "meta analysis"]),
    (4, "RCT",                        ["randomized controlled trial", "randomised controlled trial", "randomized clinical trial", "rct"]),
    (5, "Observational / registry",   ["observational", "cohort", "case-control", "registry study", "real-world"]),
    (6, "Trial registry entry",       ["trial registry", "clinicaltrials.gov", "registry entry", "study registration"]),
    (7, "Narrative / commentary / case", ["narrative review", "commentary", "case report", "case series", "editorial", "letter"]),
]

# Fallback when document_type metadata is missing/unrecognised (the enriched
# dataset is being rolled out on the server; not every local DB has it yet).
# Conservative defaults: EPMC/Elsevier go to the lowest tier so unclassified
# literature doesn't outrank guidelines on a tie.
SOURCE_TIER_FALLBACK = {
    "AGO": 1, "ESMO": 1,
    "EMA": 2, "FDA": 2,
    "CTG": 6,
    "EPMC": 7, "Elsevier": 7,
}
DEFAULT_TIER = 7

# ============================================================
# RECENCY
# ============================================================
# Recency is one of the three signals fused in the final ranking (see the
# RANKING section below). It applies only to "paper-like" tiers (3-7).
# Guidelines (tier 1) and regulatory documents (tier 2) are versioned and
# continuously maintained by their issuing bodies, so their publication
# year carries little information about currency; we treat them as
# effectively current in the recency ordering (top rank).
#
# Rationale for treating recency as a first-class signal in medical IR:
# Shojania et al., Ann Intern Med 2007;147(4):224-33 found that ~23% of
# systematic reviews had a signal for updating within 2 years and ~50%
# within 5.5 years, motivating recency-aware ranking of biomedical
# literature.
RECENCY_APPLICABLE_TIERS = {3, 4, 5, 6, 7}


# ============================================================
# RANKING — Reciprocal Rank Fusion
# ============================================================
# Final ranking fuses three independent signals via Reciprocal Rank Fusion
# (RRF). RRF is parameter-free apart from a single constant k, removing the
# need to hand-tune relative weights:
#
#     RRF_score(d) = sum_i  1 / (k + rank_i(d))
#
# where rank_i is the document's 1-indexed rank in signal i (ties allowed),
# and k = 60 is the value Cormack et al. found robust across TREC tracks.
#
# Signals fused:
#   1. Cross-encoder relevance — Qwen3-Reranker-8B over the merged vector
#      candidate pool; cf. Nogueira & Cho, "Passage Re-ranking with BERT",
#      arXiv:1901.04085, 2019. The reranker provides one query-relevance
#      score that is comparable across sources, since the merged L2
#      distances from independently-built per-source ChromaDB collections
#      are not directly comparable.
#   2. Evidence tier — based on the OCEBM Levels of Evidence (Oxford
#      Centre for Evidence-Based Medicine, 2011) and the GRADE working
#      group's evidence hierarchy (Guyatt et al., BMJ 2008;336:924-26).
#      See EVIDENCE_TIER_RULES above.
#   3. Recency — Shojania et al., Ann Intern Med 2007;147(4):224-33.
#      Applies only to RECENCY_APPLICABLE_TIERS; tiers 1-2 are treated as
#      effectively current.
#
# Supporting literature:
#   - Cormack, Clarke & Buettcher, "Reciprocal rank fusion outperforms
#     Condorcet and individual rank learning methods", SIGIR 2009.
#   - Nogueira & Cho, "Passage Re-ranking with BERT", arXiv:1901.04085.
#   - OCEBM Levels of Evidence Working Group, "The Oxford 2011 Levels of
#     Evidence", Oxford Centre for Evidence-Based Medicine, 2011.
#   - Guyatt et al., "GRADE: an emerging consensus on rating quality of
#     evidence and strength of recommendations", BMJ 2008;336:924-26.
#   - Shojania et al., "How quickly do systematic reviews go out of
#     date?", Ann Intern Med 2007;147(4):224-33.

RRF_K = 60
RRF_USE_RERANK = True
RRF_USE_TIER = True
RRF_USE_RECENCY = True
