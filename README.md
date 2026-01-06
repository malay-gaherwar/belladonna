# Belladonna

A research codebase to aggregate, structure, and evaluate breast-cancer knowledge, and to build a tiered benchmark for LLMs.

## Quick start
```bash
# 1) create and activate env (conda)
conda env create -f environment.yml
conda activate belladonna


```

## Data 
- All data will be saved in `artifacts/` and is ignored by git.



## Scripts

- `create_factoid.py` – Generates structured scientific factoids from processed biomedical text and metadata.
- `dedupe_factoids.py` – Identifies and removes semantically redundant factoids using similarity-based deduplication.
- `download_pubmed.py` – Retrieves PubMed articles and associated metadata via the NCBI API.
- `download_EPMC.py` – Downloads open-access full-text articles from Europe PMC. Works properly. Accepts three CLI flags – --query "<search terms>" to set the Europe PMC query string, --limit N to specify how many results (currently capped at 1000 per request) to retrieve, and --outdir <path> to choose the directory where the raw XML and plain‑text files are saved.
- `embed_factoids.py` – Computes vector embeddings for factoids and stores them in a Chroma vector database.
- `factoids_utils.py` – Shared utility functions for factoid parsing, formatting, and validation.
- `fetch_elsevier.py` – Fetches paper from Elsevier APIs where access is available.
- `query_embedding.py` – Generates embeddings for user queries to enable semantic retrieval.
- `rag_factoids.py` – Performs retrieval-augmented generation over the factoid database to answer biomedical questions. Requires --query "<question>" (mandatory).
- `webscrape_ESMO.py` – Scrapes guideline and meeting content from ESMO web sources. Not working.
- `data_processing.py` – Handles text normalization, sentence splitting, and preprocessing prior to extraction of factoids

