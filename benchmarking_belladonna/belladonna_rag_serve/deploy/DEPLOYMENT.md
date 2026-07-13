# BELLADONNA RAG — production deployment

This is the handoff doc for the developer who deploys the RAG on the
8 GB production box. The vector index has already been built on the
research box; you only need to ship it and stand up the two processes.

## What you're deploying

```
       8 GB production box
   ┌───────────────────────────┐
   │                           │
   │   qdrant (Docker)         │  ← INT8 1024-dim vectors, ~4 GB RAM
   │     127.0.0.1:6333        │
   │                           │
   │   belladonna_rag uvicorn  │  ← FastAPI, ~0.5 GB RAM
   │     0.0.0.0:8001          │
   │                           │
   └───────────────────────────┘
            ▲                  ▲
            │ /chat, /query    │ embeddings + rerank
            │                  │
       static website        inference server
       (rag/index.html)      (Qwen3-Embedding-8B + GPT-OSS + reranker)
```

## Prerequisites on the production box

- Linux x86_64 (Ubuntu 22.04+ tested)
- Docker (>=20.10) — `sudo apt install -y docker.io && sudo usermod -aG docker $USER`
- Python 3.11 with the `belladonna_rag/requirements.txt` deps
- 8 GB RAM minimum, 12 GB recommended
- 30 GB free disk (Qdrant storage is ~13 GB; leave headroom)
- Network access to the inference server (`BASE_URL`)

## One-time setup

### 1. Ship the Qdrant index from research → production

On the research box:

```bash
# Stop the container so the storage is in a consistent state.
docker stop belladonna-qdrant

# Tar the storage directory (preserves permissions, atomic on resume).
tar -C /home/malay/Documents/belladonna_v1_2 -czf qdrant_storage.tar.gz qdrant_storage/

# Move to production. Pick whichever transport fits your network.
scp qdrant_storage.tar.gz prod-box:/srv/belladonna/

# Restart locally.
docker start belladonna-qdrant
```

On the production box:

```bash
cd /srv/belladonna
tar -xzf qdrant_storage.tar.gz
# Should yield ./qdrant_storage/collections/belladonna_<src>/ and friends.
ls qdrant_storage/collections/
```

### 2. Lay out the deployment tree

```
/srv/belladonna/
├── belladonna_rag/         # this directory, copied from git
│   ├── app.py
│   ├── retriever.py
│   ├── qdrant_retriever.py
│   ├── config.py
│   └── ...
├── deploy/
│   └── docker-compose.yml  # this file
├── qdrant_storage/         # rsync'd index from step 1
└── .env                    # see below
```

### 3. Environment variables

Create `/srv/belladonna/.env`:

```bash
# Backend selector — this is the whole switch.
BELLADONNA_VECTOR_BACKEND=qdrant

# Qdrant URL (default; only override if running it on a different host).
BELLADONNA_QDRANT_URL=http://127.0.0.1:6333

# Inference server (your existing Qwen3 / GPT-OSS / reranker endpoint).
BASE_URL=http://<inference-host>/v1/
VIRTUAL_API_KEY=<key>
```

### 4. Start Qdrant

```bash
cd /srv/belladonna
docker compose -f deploy/docker-compose.yml up -d
docker compose -f deploy/docker-compose.yml logs -f qdrant   # tail to confirm
```

Health check:

```bash
curl -fsS http://127.0.0.1:6333/healthz
# expects: "healthz check passed"

curl -s http://127.0.0.1:6333/collections | jq '.result.collections | map(.name)'
# expects all 7: belladonna_ago, belladonna_esmo, belladonna_ema,
#                belladonna_fda, belladonna_ctg, belladonna_epmc,
#                belladonna_elsevier
```

### 5. Start the FastAPI app

```bash
cd /srv/belladonna/belladonna_rag
set -a && source ../.env && set +a
conda run --no-capture-output -n belladonna uvicorn app:app \
    --host 0.0.0.0 --port 8001 --workers 2
```

Or via systemd (preferred) — see `deploy/belladonna-rag.service.template`.

End-to-end health check:

```bash
curl -fsS http://127.0.0.1:8001/health
curl -s http://127.0.0.1:8001/status | jq
curl -s -X POST http://127.0.0.1:8001/query \
    -H 'Content-Type: application/json' \
    -d '{"question":"How often should women have a screening mammogram?","top_k":3}' \
    | jq '.routed_sources, .answer'
```

## Resource expectations

- **Qdrant**: 3-4 GB resident, 1-2 GB working set during typical queries
- **uvicorn / FastAPI**: 300-500 MB resident
- **Headroom**: 2-3 GB for OS, page cache, occasional spikes during compaction

If memory pressure becomes an issue:

1. Cut `--workers` to 1.
2. Set `vm.overcommit_memory=1` to absorb transient spikes from Qdrant's optimizer.
3. Add a swap file (`fallocate -l 4G /swapfile && mkswap /swapfile && swapon /swapfile`) — not strictly needed, but cheap insurance.

## Rolling back to Chroma

The Chroma datasets are still intact on the research box (under
`/home/malay/Documents/belladonna_v1_2/<SOURCE>/embeddings/`). To revert,
ship those instead of `qdrant_storage`, set
`BELLADONNA_VECTOR_BACKEND=chroma` in `.env`, restart. EPMC and Elsevier
will fail to load on an 8 GB box — that's the whole reason we migrated.

## Updating the index

When new factoids are added on the research side:

1. Run the embedding pipeline as before (writes new vectors into the
   existing Chroma sqlite).
2. On the research box, re-run:

   ```bash
   conda run -n belladonna python \
       belladonna_rag/migration/chroma_to_qdrant.py --all
   ```

   The script is idempotent — UUID5 point IDs mean re-running upserts
   existing points and adds new ones.
3. Re-ship the `qdrant_storage` directory to production.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/status` shows `ok: false` for one source | Qdrant collection missing — index wasn't shipped correctly | Re-rsync `qdrant_storage/collections/<name>/` |
| Slow first query (5-10s), fast after | Cold cache — quantized vectors aren't in RAM yet | Hit each collection with a dummy `/query` at startup as a warm-up |
| Container restarts repeatedly | Memory limit too tight — Qdrant OOM | Raise `services.qdrant.deploy.resources.limits.memory` in the compose file |
| Embedding API timeouts | Inference endpoint slow or down | Check `BASE_URL`; the RAG falls back to vector order on reranker failure but cannot recover from embed failure |
