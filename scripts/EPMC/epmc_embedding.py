#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from openai import AsyncOpenAI


INPUT_DIR = Path("artifacts/EPMC/factoids")
OUTPUT_DIR = Path("artifacts/EPMC/embeddings")

MAX_FILES = None  # Keep as None for all files

MODEL_NAME = "Qwen3-Embedding-8B"
COLLECTION_NAME = "epmc_factoids_qwen_embeddings"
BATCH_SIZE = 64
CONCURRENCY = 50


# ============================================================
# CLIENT
# ============================================================

def get_client() -> AsyncOpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("VIRTUAL_API_KEY not set")
    if not base_url:
        raise RuntimeError("BASE_URL not set")

    return AsyncOpenAI(api_key=api_key, base_url=base_url)


# ============================================================
# EMBEDDINGS
# ============================================================

async def get_embeddings_batch(client: AsyncOpenAI, texts: List[str]) -> List[List[float]]:
    response = await client.embeddings.create(
        model=MODEL_NAME,
        input=texts,
    )
    return [x.embedding for x in response.data]


# ============================================================
# CHROMA
# ============================================================

def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(OUTPUT_DIR))
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


# ============================================================
# DATA PREP
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_factoid_id(file_name: str, local_id: Any) -> str:
    return f"{Path(file_name).stem}_{local_id}"


def sanitize_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def sanitize_metadata_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): sanitize_metadata_value(v) for k, v in d.items()}


def pick_metadata(metadata: Dict[str, Any], *names: str) -> Any:
    """Return the first non-empty value among `names`, case-insensitively.

    EPMC factoid JSON has used inconsistent metadata key casing across
    builds (e.g. TITLE vs document_title, DOI vs doi, YEAR vs
    document_year). The Belladonna RAG reads fixed flat keys, so map
    tolerantly here rather than assume one schema.
    """
    if not isinstance(metadata, dict):
        return ""
    lowered = {str(k).lower(): v for k, v in metadata.items()}
    for n in names:
        v = lowered.get(n.lower())
        if v not in (None, "", "None"):
            return v
    return ""


def prepare_rows(data: Dict[str, Any], json_path: Path) -> List[Dict[str, Any]]:
    metadata = data.get("metadata", {}) or {}
    factoids = data.get("factoids", []) or []

    file_name = metadata.get("file_name") or json_path.name
    rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(factoids, start=1):
        if not isinstance(item, dict):
            continue

        fid = item.get("id", idx)
        text = str(item.get("factoid_text", "")).strip()

        if not text:
            continue

        factoid_id = build_factoid_id(file_name, fid)

        embedding_text = " | ".join(
            str(x) for x in [
                metadata.get("source_family"),
                metadata.get("document_title"),
                metadata.get("document_type"),
                metadata.get("document_year"),
                metadata.get("journal"),
                metadata.get("publication_year"),
                metadata.get("pmcid"),
                metadata.get("pmid"),
                metadata.get("doi"),
                metadata.get("title"),
                item.get("section"),
                item.get("section_title"),
                item.get("factoid_type"),
                item.get("topic"),
                text,
            ] if x not in (None, "", "None")
        )

        # Flat keys the Belladonna RAG retriever reads directly
        # (belladonna_rag/retriever.py). These MUST mirror the other
        # sources, e.g. scripts/AGO/ago_embedding.py — otherwise EPMC
        # evidence shows blank title/year/DOI in the chatbot. The
        # doc_/factoid_ prefixed copies below are kept for completeness.
        row_metadata: Dict[str, Any] = {
            "factoid_id": factoid_id,
            "source_json": json_path.name,
            "source_family": pick_metadata(metadata, "source_family") or "EPMC",
            "document_title": pick_metadata(metadata, "document_title", "title"),
            "document_type": pick_metadata(metadata, "document_type"),
            "document_year": pick_metadata(
                metadata, "document_year", "year", "publication_year"),
            "file_name": file_name,
            "doi": pick_metadata(metadata, "doi"),
            "source_pdf_name": pick_metadata(metadata, "source_pdf_name"),
        }

        for k, v in metadata.items():
            row_metadata[f"doc_{k}"] = v

        for k, v in item.items():
            if k == "factoid_text":
                continue
            row_metadata[f"factoid_{k}"] = v

        rows.append(
            {
                "id": factoid_id,
                "document": text,
                "embedding_text": embedding_text,
                "metadata": sanitize_metadata_dict(row_metadata),
            }
        )

    return rows


# ============================================================
# UPSERT
# ============================================================

async def process_batch(
    sem: asyncio.Semaphore,
    collection,
    client: AsyncOpenAI,
    batch: List[Dict[str, Any]],
    completed_counter: Dict[str, int],
    total: int,
) -> None:
    async with sem:
        texts = [r["embedding_text"] for r in batch]
        embeddings = await get_embeddings_batch(client, texts)

        ids = [r["id"] for r in batch]
        docs = [r["document"] for r in batch]
        metas = [r["metadata"] for r in batch]

        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

        completed_counter["done"] += len(batch)
        print(f"Upserted {completed_counter['done']}/{total}")


async def upsert_rows(collection, client: AsyncOpenAI, rows: List[Dict[str, Any]]):
    total = len(rows)
    if total == 0:
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    completed_counter = {"done": 0}

    tasks = []
    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        tasks.append(
            asyncio.create_task(
                process_batch(
                    sem=sem,
                    collection=collection,
                    client=client,
                    batch=batch,
                    completed_counter=completed_counter,
                    total=total,
                )
            )
        )

    await asyncio.gather(*tasks)


# ============================================================
# MAIN
# ============================================================

async def main():
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Input dir not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise RuntimeError("No JSON files found")

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    print(f"Processing {len(files)} file(s)")
    print(f"MODEL: {MODEL_NAME}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"CONCURRENCY: {CONCURRENCY}")

    client = get_client()
    collection = get_collection()

    total_factoids = 0

    for file in files:
        print(f"\nProcessing: {file.name}")

        data = load_json(file)
        rows = prepare_rows(data, file)

        print(f"Factoids in file: {len(rows)}")

        await upsert_rows(collection, client, rows)

        total_factoids += len(rows)

    print("\nDone.")
    print(f"Total factoids embedded: {total_factoids}")
    print(f"Chroma DB location: {OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())