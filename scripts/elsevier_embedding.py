#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from openai import AsyncOpenAI


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path("artifacts/elsevier/factoids")
OUTPUT_DIR = Path("artifacts/elsevier/embeddings")

MAX_FILES: Optional[int] = None  # None = all files

MODEL_NAME = "Qwen3-Embedding-8B"
COLLECTION_NAME = "elsevier_factoids_qwen_embeddings"

BATCH_SIZE = 64
REQUEST_CONCURRENCY = 70


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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


def sanitize_metadata_value(value: Any) -> Optional[Any]:
    """
    Chroma metadata values should be scalar-ish.
    Convert lists/dicts to JSON strings. Drop empty values.
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    if isinstance(value, (list, dict)):
        if not value:
            return None
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    return str(value)


def flatten_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in metadata.items():
        cleaned = sanitize_metadata_value(value)
        if cleaned is not None:
            out[key] = cleaned
    return out


def build_embedding_text(metadata: Dict[str, Any], factoid_text: str) -> str:
    """
    Text used for embedding. Include rich metadata context so retrieval can use
    title, journal, subtype, DOI, authors, etc.
    """
    authors = metadata.get("authors")
    if isinstance(authors, list):
        authors_text = ", ".join(str(x) for x in authors if x)
    else:
        authors_text = str(authors).strip() if authors not in (None, "") else ""

    fields = [
        metadata.get("source_family"),
        metadata.get("source_format"),
        metadata.get("document_title"),
        metadata.get("document_type"),
        metadata.get("document_subtype"),
        metadata.get("document_year"),
        metadata.get("journal"),
        metadata.get("doi"),
        metadata.get("pii"),
        metadata.get("eid"),
        metadata.get("pubmed_id"),
        metadata.get("scopus_id"),
        metadata.get("issn"),
        metadata.get("volume"),
        metadata.get("issue"),
        metadata.get("starting_page"),
        metadata.get("ending_page"),
        metadata.get("page_range"),
        metadata.get("cover_date"),
        metadata.get("cover_display_date"),
        metadata.get("publisher"),
        authors_text,
        metadata.get("abstract_from_coredata"),
        factoid_text,
    ]

    return " | ".join(str(x) for x in fields if x not in (None, "", "None"))


def prepare_rows(json_path: Path, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = data.get("metadata", {})
    factoids = data.get("factoids", [])

    if not isinstance(metadata, dict):
        metadata = {}

    if not isinstance(factoids, list):
        factoids = []

    flat_base_metadata = flatten_metadata(metadata)
    file_name = str(metadata.get("file_name") or json_path.name)

    rows: List[Dict[str, Any]] = []

    for item in factoids:
        if not isinstance(item, dict):
            continue

        local_id = item.get("id")
        text = str(item.get("factoid_text", "")).strip()

        if local_id in (None, "") or not text:
            continue

        factoid_id = build_factoid_id(file_name, local_id)

        row_metadata = dict(flat_base_metadata)
        row_metadata.update(
            {
                "factoid_id": factoid_id,
                "factoid_local_id": str(local_id),
                "source_json_file": json_path.name,
            }
        )

        rows.append(
            {
                "id": factoid_id,
                "document": text,
                "embedding_text": build_embedding_text(metadata, text),
                "metadata": row_metadata,
            }
        )

    return rows


# ============================================================
# STATS
# ============================================================

class Stats:
    def __init__(self, total_files: int) -> None:
        self.total_files = total_files
        self.processed_files = 0
        self.failed_files = 0
        self.total_factoids = 0
        self.total_batches = 0
        self.start_time = time.time()
        self.lock = asyncio.Lock()

    async def record_success(self, file_name: str, factoid_count: int, batch_count: int) -> None:
        async with self.lock:
            self.processed_files += 1
            self.total_factoids += factoid_count
            self.total_batches += batch_count
            print(
                f"[OK] {file_name} -> factoids={factoid_count} | batches={batch_count} | "
                f"processed_files={self.processed_files}/{self.total_files}"
            )

    async def record_failure(self, file_name: str, error: str) -> None:
        async with self.lock:
            self.processed_files += 1
            self.failed_files += 1
            print(
                f"[ERROR] {file_name} -> {error} | "
                f"processed_files={self.processed_files}/{self.total_files}"
            )

    def final_print(self) -> None:
        elapsed = time.time() - self.start_time
        rate = self.processed_files / elapsed if elapsed > 0 else 0.0

        print("\n========== PERFORMANCE ==========")
        print(f"Processed files:   {self.processed_files}/{self.total_files}")
        print(f"Failed files:      {self.failed_files}")
        print(f"Total factoids:    {self.total_factoids}")
        print(f"Total batches:     {self.total_batches}")
        print(f"Total time:        {elapsed:.2f}s")
        print(f"Files/sec:         {rate:.2f}")
        print("=================================")


# ============================================================
# UPSERT
# ============================================================

async def upsert_rows(
    collection,
    client: AsyncOpenAI,
    rows: List[Dict[str, Any]],
) -> int:
    total = len(rows)
    if total == 0:
        return 0

    batch_count = 0

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]

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

        batch_count += 1
        print(f"  Upserted {min(i + BATCH_SIZE, total)}/{total}")

    return batch_count


# ============================================================
# WORKER
# ============================================================

async def worker(
    worker_id: int,
    queue: asyncio.Queue,
    client: AsyncOpenAI,
    collection,
    stats: Stats,
) -> None:
    while True:
        json_path = await queue.get()

        if json_path is None:
            queue.task_done()
            return

        try:
            print(f"\n[worker={worker_id}] Processing: {json_path.name}")

            data = load_json(json_path)
            rows = prepare_rows(json_path, data)

            print(f"[worker={worker_id}] Factoids in file: {len(rows)}")

            batch_count = await upsert_rows(collection, client, rows)
            await stats.record_success(json_path.name, len(rows), batch_count)

        except Exception as e:
            await stats.record_failure(json_path.name, f"{type(e).__name__}: {e}")

        finally:
            queue.task_done()


# ============================================================
# MAIN
# ============================================================

async def main() -> None:
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Input dir not found: {INPUT_DIR}")

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON files found in {INPUT_DIR}")

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    print(f"Input dir:            {INPUT_DIR}")
    print(f"Output dir:           {OUTPUT_DIR}")
    print(f"Collection:           {COLLECTION_NAME}")
    print(f"Model:                {MODEL_NAME}")
    print(f"Batch size:           {BATCH_SIZE}")
    print(f"Request concurrency:  {REQUEST_CONCURRENCY}")
    print(f"Files to process:     {len(files)}")

    client = get_client()
    collection = get_collection()
    stats = Stats(total_files=len(files))

    queue: asyncio.Queue = asyncio.Queue()

    for file in files:
        queue.put_nowait(file)

    workers = [
        asyncio.create_task(worker(i + 1, queue, client, collection, stats))
        for i in range(REQUEST_CONCURRENCY)
    ]

    for _ in workers:
        queue.put_nowait(None)

    await queue.join()
    await asyncio.gather(*workers)

    print("\nDone.")
    print(f"Chroma DB location: {OUTPUT_DIR}")
    stats.final_print()


if __name__ == "__main__":
    asyncio.run(main())