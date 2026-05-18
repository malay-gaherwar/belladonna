#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from openai import OpenAI



INPUT_DIR = Path("artifacts/AGO/factoids")
OUTPUT_DIR = Path("artifacts/AGO/embeddings")

MAX_FILES = None  # Keep as None for all files

MODEL_NAME = "Qwen3-Embedding-8B"
COLLECTION_NAME = "ago_factoids_qwen_embeddings"
BATCH_SIZE = 64


# ============================================================
# CLIENT
# ============================================================

def get_client() -> OpenAI:
    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")

    if not api_key:
        raise RuntimeError("VIRTUAL_API_KEY not set")
    if not base_url:
        raise RuntimeError("BASE_URL not set")

    return OpenAI(api_key=api_key, base_url=base_url)


# ============================================================
# EMBEDDINGS
# ============================================================

def get_embeddings_batch(client: OpenAI, texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
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


def build_factoid_id(file_name: str, local_id: int) -> str:
    return f"{Path(file_name).stem}_{local_id}"


def prepare_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    metadata = data.get("metadata", {})
    factoids = data.get("factoids", [])

    rows = []

    for item in factoids:
        fid = item.get("id")
        text = item.get("factoid_text", "").strip()

        if not fid or not text:
            continue

        factoid_id = build_factoid_id(metadata.get("file_name", "unknown"), fid)

        embedding_text = " | ".join(
            str(x) for x in [
                metadata.get("source_family"),
                metadata.get("document_title"),
                metadata.get("document_type"),
                metadata.get("document_year"),
                text
            ] if x not in (None, "", "None")
        )

        rows.append(
            {
                "id": factoid_id,
                "document": text,
                "embedding_text": embedding_text,
                "metadata": {
                            "factoid_id": factoid_id,
                            "source_family": metadata.get("source_family"),
                            "document_title": metadata.get("document_title"),
                            "document_type": metadata.get("document_type"),
                            "document_year": metadata.get("document_year"),
                            "file_name": metadata.get("file_name"),
                        
                },
            }
        )

    return rows


# ============================================================
# UPSERT
# ============================================================

def upsert_rows(collection, client: OpenAI, rows: List[Dict[str, Any]]):
    total = len(rows)

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]

        texts = [r["embedding_text"] for r in batch]
        embeddings = get_embeddings_batch(client, texts)

        ids = [r["id"] for r in batch]
        docs = [r["document"] for r in batch]
        metas = [r["metadata"] for r in batch]

        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metas,
            embeddings=embeddings,
        )

        print(f"Upserted {min(i+BATCH_SIZE, total)}/{total}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Input dir not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise RuntimeError("No JSON files found")

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    print(f"Processing {len(files)} file(s)")

    client = get_client()
    collection = get_collection()

    total_factoids = 0

    for file in files:
        print(f"\nProcessing: {file.name}")

        data = load_json(file)
        rows = prepare_rows(data)

        print(f"Factoids in file: {len(rows)}")

        upsert_rows(collection, client, rows)

        total_factoids += len(rows)

    print("\nDone.")
    print(f"Total factoids embedded: {total_factoids}")
    print(f"Chroma DB location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()