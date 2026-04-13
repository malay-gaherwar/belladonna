#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from openai import OpenAI


INPUT_DIR = Path("artifacts/esmo/factoids")
OUTPUT_DIR = Path("artifacts/esmo/embeddings")

MAX_FILES = None  # Keep as None for all files

MODEL_NAME = "Qwen3-Embedding-8B"
COLLECTION_NAME = "esmo_factoids_qwen_embeddings"
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


def normalize_scalar(value: Any) -> Any:
    """
    Chroma metadata works best with scalar values.
    Convert lists/dicts to JSON strings, keep scalars as-is.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        cleaned_value = normalize_scalar(value)
        if cleaned_value is not None:
            cleaned[key] = cleaned_value
    return cleaned


def build_embedding_text(metadata: Dict[str, Any], factoid_text: str) -> str:
    parts = []

    # Put the most semantically useful fields first
    preferred_order = [
        "source_family",
        "document_family",
        "document_title",
        "document_type",
        "document_year",
        "file_name",
        "source_pdf_name",
    ]

    for key in preferred_order:
        value = metadata.get(key)
        if value not in (None, "", "None"):
            parts.append(str(value))

    # Include any additional metadata fields not already used
    for key, value in metadata.items():
        if key in preferred_order:
            continue
        if value not in (None, "", "None"):
            parts.append(f"{key}: {value}")

    parts.append(factoid_text)

    return " | ".join(parts)


def prepare_rows(data: Dict[str, Any], source_json_path: Path) -> List[Dict[str, Any]]:
    metadata = data.get("metadata", {})
    factoids = data.get("factoids", [])

    rows: List[Dict[str, Any]] = []

    for item in factoids:
        local_id = item.get("id")
        text = item.get("factoid_text", "").strip()

        if local_id is None or not text:
            continue

        file_name = metadata.get("file_name") or source_json_path.name
        factoid_id = build_factoid_id(file_name, local_id)

        # Keep all original metadata and extend it
        row_metadata = dict(metadata)
        row_metadata["factoid_id"] = factoid_id
        row_metadata["factoid_local_id"] = local_id
        row_metadata["factoid_text"] = text
        row_metadata["source_json_file"] = source_json_path.name
        row_metadata["source_json_stem"] = source_json_path.stem

        cleaned_metadata = clean_metadata(row_metadata)
        embedding_text = build_embedding_text(cleaned_metadata, text)

        rows.append(
            {
                "id": factoid_id,
                "document": text,
                "embedding_text": embedding_text,
                "metadata": cleaned_metadata,
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

        print(f"Upserted {min(i + BATCH_SIZE, total)}/{total}")


# ============================================================
# MAIN
# ============================================================

def main():
    if not INPUT_DIR.exists():
        raise RuntimeError(f"Input dir not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON files found in {INPUT_DIR}")

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    print(f"Processing {len(files)} file(s)")

    client = get_client()
    collection = get_collection()

    total_factoids = 0

    for file in files:
        print(f"\nProcessing: {file.name}")

        data = load_json(file)
        rows = prepare_rows(data, file)

        print(f"Factoids in file: {len(rows)}")

        if rows:
            upsert_rows(collection, client, rows)

        total_factoids += len(rows)

    print("\nDone.")
    print(f"Total factoids embedded: {total_factoids}")
    print(f"Chroma DB location: {OUTPUT_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()