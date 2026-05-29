#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from openai import OpenAI

# ============================================================
# PATHS (CTG)
# ============================================================

INPUT_DIR = Path("artifacts/CTG/factoids")
OUTPUT_DIR = Path("artifacts/CTG/embeddings")

MAX_FILES = None  # None = all files

MODEL_NAME = "Qwen3-Embedding-8B"
COLLECTION_NAME = "ctg_factoids_qwen_embeddings"
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


def build_factoid_id(file_name: str, local_id: int) -> str:
    # stable id: <source file stem>_<factoid local id>
    return f"{Path(file_name).stem}_{local_id}"


def flatten_metadata(d: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flatten nested dicts to satisfy Chroma's scalar-metadata constraint.
    Nested dicts get keys joined with '_'; lists/other non-scalars are
    JSON-stringified; None values are dropped.
    """
    flat: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}_{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_metadata(v, new_key))
        elif v is None:
            continue
        elif isinstance(v, (str, int, float, bool)):
            flat[new_key] = v
        else:
            flat[new_key] = json.dumps(v, ensure_ascii=False)
    return flat


def prepare_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    CTG factoids file shape (example):

    {
      "metadata": {
        "source_family": "Clinical Trials",
        "document_title": "...",
        "document_type": "Clinical Trials",
        "document_year": 2003,
        "file_name": "0000007_NCT00005886.json",
        "license_info": {"copyright": "...", "commercial_use": "yes", ...}
      },
      "factoids": [{"id": 1, "factoid_text": "..."}, ...]
    }
    """
    metadata = data.get("metadata", {}) or {}
    factoids = data.get("factoids", []) or []

    rows: List[Dict[str, Any]] = []

    # Flatten once per file: nested dicts (license_info, …) become keys
    # like license_info_copyright. Required so Chroma accepts the metadata.
    flat_meta = flatten_metadata(metadata)

    for item in factoids:
        fid = item.get("id")
        text = (item.get("factoid_text") or "").strip()
        if not fid or not text:
            continue

        factoid_id = build_factoid_id(metadata.get("file_name", "unknown"), int(fid))

        # Embedding text: include ALL metadata fields (as key=value) + factoid text.
        meta_kv = []
        for k in sorted(flat_meta.keys()):
            v = flat_meta[k]
            if v in (None, "", "None"):
                continue
            meta_kv.append(f"{k}={v}")

        embedding_text = " | ".join(meta_kv + [text])

        chroma_meta = {
            **flat_meta,
            "factoid_id": factoid_id,
            "local_factoid_id": int(fid),
        }

        rows.append(
            {
                "id": factoid_id,
                "document": text,
                "embedding_text": embedding_text,
                "metadata": chroma_meta,
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

    files = sorted(INPUT_DIR.glob("*.json"))
    if not files:
        raise RuntimeError(f"No JSON files found in {INPUT_DIR}")

    if MAX_FILES is not None:
        files = files[:MAX_FILES]

    print(f"Processing {len(files)} file(s)")
    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {MODEL_NAME}")

    client = get_client()
    collection = get_collection()

    total_factoids = 0

    for file in files:
        print(f"\nProcessing: {file.name}")
        data = load_json(file)
        rows = prepare_rows(data)

        print(f"Factoids in file: {len(rows)}")
        if not rows:
            continue

        upsert_rows(collection, client, rows)
        total_factoids += len(rows)

    print("\nDone.")
    print(f"Total factoids embedded: {total_factoids}")
    print(f"Chroma DB location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()