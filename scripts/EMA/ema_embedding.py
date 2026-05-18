#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from openai import OpenAI


INPUT_DIR = Path("artifacts/EMA/factoids")
OUTPUT_DIR = Path("artifacts/EMA/embeddings")

MAX_FILES = None  # Keep as None for all files

MODEL_NAME = "Qwen3-Embedding-8B"
COLLECTION_NAME = "ema_factoids_qwen_embeddings"
BATCH_SIZE = 64

FACTOID_START = "<<<FACTOID>>>"
FACTOID_END = "<<<END_FACTOID>>>"


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
# HELPERS
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_whitespace(value)
    if isinstance(value, (list, tuple, set)):
        return normalize_whitespace(" ".join(clean_text(v) for v in value if v is not None))
    if isinstance(value, dict):
        return normalize_whitespace(" ".join(clean_text(v) for v in value.values() if v is not None))
    return normalize_whitespace(str(value))


def build_factoid_id(file_name: str, local_id: int) -> str:
    return f"{Path(file_name).stem}_{local_id}"


def safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def split_embedded_factoids(text: str) -> List[str]:
    """
    Handles malformed rows where one factoid_text accidentally contains
    multiple wrapped factoids, e.g.:
      text <<<END_FACTOID>>> <<<FACTOID>>> text2 ...
    """
    if not text:
        return []

    if FACTOID_START in text or FACTOID_END in text:
        matches = re.findall(
            re.escape(FACTOID_START) + r"(.*?)" + re.escape(FACTOID_END),
            text,
            flags=re.DOTALL,
        )
        cleaned = [normalize_whitespace(m) for m in matches if normalize_whitespace(m)]
        if cleaned:
            return cleaned

    return [normalize_whitespace(text)] if normalize_whitespace(text) else []


def get_record_lookup(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build a lookup from flattened factoid rows back to record-level outputs.

    Match priority:
      1. entity_name + active_substance + url
      2. entity_name + active_substance
      3. entity_name
    """
    lookup: Dict[str, Dict[str, Any]] = {}

    for record in data.get("record_level_outputs", []):
        if not isinstance(record, dict):
            continue

        entity_name = clean_text(record.get("entity_name"))
        active_substance = clean_text(record.get("active_substance"))
        url = clean_text(record.get("url"))

        keys = []

        if entity_name and active_substance and url:
            keys.append(f"{entity_name}|||{active_substance}|||{url}")
        if entity_name and active_substance:
            keys.append(f"{entity_name}|||{active_substance}")
        if entity_name:
            keys.append(entity_name)

        for key in keys:
            lookup.setdefault(key, record)

    return lookup


def find_matching_record(
    item: Dict[str, Any],
    record_lookup: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    entity_name = clean_text(item.get("entity_name"))
    active_substance = clean_text(item.get("active_substance"))
    url = clean_text(item.get("url"))

    candidate_keys = []

    if entity_name and active_substance and url:
        candidate_keys.append(f"{entity_name}|||{active_substance}|||{url}")
    if entity_name and active_substance:
        candidate_keys.append(f"{entity_name}|||{active_substance}")
    if entity_name:
        candidate_keys.append(entity_name)

    for key in candidate_keys:
        record = record_lookup.get(key)
        if record is not None:
            return record

    return None


def build_embedding_text(
    top_metadata: Dict[str, Any],
    factoid_item: Dict[str, Any],
    source_summary: Dict[str, Any],
) -> str:
    parts = [
        clean_text(top_metadata.get("source_family")),
        clean_text(top_metadata.get("document_title")),
        clean_text(top_metadata.get("document_type")),
        clean_text(top_metadata.get("document_year")),
        clean_text(factoid_item.get("entity_name")),
        clean_text(factoid_item.get("active_substance")),
        clean_text(factoid_item.get("status")),
        clean_text(factoid_item.get("procedure_number")),
        clean_text(factoid_item.get("ema_number")),
        clean_text(source_summary.get("therapeutic_indication")),
        clean_text(source_summary.get("therapeutic_area")),
        clean_text(source_summary.get("classification")),
        clean_text(source_summary.get("opinion_status")),
        clean_text(source_summary.get("holder")),
        clean_text(source_summary.get("regulatory_outcome")),
        clean_text(source_summary.get("first_published_date")),
        clean_text(source_summary.get("last_updated_date")),
        clean_text(source_summary.get("marketing_authorisation_date")),
        clean_text(source_summary.get("extra_context")),
        clean_text(factoid_item.get("factoid_text")),
    ]

    return " | ".join(p for p in parts if p not in ("", "None"))


# ============================================================
# DATA PREP
# ============================================================

def prepare_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    top_metadata = data.get("metadata", {})
    factoids = data.get("factoids", [])
    record_lookup = get_record_lookup(data)

    rows: List[Dict[str, Any]] = []
    expanded_counter = 0

    for item in factoids:
        if not isinstance(item, dict):
            continue

        fid = safe_int(item.get("id"))
        raw_text = clean_text(item.get("factoid_text"))

        if fid is None or not raw_text:
            continue

        split_texts = split_embedded_factoids(raw_text)
        matched_record = find_matching_record(item, record_lookup)
        source_summary = (
            matched_record.get("source_summary", {})
            if isinstance(matched_record, dict) and isinstance(matched_record.get("source_summary"), dict)
            else {}
        )

        for j, text in enumerate(split_texts, start=1):
            expanded_counter += 1

            if len(split_texts) == 1:
                factoid_id = build_factoid_id(top_metadata.get("file_name", "unknown"), fid)
                local_split_index = None
            else:
                factoid_id = build_factoid_id(
                    top_metadata.get("file_name", "unknown"),
                    int(f"{fid}{j}")
                )
                local_split_index = j

            factoid_payload = dict(item)
            factoid_payload["factoid_text"] = text

            embedding_text = build_embedding_text(
                top_metadata=top_metadata,
                factoid_item=factoid_payload,
                source_summary=source_summary,
            )

            row_metadata = {
                # -------- top-level file metadata --------
                "factoid_id": factoid_id,
                "source_family": clean_text(top_metadata.get("source_family")),
                "document_title": clean_text(top_metadata.get("document_title")),
                "document_type": clean_text(top_metadata.get("document_type")),
                "document_year": safe_int(top_metadata.get("document_year")),
                "file_name": clean_text(top_metadata.get("file_name")),

                # -------- flattened factoid-level metadata --------
                "original_factoid_id": fid,
                "split_factoid_index": local_split_index,
                "entity_name": clean_text(item.get("entity_name")),
                "active_substance": clean_text(item.get("active_substance")),
                "status": clean_text(item.get("status")),
                "procedure_number": clean_text(item.get("procedure_number")),
                "ema_number": clean_text(item.get("ema_number")),
                "url": clean_text(item.get("url")),

                # -------- record/source summary metadata --------
                "therapeutic_indication": clean_text(source_summary.get("therapeutic_indication")),
                "therapeutic_area": clean_text(source_summary.get("therapeutic_area")),
                "classification": clean_text(source_summary.get("classification")),
                "opinion_status": clean_text(source_summary.get("opinion_status")),
                "holder": clean_text(source_summary.get("holder")),
                "regulatory_outcome": clean_text(source_summary.get("regulatory_outcome")),
                "first_published_date": clean_text(source_summary.get("first_published_date")),
                "last_updated_date": clean_text(source_summary.get("last_updated_date")),
                "marketing_authorisation_date": clean_text(source_summary.get("marketing_authorisation_date")),
                "extra_context": clean_text(source_summary.get("extra_context")),
            }

            # Chroma metadata should not contain None values
            row_metadata = {
                k: v for k, v in row_metadata.items()
                if v not in (None, "")
            }

            rows.append(
                {
                    "id": factoid_id,
                    "document": text,
                    "embedding_text": embedding_text,
                    "metadata": row_metadata,
                }
            )

    return rows


# ============================================================
# UPSERT
# ============================================================

def upsert_rows(collection, client: OpenAI, rows: List[Dict[str, Any]]):
    total = len(rows)

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]

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
    print(f"Input dir: {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")

    client = get_client()
    collection = get_collection()

    total_factoids = 0

    for file in files:
        print(f"\nProcessing: {file.name}")

        data = load_json(file)
        rows = prepare_rows(data)

        print(f"Rows prepared from file: {len(rows)}")

        if not rows:
            print("No valid factoid rows found, skipping.")
            continue

        upsert_rows(collection, client, rows)
        total_factoids += len(rows)

    print("\nDone.")
    print(f"Total factoid rows embedded: {total_factoids}")
    print(f"Chroma DB location: {OUTPUT_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()