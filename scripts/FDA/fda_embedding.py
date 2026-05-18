#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

def resolve_artifact_root() -> Path:
    configured = os.getenv("FDA_ARTIFACT_ROOT")
    if configured:
        return Path(configured)

    lowercase = Path("artifacts/fda")
    uppercase = Path("artifacts/FDA")
    if lowercase.exists():
        return lowercase
    if uppercase.exists():
        return uppercase
    return lowercase


ARTIFACT_ROOT = resolve_artifact_root()
INPUT_FACTOIDS_JSON = ARTIFACT_ROOT / "factoids" / "fda_factoids.json"
OUTPUT_DIR = ARTIFACT_ROOT / "embeddings"

MODEL_NAME = "Qwen3-Embedding-8B"
COLLECTION_NAME = "fda_factoid_qwen_embeddings"
BATCH_SIZE = 64
MAX_EMBED_CHARS = 12000


# ============================================================
# CLIENT
# ============================================================

def get_client() -> OpenAI:
    from openai import OpenAI

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
    import chromadb

    chroma_client = chromadb.PersistentClient(path=str(OUTPUT_DIR))
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_factoid_payload(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise RuntimeError(f"Factoid input must be a JSON object: {path}")

    factoids = payload.get("factoids")
    if not isinstance(factoids, list):
        raise RuntimeError(f"Factoid input missing list field 'factoids': {path}")

    return payload


# ============================================================
# HELPERS
# ============================================================

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def clean_metadata_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return " | ".join(str(x) for x in value if x not in (None, "", "None"))
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def first_nonempty(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, list):
            for x in value:
                if isinstance(x, str) and x.strip():
                    return x.strip()
    return ""


def safe_slug(value: str) -> str:
    value = (value or "").lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def build_factoid_record_id(factoid: Dict[str, Any], index: int) -> str:
    generic_name = first_nonempty(factoid.get("generic_name"), "unknown")
    label_id = first_nonempty(
        factoid.get("source_label_set_id"),
        factoid.get("spl_set_id"),
        factoid.get("source_label_id"),
        factoid.get("file_name"),
        f"row_{index}",
    )
    factoid_id = first_nonempty(factoid.get("id"), index)

    return (
        f"{safe_slug(generic_name)}_"
        f"{safe_slug(str(label_id))}_"
        f"factoid_{safe_slug(str(factoid_id))}"
    )


def source_context_parts(factoid: Dict[str, Any]) -> List[str]:
    return [
        first_nonempty(factoid.get("generic_name")),
        first_nonempty(factoid.get("brand_name")),
        first_nonempty(factoid.get("drug_class")),
        first_nonempty(factoid.get("fda_bc_indication")),
        first_nonempty(factoid.get("ema_bc_indication")),
        first_nonempty(factoid.get("fda_ema_status")),
        first_nonempty(factoid.get("use_type")),
        first_nonempty(factoid.get("bc_notes")),
        first_nonempty(factoid.get("label_date")),
        first_nonempty(factoid.get("application_number")),
        first_nonempty(factoid.get("manufacturer_name")),
    ]


def build_embedding_text(factoid: Dict[str, Any]) -> str:
    factoid_text = normalize_ws(str(factoid.get("factoid_text") or ""))
    prefix = " | ".join(x for x in source_context_parts(factoid) if x)
    full = f"{prefix}\n\nFactoid: {factoid_text}".strip()

    if len(full) > MAX_EMBED_CHARS:
        full = full[:MAX_EMBED_CHARS]

    return full


def build_document_text(factoid: Dict[str, Any]) -> str:
    lines = [
        f"Generic name: {first_nonempty(factoid.get('generic_name'))}",
        f"Brand name: {first_nonempty(factoid.get('brand_name'))}",
        f"Drug class: {first_nonempty(factoid.get('drug_class'))}",
        f"FDA BC indication: {first_nonempty(factoid.get('fda_bc_indication'))}",
        f"EMA BC indication: {first_nonempty(factoid.get('ema_bc_indication'))}",
        f"FDA EMA status: {first_nonempty(factoid.get('fda_ema_status'))}",
        f"Use type: {first_nonempty(factoid.get('use_type'))}",
        f"BC notes: {first_nonempty(factoid.get('bc_notes'))}",
        f"Label date: {first_nonempty(factoid.get('label_date'))}",
        f"Application number: {first_nonempty(factoid.get('application_number'))}",
        f"Manufacturer name: {first_nonempty(factoid.get('manufacturer_name'))}",
        "",
        f"Factoid: {normalize_ws(str(factoid.get('factoid_text') or ''))}",
    ]
    return "\n".join(lines).strip()


def build_metadata(
    factoid: Dict[str, Any],
    record_id: str,
    index: int,
    payload_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    meta = {
        "record_id": record_id,
        "factoid_id": clean_metadata_value(factoid.get("id")),
        "factoid_index": index,
        "source_family": "FDA",
        "source_kind": "factoid",
        "source_factoid_file": INPUT_FACTOIDS_JSON.name,
        "factoid_model_name": clean_metadata_value(payload_metadata.get("model_name")),
        "classification_date": clean_metadata_value(payload_metadata.get("classification_date")),
        "file_name": clean_metadata_value(factoid.get("file_name")),
        "generic_name": clean_metadata_value(factoid.get("generic_name")),
        "brand_name": clean_metadata_value(factoid.get("brand_name")),
        "drug_class": clean_metadata_value(factoid.get("drug_class")),
        "application_number": clean_metadata_value(factoid.get("application_number")),
        "manufacturer_name": clean_metadata_value(factoid.get("manufacturer_name")),
        "label_date": clean_metadata_value(factoid.get("label_date")),
        "source_label_set_id": clean_metadata_value(factoid.get("source_label_set_id")),
        "source_label_id": clean_metadata_value(factoid.get("source_label_id")),
        "spl_id": clean_metadata_value(factoid.get("spl_id")),
        "spl_set_id": clean_metadata_value(factoid.get("spl_set_id")),
        "fda_bc_indication": clean_metadata_value(factoid.get("fda_bc_indication")),
        "ema_bc_indication": clean_metadata_value(factoid.get("ema_bc_indication")),
        "fda_ema_status": clean_metadata_value(factoid.get("fda_ema_status")),
        "use_type": clean_metadata_value(factoid.get("use_type")),
        "bc_notes": clean_metadata_value(factoid.get("bc_notes")),
        "metadata_link_status": clean_metadata_value(factoid.get("metadata_link_status")),
        "metadata_link_method": clean_metadata_value(factoid.get("metadata_link_method")),
        "metadata_match_candidates": clean_metadata_value(factoid.get("metadata_match_candidates")),
    }

    return {k: v for k, v in meta.items() if v is not None}


def prepare_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    payload_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    seen_ids: set[str] = set()

    for i, factoid in enumerate(payload.get("factoids", []), start=1):
        if not isinstance(factoid, dict):
            continue

        factoid_text = normalize_ws(str(factoid.get("factoid_text") or ""))
        if not factoid_text:
            continue

        record_id = build_factoid_record_id(factoid, i)
        if record_id in seen_ids:
            record_id = f"{record_id}_{i}"
        seen_ids.add(record_id)

        embedding_text = build_embedding_text(factoid)
        if not embedding_text.strip():
            continue

        rows.append(
            {
                "id": record_id,
                "document": build_document_text(factoid),
                "embedding_text": embedding_text,
                "metadata": build_metadata(
                    factoid=factoid,
                    record_id=record_id,
                    index=i,
                    payload_metadata=payload_metadata,
                ),
            }
        )

    return rows


# ============================================================
# UPSERT
# ============================================================

def upsert_rows(collection, client: OpenAI, rows: List[Dict[str, Any]]) -> None:
    total = len(rows)

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]

        texts = [r["embedding_text"] for r in batch]
        embeddings = get_embeddings_batch(client, texts)

        collection.upsert(
            ids=[r["id"] for r in batch],
            documents=[r["document"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
            embeddings=embeddings,
        )

        print(f"Upserted {min(i + BATCH_SIZE, total)}/{total}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_FACTOIDS_JSON.exists():
        raise RuntimeError(
            f"Factoid input not found: {INPUT_FACTOIDS_JSON}. "
            "Run scripts/FDA/factoids_fda.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading factoids: {INPUT_FACTOIDS_JSON}")
    payload = load_factoid_payload(INPUT_FACTOIDS_JSON)

    payload_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    if (
        "input_jsonl" not in payload_metadata
        and os.getenv("ALLOW_NON_SEEDED_FDA_FACTOIDS") != "1"
    ):
        raise RuntimeError(
            "Factoid file does not look like the seeded FDA factoid output. "
            "Run scripts/FDA/factoids_fda.py to regenerate curated factoids before embedding. "
            "Set ALLOW_NON_SEEDED_FDA_FACTOIDS=1 only if you intentionally want to embed "
            "legacy broad FDA factoids."
        )

    factoid_count = len(payload.get("factoids", []))
    print(f"Factoids loaded: {factoid_count}")

    client = get_client()
    collection = get_collection()

    rows = prepare_rows(payload)
    if not rows:
        raise RuntimeError("No factoid rows prepared for embeddings")

    print(f"Rows prepared for embeddings: {len(rows)}")
    upsert_rows(collection, client, rows)

    manifest = {
        "input_factoids_json": str(INPUT_FACTOIDS_JSON),
        "output_dir": str(OUTPUT_DIR),
        "collection_name": COLLECTION_NAME,
        "embedding_model": MODEL_NAME,
        "factoids_loaded": factoid_count,
        "rows_embedded": len(rows),
    }
    (OUTPUT_DIR / "fda_factoid_embedding_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Total factoids embedded: {len(rows)}")
    print(f"Chroma DB location: {OUTPUT_DIR}")
    print(f"Collection name: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
