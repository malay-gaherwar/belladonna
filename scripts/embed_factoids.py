#!/usr/bin/env python3
import json
from pathlib import Path
import argparse

import chromadb
from openai import OpenAI
import os


# ============================================================
# INLINE CONFIG
# ============================================================

CONFIG = {
    "API": {
        "BASE_URL": os.environ.get("BASE_URL"),
        "API_KEY": os.environ.get("VIRTUAL_API_KEY"),
        "Model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    },
    "GENERATION": {
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 7000,
    },
}


# ============================================================
# LOAD FACTOIDS
# ============================================================

def load_factoids(jsonl_path: Path):
    """Load deduplicated factoids JSONL file."""
    out = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


# ============================================================
# EMBEDDING FUNCTION
# ============================================================

def get_embedding(client: OpenAI, model_name: str, text: str):
    """
    Query your local embedding model (Qwen3-Embedding-8B)
    via the OpenAI-compatible /embeddings endpoint.
    """
    response = client.embeddings.create(
        model=model_name,
        input=text,
    )
    return response.data[0].embedding


# ============================================================
# CHROMA SETUP
# ============================================================

def build_chroma_collection(persist_dir: Path, collection_name: str):
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))

    # Remove old collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    # No embedding_function — we pass in our own vectors
    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Qwen embeddings for factoids"},
        embedding_function=None,
    )
    return collection


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Embed factoids using Qwen3-Embedding-8B and store in Chroma.")

    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("artifacts/factoids_dedup.jsonl"),
        help="Path to deduplicated factoids JSONL.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("artifacts/chroma_qwen"),
        help="Directory for Chroma DB.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="factoids_qwen_embeddings",
        help="Name of the Chroma collection.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen3-Embedding-8B",
        help="Embedding model name.",
    )

    args = parser.parse_args()

    # Load factoids
    factoids = load_factoids(args.input_jsonl)
    print(f"Loaded {len(factoids)} factoids from {args.input_jsonl}")

    # Prepare Chroma
    collection = build_chroma_collection(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    # OpenAI-compatible local client
    client = OpenAI(
        api_key=CONFIG["API"]["API_KEY"],
        base_url=CONFIG["API"]["BASE_URL"],
    )

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    # Embed & collect
    for f in factoids:
        fid = str(f["factoid_id"])
        text = f["text"]

        emb = get_embedding(client, args.embedding_model, text)

        ids.append(fid)
        documents.append(text)
        metadatas.append({
            "factoid_id": f["factoid_id"],
            "pmid": f.get("pmid"),
            "doi": f.get("doi"),
            "index": f.get("index"),
        })
        embeddings.append(emb)

    # Store in Chroma
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print(f"Stored {len(ids)} embedded factoids in Chroma at {args.persist_dir}")


if __name__ == "__main__":
    main()
