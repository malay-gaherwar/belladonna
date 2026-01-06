#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import chromadb
from openai import OpenAI


# ============================================================
# INLINE CONFIG (as requested, with fixed model name)
# ============================================================

CONFIG = {
    "API": {
        "BASE_URL": os.environ.get("BASE_URL"), 
        "API_KEY": os.environ.get("VIRTUAL_API_KEY"),
        "Model": "Qwen3-Embedding-8B",
    },
    "GENERATION": {
        "temperature": 1.0,
        "top_p": 0.9,
        "max_tokens": 7000,
    },
}


# ============================================================
# EMBEDDING + CHROMA HELPERS
# ============================================================

def get_embedding(client: OpenAI, text: str):
    """Get embedding for a query string using Qwen3-Embedding-8B."""
    response = client.embeddings.create(
        model=CONFIG["API"]["Model"],
        input=text,
    )
    return response.data[0].embedding


def load_chroma_collection(persist_dir: Path, collection_name: str):
    """Load existing Chroma collection with precomputed embeddings."""
    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=None,  # embeddings are stored, not computed by Chroma
    )
    return collection


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Query factoid Chroma DB using Qwen3-Embedding-8B."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="which mutations are common in Asian patients",
        help="Natural language query.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("artifacts/chroma_qwen"),
        help="Directory where Chroma DB is stored.",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        default="factoids_qwen_embeddings",
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest factoids to retrieve.",
    )

    args = parser.parse_args()

    if not CONFIG["API"]["API_KEY"]:
        raise RuntimeError(
            "VIRTUAL_API_KEY environment variable is not set. "
            "Export it before running this script."
        )

    # 1. Init OpenAI-compatible client
    client = OpenAI(
        api_key=CONFIG["API"]["API_KEY"],
        base_url=CONFIG["API"]["BASE_URL"],
    )

    # 2. Get query embedding
    query_emb = get_embedding(client, args.query)

    # 3. Load Chroma collection
    collection = load_chroma_collection(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    # 4. Query Chroma using the embedding
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=args.top_k,
    )

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    print(f"\nQuery: {args.query}")
    print(f"Top {len(docs)} matching factoids:\n")

    for rank, (fid, doc, meta) in enumerate(zip(ids, docs, metas), start=1):
        print(f"=== Result {rank} ===")
        print(f"factoid_id: {meta.get('factoid_id', fid)}")
        print(f"pmid      : {meta.get('pmid')}")
        print(f"doi       : {meta.get('doi')}")
        print(f"index     : {meta.get('index')}")
        print(f"text      : {doc}")
        print()

if __name__ == "__main__":
    main()
