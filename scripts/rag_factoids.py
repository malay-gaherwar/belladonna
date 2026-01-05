#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import chromadb
from openai import OpenAI


# ============================================================
# INLINE CONFIG (self-contained, as requested)
# ============================================================

CONFIG = {
    "API": {
        "BASE_URL": "http://192.168.33.27/v1/",
        "API_KEY": os.environ.get("VIRTUAL_API_KEY"),
        "EMBEDDING_MODEL": "Qwen3-Embedding-8B",
        "GENERATION_MODEL": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    },
    "GENERATION": {
        "temperature": 0.2,   # low temp for factual RAG
        "top_p": 0.9,
        "max_tokens": 1024,
    },
}


# ============================================================
# HELPERS
# ============================================================

def get_embedding(client: OpenAI, text: str):
    """Embed text using Qwen3."""
    response = client.embeddings.create(
        model=CONFIG["API"]["EMBEDDING_MODEL"],
        input=text,
    )
    return response.data[0].embedding


def load_chroma_collection(persist_dir: Path, collection_name: str):
    """Load existing Chroma collection (no embedding function)."""
    chroma_client = chromadb.PersistentClient(path=str(persist_dir))
    return chroma_client.get_collection(
        name=collection_name,
        embedding_function=None,
    )


def format_context(documents, metadatas):
    """Format retrieved factoids into a clean RAG context block."""
    chunks = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        src = []
        if meta.get("pmid"):
            src.append(meta["pmid"])
        if meta.get("doi"):
            src.append(meta["doi"])

        src_str = "; ".join(src) if src else "unknown source"

        chunks.append(
            f"[Factoid {i} | {src_str}]\n{doc}"
        )
    return "\n\n".join(chunks)


# ============================================================
# MAIN RAG PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RAG over factoid Chroma DB")

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="User question (natural language).",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("artifacts/chroma_qwen"),
        help="Chroma persistence directory.",
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
        help="Number of factoids to retrieve.",
    )

    args = parser.parse_args()

    if not CONFIG["API"]["API_KEY"]:
        raise RuntimeError(
            "VIRTUAL_API_KEY environment variable is not set."
        )

    # 1. Init OpenAI-compatible client
    client = OpenAI(
        api_key=CONFIG["API"]["API_KEY"],
        base_url=CONFIG["API"]["BASE_URL"],
    )

    # 2. Embed query
    query_embedding = get_embedding(client, args.query)

    # 3. Load Chroma + retrieve
    collection = load_chroma_collection(
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=args.top_k,
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        print("No relevant factoids found.")
        return

    # 4. Build RAG context
    context = format_context(documents, metadatas)

    # 5. RAG prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are a biomedical expert assistant. "
                "Answer the question using ONLY the provided factoids. "
                "If the answer is not contained in the factoids, say so explicitly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{args.query}\n\n"
                f"Factoids:\n{context}\n\n"
                "Answer concisely and cite evidence from the factoids."
            ),
        },
    ]

    # 6. Generate answer
    response = client.chat.completions.create(
        model=CONFIG["API"]["GENERATION_MODEL"],
        messages=messages,
        temperature=CONFIG["GENERATION"]["temperature"],
        top_p=CONFIG["GENERATION"]["top_p"],
        max_tokens=CONFIG["GENERATION"]["max_tokens"],
    )

    answer = response.choices[0].message.content

    # 7. Output
    print("\n=== RAG ANSWER ===\n")
    print(answer)
    print("\n=== RETRIEVED FACTOIDS ===\n")
    print(context)


if __name__ == "__main__":
    main()
