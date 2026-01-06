# scripts/dedupe_factoids.py

#!/usr/bin/env python

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from factoids_utils import FactoidRecord, load_all_factoids


def build_chroma_collection(
    records: List[FactoidRecord],
    persist_dir: Path,
    collection_name: str = "factoids",
):
    """
    Build (or rebuild) a Chroma collection for factoids.
    """
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))

    # For reproducibility while developing: drop old collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "Factoid-level collection for dedup and search"},
    )

    # Add in reasonably sized batches so we don't blow anything up
    batch_size = 256
    for start in range(0, len(records), batch_size):
        end = start + batch_size
        batch = records[start:end]
        collection.add(
            ids=[str(r.factoid_id) for r in batch],
            documents=[r.text for r in batch],
            metadatas=[
                {
                    "pmid": r.pmid,
                    "index": r.index,
                    "doi": r.doi,
                    # no "source" here anymore
                }
                for r in batch
            ],
        )

    return collection


def dedupe_factoids(
    records: List[FactoidRecord],
    collection,
    similarity_threshold: float = 0.9,
    top_k: int = 5,
):
    """
    Use Chroma similarity search to find near duplicates.

    - For each factoid in original order:
      - If it's already known as a duplicate, skip.
      - Otherwise treat it as 'canonical' and find other factoids with
        cosine similarity >= similarity_threshold.
    """
    # Chroma returns *distance*; default distance is cosine distance ~ (1 - cosine_sim)
    max_distance = 1.0 - similarity_threshold

    # Key everything by string factoid_id, because Chroma ids are strings
    id_to_record: Dict[str, FactoidRecord] = {
        str(r.factoid_id): r for r in records
    }
    canonical_for: Dict[str, str] = {}  # factoid_id(str) -> canonical_factoid_id(str)

    for r in records:
        fid = str(r.factoid_id)
        if fid in canonical_for:
            # already assigned as duplicate of an earlier canonical
            continue

        # r becomes canonical for its cluster
        canonical_for[fid] = fid

        res = collection.query(
            query_texts=[r.text],
            n_results=top_k + 1,  # include itself
        )

        # Chroma returns lists-of-lists
        neighbour_ids = res["ids"][0]
        distances = res["distances"][0]

        for nid, dist in zip(neighbour_ids, distances):
            # All ids from Chroma are strings
            if nid == fid:
                continue
            if dist is None:
                continue
            if dist > max_distance:
                continue
            if nid in canonical_for:
                # already attached to someone else, don't move it
                continue
            canonical_for[nid] = fid

    # Group by canonical id
    groups = defaultdict(list)
    for fid in id_to_record.keys():
        root = canonical_for.get(fid, fid)
        groups[root].append(fid)

    # Build deduplicated record list: keep one per group (the canonical)
    dedup_records: List[FactoidRecord] = []
    for canonical_id, fids in groups.items():
        # we keep the canonical record
        dedup_records.append(id_to_record[canonical_id])

    return canonical_for, groups, dedup_records


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate factoids using Chroma + embeddings."
    )
    parser.add_argument(
        "--factoid-dir",
        type=Path,
        default=Path("artifacts/factoids"),
        help="Directory containing per-paper factoid JSON files.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("artifacts/chroma_factoids"),
        help="Directory for Chroma persistent storage.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.9,
        help="Cosine similarity threshold for near-duplicates (0–1).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of neighbours to inspect per factoid.",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path("artifacts/factoids_dedup.jsonl"),
        help="Where to write the deduplicated factoids (JSONL).",
    )
    parser.add_argument(
        "--duplicates-jsonl",
        type=Path,
        default=Path("artifacts/factoid_duplicates.jsonl"),
        help="Where to write all duplicate clusters (JSONL).",
    )

    args = parser.parse_args()

    records = load_all_factoids(args.factoid_dir)
    total_before = len(records)
    print(f"Loaded {total_before} factoids from {args.factoid_dir}")

    if not records:
        print("No factoids found, nothing to dedupe.")
        return

    collection = build_chroma_collection(
        records,
        persist_dir=args.persist_dir,
        collection_name="factoids",
    )

    canonical_for, groups, dedup_records = dedupe_factoids(
        records,
        collection,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
    )

    total_after = len(dedup_records)

    print(f"Total factoids BEFORE dedup: {total_before}")
    print(f"Total factoids AFTER  dedup: {total_after}")
    print(
        f"Reduction: {total_before - total_after} factoids "
        f"({(1 - total_after / total_before) * 100:.2f}% fewer)"
    )

    # --- PRINT ALL DUPLICATES SO YOU CAN SEE WHAT WAS REMOVED ---
    multi_groups = [g for g in groups.values() if len(g) > 1]
    print(f"Found {len(multi_groups)} duplicate clusters (size >= 2).")

    if multi_groups:
        print("\n=== FULL LIST OF DUPLICATE CLUSTERS ===")
        for cluster_idx, fids in enumerate(multi_groups, start=1):
            canonical_id = fids[0]
            canonical_record = next(
                x for x in records if str(x.factoid_id) == canonical_id
            )
            print(f"\n[Cluster {cluster_idx}] Canonical factoid_id={canonical_record.factoid_id}")
            print(f"  CANONICAL: {canonical_record.text}")
            for fid in fids[1:]:
                r = next(x for x in records if str(x.factoid_id) == fid)
                print(f"  DUPLICATE (factoid_id={r.factoid_id}): {r.text}")

    # --- ALSO SAVE DUPLICATE CLUSTERS TO JSONL FOR LATER INSPECTION ---
    args.duplicates_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.duplicates_jsonl.open("w") as dup_f:
        for fids in multi_groups:
            canonical_id = fids[0]
            canonical_record = next(
                x for x in records if str(x.factoid_id) == canonical_id
            )
            cluster_obj = {
                "canonical_factoid_id": canonical_record.factoid_id,
                "canonical_text": canonical_record.text,
                "duplicates": [],
            }
            for fid in fids[1:]:
                r = next(x for x in records if str(x.factoid_id) == fid)
                cluster_obj["duplicates"].append(
                    {
                        "factoid_id": r.factoid_id,
                        "pmid": r.pmid,
                        "doi": r.doi,
                        "index": r.index,
                        "text": r.text,
                    }
                )
            dup_f.write(json.dumps(cluster_obj) + "\n")

    print(f"\nWrote duplicate clusters to {args.duplicates_jsonl}")

    # --- Write deduplicated factoids to JSONL ---
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for r in dedup_records:
            f.write(
                json.dumps(
                    {
                        "factoid_id": r.factoid_id,  # numeric global id
                        "pmid": r.pmid,
                        "doi": r.doi,
                        "pmcid": r.pmcid,
                        "title": r.title,
                        "journal": r.journal,
                        "year": r.year,
                        "index": r.index,
                        "text": r.text,
                    }
                )
                + "\n"
            )

    print(f"Wrote deduplicated factoids to {args.out_jsonl}")


if __name__ == "__main__":
    main()
