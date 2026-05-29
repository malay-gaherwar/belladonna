#!/usr/bin/env python3
"""Update CTG Chroma metadata in-place without recomputing embeddings.

Use this after changing the metadata schema in ctg_embedding.prepare_rows()
when the embedding *vectors* are still semantically valid. Much faster than
re-running ctg_embedding.py because there are no embedding API calls — only
local sqlite metadata updates.

Caveat: the embedding text passed to the embedder concatenates metadata
fields. Strictly, changing how metadata is serialized into that text would
produce slightly different vectors. In practice the difference is tiny.
If you need exact vectors for the new metadata, re-run ctg_embedding.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

CTG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CTG_DIR))

import ctg_embedding as ce  # type: ignore  # noqa: E402
import chromadb  # noqa: E402

BATCH = 500


def main() -> int:
    if not ce.INPUT_DIR.exists():
        raise RuntimeError(f"Input dir not found: {ce.INPUT_DIR}")
    if not ce.OUTPUT_DIR.exists():
        raise RuntimeError(f"Chroma dir not found: {ce.OUTPUT_DIR}")

    files = sorted(ce.INPUT_DIR.glob("*.json"))
    print(f"Input factoid files: {len(files)}")

    chroma = chromadb.PersistentClient(path=str(ce.OUTPUT_DIR))
    collection = chroma.get_or_create_collection(
        name=ce.COLLECTION_NAME, embedding_function=None
    )
    print(f"Chroma starting count: {collection.count()}")

    print("Fetching existing chroma IDs...", flush=True)
    existing_ids = set(collection.get(include=[])["ids"])
    print(f"Existing IDs in chroma: {len(existing_ids)}", flush=True)

    batch_ids: list[str] = []
    batch_metas: list[dict] = []
    total_updated = 0
    skipped_no_match = 0
    files_processed = 0

    for fp in files:
        data = ce.load_json(fp)
        rows = ce.prepare_rows(data)
        for r in rows:
            if r["id"] not in existing_ids:
                skipped_no_match += 1
                continue
            batch_ids.append(r["id"])
            batch_metas.append(r["metadata"])

        files_processed += 1

        if len(batch_ids) >= BATCH:
            collection.update(ids=batch_ids, metadatas=batch_metas)
            total_updated += len(batch_ids)
            batch_ids.clear()
            batch_metas.clear()
            print(
                f"Updated {total_updated} rows "
                f"({files_processed}/{len(files)} files)",
                flush=True,
            )

    if batch_ids:
        collection.update(ids=batch_ids, metadatas=batch_metas)
        total_updated += len(batch_ids)

    print("\nDone.")
    print(f"Files processed:               {files_processed}")
    print(f"Rows updated:                  {total_updated}")
    print(f"Rows skipped (id not in chroma): {skipped_no_match}")
    print(f"Chroma final count:            {collection.count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
