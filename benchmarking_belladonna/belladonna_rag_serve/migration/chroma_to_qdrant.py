"""One-time migration: Chroma (4096-dim Qwen3 vectors) -> Qdrant (1024-dim, INT8 quantized).

Why this script bypasses chromadb entirely
------------------------------------------
The chromadb Rust loader allocates a ~1.5x growth buffer for the HNSW index
on read; for EPMC (22 GB index) and Elsevier (25 GB index) it tries to mmap
a ~34 GB anonymous block and the kernel refuses on memory-constrained boxes.
That makes the embeddings *unreachable through chromadb itself*.

But the raw float32 vectors are sitting in plain `data_level0.bin` files in
the standard hnswlib on-disk layout. We mmap those directly, look up each
vector's position via `index_metadata.pickle` (chroma's id-to-hnsw-label
map), and ingest into Qdrant. No need to ever load the broken index.

Per-element layout in data_level0.bin (hnswlib level-0):
  [4 B link-list size][M_max*4 B neighbor ids][dim*4 B float32 vector][8 B label]
For chroma at dim=4096, M=16 -> M_max=32 -> stride = 4 + 128 + 16384 + 8 = 16524.
Vector starts at offset 132 within each element.

Resumability
------------
Point IDs in Qdrant are UUID5(namespace, chroma_embedding_id), so re-running
upserts the same point rather than duplicating it. The script can be killed
and restarted; finished batches stay finished.

CLI:
    python migration/chroma_to_qdrant.py --source AGO --dry-run
    python migration/chroma_to_qdrant.py --source AGO
    python migration/chroma_to_qdrant.py --source EPMC --batch 256
    python migration/chroma_to_qdrant.py --all
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import pickle
import sqlite3
import struct
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_ROOT = Path("/home/malay/Documents/belladonna_v1_2")

# One Chroma collection per source.
SOURCE_COLLECTION_NAMES = {
    "AGO":      "ago_factoids_qwen_embeddings",
    "ASCO":     "asco_factoids_qwen_embeddings",
    "CTG":      "ctg_factoids_qwen_embeddings",
    "Elsevier": "elsevier_factoids_qwen_embeddings",
    "EMA":      "ema_factoids_qwen_embeddings",
    "EPMC":     "epmc_factoids_qwen_embeddings",
    "ESMO":     "esmo_factoids_qwen_embeddings",
    "FDA":      "fda_factoid_qwen_embeddings",
}

# Source HNSW vectors -> truncated target.
ORIG_DIM   = 4096
TARGET_DIM = 1024

# hnswlib level-0 element layout for M=16 (Chroma default).
# Recomputed at runtime per source from header.bin to stay honest.
DEFAULT_M_MAX        = 32
ELEMENT_STRIDE_BYTES = 4 + DEFAULT_M_MAX * 4 + ORIG_DIM * 4 + 8   # 16524
VECTOR_OFFSET_BYTES  = 4 + DEFAULT_M_MAX * 4                      # 132
VECTOR_NBYTES        = ORIG_DIM * 4                               # 16384

# Stable UUID namespace per source so we can rebuild idempotently.
NAMESPACE_BELLADONNA = uuid.UUID("8d2f8b1a-2b3c-4f5e-9d6a-7f8a9b0c1d2e")


# ============================================================
# CHROMA SQLITE READERS
# ============================================================

def open_chroma_sqlite(source: str) -> sqlite3.Connection:
    """Open the source's chroma.sqlite3 read-only via URI."""
    path = CHROMA_ROOT / source / "embeddings" / "chroma.sqlite3"
    if not path.exists():
        raise FileNotFoundError(f"Chroma db missing: {path}")
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def hnsw_segment_dir(source: str, con: sqlite3.Connection) -> Path:
    """The uuid-named directory that contains the persisted HNSW files."""
    row = con.execute(
        "SELECT id FROM segments WHERE type LIKE '%hnsw%' LIMIT 1"
    ).fetchone()
    if not row:
        raise RuntimeError(f"No hnsw segment registered in {source}")
    seg_id = row[0]
    d = CHROMA_ROOT / source / "embeddings" / seg_id
    if not d.exists():
        raise RuntimeError(f"Segment dir not found: {d}")
    return d


def load_label_to_id(seg_dir: Path) -> Dict[int, str]:
    """Pickle's label_to_id maps the 8-byte label stored at the END of each
    on-disk element to the chroma embedding_id (string). We iterate
    data_level0.bin linearly by position, read each element's label, and
    look up the chroma id here. Labels are NOT positions — chroma may
    reorder elements during compaction, so trusting position == label
    silently corrupts the migration (the vector lookup will yield somebody
    else's embedding)."""
    pkl = seg_dir / "index_metadata.pickle"
    with open(pkl, "rb") as fh:
        obj = pickle.load(fh)
    m = obj.get("label_to_id")
    if not isinstance(m, dict):
        raise RuntimeError(f"Unexpected pickle layout: keys={list(obj.keys())}")
    return {int(k): str(v) for k, v in m.items()}


def read_hnsw_header(seg_dir: Path) -> Dict[str, Any]:
    """We use it for the element stride sanity-check. The Rust loader's
    header layout isn't perfectly documented, so we derive stride from
    file_size / n_elements instead of trusting fields. The header read here
    is only used to surface size mismatches early."""
    hdr = seg_dir / "header.bin"
    size = hdr.stat().st_size
    with open(hdr, "rb") as fh:
        data = fh.read()
    return {"size_bytes": size, "raw": data}


def compute_element_stride(seg_dir: Path, n_persisted: int) -> int:
    """Derive bytes-per-element from data_level0.bin / persisted count.
    Chroma's HNSW writes pad to max_elements (capacity), not n_persisted,
    so the file is usually a clean multiple of stride * capacity. We check
    both common cases (M_max=32 / 16) and trust the standard layout."""
    p = seg_dir / "data_level0.bin"
    file_size = p.stat().st_size

    # Standard hnswlib-style: stride = 4 (link count) + M_max*4 + dim*4 + 8 (label).
    candidates = []
    for m_max in (32, 16, 64):
        stride = 4 + m_max * 4 + ORIG_DIM * 4 + 8
        if file_size % stride == 0:
            capacity = file_size // stride
            if capacity >= n_persisted:
                candidates.append((m_max, stride, capacity))

    if not candidates:
        raise RuntimeError(
            f"Couldn't derive stride: file_size={file_size}, persisted={n_persisted}"
        )
    # Prefer the smallest capacity (= tightest fit = correct M_max).
    candidates.sort(key=lambda t: t[2])
    m_max, stride, capacity = candidates[0]
    return stride


def open_data_level0(seg_dir: Path) -> Tuple[mmap.mmap, int]:
    """Memory-map data_level0.bin read-only."""
    p = seg_dir / "data_level0.bin"
    fd = os.open(p, os.O_RDONLY)
    mm = mmap.mmap(fd, 0, prot=mmap.PROT_READ)
    os.close(fd)
    return mm, p.stat().st_size


def _row_to_value(sv: Any, iv: Any, fv: Any, bv: Any) -> Any:
    if sv is not None:
        return sv
    if iv is not None:
        return iv
    if fv is not None:
        return fv
    if bv is not None:
        return bool(bv)
    return None


def fetch_metadata_by_id(con: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    """Build a full {sqlite_int_id: {key: value}} dict in one pass.

    Memory cost: ~100 bytes × n_metadata_rows. Fine for sources up to ~25 M
    rows (EPMC). Becomes ~4.5 GB and OOM-risk for Elsevier (44.6 M rows).
    For big sources, prefer `fetch_metadata_for_ids` instead."""
    out: Dict[int, Dict[str, Any]] = defaultdict(dict)
    cur = con.execute(
        "SELECT id, key, string_value, int_value, float_value, bool_value "
        "FROM embedding_metadata"
    )
    for eid, key, sv, iv, fv, bv in cur:
        out[eid][key] = _row_to_value(sv, iv, fv, bv)
    return out


def fetch_metadata_for_ids(
    con: sqlite3.Connection, int_ids: List[int]
) -> Dict[int, Dict[str, Any]]:
    """Chunked variant: fetch metadata for at most a few thousand embeddings
    at a time. Bounded memory regardless of total corpus size."""
    if not int_ids:
        return {}
    out: Dict[int, Dict[str, Any]] = defaultdict(dict)
    placeholders = ",".join("?" * len(int_ids))
    cur = con.execute(
        f"SELECT id, key, string_value, int_value, float_value, bool_value "
        f"FROM embedding_metadata WHERE id IN ({placeholders})",
        int_ids,
    )
    for eid, key, sv, iv, fv, bv in cur:
        out[eid][key] = _row_to_value(sv, iv, fv, bv)
    return out


# Sources whose embedding_metadata table is too large to slurp into a single
# Python dict. We stream those in chunks instead.
LARGE_METADATA_SOURCES = {"EPMC", "Elsevier"}
METADATA_CHUNK_SIZE = 2000


def fetch_embedding_rows(con: sqlite3.Connection) -> List[Tuple[int, str]]:
    """(int id, chroma_embedding_id) per persisted vector."""
    return list(
        con.execute("SELECT id, embedding_id FROM embeddings ORDER BY id").fetchall()
    )


def fetch_queue_rows(con: sqlite3.Connection) -> List[Tuple[str, bytes, str]]:
    """(chroma_id, vector_blob, metadata_json) for not-yet-compacted writes."""
    return list(
        con.execute(
            "SELECT id, vector, metadata FROM embeddings_queue "
            "WHERE operation = 0"   # 0 = ADD (1 = UPDATE, 2 = DELETE)
        ).fetchall()
    )


# ============================================================
# VECTOR TRUNCATION
# ============================================================

def truncate_and_normalize(vec_f32: np.ndarray, target_dim: int) -> np.ndarray:
    """Slice to first `target_dim` floats (Matryoshka) and L2-renormalize so
    cosine distance is the right metric in Qdrant."""
    v = np.asarray(vec_f32[:target_dim], dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm > 0:
        v = v / norm
    return v


# ============================================================
# QDRANT
# ============================================================

def qdrant_collection_name(source: str) -> str:
    """`belladonna_<src_lower>` — clean prefix, distinguishable from Chroma names."""
    return f"belladonna_{source.lower()}"


def ensure_collection(client: QdrantClient, name: str) -> None:
    """Create the collection with INT8 quantization + on-disk vectors + HNSW.
    Idempotent: skips creation if a collection with `name` already exists."""
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=TARGET_DIM,
            distance=qm.Distance.COSINE,
            on_disk=True,                 # full f32 vectors on SSD, paged in for rescoring
        ),
        quantization_config=qm.ScalarQuantization(
            scalar=qm.ScalarQuantizationConfig(
                type=qm.ScalarType.INT8,
                quantile=0.99,            # tail-clip outliers, better INT8 resolution
                always_ram=True,          # quantized vectors live in RAM (hot path)
            ),
        ),
        hnsw_config=qm.HnswConfigDiff(
            m=16,
            ef_construct=100,
            on_disk=False,                # graph in RAM for query speed
        ),
        optimizers_config=qm.OptimizersConfigDiff(
            default_segment_number=2,     # parallelism during ingest
            indexing_threshold=20000,     # build HNSW once we have decent data
        ),
    )


def point_id_for(chroma_id: str) -> str:
    """Deterministic UUID per chroma embedding id, so re-running is upsert
    not duplicate-insert."""
    return str(uuid.uuid5(NAMESPACE_BELLADONNA, chroma_id))


def upsert_batch(
    client: QdrantClient,
    collection: str,
    chroma_ids: List[str],
    vectors: List[np.ndarray],
    payloads: List[Dict[str, Any]],
) -> None:
    points = [
        qm.PointStruct(
            id=point_id_for(cid),
            vector=v.tolist(),
            payload={**p, "chroma_id": cid},
        )
        for cid, v, p in zip(chroma_ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection, points=points, wait=False)


# ============================================================
# MIGRATION CORE
# ============================================================

def iter_persisted_vectors(
    seg_dir: Path,
    label_to_id: Dict[int, str],
    stride: int,
    file_size: int,
    chroma_id_to_int: Dict[str, int],
    metadata_provider,
    chunk_size: int = METADATA_CHUNK_SIZE,
) -> Iterator[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """Yield (chroma_id, 4096-dim vector, metadata) by walking
    data_level0.bin linearly. For each element we read the label from the
    element's trailing 8 bytes and look it up in `label_to_id` to recover
    the chroma id; the id then joins to sqlite for metadata.

    Position-in-file is NOT the same as the pickle's `id_to_label` value
    (chroma's labels are logical names assigned at insert time, not slots
    in the file). Walking by position + reading the trailing label is the
    only reliable extraction order.

    `metadata_provider(int_ids)` returns a {int_id: metadata-dict} mapping
    for the chunk. Use `_all_preloaded_metadata_provider` (whole-corpus
    dict in RAM, fast) for small sources, or `_chunked_sqlite_provider`
    (one SELECT per chunk, bounded RAM) for big ones."""
    label_offset = 4 + DEFAULT_M_MAX * 4 + ORIG_DIM * 4   # = 16516
    n_elements = file_size // stride
    mm, _ = open_data_level0(seg_dir)
    try:
        chunk_records: List[Tuple[str, np.ndarray, Optional[int]]] = []
        chunk_int_ids: List[int] = []

        def _drain():
            metas = metadata_provider(chunk_int_ids)
            for chroma_id, vec, int_id in chunk_records:
                md = metas.get(int_id, {}) if int_id is not None else {}
                yield chroma_id, vec, md

        for pos in range(n_elements):
            base = pos * stride
            vec_buf = mm[base + VECTOR_OFFSET_BYTES : base + VECTOR_OFFSET_BYTES + VECTOR_NBYTES]
            label = struct.unpack_from("<Q", mm, base + label_offset)[0]
            chroma_id = label_to_id.get(label)
            if chroma_id is None:
                # Element exists on disk but isn't claimed by the pickle.
                # Could be a deleted-then-overwritten slot. Skip silently.
                continue
            vec = np.frombuffer(vec_buf, dtype=np.float32)
            int_id = chroma_id_to_int.get(chroma_id)
            chunk_records.append((chroma_id, vec, int_id))
            if int_id is not None:
                chunk_int_ids.append(int_id)

            if len(chunk_records) >= chunk_size:
                yield from _drain()
                chunk_records = []
                chunk_int_ids = []

        if chunk_records:
            yield from _drain()
    finally:
        mm.close()


def _all_preloaded_metadata_provider(metadata_by_id: Dict[int, Dict[str, Any]]):
    """Closure: returns the slice {id: md} for the requested int_ids out of
    a fully-preloaded {id: md} dict. Constant-time per id."""
    def provider(int_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        return {i: metadata_by_id.get(i, {}) for i in int_ids}
    return provider


def _chunked_sqlite_provider(con_path: str):
    """Closure: opens a *fresh* read-only sqlite handle per call to avoid
    keeping the connection across the main thread's mmap iteration. Each
    call hits sqlite once for the chunk's IDs."""
    def provider(int_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        if not int_ids:
            return {}
        con = sqlite3.connect(con_path, uri=True)
        try:
            return fetch_metadata_for_ids(con, int_ids)
        finally:
            con.close()
    return provider


def iter_queued_vectors(
    queue_rows: List[Tuple[str, bytes, str]],
) -> Iterator[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """Yield (chroma_id, vector, metadata) for vectors still in the WAL
    (haven't been compacted into data_level0.bin yet)."""
    for chroma_id, vec_blob, md_json in queue_rows:
        if not vec_blob:
            continue
        vec = np.frombuffer(vec_blob, dtype=np.float32)
        try:
            md = json.loads(md_json) if md_json else {}
        except json.JSONDecodeError:
            md = {}
        yield chroma_id, vec, md


def build_payload(chroma_md: Dict[str, Any], source: str) -> Dict[str, Any]:
    """The Qdrant payload is what becomes `metadata` on the retrieved hit.
    We pull the chroma:document into a top-level `factoid_text` field
    (matches the existing retriever's expectations) and keep everything else
    as-is — we don't strip license/title/year keys."""
    text = chroma_md.get("chroma:document") or ""
    payload = {k: v for k, v in chroma_md.items() if k != "chroma:document"}
    payload["factoid_text"] = text
    payload["source"] = source
    return payload


def migrate_source(
    source: str,
    *,
    client: QdrantClient,
    batch: int = 256,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Migrate one source end-to-end. Returns a stats dict."""
    t0 = time.time()
    coll = qdrant_collection_name(source)

    con = open_chroma_sqlite(source)
    seg_dir = hnsw_segment_dir(source, con)
    label_to_id = load_label_to_id(seg_dir)
    embedding_rows = fetch_embedding_rows(con)
    queue_rows = fetch_queue_rows(con)

    # chroma_id -> sqlite int id, for joining vector reads to metadata.
    chroma_id_to_int = {chroma_id: int_id for int_id, chroma_id in embedding_rows}

    # Pick the metadata strategy. For huge embedding_metadata tables, the
    # full {id: md} dict pushes 4+ GB and OOMs the migration; we chunk
    # via fresh sqlite handles instead.
    sqlite_path = str(CHROMA_ROOT / source / "embeddings" / "chroma.sqlite3")
    sqlite_uri = f"file:{sqlite_path}?mode=ro"
    if source in LARGE_METADATA_SOURCES:
        con.close()
        metadata_provider = _chunked_sqlite_provider(sqlite_uri)
        metadata_strategy = "chunked"
    else:
        metadata_by_id = fetch_metadata_by_id(con)
        con.close()
        metadata_provider = _all_preloaded_metadata_provider(metadata_by_id)
        metadata_strategy = "preloaded"

    n_persisted = len(label_to_id)
    stride = compute_element_stride(seg_dir, n_persisted)
    file_size = (seg_dir / "data_level0.bin").stat().st_size

    print(
        f"[{source}] embeddings={len(embedding_rows)} "
        f"persisted={n_persisted} queued={len(queue_rows)} "
        f"stride={stride}B data_level0={file_size / 1e9:.2f}GB "
        f"metadata={metadata_strategy}"
    )

    if dry_run:
        return {
            "source": source,
            "embeddings": len(embedding_rows),
            "persisted": n_persisted,
            "queued": len(queue_rows),
            "stride": stride,
            "data_level0_bytes": file_size,
            "dry_run": True,
        }

    ensure_collection(client, coll)

    ids: List[str] = []
    vecs: List[np.ndarray] = []
    pays: List[Dict[str, Any]] = []
    n_written = 0
    n_skipped = 0
    last_log = time.time()

    def _flush() -> None:
        nonlocal ids, vecs, pays, n_written
        if not ids:
            return
        upsert_batch(client, coll, ids, vecs, pays)
        n_written += len(ids)
        ids, vecs, pays = [], [], []

    def _chain() -> Iterator[Tuple[str, np.ndarray, Dict[str, Any]]]:
        yield from iter_persisted_vectors(
            seg_dir, label_to_id, stride, file_size,
            chroma_id_to_int, metadata_provider,
        )
        yield from iter_queued_vectors(queue_rows)

    stream = _chain()

    for i, (cid, raw_vec, md) in enumerate(stream):
        if limit is not None and i >= limit:
            break
        if raw_vec.size < TARGET_DIM:
            n_skipped += 1
            continue
        v1024 = truncate_and_normalize(raw_vec, TARGET_DIM)
        ids.append(cid)
        vecs.append(v1024)
        pays.append(build_payload(md, source))
        if len(ids) >= batch:
            _flush()
            if time.time() - last_log > 5:
                rate = n_written / max(time.time() - t0, 1e-6)
                print(f"  [{source}] {n_written:,} written  ({rate:,.0f}/s)")
                last_log = time.time()

    _flush()
    # Force one final wait so the count() below is accurate.
    client.upsert(
        collection_name=coll,
        points=[],
        wait=True,
    ) if False else None

    elapsed = time.time() - t0
    final_count = client.count(coll, exact=False).count
    print(
        f"[{source}] done: wrote {n_written:,} (skipped {n_skipped}), "
        f"qdrant count~={final_count:,}, took {elapsed:.1f}s "
        f"({n_written / max(elapsed, 1e-6):,.0f}/s)"
    )
    return {
        "source": source,
        "written": n_written,
        "skipped": n_skipped,
        "qdrant_count": final_count,
        "elapsed_s": elapsed,
    }


# ============================================================
# CLI
# ============================================================

def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", help="single source name (e.g. AGO, EPMC)")
    p.add_argument("--all", action="store_true", help="migrate all sources")
    p.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N vectors (debugging)")
    p.add_argument("--dry-run", action="store_true",
                   help="probe only; don't write to Qdrant")
    args = p.parse_args()

    if not args.source and not args.all:
        p.error("specify --source X or --all")
    if args.all and args.source:
        p.error("--all and --source are mutually exclusive")

    client = QdrantClient(url=args.qdrant_url, timeout=120.0)

    if args.all:
        # Smaller sources first so failures surface fast.
        order = ["AGO", "ASCO", "ESMO", "EMA", "FDA", "CTG", "EPMC", "Elsevier"]
        results = []
        for src in order:
            if src not in SOURCE_COLLECTION_NAMES:
                continue
            chroma_db = CHROMA_ROOT / src / "embeddings" / "chroma.sqlite3"
            if not chroma_db.exists():
                print(f"[{src}] skipped — no chroma db at {chroma_db}")
                continue
            try:
                results.append(migrate_source(
                    src, client=client, batch=args.batch,
                    limit=args.limit, dry_run=args.dry_run,
                ))
            except Exception as exc:
                print(f"[{src}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                results.append({"source": src, "error": str(exc)})
        print("\n=== summary ===")
        for r in results:
            print(json.dumps(r))
        return

    migrate_source(
        args.source, client=client, batch=args.batch,
        limit=args.limit, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    _main()
