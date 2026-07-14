#!/usr/bin/env python3
"""
BELLADONNA Factoid Atlas — data packer.

Joins ids + UMAP xy + 4-dimension labels + per-factoid metadata into the binary
files the web renderer streams:

    points.bin    float32 [N,2]            world coords (x,y)
    attrs.bin     packed record per point  (layout below, little-endian)
    docs.json     [{t,s,x,y}]              per-index hover/detail payload
    clusters.json [{id,count,terms}]       topic legend (if topics provided)
    manifest.json everything the UI needs to colour, legend and filter

attrs.bin record (ATTR_BYTES = 10 bytes/point):
    0  src          u8     source id
    1  drug_class   u8     taxonomy index
    2  biomarker    u8     taxonomy index
    3  setting      u8     taxonomy index
    4  evidence     u8     taxonomy index
    5  cluster      u8     k-means topic id (0 if none; k must be <= 255)
    6  year         u16    document year (0 = unknown)
    8  _pad         u16

Inputs:
    --stem    proto         -> proto.ids.json, proto.xy.npy, [proto.clusters.npy]
    --meta    proto.jsonl   -> {id, source, year, doc_id, text}
    --labels  proto.labels.jsonl -> {id, drug_class, biomarker, setting, evidence}
    --topics  proto.topics.json  (optional, from build_layout.py)
    --out     web/data           (output directory)

Example:
    python build_atlas_data.py --stem proto --meta proto.jsonl \
        --labels proto.labels.jsonl --topics proto.topics.json --out web/data
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

import taxonomy as tax

# attrs.bin record layout, computed from the ACTIVE dimensions:
#   byte 0       src (u8)
#   bytes 1..D   one u8 per active dimension (in DIMENSIONS order)
#   byte 1+D     cluster (u8)
#   year (u16)   at the next even offset
_U8_FIELDS = ["src"] + list(tax.DIMENSIONS) + ["cluster"]
_YEAR_OFF = len(_U8_FIELDS) + (len(_U8_FIELDS) % 2)   # 2-byte align
ATTR_BYTES = _YEAR_OFF + 2
ATTR_LAYOUT = {f: ["u8", i] for i, f in enumerate(_U8_FIELDS)}
ATTR_LAYOUT["year"] = ["u16", _YEAR_OFF]

# Source ids + colours (kept consistent with the existing knowledge-graph atlas
# / generate_conference_visualizations.py).
SOURCES = [
    ("AGO", "#2a9d8f"), ("ASCO", "#1d3557"), ("CTG", "#f4a261"), ("EMA", "#ffb703"),
    ("EPMC", "#e61a8d"), ("ESMO", "#457b9d"), ("Elsevier", "#3c5163"), ("FDA", "#d62828"),
]
SRC_ID = {name: i for i, (name, _) in enumerate(SOURCES)}

DOC_TEXT_MAXLEN = 320     # truncate factoid text stored in docs
DOC_SHARD_SIZE = 50000    # factoids per docs/<k>.json shard
DOC_SHARD_THRESHOLD = 120000   # above this N, shard docs (else single docs.json)


def load_jsonl_map(path: Path) -> dict:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pack atlas binaries for the web renderer.")
    ap.add_argument("--stem", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--topics", default="")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stem = Path(args.stem)
    ids = json.loads(stem.with_suffix(".ids.json").read_text())
    xy = np.load(stem.with_suffix(".xy.npy")).astype(np.float32)
    assert len(ids) == len(xy), "ids/xy length mismatch"
    N = len(ids)

    clusters = None
    cl_path = stem.with_suffix(".clusters.npy")
    if cl_path.exists():
        clusters = np.load(cl_path)

    meta = load_jsonl_map(Path(args.meta))
    labels = load_jsonl_map(Path(args.labels))

    idx = {d: tax.label_index(d) for d in tax.DIMENSIONS}
    default_code = {d: tax.default_code(d) for d in tax.DIMENSIONS}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # --- normalise xy to a tidy world box (center 0, unit-ish span) ---
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    span = float(np.max(mx - mn)) or 1.0
    xy_norm = (xy - (mn + mx) / 2.0) / span  # roughly [-0.5, 0.5]

    # --- points.bin ---
    (out / "points.bin").write_bytes(xy_norm.astype("<f4").tobytes())

    # --- attrs.bin + docs.json ---
    attrs = bytearray(N * ATTR_BYTES)
    docs = []
    src_count = [0] * len(SOURCES)
    years = []
    for i, fid in enumerate(ids):
        m = meta.get(fid, {})
        lab = labels.get(fid, {})
        src = m.get("source", "")
        sid = SRC_ID.get(src, 0)
        src_count[sid] += 1
        cl = int(clusters[i]) if clusters is not None else 0
        yr = int(m.get("year") or 0)
        if yr:
            years.append(yr)
        # u8 fields in layout order: src, <each dimension>, cluster
        u8vals = [sid]
        u8vals += [idx[d].get(lab.get(d, default_code[d]), 0) for d in tax.DIMENSIONS]
        u8vals += [min(cl, 255)]
        struct.pack_into("<" + "B" * len(u8vals), attrs, i * ATTR_BYTES, *u8vals)
        struct.pack_into("<H", attrs, i * ATTR_BYTES + _YEAR_OFF, yr if 0 < yr < 65535 else 0)
        txt = (m.get("text") or "")[:DOC_TEXT_MAXLEN]
        docs.append({"t": txt, "s": sid, "x": m.get("doc_id", ""), "y": yr})

    (out / "attrs.bin").write_bytes(bytes(attrs))

    # --- docs: single file for small N, sharded for large N (browser can't load
    #     a ~1 GB docs.json whole; the renderer lazy-loads shards on hover/click) ---
    if N > DOC_SHARD_THRESHOLD:
        shard_dir = out / "docs"
        shard_dir.mkdir(parents=True, exist_ok=True)
        for old in shard_dir.glob("*.json"):
            old.unlink()
        n_shards = (N + DOC_SHARD_SIZE - 1) // DOC_SHARD_SIZE
        for s in range(n_shards):
            chunk = docs[s * DOC_SHARD_SIZE:(s + 1) * DOC_SHARD_SIZE]
            (shard_dir / f"{s}.json").write_text(json.dumps(chunk, ensure_ascii=False), encoding="utf-8")
        (out / "docs.json").unlink(missing_ok=True)
        docs_shards = {"size": DOC_SHARD_SIZE, "count": n_shards}
        print(f"  docs: {n_shards} shards of {DOC_SHARD_SIZE} -> {shard_dir}/")
    else:
        (out / "docs.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
        docs_shards = None

    # --- clusters.json (topic legend) ---
    if args.topics and Path(args.topics).exists():
        topics = json.loads(Path(args.topics).read_text())
        (out / "clusters.json").write_text(json.dumps(topics), encoding="utf-8")
        k = len(topics)
    else:
        (out / "clusters.json").write_text("[]", encoding="utf-8")
        k = 0

    # --- manifest.json ---
    y0 = min(years) if years else 1990
    y1 = max(years) if years else 2026
    manifest = {
        "version": "factoid-1.0",
        "n": N,
        "factoidsTotal": N,
        "docsShards": docs_shards,   # {size,count} when sharded, else null (single docs.json)
        "attrRecordBytes": ATTR_BYTES,
        "attrLayout": ATTR_LAYOUT,
        "sources": [
            {"id": i, "name": name, "color": color, "count": src_count[i]}
            for i, (name, color) in enumerate(SOURCES)
        ],
        "dimensions": [
            {
                "key": d,
                "title": tax.DIM_TITLES[d],
                "labels": [
                    {"code": e["code"], "name": e["name"], "color": e["color"]}
                    for e in tax.TAXONOMY[d]
                ],
            }
            for d in tax.DIMENSIONS
        ],
        "k": k,
        "yearRange": [y0, y1],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # also drop taxonomy.json next to the data (handy for the page)
    tax.dump_json(out / "taxonomy.json")

    print(f"[pack] N={N} -> {out}")
    print(f"  points.bin {(out/'points.bin').stat().st_size/1e6:.1f} MB")
    print(f"  attrs.bin  {(out/'attrs.bin').stat().st_size/1e6:.1f} MB")
    if docs_shards:
        tot = sum(p.stat().st_size for p in (out/'docs').glob('*.json'))
        print(f"  docs/      {docs_shards['count']} shards, {tot/1e6:.1f} MB total")
    else:
        print(f"  docs.json  {(out/'docs.json').stat().st_size/1e6:.1f} MB")
    print(f"  sources: " + ", ".join(f"{SOURCES[i][0]}={src_count[i]}" for i in range(len(SOURCES)) if src_count[i]))


if __name__ == "__main__":
    main()
