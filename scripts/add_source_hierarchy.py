#!/usr/bin/env python3
"""
Add a `source_hierarchy` field to the metadata of every factoid JSON file
across all sources.

Hierarchy (level -> label):
    1  Guidelines                                      (AGO, ESMO, ASCO)
    2  Regulatory documents                            (EMA, FDA)
    3  Systematic reviews and meta-analyses            (EPMC, Elsevier)
    4  Randomized clinical trials                      (EPMC, Elsevier)
    5  Observational and registry studies              (EPMC, Elsevier)
    6  Trial registry entries                          (ClinicalTrials.gov)
    7  Narrative reviews, commentaries, case reports   (EPMC, Elsevier)

For sources with a deterministic level (1, 2, 6) the label is applied directly.

For EPMC and Elsevier the level is decided by GPT-OSS-120B based on signals
pulled from the UPSTREAM filtered XML (not the factoid file):
    - title
    - journal
    - JATS article-type / Elsevier pubType + document-type / document-subtype
    - article categories / subject groups
    - abstract
    - first ~1500 chars of the methods section (when present)
If the upstream XML cannot be found, the script falls back to whatever
metadata + factoid sample already lives inside the factoid JSON.

Usage:
    python scripts/add_source_hierarchy.py                       # all sources
    python scripts/add_source_hierarchy.py --sources EPMC --limit 5
    python scripts/add_source_hierarchy.py --dry-run             # don't write
    python scripts/add_source_hierarchy.py --force               # re-label

Environment (only required when EPMC/Elsevier run):
    VIRTUAL_API_KEY, BASE_URL
    MODEL_NAME           (default: GPT-OSS-120B)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# HIERARCHY
# ============================================================

HIERARCHY_LABELS: Dict[int, str] = {
    1: "Guidelines",
    2: "Regulatory documents",
    3: "Systematic reviews and meta-analyses",
    4: "Randomized clinical trials",
    5: "Observational and registry studies",
    6: "Trial registry entries",
    7: "Narrative reviews, commentaries, case reports",
}

LLM_ALLOWED_LEVELS = {3, 4, 5, 7}


def hierarchy_block(level: int) -> Dict[str, Any]:
    if level not in HIERARCHY_LABELS:
        raise ValueError(f"Unknown hierarchy level: {level}")
    return {"level": level, "label": HIERARCHY_LABELS[level]}


# ============================================================
# SOURCE CONFIG
# ============================================================
# Each source declares:
#   - paths:    candidate directories holding factoid JSONs (first existing wins)
#   - strategy: "direct" or "llm"
#   - level:    used only for "direct"
#   - xml_paths: (llm only) candidate dirs holding the upstream filtered XML
#   - xml_kind:  (llm only) "jats" (EPMC) or "elsevier"

SOURCES: Dict[str, Dict[str, Any]] = {
    "AGO":      {"paths": ["artifacts/AGO/factoids"],
                 "strategy": "direct", "level": 1},
    "ESMO":     {"paths": ["artifacts/esmo/factoids", "artifacts/ESMO/factoids"],
                 "strategy": "direct", "level": 1},
    "ASCO":     {"paths": ["ASCO/factoids", "artifacts/ASCO/factoids"],
                 "strategy": "direct", "level": 1},
    "EMA":      {"paths": ["artifacts/EMA/factoids"],
                 "strategy": "direct", "level": 2},
    "FDA":      {"paths": ["artifacts/FDA/factoids", "artifacts/FDA"],
                 "strategy": "direct", "level": 2,
                 "file_filter": "*factoids*.json"},
    "CTG":      {"paths": ["artifacts/CTG/factoids"],
                 "strategy": "direct", "level": 6},
    "EPMC":     {"paths": ["artifacts/EPMC/factoids",
                           "artifacts/epmc_fulltext/factoids",
                           "artifacts/factoids"],
                 "strategy": "llm",
                 "xml_paths": ["artifacts/EPMC/filtered_xml",
                               "artifacts/epmc_fulltext/filtered_xml"],
                 "xml_kind": "jats"},
    "Elsevier": {"paths": ["artifacts/elsevier/factoids", "artifacts/Elsevier/factoids"],
                 "strategy": "llm",
                 "xml_paths": ["artifacts/elsevier/filtered_xml",
                               "artifacts/Elsevier/filtered_xml"],
                 "xml_kind": "elsevier"},
}


# ============================================================
# LLM CONFIG
# ============================================================

MODEL_NAME = os.getenv("MODEL_NAME", "GPT-OSS-120B")
MAX_COMPLETION_TOKENS = 2048
LLM_CONCURRENCY = 60
LLM_TIMEOUT_SECONDS = 120
LLM_MAX_RETRIES = 3

ABSTRACT_CHAR_LIMIT = 4000
METHODS_CHAR_LIMIT = 1500
FALLBACK_SAMPLE_FACTOIDS = 8
FALLBACK_SAMPLE_CHAR_LIMIT = 280


# ============================================================
# XML EXTRACTION HELPERS
# ============================================================

def _local(tag: str) -> str:
    """Strip namespace prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _iter_local(root: ET.Element, name: str):
    for el in root.iter():
        if _local(el.tag) == name:
            yield el


def _text_in(root: Optional[ET.Element], name: str) -> Optional[str]:
    if root is None:
        return None
    for el in _iter_local(root, name):
        text = "".join(el.itertext()).strip()
        if text:
            return text
    return None


def _all_text(el: Optional[ET.Element]) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()


def _normalize(text: str, limit: Optional[int] = None) -> str:
    out = re.sub(r"\s+", " ", text or "").strip()
    if limit is not None and len(out) > limit:
        out = out[:limit].rstrip() + "..."
    return out


# ---------- JATS (EPMC) ----------

def extract_signals_jats(xml_path: Path) -> Dict[str, Any]:
    """Pull classifier-relevant signals from an EPMC JATS XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    article = next(_iter_local(root, "article"), root)
    article_type = (article.attrib.get("article-type") or "").strip()

    title = _text_in(root, "article-title") or ""
    journal = (_text_in(root, "journal-title")
               or _text_in(root, "abbrev-journal-title") or "")

    categories: List[str] = []
    for sg in _iter_local(root, "subj-group"):
        for subj in _iter_local(sg, "subject"):
            t = _all_text(subj)
            if t:
                categories.append(t)

    kwds: List[str] = []
    for kw in _iter_local(root, "kwd"):
        t = _all_text(kw)
        if t:
            kwds.append(t)

    abstract_parts: List[str] = []
    for ab in _iter_local(root, "abstract"):
        if ab.attrib.get("abstract-type") == "graphical":
            continue
        block_parts: List[str] = []
        for child in ab.iter():
            tag = _local(child.tag)
            if tag == "title":
                block_parts.append(_all_text(child).upper())
            elif tag == "p":
                block_parts.append(_all_text(child))
        abstract_parts.append(" ".join(b for b in block_parts if b))
    abstract = " | ".join(p for p in abstract_parts if p)

    methods_text = ""
    for sec in _iter_local(root, "sec"):
        sec_type = (sec.attrib.get("sec-type") or "").lower()
        title_el = next((c for c in sec if _local(c.tag) == "title"), None)
        title_text = _all_text(title_el).lower() if title_el is not None else ""
        if "method" in sec_type or "method" in title_text:
            methods_text = _all_text(sec)
            break

    return {
        "title": _normalize(title),
        "journal": _normalize(journal),
        "article_type": article_type,
        "document_subtype": "",
        "categories": [c for c in (_normalize(c) for c in categories) if c],
        "keywords": [k for k in (_normalize(k) for k in kwds) if k][:12],
        "abstract": _normalize(abstract, ABSTRACT_CHAR_LIMIT),
        "methods_snippet": _normalize(methods_text, METHODS_CHAR_LIMIT),
    }


# ---------- Elsevier ----------

def extract_signals_elsevier(xml_path: Path) -> Dict[str, Any]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    coredata = next(_iter_local(root, "coredata"), None)
    article = next(_iter_local(root, "article"), None)
    head = None
    if article is not None:
        head = next((c for c in article if _local(c.tag) == "head"), None)

    title = (_text_in(coredata, "title")
             or _text_in(head, "title") or "")
    journal = _text_in(coredata, "publicationName") or ""
    pub_type = _text_in(coredata, "pubType") or ""
    aggregation_type = _text_in(coredata, "aggregationType") or ""
    document_type = _text_in(root, "document-type") or ""
    document_subtype = _text_in(root, "document-subtype") or ""

    coredata_abstract = _text_in(coredata, "description") or ""

    abstract_chunks: List[str] = [coredata_abstract] if coredata_abstract else []
    if head is not None:
        for ab in (c for c in head if _local(c.tag) == "abstract"):
            block_parts: List[str] = []
            for child in ab.iter():
                tag = _local(child.tag)
                if tag in {"section-title", "simple-para", "para"}:
                    txt = _all_text(child)
                    if txt:
                        block_parts.append(txt)
            if block_parts:
                abstract_chunks.append(" ".join(block_parts))
    abstract = " | ".join(p for p in abstract_chunks if p)

    methods_text = ""
    if article is not None:
        body = next((c for c in article if _local(c.tag) == "body"), None)
        if body is not None:
            for sec in _iter_local(body, "section"):
                title_el = next((c for c in sec if _local(c.tag) == "section-title"),
                                None)
                if title_el is None:
                    continue
                title_text = _all_text(title_el).lower()
                if "method" in title_text or "material" in title_text:
                    methods_text = _all_text(sec)
                    break

    article_type_hint = pub_type or aggregation_type or document_type

    return {
        "title": _normalize(title),
        "journal": _normalize(journal),
        "article_type": _normalize(article_type_hint),
        "document_subtype": _normalize(document_subtype),
        "categories": [],
        "keywords": [],
        "abstract": _normalize(abstract, ABSTRACT_CHAR_LIMIT),
        "methods_snippet": _normalize(methods_text, METHODS_CHAR_LIMIT),
    }


def extract_signals_from_xml(xml_path: Path, xml_kind: str) -> Dict[str, Any]:
    if xml_kind == "jats":
        return extract_signals_jats(xml_path)
    if xml_kind == "elsevier":
        return extract_signals_elsevier(xml_path)
    raise ValueError(f"Unknown xml_kind: {xml_kind}")


# ---------- XML lookup ----------

def stem_to_xml_path(stem_without_factoids: str, xml_dirs: List[Path]) -> Optional[Path]:
    for d in xml_dirs:
        cand = d / f"{stem_without_factoids}.xml"
        if cand.exists():
            return cand
    return None


def factoid_stem_to_doc_id(file_stem: str) -> str:
    # `PMC12104954_factoids` -> `PMC12104954`
    return re.sub(r"_factoids$", "", file_stem)


# ============================================================
# FALLBACK SIGNALS (when no upstream XML is available)
# ============================================================

def signals_from_factoid_json(data: Dict[str, Any]) -> Dict[str, Any]:
    meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    title = (meta.get("document_title") or meta.get("TITLE")
             or meta.get("title") or "").strip()
    journal = (meta.get("journal") or meta.get("JOURNAL") or "").strip()
    doc_type = (meta.get("document_type") or "").strip()
    doc_subtype = (meta.get("document_subtype") or "").strip()

    sample_lines: List[str] = []
    for f in (data.get("factoids") or [])[:FALLBACK_SAMPLE_FACTOIDS]:
        if not isinstance(f, dict):
            continue
        text = f.get("factoid_text") or f.get("text") or ""
        text = _normalize(text, FALLBACK_SAMPLE_CHAR_LIMIT)
        if text:
            sample_lines.append(text)

    return {
        "title": title,
        "journal": journal,
        "article_type": doc_type,
        "document_subtype": doc_subtype,
        "categories": [],
        "keywords": [],
        "abstract": "",
        "methods_snippet": "",
        "factoid_sample": sample_lines,
    }


# ============================================================
# PROMPT
# ============================================================

def build_prompts(signals: Dict[str, Any]) -> Tuple[str, str]:
    system = (
        "You classify a single biomedical document into exactly one of four "
        "study-type buckets and reply with one digit.\n"
        "\n"
        "Buckets:\n"
        "  3 = Systematic review or meta-analysis "
        "(explicit systematic search strategy, PRISMA-style methodology, "
        "or pooled quantitative synthesis across multiple studies).\n"
        "  4 = Randomized clinical trial (primary report of a randomized "
        "interventional trial; the abstract / methods describe randomization "
        "of participants to arms).\n"
        "  5 = Observational or registry study (cohort, case-control, "
        "cross-sectional, registry analysis, real-world data, biomarker "
        "association study, prospective or retrospective observational study, "
        "single-arm trial without randomization).\n"
        "  7 = Narrative review, commentary, editorial, perspective, opinion, "
        "case report, or case series.\n"
        "\n"
        "Decision rules:\n"
        "- Trust the abstract's wording over the article-type hint when they "
        "  disagree. The hint may be generic ('research-article', 'fla').\n"
        "- A document that calls itself a 'systematic review' or 'meta-analysis' "
        "  is 3 only if the abstract/methods describe a systematic search; "
        "  otherwise classify as 7.\n"
        "- A randomized phase II/III interventional trial primary report is 4. "
        "  Single-arm or non-randomized trials are 5.\n"
        "- In-vitro, animal-only, computational, or methodology papers without "
        "  human study design described should be 5 (observational/laboratory) "
        "  if they collect data, else 7.\n"
        "- If genuinely uncertain after weighing all signals, pick 7.\n"
        "\n"
        "Output exactly one digit (3, 4, 5, or 7). No words. No punctuation."
    )

    cats = ", ".join(signals.get("categories") or []) or "N/A"
    kwds = ", ".join(signals.get("keywords") or []) or "N/A"
    article_type = signals.get("article_type") or "N/A"
    doc_subtype = signals.get("document_subtype") or "N/A"
    abstract = signals.get("abstract") or ""
    methods = signals.get("methods_snippet") or ""

    parts = [
        f"Title: {signals.get('title') or 'N/A'}",
        f"Journal: {signals.get('journal') or 'N/A'}",
        f"Article-type hint: {article_type}",
        f"Document subtype: {doc_subtype}",
        f"Categories: {cats}",
        f"Keywords: {kwds}",
    ]

    if abstract:
        parts.append(f"\nAbstract:\n{abstract}")
    else:
        sample = signals.get("factoid_sample") or []
        if sample:
            joined = "\n".join(f"- {s}" for s in sample)
            parts.append(
                "\nAbstract not available. Below is a sample of factoids "
                "extracted from the paper; use them as weak signal only:\n"
                f"{joined}"
            )

    if methods:
        parts.append(f"\nMethods (first ~{METHODS_CHAR_LIMIT} chars):\n{methods}")

    parts.append("\nReply with one digit only: 3, 4, 5, or 7.")
    user = "\n".join(parts)
    return system, user


_LEVEL_RE = re.compile(r"[3457]")


def parse_level(text: str) -> Optional[int]:
    if not text:
        return None
    m = _LEVEL_RE.search(text)
    if not m:
        return None
    return int(m.group(0))


# ============================================================
# IO
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def discover_files(cfg: Dict[str, Any], root: Path) -> List[Path]:
    pattern = cfg.get("file_filter", "*_factoids.json")
    for rel in cfg["paths"]:
        d = (root / rel).resolve()
        if d.exists() and d.is_dir():
            files = sorted(d.glob(pattern))
            if files:
                return files
    return []


def resolve_xml_dirs(cfg: Dict[str, Any], root: Path) -> List[Path]:
    out: List[Path] = []
    for rel in cfg.get("xml_paths", []) or []:
        d = (root / rel).resolve()
        if d.exists() and d.is_dir():
            out.append(d)
    return out


def already_labelled(data: Dict[str, Any]) -> bool:
    meta = data.get("metadata")
    if not isinstance(meta, dict):
        return False
    sh = meta.get("source_hierarchy")
    return isinstance(sh, dict) and "level" in sh and "label" in sh


def inject_hierarchy(data: Dict[str, Any], level: int) -> Dict[str, Any]:
    if not isinstance(data.get("metadata"), dict):
        data["metadata"] = {}
    data["metadata"]["source_hierarchy"] = hierarchy_block(level)
    return data


# ============================================================
# DIRECT LABELLING
# ============================================================

def label_direct(files: List[Path], level: int, force: bool, dry_run: bool,
                 source_name: str) -> Dict[str, int]:
    stats = {"total": len(files), "labelled": 0, "skipped": 0, "errors": 0}
    for path in files:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[{source_name}] [ERROR] read {path.name}: {e}")
            stats["errors"] += 1
            continue

        if already_labelled(data) and not force:
            stats["skipped"] += 1
            continue

        inject_hierarchy(data, level)
        if not dry_run:
            try:
                write_json_atomic(path, data)
            except Exception as e:
                print(f"[{source_name}] [ERROR] write {path.name}: {e}")
                stats["errors"] += 1
                continue
        stats["labelled"] += 1
    return stats


# ============================================================
# LLM LABELLING
# ============================================================

async def classify_one(client, signals: Dict[str, Any]) -> Optional[int]:
    system, user = build_prompts(signals)

    last_err: Optional[Exception] = None
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            async def _do_request() -> str:
                resp = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                return resp.choices[0].message.content or ""

            text = await asyncio.wait_for(_do_request(), timeout=LLM_TIMEOUT_SECONDS)
            level = parse_level(text)
            if level in LLM_ALLOWED_LEVELS:
                return level
            last_err = ValueError(f"Unparseable response: {text!r}")
        except Exception as e:
            last_err = e
        await asyncio.sleep(0.5 * attempt)

    print(f"[LLM] giving up after {LLM_MAX_RETRIES} attempts: {last_err}")
    return None


def build_signals_for_file(path: Path, xml_dirs: List[Path], xml_kind: str,
                           data: Dict[str, Any], source_name: str
                           ) -> Tuple[Dict[str, Any], bool]:
    """Return (signals, used_xml). Falls back to factoid JSON if XML missing."""
    doc_id = factoid_stem_to_doc_id(path.stem)
    xml_path = stem_to_xml_path(doc_id, xml_dirs)

    if xml_path is not None:
        try:
            return extract_signals_from_xml(xml_path, xml_kind), True
        except Exception as e:
            print(f"[{source_name}] [WARN] XML parse failed {xml_path.name}: {e}"
                  " -> falling back to factoid metadata")

    return signals_from_factoid_json(data), False


async def process_llm_file(client, sem: asyncio.Semaphore, path: Path,
                           xml_dirs: List[Path], xml_kind: str,
                           force: bool, dry_run: bool, source_name: str,
                           stats: Dict[str, int]) -> None:
    async with sem:
        try:
            data = load_json(path)
        except Exception as e:
            print(f"[{source_name}] [ERROR] read {path.name}: {e}")
            stats["errors"] += 1
            return

        if already_labelled(data) and not force:
            stats["skipped"] += 1
            return

        signals, used_xml = build_signals_for_file(
            path, xml_dirs, xml_kind, data, source_name
        )
        if not used_xml:
            stats["xml_missing"] = stats.get("xml_missing", 0) + 1

        if not (signals.get("title") or signals.get("abstract")
                or signals.get("factoid_sample")):
            print(f"[{source_name}] [SKIP] {path.name} -> no usable signals")
            stats["errors"] += 1
            return

        level = await classify_one(client, signals)
        if level is None:
            print(f"[{source_name}] [SKIP] {path.name} -> classifier failed")
            stats["errors"] += 1
            return

        inject_hierarchy(data, level)
        if not dry_run:
            try:
                write_json_atomic(path, data)
            except Exception as e:
                print(f"[{source_name}] [ERROR] write {path.name}: {e}")
                stats["errors"] += 1
                return

        stats["labelled"] += 1
        stats[f"level_{level}"] = stats.get(f"level_{level}", 0) + 1

        title = signals.get("title") or ""
        print(f"[{source_name}] [{level}] {path.name}  ::  "
              f"{title[:90]}{'...' if len(title) > 90 else ''}")

        if stats["labelled"] % 50 == 0:
            print(f"[{source_name}] progress: labelled={stats['labelled']} "
                  f"skipped={stats['skipped']} errors={stats['errors']} "
                  f"xml_missing={stats.get('xml_missing', 0)}")


async def label_llm(files: List[Path], xml_dirs: List[Path], xml_kind: str,
                    force: bool, dry_run: bool, source_name: str
                    ) -> Dict[str, int]:
    stats: Dict[str, int] = {"total": len(files), "labelled": 0,
                             "skipped": 0, "errors": 0, "xml_missing": 0}
    if not files:
        return stats

    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("[LLM] openai package not installed; cannot run LLM step.")
        stats["errors"] = len(files)
        return stats

    api_key = os.getenv("VIRTUAL_API_KEY")
    base_url = os.getenv("BASE_URL")
    if not api_key or not base_url:
        print("[LLM] VIRTUAL_API_KEY / BASE_URL not set; skipping LLM step.")
        stats["errors"] = len(files)
        return stats

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    sem = asyncio.Semaphore(LLM_CONCURRENCY)

    tasks = [
        process_llm_file(client, sem, p, xml_dirs, xml_kind,
                         force, dry_run, source_name, stats)
        for p in files
    ]
    await asyncio.gather(*tasks)
    return stats


# ============================================================
# DRIVER
# ============================================================

def run_source(source_name: str, root: Path, force: bool, dry_run: bool,
               limit: Optional[int]) -> Dict[str, int]:
    cfg = SOURCES[source_name]
    files = discover_files(cfg, root)
    if limit is not None:
        files = files[:limit]

    if not files:
        print(f"[{source_name}] no factoid files found under any of: "
              f"{cfg['paths']}")
        return {"total": 0, "labelled": 0, "skipped": 0, "errors": 0}

    print(f"[{source_name}] found {len(files)} file(s); strategy={cfg['strategy']}")

    if cfg["strategy"] == "direct":
        return label_direct(files, cfg["level"], force, dry_run, source_name)

    if cfg["strategy"] == "llm":
        xml_dirs = resolve_xml_dirs(cfg, root)
        if not xml_dirs:
            print(f"[{source_name}] [WARN] no upstream XML dir found under "
                  f"{cfg.get('xml_paths')} -> falling back to factoid metadata only")
        else:
            print(f"[{source_name}] upstream XML dirs: {[str(d) for d in xml_dirs]}")
        return asyncio.run(
            label_llm(files, xml_dirs, cfg["xml_kind"], force, dry_run, source_name)
        )

    raise ValueError(f"Unknown strategy for {source_name}: {cfg['strategy']}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(),
                        help="Project root containing artifacts/ (default: cwd)")
    parser.add_argument("--sources", nargs="+",
                        choices=list(SOURCES.keys()),
                        help="Restrict to a subset of sources (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-label files that already have source_hierarchy")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do everything except writing the JSON back")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N files per source (for testing)")
    args = parser.parse_args()

    chosen = args.sources if args.sources else list(SOURCES.keys())

    print("=" * 70)
    print("ADD SOURCE HIERARCHY")
    print("=" * 70)
    print(f"Root:      {args.root.resolve()}")
    print(f"Sources:   {chosen}")
    print(f"Force:     {args.force}")
    print(f"Dry-run:   {args.dry_run}")
    print(f"Limit:     {args.limit}")
    print()

    overall_start = time.time()
    summary: Dict[str, Dict[str, int]] = {}

    for source_name in chosen:
        t0 = time.time()
        stats = run_source(source_name, args.root, args.force,
                           args.dry_run, args.limit)
        summary[source_name] = stats
        print(f"[{source_name}] done in {time.time() - t0:.1f}s -> {stats}\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = {"total": 0, "labelled": 0, "skipped": 0, "errors": 0}
    for src, s in summary.items():
        print(f"  {src:10s} {s}")
        for k in total:
            total[k] += s.get(k, 0)
    print(f"  {'TOTAL':10s} {total}")
    print(f"Elapsed: {time.time() - overall_start:.1f}s")

    return 0 if total["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
