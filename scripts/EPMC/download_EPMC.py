#!/usr/bin/env python3
"""
Download full text from Europe PMC using cursorMark deep paging and save both
raw JATS XML and extracted plain text.

This single script downloads BOTH:
  1. Open-access published articles  (OPEN_ACCESS:Y, fetched by PMCID)
  2. Preprints                       (PUB_TYPE:Preprint, source=="PPR", fetched by PPR id)

Both are written to the SAME output directory. Each document is labelled with a
DOC_TYPE ("article" or "preprint") in the .txt metadata header AND in the JSONL
manifest, so preprints stay distinguishable downstream even though they share a
folder. Filenames are also self-distinguishing (PMCxxxxxxx vs PPRxxxxxxx).

Key features:
- Uses /search with cursorMark to iterate beyond 1000 results.
- Downloads full text via /{ID}/fullTextXML (works for OA/PMCID items and PPR ids).
- Saves:
    artifacts/epmc_fulltext/PMCxxxxxxx.txt   (article, with metadata header)
    artifacts/epmc_fulltext/PPRxxxxxxx.txt   (preprint, with metadata header)
    artifacts/epmc_fulltext/xml/<ID>.xml     (raw JATS XML)
- Prints real-time progress: how many records streamed and how many full texts downloaded.
- Creates logs/ and writes a timestamped log file named after this script.

Examples:
    python scripts/EPMC/download_EPMC.py
    python scripts/EPMC/download_EPMC.py --query '("breast cancer") AND OPEN_ACCESS:Y'
    python scripts/EPMC/download_EPMC.py --no-preprints          # articles only
    python scripts/EPMC/download_EPMC.py --no-articles           # preprints only
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence, Union, cast

import requests
import xml.etree.ElementTree as ET


# ---- Requests typing helpers (keeps mypy happy) ----
ParamScalar = Union[str, bytes, int, float]
ParamValue = Union[ParamScalar, Sequence[ParamScalar], None]
Params = Mapping[str, ParamValue]

SEARCH_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/{rec_id}/fullTextXML"

DOC_TYPE_ARTICLE = "article"
DOC_TYPE_PREPRINT = "preprint"


@dataclass(frozen=True)
class Article:
    id: str
    source: str
    title: str
    author_string: str | None
    journal_title: str | None
    pub_year: str | None
    pmcid: str | None
    doi: str | None
    cited_by_count: int | None
    is_open_access: bool

    @staticmethod
    def from_epmc(d: Mapping[str, Any]) -> "Article":
        return Article(
            id=str(d.get("id", "")),
            source=str(d.get("source", "")),
            title=str(d.get("title", "")),
            author_string=cast(str | None, d.get("authorString")),
            journal_title=cast(str | None, d.get("journalTitle")),
            pub_year=cast(str | None, d.get("pubYear")),
            pmcid=cast(str | None, d.get("pmcid")),
            doi=cast(str | None, d.get("doi")),
            cited_by_count=int(d["citedByCount"]) if "citedByCount" in d else None,
            is_open_access=bool(d.get("isOpenAccess", False)),
        )


@dataclass
class Counters:
    streamed: int = 0
    downloaded: int = 0
    skipped_not_downloadable: int = 0
    skipped_already: int = 0
    http_errors: int = 0
    other_errors: int = 0


def make_logger(script_path: str | None) -> tuple[Path, Callable[[str], None]]:
    """
    Create logs/ and a log file named: <script_stem>_YYYYmmdd_HHMMSS.log
    Returns (log_path, log_fn).
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(script_path).stem if script_path else "epmc_fulltext"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{stem}_{ts}.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log_path, log


def epmc_search_page(query: str, page_size: int, cursor_mark: str, session: requests.Session) -> tuple[list[Article], str]:
    """
    Fetch one page using cursorMark deep paging.
    Returns (articles, next_cursor_mark).
    """
    params: Params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "lite",
        "cursorMark": cursor_mark,
    }
    r = session.get(SEARCH_BASE, params=params, timeout=30)
    r.raise_for_status()

    data = cast(dict[str, Any], r.json())
    results = cast(list[dict[str, Any]], (data.get("resultList") or {}).get("result", []) or [])
    next_cursor = cast(str, data.get("nextCursorMark", ""))

    return [Article.from_epmc(x) for x in results], next_cursor


def iter_epmc_all(query: str, page_size: int, sleep_s: float, session: requests.Session) -> Iterator[Article]:
    """
    Stream all results for a query using cursorMark deep paging.
    Stops when cursor stops advancing.
    """
    cursor = "*"
    while True:
        page, next_cursor = epmc_search_page(query=query, page_size=page_size, cursor_mark=cursor, session=session)
        if not page:
            return

        for art in page:
            yield art

        if not next_cursor or next_cursor == cursor:
            return

        cursor = next_cursor
        if sleep_s > 0:
            time.sleep(sleep_s)


def fetch_fulltext_xml(rec_id: str, session: requests.Session) -> bytes:
    url = FULLTEXT_BASE.format(rec_id=rec_id)
    r = session.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def _itertext(elem: ET.Element) -> str:
    # Robust, whitespace-normalized text extraction
    text_parts: list[str] = []
    for t in elem.itertext():
        s = " ".join(t.split())
        if s:
            text_parts.append(s)
    return " ".join(text_parts)


def _findall_anyns(root: ET.Element, tag: str) -> list[ET.Element]:
    # Namespace-agnostic search: matches .//{*}tag
    return list(root.findall(f".//{{*}}{tag}"))


def jats_xml_to_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)

    # Title
    titles = _findall_anyns(root, "article-title")
    title = _itertext(titles[0]) if titles else ""

    # Abstract(s)
    abstracts = _findall_anyns(root, "abstract")
    abstract_txt = "\n\n".join(_itertext(a) for a in abstracts) if abstracts else ""

    # Body: exclude some noisy blocks by clearing their content
    body_elems = _findall_anyns(root, "body")
    body_txt_parts: list[str] = []

    for body in body_elems:
        for unwanted in ("fig", "table-wrap", "ref-list", "alternatives", "supplementary-material"):
            for node in list(body.findall(f".//{{*}}{unwanted}")):
                node.clear()

        secs = body.findall(".//{*}sec")
        if secs:
            for sec in secs:
                stitles = _findall_anyns(sec, "title")
                if stitles:
                    body_txt_parts.append(_itertext(stitles[0]))
                for p in sec.findall(".//{*}p"):
                    body_txt_parts.append(_itertext(p))
        else:
            for p in body.findall(".//{*}p"):
                body_txt_parts.append(_itertext(p))

    body_txt = "\n\n".join(x for x in body_txt_parts if x)

    blocks: list[str] = []
    if title:
        blocks.append(title)
    if abstract_txt:
        blocks.append("ABSTRACT\n" + abstract_txt)
    if body_txt:
        blocks.append("MAIN TEXT\n" + body_txt)

    return "\n\n".join(blocks).strip()


def save_text(outdir: Path, rec_id: str, header_meta: Mapping[str, str], content: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{rec_id}.txt"

    header_lines = [f"{k}: {v}" for k, v in header_meta.items() if v]
    header = "\n".join(header_lines)
    sep = "\n" + ("-" * 80) + "\n"

    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write(sep)
        f.write(content)
        f.write("\n")

    return path


def save_raw_xml(outdir: Path, rec_id: str, xml_bytes: bytes) -> Path:
    xml_dir = outdir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    path = xml_dir / f"{rec_id}.xml"
    with path.open("wb") as f:
        f.write(xml_bytes)
    return path


def already_downloaded(outdir: Path, rec_id: str) -> bool:
    """
    Skip re-downloading if both XML and TXT already exist.
    """
    txt_path = outdir / f"{rec_id}.txt"
    xml_path = outdir / "xml" / f"{rec_id}.xml"
    return txt_path.exists() and xml_path.exists()


def article_record_id(a: Article) -> str | None:
    """Articles are fetched/saved by PMCID; require OA + a PMCID."""
    if a.is_open_access and a.pmcid:
        return a.pmcid
    return None


def preprint_record_id(a: Article) -> str | None:
    """Preprints are fetched/saved by their PPR id; require source==PPR."""
    if a.source == "PPR" and a.id.startswith("PPR"):
        return a.id
    return None


def run_pass(
    *,
    query: str,
    doc_type: str,
    record_id_of: Callable[[Article], str | None],
    args: argparse.Namespace,
    session: requests.Session,
    outdir: Path,
    manifest_path: Path,
    log: Callable[[str], None],
    counters: Counters,
) -> None:
    """
    Stream one Europe PMC query, append every record to the manifest (tagged
    with doc_type), and download+save full text for the downloadable ones.
    Returns early once the global --max-downloads cap is hit.
    """
    log(f"=== Pass: {doc_type} ===")
    log(f"Query: {query}")

    for a in iter_epmc_all(query, page_size=args.page_size, sleep_s=args.sleep, session=session):
        counters.streamed += 1

        # Manifest: every streamed record, labelled with doc_type.
        try:
            with manifest_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps({**a.__dict__, "doc_type": doc_type}, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            log(f"Manifest write failed (continuing): {e}")

        if counters.streamed % 100 == 0:
            log(
                f"Progress [{doc_type}]: streamed={counters.streamed:,} | "
                f"downloaded={counters.downloaded:,} | "
                f"skipped(not downloadable)={counters.skipped_not_downloadable:,} | "
                f"skipped(already)={counters.skipped_already:,}"
            )

        rec_id = record_id_of(a)
        if not rec_id:
            counters.skipped_not_downloadable += 1
            continue

        if already_downloaded(outdir, rec_id):
            counters.skipped_already += 1
            continue

        try:
            xml_bytes = fetch_fulltext_xml(rec_id, session=session)
            raw_xml_path = save_raw_xml(outdir, rec_id, xml_bytes)

            plain = jats_xml_to_text(xml_bytes)
            meta = {
                "DOC_TYPE": doc_type,
                "TITLE": a.title,
                "PMCID": a.pmcid or "",
                "DOI": a.doi or "",
                "ID": f"{a.source}:{a.id}",
                "JOURNAL": a.journal_title or "",
                "YEAR": a.pub_year or "",
                "AUTHORS": a.author_string or "",
            }
            txt_path = save_text(outdir, rec_id, meta, plain)

            counters.downloaded += 1
            log(
                f"Downloaded #{counters.downloaded:,} [{doc_type}] "
                f"(streamed={counters.streamed:,}): {rec_id} | XML={raw_xml_path} | TXT={txt_path}"
            )

            if args.max_downloads and counters.downloaded >= args.max_downloads:
                log("Reached --max-downloads cap; stopping.")
                return

        except requests.HTTPError as e:
            counters.http_errors += 1
            status = e.response.status_code if e.response is not None else "ERR"
            log(f"HTTP error for {rec_id}: {status} (continuing)")
        except Exception as e:  # noqa: BLE001
            counters.other_errors += 1
            log(f"Error for {rec_id}: {e} (continuing)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download Europe PMC OA articles AND preprints (PPR) using cursorMark deep paging."
    )
    ap.add_argument(
        "--query",
        default='("breast cancer") AND OPEN_ACCESS:Y',
        help="Europe PMC query for the ARTICLE pass. Recommended: add OPEN_ACCESS:Y",
    )
    ap.add_argument(
        "--ppr-query",
        default='("breast cancer") AND PUB_TYPE:Preprint AND HAS_FT:Y',
        help="Europe PMC query for the PREPRINT pass. Recommended: PUB_TYPE:Preprint AND HAS_FT:Y",
    )
    ap.add_argument(
        "--no-articles",
        action="store_true",
        help="Skip the open-access article pass (download preprints only).",
    )
    ap.add_argument(
        "--no-preprints",
        action="store_true",
        help="Skip the preprint pass (download articles only).",
    )
    ap.add_argument(
        "--outdir",
        default="artifacts/epmc_fulltext",
        help="Output directory for .txt and raw XML files (shared by articles + preprints).",
    )
    ap.add_argument(
        "--page-size",
        type=int,
        default=1000,
        help="Search page size for cursorMark paging (use 1000 unless you have a reason).",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.1,
        help="Sleep seconds between search pages (politeness / rate limiting).",
    )
    ap.add_argument(
        "--max-downloads",
        type=int,
        default=0,
        help="Safety cap: stop after downloading N full texts total (across both passes). 0 means no cap.",
    )
    ap.add_argument(
        "--manifest",
        default="artifacts/epmc_manifest.jsonl",
        help="JSONL file to append every streamed record (metadata + doc_type).",
    )
    args = ap.parse_args()

    if args.no_articles and args.no_preprints:
        ap.error("--no-articles and --no-preprints together leave nothing to do.")

    outdir = Path(args.outdir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    log_path, log = make_logger(__file__ if "__file__" in globals() else None)
    log(f"Log file: {log_path}")
    log(f"Outdir: {outdir}")
    log(f"Page size: {args.page_size}")
    log(f"Sleep: {args.sleep}s")
    if args.max_downloads:
        log(f"Max downloads (total): {args.max_downloads}")

    session = requests.Session()
    counters = Counters()

    if not args.no_articles:
        run_pass(
            query=args.query,
            doc_type=DOC_TYPE_ARTICLE,
            record_id_of=article_record_id,
            args=args,
            session=session,
            outdir=outdir,
            manifest_path=manifest_path,
            log=log,
            counters=counters,
        )

    capped = bool(args.max_downloads and counters.downloaded >= args.max_downloads)
    if not args.no_preprints and not capped:
        run_pass(
            query=args.ppr_query,
            doc_type=DOC_TYPE_PREPRINT,
            record_id_of=preprint_record_id,
            args=args,
            session=session,
            outdir=outdir,
            manifest_path=manifest_path,
            log=log,
            counters=counters,
        )

    log("Done.")
    log(
        f"Totals: streamed={counters.streamed:,} downloaded={counters.downloaded:,} "
        f"skipped(not downloadable)={counters.skipped_not_downloadable:,} "
        f"skipped(already)={counters.skipped_already:,} "
        f"http_errors={counters.http_errors:,} other_errors={counters.other_errors:,}"
    )
    log(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
