#!/usr/bin/env python3
"""
EPMC fulltext downloader with cursorMark deep paging + resume options.

Resume options:
1) Best (exact): provide --start-cursor <cursorMark token>
2) Fallback: provide --start-streamed N --skip-streamed
   (This will re-stream from the beginning but do nothing until item N is reached.)

Defaults:
- start_streamed defaults to 66129 and skip-streamed is enabled by default,
  so it effectively resumes at ~66,129th streamed record.

Outputs:
- artifacts/epmc_fulltext/PMCxxxxxxx.txt
- artifacts/epmc_fulltext/xml/PMCxxxxxxx.xml
- logs/<script>_YYYYmmdd_HHMMSS.log
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Union, cast

import requests
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---- Requests typing helpers (keeps mypy happy) ----
ParamScalar = Union[str, bytes, int, float]
ParamValue = Union[ParamScalar, Sequence[ParamScalar], None]
Params = Mapping[str, ParamValue]

SEARCH_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

# Your requested starting number (fallback resume mode)
START_STREAMED_DEFAULT = 66129


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


def make_logger(script_path: str | None) -> tuple[Path, callable]:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(script_path).stem if script_path else "epmc_fulltext"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{stem}_{ts}.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log_path, log


def build_session() -> requests.Session:
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=100, pool_maxsize=100)

    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update(
        {"User-Agent": "belladonna-epmc-harvester/1.0 (+https://github.com/KatherLab/belladonna)"}
    )
    return s


def epmc_search_page(
    query: str,
    page_size: int,
    cursor_mark: str,
    session: requests.Session,
    timeout_s: float,
) -> tuple[list[Article], str]:
    params: Params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "resultType": "lite",
        "cursorMark": cursor_mark,
    }
    r = session.get(SEARCH_BASE, params=params, timeout=timeout_s)
    r.raise_for_status()

    data = cast(dict[str, Any], r.json())
    results = cast(list[dict[str, Any]], (data.get("resultList") or {}).get("result", []) or [])
    next_cursor = cast(str, data.get("nextCursorMark", ""))

    return [Article.from_epmc(x) for x in results], next_cursor


def iter_epmc_all(
    query: str,
    page_size: int,
    sleep_s: float,
    session: requests.Session,
    timeout_s: float,
    start_cursor: str,
) -> Iterator[Article]:
    cursor = start_cursor
    while True:
        page, next_cursor = epmc_search_page(
            query=query,
            page_size=page_size,
            cursor_mark=cursor,
            session=session,
            timeout_s=timeout_s,
        )
        if not page:
            return

        for art in page:
            yield art

        if not next_cursor or next_cursor == cursor:
            return

        cursor = next_cursor
        if sleep_s > 0:
            time.sleep(sleep_s)


def fetch_fulltext_xml(pmcid: str, session: requests.Session, timeout_s: float) -> bytes:
    url = FULLTEXT_BASE.format(pmcid=pmcid)
    r = session.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def _itertext(elem: ET.Element) -> str:
    text_parts: list[str] = []
    for t in elem.itertext():
        s = " ".join(t.split())
        if s:
            text_parts.append(s)
    return " ".join(text_parts)


def _findall_anyns(root: ET.Element, tag: str) -> list[ET.Element]:
    return list(root.findall(f".//{{*}}{tag}"))


def jats_xml_to_text(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes)

    titles = _findall_anyns(root, "article-title")
    title = _itertext(titles[0]) if titles else ""

    abstracts = _findall_anyns(root, "abstract")
    abstract_txt = "\n\n".join(_itertext(a) for a in abstracts) if abstracts else ""

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


def save_text(outdir: Path, pmcid: str, header_meta: Mapping[str, str], content: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{pmcid}.txt"

    header_lines = [f"{k}: {v}" for k, v in header_meta.items() if v]
    header = "\n".join(header_lines)
    sep = "\n" + ("-" * 80) + "\n"

    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        f.write(sep)
        f.write(content)
        f.write("\n")

    return path


def save_raw_xml(outdir: Path, pmcid: str, xml_bytes: bytes) -> Path:
    xml_dir = outdir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)
    path = xml_dir / f"{pmcid}.xml"
    with path.open("wb") as f:
        f.write(xml_bytes)
    return path


def already_downloaded(outdir: Path, pmcid: str) -> bool:
    txt_path = outdir / f"{pmcid}.txt"
    xml_path = outdir / "xml" / f"{pmcid}.xml"
    return txt_path.exists() and xml_path.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Europe PMC full texts using cursorMark deep paging (resume-friendly).")
    ap.add_argument(
        "--query",
        default='("breast cancer") AND OPEN_ACCESS:Y',
        help="Europe PMC query string. Recommended for full-text harvest: add OPEN_ACCESS:Y",
    )
    ap.add_argument("--outdir", default="artifacts/epmc_fulltext", help="Output directory for .txt and raw XML files.")
    ap.add_argument("--page-size", type=int, default=1000, help="Search page size for cursorMark paging.")
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep seconds between search pages.")
    ap.add_argument("--timeout-search", type=float, default=30.0, help="Timeout seconds for /search calls.")
    ap.add_argument("--timeout-fulltext", type=float, default=90.0, help="Timeout seconds for fullTextXML calls.")
    ap.add_argument("--max-downloads", type=int, default=0, help="Stop after N full texts. 0 means no cap.")
    ap.add_argument("--manifest", default="artifacts/epmc_manifest.jsonl", help="JSONL file to append every streamed record (metadata).")

    # Resume controls
    ap.add_argument(
        "--start-cursor",
        default="",
        help="Exact resume: start from this cursorMark token (best option if you saved it).",
    )
    ap.add_argument(
        "--start-streamed",
        type=int,
        default=START_STREAMED_DEFAULT,
        help=f"Fallback resume: pretend we already streamed N records (default {START_STREAMED_DEFAULT}).",
    )
    ap.add_argument(
        "--skip-streamed",
        action="store_true",
        default=True,
        help="Fallback resume: while streamed < start-streamed, do NOT download anything (just advance). Default: enabled.",
    )

    args = ap.parse_args()

    log_path, log = make_logger(__file__ if "__file__" in globals() else None)
    log(f"Log file: {log_path}")
    log(f"Query: {args.query}")
    log(f"Outdir: {args.outdir}")

    if args.start_cursor:
        log(f"Starting from cursorMark token (exact resume).")
    else:
        log(f"Starting from streamed={args.start_streamed:,} using skip-streamed={args.skip_streamed} (fallback resume).")
        log("NOTE: This will re-stream from the beginning, but skip processing until the counter is reached.")

    outdir = Path(args.outdir)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    session = build_session()

    streamed = 0
    downloaded = 0
    skipped_no_pmcid_or_not_oa = 0
    skipped_already = 0
    http_errors = 0
    other_errors = 0

    start_cursor = args.start_cursor if args.start_cursor else "*"

    for a in iter_epmc_all(
        args.query,
        page_size=args.page_size,
        sleep_s=args.sleep,
        session=session,
        timeout_s=args.timeout_search,
        start_cursor=start_cursor,
    ):
        streamed += 1

        # Manifest (best effort)
        try:
            with manifest_path.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(a.__dict__, ensure_ascii=False) + "\n")
        except Exception:
            pass

        # Progress
        if streamed % 1000 == 0:
            log(
                f"Progress: streamed={streamed:,} | downloaded={downloaded:,} | "
                f"skipped(no OA/PMCID)={skipped_no_pmcid_or_not_oa:,} | skipped(already)={skipped_already:,} | "
                f"http_errors={http_errors:,} | other_errors={other_errors:,}"
            )

        # Fallback resume mode: do nothing until we reach the requested starting count
        if not args.start_cursor and args.skip_streamed and streamed < args.start_streamed:
            continue

        if not (a.is_open_access and a.pmcid):
            skipped_no_pmcid_or_not_oa += 1
            continue

        if already_downloaded(outdir, a.pmcid):
            skipped_already += 1
            continue

        try:
            xml_bytes = fetch_fulltext_xml(a.pmcid, session=session, timeout_s=args.timeout_fulltext)
            raw_xml_path = save_raw_xml(outdir, a.pmcid, xml_bytes)

            plain = jats_xml_to_text(xml_bytes)
            meta = {
                "TITLE": a.title,
                "PMCID": a.pmcid,
                "DOI": a.doi or "",
                "ID": f"{a.source}:{a.id}",
                "JOURNAL": a.journal_title or "",
                "YEAR": a.pub_year or "",
                "AUTHORS": a.author_string or "",
            }
            txt_path = save_text(outdir, a.pmcid, meta, plain)

            downloaded += 1
            log(f"Downloaded #{downloaded:,} (streamed={streamed:,}): {a.pmcid} | XML={raw_xml_path} | TXT={txt_path}")

            if args.max_downloads and downloaded >= args.max_downloads:
                log("Reached --max-downloads cap; stopping.")
                break

        except requests.HTTPError as e:
            http_errors += 1
            status = e.response.status_code if e.response is not None else "ERR"
            log(f"HTTP error for {a.pmcid}: {status} (continuing)")
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.Timeout,
        ) as e:
            other_errors += 1
            log(f"Transient network error for {a.pmcid}: {type(e).__name__}: {e} (continuing)")
            time.sleep(2.0)
        except Exception as e:  # noqa: BLE001
            other_errors += 1
            log(f"Error for {a.pmcid}: {e} (continuing)")

    log("Done.")
    log(
        f"Totals: streamed={streamed:,} downloaded={downloaded:,} "
        f"skipped(no OA/PMCID)={skipped_no_pmcid_or_not_oa:,} skipped(already)={skipped_already:,} "
        f"http_errors={http_errors:,} other_errors={other_errors:,}"
    )


if __name__ == "__main__":
    main()
