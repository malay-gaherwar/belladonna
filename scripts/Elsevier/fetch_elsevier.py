#!/usr/bin/env python3
"""
Elsevier / Scopus ingestion pipeline (LOSSLESS, CURSOR PAGINATION).

Implements cursor pagination EXACTLY as documented by Elsevier:
- First request uses cursor=*
- Subsequent requests follow the 'link ref=next' URL verbatim
- No offset paging
- No fallback

FULLTEXT:
  artifacts/elsevier/xml/<SCOPUS_ID>.xml

META-ONLY:
  artifacts/elsevier/only_meta/<SCOPUS_ID>.json
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import requests


# ======================================================================
# CONFIG
# ======================================================================
CONFIG = {
    "SCOPUS_SEARCH_BASE": "https://api.elsevier.com/content/search/scopus",
    "ARTICLE_RETRIEVAL_BASE": "https://api.elsevier.com/content/article",
    "TIMEOUT": 60,

    "QUERY": 'TITLE-ABS-KEY("breast cancer")',
    "COUNT": 25,                 # max allowed
    "SLEEP_SECONDS": 0.25,

    # >>> NEW <<<
    "START_STREAMED": 123400,    # restart from here (set to 0 for fresh run)
    "MAX_STREAM": 0,             # 0 = unlimited

    "OUTDIR": "artifacts/elsevier",
    "LOG_PREFIX": "fetch_elsevier",

    # >>> NEW <<<
    "MAX_RETRIES": 6,            # for 5xx errors
    "BACKOFF_BASE": 1.0,         # seconds
}


# ======================================================================
# Logging
# ======================================================================
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS_DIR / f"{CONFIG['LOG_PREFIX']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ======================================================================
# Helpers
# ======================================================================
def safe_id(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def extract_scopus_id(raw: Optional[str]) -> str:
    if not raw:
        return ""
    return raw.replace("SCOPUS_ID:", "").strip()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _repair_next_url_if_needed(next_url: str, query: str, count: int) -> str:
    """
    Some rare Scopus API responses return a malformed next link missing `query=...`
    (e.g. .../search/scopus?cursor=XYZ). Scopus Search requires `query`.
    If missing, we add it back (and count if missing) without changing anything else.
    """
    parsed = urlparse(next_url)
    qs = parse_qs(parsed.query, keep_blank_values=True)

    changed = False

    if "query" not in qs or not qs["query"] or not qs["query"][0]:
        qs["query"] = [query]
        changed = True

    if "count" not in qs or not qs["count"] or not qs["count"][0]:
        qs["count"] = [str(count)]
        changed = True

    if not changed:
        return next_url

    new_query = urlencode(qs, doseq=True)
    repaired = urlunparse(parsed._replace(query=new_query))
    return repaired


# ======================================================================
# Cursor-based Scopus Search (OFFICIAL METHOD + RETRIES)
# ======================================================================
def iter_scopus_search(
    session: requests.Session,
    api_key: str,
    query: str,
    count: int,
    sleep_s: float,
) -> Iterator[Dict[str, Any]]:
    """
    Cursor pagination following Elsevier documentation:
    - Start with cursor=*
    - Then follow link ref='next' exactly
    - Retries on HTTP 5xx with exponential backoff
    """

    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json",
    }

    next_url = (
        f"{CONFIG['SCOPUS_SEARCH_BASE']}?"
        f"query={query.replace(' ', '+')}&cursor=*&count={count}&sort=coverDate"
    )

    page = 0

    while next_url:
        page += 1
        log(f"Fetching page {page}")

        # >>> NEW: retry loop <<<
        for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
            try:
                r = session.get(next_url, headers=headers, timeout=CONFIG["TIMEOUT"])
                r.raise_for_status()
                break
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else None
                if status and 500 <= status < 600 and attempt < CONFIG["MAX_RETRIES"]:
                    backoff = (
                        CONFIG["BACKOFF_BASE"]
                        * (2 ** (attempt - 1))
                        + random.uniform(0, 0.3)
                    )
                    log(
                        f"HTTP {status} on page fetch "
                        f"(attempt {attempt}/{CONFIG['MAX_RETRIES']}), "
                        f"retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    continue
                raise

        data = r.json()
        # Add this to iter_scopus_search in your script:
        actual_query = data.get('search-results', {}).get('opensearch:Query', {}).get('@searchTerms')
        log(f"API verified query: {actual_query}")
        sr = data.get("search-results", {}) or {}
        entries = sr.get("entry", []) or []
        if not entries:
            return

        for e in entries:
            yield e

        next_url = None
        for link in sr.get("link", []):
            if link.get("@ref") == "next" and link.get("@href"):
                next_url = link["@href"]
                break
        if next_url:
            # Scopus occasionally drops `query=` from the next link; re-inject it.
            next_url = _repair_next_url_if_needed(next_url, query=query, count=count)
            log(f"Next URL: {next_url}")
        if sleep_s > 0:
            time.sleep(sleep_s)


# ======================================================================
# Article Retrieval API (RAW XML)
# ======================================================================
def fetch_fulltext_xml(
    session: requests.Session,
    api_key: str,
    doi: str,
    pii: str,
) -> bytes:
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/xml",
    }
    params = {"view": "FULL"}

    if doi:
        url = f"{CONFIG['ARTICLE_RETRIEVAL_BASE']}/doi/{doi}"
    elif pii:
        url = f"{CONFIG['ARTICLE_RETRIEVAL_BASE']}/pii/{pii}"
    else:
        raise ValueError("No DOI/PII")

    r = session.get(url, headers=headers, params=params, timeout=CONFIG["TIMEOUT"])
    r.raise_for_status()
    return r.content


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    api_key = os.getenv("ELSEVIER_API_KEY")
    if not api_key:
        log("ERROR: ELSEVIER_API_KEY not set")
        sys.exit(1)

    outdir = Path(CONFIG["OUTDIR"])
    ensure_dir(outdir / "xml")
    ensure_dir(outdir / "only_meta")

    session = requests.Session()

    streamed = 0
    downloaded = 0
    saved_full = 0
    saved_meta = 0
    skipped_no_id = 0
    errors = 0

    log(f"QUERY: {CONFIG['QUERY']}")
    log("Pagination: CURSOR (official)")
    log(f"COUNT: {CONFIG['COUNT']}")
    log(f"START_STREAMED: {CONFIG['START_STREAMED']}")

    for entry in iter_scopus_search(
        session=session,
        api_key=api_key,
        query=CONFIG["QUERY"],
        count=CONFIG["COUNT"],
        sleep_s=CONFIG["SLEEP_SECONDS"],
    ):
        streamed += 1

        # >>> NEW: restart logic <<<
        if streamed <= CONFIG["START_STREAMED"]:
            continue

        if CONFIG["MAX_STREAM"] and streamed > CONFIG["MAX_STREAM"]:
            break

        scopus_id = extract_scopus_id(entry.get("dc:identifier"))
        if not scopus_id:
            skipped_no_id += 1
            continue

        scid = safe_id(scopus_id)
        xml_path = outdir / "xml" / f"{scid}.xml"
        meta_path = outdir / "only_meta" / f"{scid}.json"

        downloaded += 1

        try:
            xml = fetch_fulltext_xml(
                session,
                api_key,
                entry.get("prism:doi", ""),
                entry.get("pii", ""),
            )
            xml_path.write_bytes(xml)
            saved_full += 1

            log(
                f"Downloaded #{downloaded:,} "
                f"(streamed={streamed:,}, full-text={saved_full:,}, meta={saved_meta:,}): "
                f"{scopus_id} | XML"
            )

        except requests.HTTPError:
            meta_path.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved_meta += 1

            log(
                f"Downloaded #{downloaded:,} "
                f"(streamed={streamed:,}, full-text={saved_full:,}, meta={saved_meta:,}): "
                f"{scopus_id} | META_ONLY"
            )

        except Exception as e:
            errors += 1
            log(f"ERROR processing {scopus_id}: {e}")

    log("Finished.")
    log(
        "Totals: "
        f"streamed={streamed:,}, "
        f"downloaded={downloaded:,}, "
        f"saved full-text={saved_full:,}, "
        f"saved meta-data={saved_meta:,}, "
        f"skipped(no id)={skipped_no_id:,}, "
        f"errors={errors:,}"
    )


if __name__ == "__main__":
    main()
