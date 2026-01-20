#!/usr/bin/env python3
"""
Elsevier / Scopus ingestion pipeline (LOSSLESS, OFFSET PAGINATION).

FULLTEXT:
  artifacts/elsevier/xml/<SCOPUS_ID>.xml

META-ONLY:
  artifacts/elsevier/only_meta/<SCOPUS_ID>.json

Notes:
- Offset-based pagination (start/count)
- Meta-only is expected and NOT an error
- Raw formats only (no parsing, no transformation)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import requests


# ======================================================================
# CONFIG
# ======================================================================
CONFIG = {
    "SCOPUS_SEARCH_BASE": "https://api.elsevier.com/content/search/scopus",
    "ARTICLE_RETRIEVAL_BASE": "https://api.elsevier.com/content/article",
    "TIMEOUT": 60,

    "QUERY": "breast cancer",
    "PAGE_SIZE": 25,          # Scopus max
    "SLEEP_SECONDS": 0.25,

    "START_AT": 0,            # resume offset
    "MAX_STREAM": 0,          # 0 = unlimited

    "OUTDIR": "artifacts/elsevier",
    "LOG_PREFIX": "fetch_elsevier",
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


# ======================================================================
# Offset-based Scopus Search
# ======================================================================
def iter_scopus_search(
    session: requests.Session,
    api_key: str,
    query: str,
    page_size: int,
    sleep_s: float,
) -> Iterator[Dict[str, Any]]:
    start = 0

    while True:
        params = {
            "query": query,
            "count": page_size,
            "start": start,
            "view": "COMPLETE",
        }
        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json",
        }

        r = session.get(
            CONFIG["SCOPUS_SEARCH_BASE"],
            headers=headers,
            params=params,
            timeout=CONFIG["TIMEOUT"],
        )
        r.raise_for_status()
        data = r.json()

        entries = data.get("search-results", {}).get("entry", []) or []
        if not entries:
            return

        for e in entries:
            yield e

        start += len(entries)
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
    log(f"PAGE_SIZE: {CONFIG['PAGE_SIZE']}")
    log(f"START_AT: {CONFIG['START_AT']}")
    log(f"MAX_STREAM: {CONFIG['MAX_STREAM'] or 'unlimited'}")

    for entry in iter_scopus_search(
        session=session,
        api_key=api_key,
        query=CONFIG["QUERY"],
        page_size=CONFIG["PAGE_SIZE"],
        sleep_s=CONFIG["SLEEP_SECONDS"],
    ):
        streamed += 1

        if streamed <= CONFIG["START_AT"]:
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
                f"(streamed={streamed:,}, saved full-text={saved_full:,}, saved meta-data={saved_meta:,}): "
                f"{scopus_id} | XML={xml_path}"
            )

        except requests.HTTPError:
            meta_path.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved_meta += 1

            log(
                f"Downloaded #{downloaded:,} "
                f"(streamed={streamed:,}, saved full-text={saved_full:,}, saved meta-data={saved_meta:,}): "
                f"{scopus_id} | META_ONLY={meta_path}"
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
