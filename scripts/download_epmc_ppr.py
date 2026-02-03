#!/usr/bin/env python3
"""
Download Europe PMC PREPRINT full text XML (PPR...) via /{id}/fullTextXML.

- Discovers records using /search + cursorMark deep paging
- Prints HIT COUNT at the start (how many records match the query)
- Downloads ONLY raw XML (no TXT extraction)
- Saves:
    artifacts/epmc_fulltext/preprints/xml/PPRxxxxxxx.xml
- Logs:
    logs/<script>_YYYYmmdd_HHMMSS.log

Default query targets breast-cancer preprints with full text:
    ("breast cancer") AND PUB_TYPE:Preprint AND HAS_FT:Y

Examples:
  python scripts/epmc_preprints_xml.py
  python scripts/epmc_preprints_xml.py --max-downloads 500
  python scripts/epmc_preprints_xml.py --query '("breast cancer") AND PUB_TYPE:Preprint AND HAS_FT:Y AND OPEN_ACCESS:Y'
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, Union, cast

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -----------------------------
# Config / endpoints
# -----------------------------
ParamScalar = Union[str, bytes, int, float]
ParamValue = Union[ParamScalar, Sequence[ParamScalar], None]
Params = Mapping[str, ParamValue]

SEARCH_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
FULLTEXT_BY_ID = "https://www.ebi.ac.uk/europepmc/webservices/rest/{id}/fullTextXML"


# -----------------------------
# Data model
# -----------------------------
@dataclass(frozen=True)
class Preprint:
    id: str               # e.g., PPR1006247
    source: str           # should be "PPR"
    title: str | None
    pub_year: str | None
    doi: str | None

    @staticmethod
    def from_epmc(d: Mapping[str, Any]) -> "Preprint":
        return Preprint(
            id=str(d.get("id", "")),
            source=str(d.get("source", "")),
            title=cast(str | None, d.get("title")),
            pub_year=cast(str | None, d.get("pubYear")),
            doi=cast(str | None, d.get("doi")),
        )


# -----------------------------
# Logging
# -----------------------------
def make_logger(script_path: str | None) -> tuple[Path, callable]:
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(script_path).stem if script_path else "epmc_preprints_xml"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{stem}_{ts}.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log_path, log


# -----------------------------
# HTTP session with retries
# -----------------------------
def build_session() -> requests.Session:
    retry = Retry(
        total=10,
        connect=10,
        read=10,
        status=10,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "belladonna-epmc-preprints/1.0"})
    return s


# -----------------------------
# CursorMark paging
# -----------------------------
def epmc_search_page(
    *,
    query: str,
    page_size: int,
    cursor_mark: str,
    sort: str,
    session: requests.Session,
    timeout_s: float,
) -> tuple[list[Preprint], str, int]:
    params: Params = {
        "query": query,
        "format": "json",
        "resultType": "lite",
        "pageSize": page_size,
        "cursorMark": cursor_mark,
        "sort": sort,  # stable sort helps deep paging
    }
    r = session.get(SEARCH_BASE, params=params, timeout=timeout_s)
    r.raise_for_status()
    data = cast(dict[str, Any], r.json())

    hit_count = int(data.get("hitCount", 0) or 0)
    results = cast(list[dict[str, Any]], (data.get("resultList") or {}).get("result", []) or [])
    next_cursor = cast(str, data.get("nextCursorMark", ""))

    return [Preprint.from_epmc(x) for x in results], next_cursor, hit_count


def iter_preprints(
    *,
    query: str,
    page_size: int,
    sleep_s: float,
    sort: str,
    session: requests.Session,
    timeout_s: float,
    start_cursor: str,
) -> Iterator[Preprint]:
    cursor = start_cursor
    while True:
        page, next_cursor, _hit = epmc_search_page(
            query=query,
            page_size=page_size,
            cursor_mark=cursor,
            sort=sort,
            session=session,
            timeout_s=timeout_s,
        )
        if not page:
            return

        for rec in page:
            yield rec

        if not next_cursor or next_cursor == cursor:
            return

        cursor = next_cursor
        if sleep_s > 0:
            time.sleep(sleep_s)


# -----------------------------
# Fulltext retrieval
# -----------------------------
def fetch_fulltext_xml_by_id(rec_id: str, session: requests.Session, timeout_s: float) -> bytes:
    url = FULLTEXT_BY_ID.format(id=rec_id)
    r = session.get(url, timeout=timeout_s)
    r.raise_for_status()
    return r.content


def already_downloaded(xml_dir: Path, ppr_id: str) -> bool:
    return (xml_dir / f"{ppr_id}.xml").exists()


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Download Europe PMC preprint (PPR) fullTextXML as raw XML only.")
    ap.add_argument(
        "--query",
        default='("breast cancer") AND PUB_TYPE:Preprint AND HAS_FT:Y',
        help="Europe PMC query. Recommended: PUB_TYPE:Preprint AND HAS_FT:Y (add OPEN_ACCESS:Y if you want).",
    )
    ap.add_argument(
        "--outdir",
        default="artifacts/epmc_fulltext/preprints",
        help="Base output directory.",
    )
    ap.add_argument("--page-size", type=int, default=1000, help="Search page size for cursorMark paging.")
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep seconds between search pages.")
    ap.add_argument("--timeout-search", type=float, default=30.0, help="Timeout seconds for /search calls.")
    ap.add_argument("--timeout-fulltext", type=float, default=90.0, help="Timeout seconds for /{id}/fullTextXML calls.")
    ap.add_argument("--sort", default="P_PDATE_D asc", help="Stable sort for cursorMark deep paging.")
    ap.add_argument("--max-downloads", type=int, default=0, help="Stop after N downloads. 0 = no cap.")
    ap.add_argument(
        "--start-cursor",
        default="*",
        help="Resume: cursorMark token. Default '*' (start).",
    )
    args = ap.parse_args()

    log_path, log = make_logger(__file__ if "__file__" in globals() else None)
    log(f"Log file: {log_path}")
    log(f"Query: {args.query}")
    log(f"Outdir: {args.outdir}")
    log(f"pageSize={args.page_size} sleep={args.sleep}s sort={args.sort}")
    log(f"start-cursor={args.start_cursor}")

    base_outdir = Path(args.outdir)
    xml_dir = base_outdir / "xml"
    xml_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()

    # ---- Print HIT COUNT at the start ----
    try:
        _page0, _next0, hit_count = epmc_search_page(
            query=args.query,
            page_size=1,           # tiny page, just to get hitCount
            cursor_mark="*",
            sort=args.sort,
            session=session,
            timeout_s=args.timeout_search,
        )
        log(f"HIT COUNT (matching records): {hit_count:,}")
        log("NOTE: This is the number of matching preprint records for the query; actual downloadable XML may be <= this.")
    except Exception as e:  # noqa: BLE001
        log(f"Could not fetch hitCount (continuing anyway): {e}")

    streamed = 0
    downloaded = 0
    skipped_non_ppr = 0
    skipped_existing = 0
    http_errors = 0
    other_errors = 0

    for rec in iter_preprints(
        query=args.query,
        page_size=args.page_size,
        sleep_s=args.sleep,
        sort=args.sort,
        session=session,
        timeout_s=args.timeout_search,
        start_cursor=args.start_cursor,
    ):
        streamed += 1

        # Only preprints
        if rec.source != "PPR" or not rec.id.startswith("PPR"):
            skipped_non_ppr += 1
            continue

        if already_downloaded(xml_dir, rec.id):
            skipped_existing += 1
            continue

        try:
            xml_bytes = fetch_fulltext_xml_by_id(rec.id, session=session, timeout_s=args.timeout_fulltext)
            (xml_dir / f"{rec.id}.xml").write_bytes(xml_bytes)

            downloaded += 1
            log(f"Downloaded #{downloaded:,} (streamed={streamed:,}): {rec.id} | year={rec.pub_year or ''} | doi={rec.doi or ''}")

            if args.max_downloads and downloaded >= args.max_downloads:
                log("Reached --max-downloads cap; stopping.")
                break

        except requests.HTTPError as e:
            http_errors += 1
            status = e.response.status_code if e.response is not None else "ERR"
            log(f"HTTP error for {rec.id}: {status} (continuing)")
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ReadTimeout,
            requests.exceptions.Timeout,
        ) as e:
            other_errors += 1
            log(f"Transient network error for {rec.id}: {type(e).__name__}: {e} (continuing)")
            time.sleep(2.0)
        except Exception as e:  # noqa: BLE001
            other_errors += 1
            log(f"Error for {rec.id}: {e} (continuing)")

        if streamed % 2000 == 0:
            log(
                f"Progress: streamed={streamed:,} downloaded={downloaded:,} "
                f"skipped_non_ppr={skipped_non_ppr:,} skipped_existing={skipped_existing:,} "
                f"http_errors={http_errors:,} other_errors={other_errors:,}"
            )

    log("Done.")
    log(
        f"Totals: streamed={streamed:,} downloaded={downloaded:,} "
        f"skipped_non_ppr={skipped_non_ppr:,} skipped_existing={skipped_existing:,} "
        f"http_errors={http_errors:,} other_errors={other_errors:,}"
    )


if __name__ == "__main__":
    main()
