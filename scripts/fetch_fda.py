#!/usr/bin/env python3
"""
Dump openFDA datasets using Search-After paging (Link rel="next").

Downloads:
  1) Drugs@FDA: https://api.fda.gov/drug/drugsfda.json
  2) Drug Labels: https://api.fda.gov/drug/label.json

Why Search-After:
- skip/limit paging only works up to ~26,000 hits.
- Search-After uses the Link header and can scroll through the entire dataset.
Docs: https://open.fda.gov/apis/paging/

Notes:
- Saves each response body as raw bytes (no parsing, no rewriting).
- Also saves response headers per page (so you keep the exact Link header chain).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests

LINK_NEXT_RE = re.compile(r'<([^>]+)>\s*;\s*rel="next"')


def extract_next_link(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    m = LINK_NEXT_RE.search(link_header)
    return m.group(1) if m else None


def save_headers(path: Path, resp: requests.Response) -> None:
    # Save headers as-is (text). No JSON parsing.
    lines = [f"{k}: {v}" for k, v in resp.headers.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stream_save_body(path: Path, resp: requests.Response) -> None:
    # Stream full body to disk (raw, original bytes).
    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def dump_dataset(
    session: requests.Session,
    *,
    name: str,
    endpoint: str,
    search: str,
    sort: str,
    limit: int,
    api_key: Optional[str],
    out_dir: Path,
    sleep_s: float,
    max_pages: int,
) -> Tuple[int, int]:
    """
    Returns (pages_saved, last_status_code).
    """
    pages_dir = out_dir / name / "pages"
    headers_dir = out_dir / name / "headers"
    pages_dir.mkdir(parents=True, exist_ok=True)
    headers_dir.mkdir(parents=True, exist_ok=True)

    base_params = [("limit", str(limit)), ("sort", sort), ("search", search)]
    if api_key:
        base_params.insert(0, ("api_key", api_key))

    headers = {
        "User-Agent": "belladonna-openfda-dump/1.0 (+https://open.fda.gov/)",
        "Accept": "application/json",
    }

    page = 0
    next_url: Optional[str] = None
    last_status = 0

    while True:
        page += 1
        if max_pages and page > max_pages:
            print(f"[{name}] reached --max-pages={max_pages}; stopping.", file=sys.stderr)
            break

        try:
            if next_url:
                resp = session.get(next_url, headers=headers, stream=True, timeout=180)
            else:
                resp = session.get(endpoint, params=base_params, headers=headers, stream=True, timeout=180)
        except requests.RequestException as e:
            print(f"[{name} page {page}] request failed: {e}", file=sys.stderr)
            return (page - 1, 0)

        last_status = resp.status_code

        body_path = pages_dir / f"{name}_page_{page:06d}.json"
        hdrs_path = headers_dir / f"{name}_page_{page:06d}.headers.txt"

        save_headers(hdrs_path, resp)
        stream_save_body(body_path, resp)

        print(f"[{name} page {page}] status={resp.status_code} saved={body_path}", file=sys.stderr)

        if not resp.ok:
            print(f"[{name} page {page}] non-OK response; stopping (raw body saved).", file=sys.stderr)
            break

        next_url = extract_next_link(resp.headers.get("Link"))
        if not next_url:
            print(f"[{name}] no Link rel='next'; finished.", file=sys.stderr)
            break

        if sleep_s > 0:
            time.sleep(sleep_s)

    return (page, last_status)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("OPENFDA_API_KEY"))
    ap.add_argument("--out-dir", default="artifacts/fda")
    ap.add_argument("--limit", type=int, default=1000, help="Max 1000 (openFDA limit).")
    ap.add_argument("--sleep", type=float, default=0.15, help="Polite delay between requests.")
    ap.add_argument("--max-pages", type=int, default=0, help="0 = no limit (debugging: set >0).")
    args = ap.parse_args()

    if not (1 <= args.limit <= 1000):
        print("Error: --limit must be between 1 and 1000.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        # 1) Drugs@FDA: match-all via _exists_:application_number; sort by application_number
        dump_dataset(
            session,
            name="drugsfda",
            endpoint="https://api.fda.gov/drug/drugsfda.json",
            search="_exists_:application_number",
            sort="application_number:asc",
            limit=args.limit,
            api_key=args.api_key,
            out_dir=out_dir,
            sleep_s=args.sleep,
            max_pages=args.max_pages,
        )

        # 2) Drug Labels: match-all via _exists_:id; sort by effective_time (label version date)
        dump_dataset(
            session,
            name="label",
            endpoint="https://api.fda.gov/drug/label.json",
            search="_exists_:id",
            sort="effective_time:asc",
            limit=args.limit,
            api_key=args.api_key,
            out_dir=out_dir,
            sleep_s=args.sleep,
            max_pages=args.max_pages,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
