#!/usr/bin/env python3
"""
Query openFDA for the phrase "breast cancer" and save the first N results
as *raw* JSON (no parsing / reformatting).

Docs:
- Query parameters (search/limit/skip): https://open.fda.gov/apis/query-parameters/
- Paging + Search-After (Link header): https://open.fda.gov/apis/paging/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import requests


DEFAULT_ENDPOINT = "https://api.fda.gov/drug/event.json"
DEFAULT_OUT_DIR = Path("artifacts") / "fda"
DEFAULT_OUT_FILE = "openfda_breast_cancer_first10.json"


def build_params(api_key: Optional[str], phrase: str, limit: int) -> List[Tuple[str, str]]:
    # Phrase match uses double quotes: "breast cancer"
    search_value = f"\"{phrase}\""
    params: List[Tuple[str, str]] = []
    if api_key:
        params.append(("api_key", api_key))
    params.append(("search", search_value))
    params.append(("limit", str(limit)))
    return params


def main() -> int:
    parser = argparse.ArgumentParser(description="Query openFDA and save raw JSON response.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"openFDA endpoint (default: {DEFAULT_ENDPOINT})")
    parser.add_argument("--phrase", default="breast cancer", help='Phrase to search (default: "breast cancer")')
    parser.add_argument("--limit", type=int, default=10, help="Number of results to retrieve (default: 10)")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENFDA_API_KEY"),
        help="openFDA API key (default: env var OPENFDA_API_KEY).",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_DIR / DEFAULT_OUT_FILE),
        help=f'Output file path (default: "{DEFAULT_OUT_DIR / DEFAULT_OUT_FILE}")',
    )

    args = parser.parse_args()

    if args.limit < 1 or args.limit > 1000:
        print("Error: --limit must be between 1 and 1000.", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    params = build_params(args.api_key, args.phrase, args.limit)

    headers = {
        "User-Agent": "openfda-raw-fetch/1.0 (+https://open.fda.gov/)",
        "Accept": "application/json",
    }

    try:
        resp = requests.get(args.endpoint, params=params, headers=headers, timeout=60)
    except requests.RequestException as e:
        print(f"Request failed: {e}", file=sys.stderr)
        return 1

    # Save EXACT bytes returned by the server (raw/original format).
    out_path.write_bytes(resp.content)

    # Print where it was saved; keep stdout clean for piping if you want—use stderr.
    print(f"Saved raw response to: {out_path}", file=sys.stderr)

    return 0 if resp.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
