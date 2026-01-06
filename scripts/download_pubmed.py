#!/usr/bin/env python3
"""
Europe PMC full-text fetcher with DEBUG mode.
- Prints request URL, status, headers, and hitCount.
- Falls back to a simpler query and to POST if GET looks blocked.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ParamValue = Union[str, int, float, None]

DEFAULT_HEADERS = {
    "User-Agent": "belladonna/0.1 (+https://example.org) requests",
    "Accept": "application/json",
}

def _pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)[:1200]
    except Exception:
        return str(obj)[:1200]

def epmc_search(query: str, rows: int = 5, debug: bool = False) -> List[Dict[str, Any]]:
    params: Dict[str, ParamValue] = {
        "query": query,
        "format": "json",
        "pageSize": rows,
        "sort": "P_PDATE_D",
        # You can force OA only by adding: "resultType": "core"
        # and using query += " AND OPEN_ACCESS:Y"
    }
    url = f"{EPMC_BASE}/search"
    try:
        r = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=30)
        if debug:
            print(f"[GET] {r.url}")
            print(f"[GET] status={r.status_code}")
            print(f"[GET] headers={_pretty(dict(r.headers))}")
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        if debug:
            # Show first 800 chars of body to detect captive portals / proxy HTML
            try:
                body = r.text[:800] if 'r' in locals() else "(no response object)"
                print("[GET] error:", e)
                print("[GET] body:", body)
            except Exception:
                pass
        # Fallback: some networks dislike query params on GET; try POST endpoint
        if debug:
            print("[POST] Falling back to POST /search with same params")
        r2 = requests.post(url, data=params, headers=DEFAULT_HEADERS, timeout=30)
        if debug:
            print(f"[POST] {r2.url} status={r2.status_code}")
            print(f"[POST] headers={_pretty(dict(r2.headers))}")
        r2.raise_for_status()
        data = r2.json()

    hit_count = int(data.get("hitCount", 0))
    if debug:
        print(f"[INFO] hitCount={hit_count}")
        if hit_count == 0:
            print("[INFO] raw payload preview:", _pretty(data))

    return data.get("resultList", {}).get("result", [])

def epmc_fulltext_xml(pmcid: str, debug: bool = False) -> Optional[str]:
    url = f"{EPMC_BASE}/{pmcid}/fullTextXML"
    r = requests.get(url, headers=DEFAULT_HEADERS, timeout=60)
    if debug:
        print(f"[XML] GET {r.url} status={r.status_code}")
    if r.status_code == 200 and r.text.strip():
        return r.text
    return None

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="breast cancer", help="Europe PMC query")
    ap.add_argument("--outdir", default="artifacts/epmc_fulltext")
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--oa-only", action="store_true", help="restrict to open-access items")
    args = ap.parse_args()

    query = args.query
    if args.oa_only:
        # Europe PMC OA filter (documented): OPEN_ACCESS:Y
        query = f"({query}) AND OPEN_ACCESS:Y"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Searching Europe PMC for: {query!r}")
    results = epmc_search(query, rows=args.rows, debug=args.debug)
    if not results:
        print("No results found.")
        return

    for i, rec in enumerate(results, 1):
        title = (rec.get("title") or "").strip().replace("\n", " ")
        pmcid = rec.get("pmcid")
        doi = rec.get("doi")
        src = rec.get("source")  # e.g., MED, PMC, AGR, bioRxiv, etc.
        print(f"{i}. {title[:90]}...")
        print(f"   DOI: {doi or '—'}  PMCID: {pmcid or '—'}  source: {src or '—'}")

        if pmcid:
            xml_text = epmc_fulltext_xml(pmcid, debug=args.debug)
            if xml_text:
                xml_path = outdir / f"{pmcid}.xml"
                xml_path.write_text(xml_text, encoding="utf-8")
                print(f"   → saved XML to {xml_path}")
            else:
                print("   (PMCID present but XML not returned—may not be OA, or transient issue)")

    print(f"Done. Output directory: {outdir.resolve()}")

if __name__ == "__main__":
    main()
