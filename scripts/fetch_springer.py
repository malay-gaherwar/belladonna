#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path

import requests

META_ENDPOINT = "https://api.springernature.com/meta/v2/json"
OA_JATS_ENDPOINT = "https://api.springernature.com/openaccess/jats"


def require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise SystemExit(f"Missing env var {name}. Did you export it in your bashrc?")
    return val


def meta_search(query: str, n: int, start: int, api_key: str, timeout: int = 30) -> dict:
    # Meta API uses q, p (page size), s (start offset) :contentReference[oaicite:3]{index=3}
    params = {"q": query, "p": n, "s": start, "api_key": api_key}
    r = requests.get(META_ENDPOINT, params=params, timeout=timeout, headers={"User-Agent": "belladonna/0.1"})
    r.raise_for_status()
    return r.json()


def oa_jats_by_doi(doi: str, api_key: str, timeout: int = 30) -> str:
    # OA JATS query format: q=(doi:"...") :contentReference[oaicite:4]{index=4}
    params = {"q": f'(doi:"{doi}")', "api_key": api_key}
    r = requests.get(OA_JATS_ENDPOINT, params=params, timeout=timeout, headers={"User-Agent": "belladonna/0.1"})
    r.raise_for_status()
    return r.text


def pick_best_url(url_list):
    # Meta records often contain a list of {format, platform, value}
    if not isinstance(url_list, list):
        return None
    for fmt in ("html", "pdf"):
        for u in url_list:
            if u.get("platform") == "web" and u.get("format") == fmt and u.get("value"):
                return u["value"].replace("http://", "https://", 1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default="breast cancer", help="Search query (passed to Springer q=...)")
    ap.add_argument("--n", type=int, default=10, help="Number of results to return")
    ap.add_argument("--start", type=int, default=1, help="Start offset (often 1 for first page)")
    ap.add_argument("--download-jats", action="store_true", help="Try downloading OA JATS for DOIs")
    ap.add_argument("--outdir", default="springer_out", help="Where to write downloaded JATS (if enabled)")
    ap.add_argument("--sleep", type=float, default=0.4, help="Delay between OA calls (avoid rate limits)")
    args = ap.parse_args()

    meta_key = require_env("SPRINGER_META_API")
    oa_key = os.getenv("SPRINGER_OA_API")  # optional unless --download-jats is set

    meta = meta_search(args.query, args.n, args.start, meta_key)
    records = meta.get("records", []) or []

    # Print a clean 10-item summary (metadata)
    results = []
    for rec in records[: args.n]:
        doi = rec.get("doi") or None
        results.append(
            {
                "title": rec.get("title"),
                "doi": doi,
                "publicationDate": rec.get("publicationDate"),
                "publicationName": rec.get("publicationName"),
                "contentType": rec.get("contentType"),
                "abstract": rec.get("abstract"),
                "url": pick_best_url(rec.get("url")),
            }
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))

    # Optionally download full text (OA only)
    if args.download_jats:
        if not oa_key:
            raise SystemExit("You used --download-jats but SPRINGER_OA_API is not set.")
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(results, start=1):
            doi = item.get("doi")
            if not doi:
                continue

            safe = doi.replace("/", "_")
            outpath = outdir / f"{safe}.jats.xml"

            try:
                xml = oa_jats_by_doi(doi, oa_key)
                outpath.write_text(xml, encoding="utf-8")
                print(f"[{i}/{len(results)}] saved JATS: {outpath}")
            except requests.HTTPError as e:
                # Common if the DOI isn’t in the OA corpus
                print(f"[{i}/{len(results)}] no OA JATS for DOI {doi} ({e.response.status_code})")
            except requests.RequestException as e:
                print(f"[{i}/{len(results)}] request failed for DOI {doi}: {e}")

            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
