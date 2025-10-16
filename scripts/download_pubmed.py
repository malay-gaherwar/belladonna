#!/usr/bin/env python3
"""
Minimal PubMed E-utilities example (ESearch → EFetch).

Typed and lint-friendly version that works with ruff/black/mypy and pre-commit.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Sequence, Union, cast

import requests

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Precise type for requests' `params` values
ParamScalar = Union[str, bytes, int, float]
ParamValue = Union[ParamScalar, Sequence[ParamScalar], None]


def esearch(query: str, api_key: str | None, email: str | None, retmax: int = 5) -> list[str]:
    """Search PubMed and return up to `retmax` PMIDs."""
    params: dict[str, ParamValue] = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "mostrecent",
    }
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
        params["tool"] = "belladonna-script"

    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    data = cast(dict[str, Any], r.json())
    ids = data.get("esearchresult", {}).get("idlist", [])
    return cast(list[str], ids)


def efetch(pmids: list[str]) -> list[str]:
    """Fetch article records (MEDLINE text blocks) for given PMIDs."""
    if not pmids:
        return []
    params: dict[str, ParamValue] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "text",
        "rettype": "medline",
    }
    r = requests.get(f"{BASE}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    # MEDLINE entries separated by blank lines (crude but fine for a demo)
    return r.text.split("\n\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="breast cancer", help="PubMed query string")
    p.add_argument(
        "--out",
        default="artifacts/pubmed_breast_cancer.txt",
        help="Output text file for fetched records",
    )
    args = p.parse_args()

    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("PUBMED_EMAIL")

    pmids = esearch(args.query, api_key, email, retmax=5)
    print(f"Found {len(pmids)} PMIDs: {', '.join(pmids)}")

    records = efetch(pmids)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(records))

    print(f"Saved details of {len(pmids)} articles → {args.out}")


if __name__ == "__main__":
    main()
