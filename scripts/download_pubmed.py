import argparse
import os
from pathlib import Path
from typing import Any, cast
import requests

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def esearch(query: str, api_key: str | None, email: str | None, retmax: int = 5) -> list[str]:
    """Search PubMed and return up to retmax PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": retmax,
        "sort": "mostrecent",  # optional, to get latest papers
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
    """Fetch article details (titles + abstracts) for given PMIDs."""
    if not pmids:
        return []
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "text",
        "rettype": "medline"
    }
    r = requests.get(f"{BASE}/efetch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    return r.text.split("\n\n")  # crude split, enough for a small demo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--query", default="breast cancer", help="PubMed query string")
    p.add_argument("--out", default="artifacts/pubmed_breast_cancer.txt")
    args = p.parse_args()

    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("PUBMED_EMAIL")

    pmids = esearch(args.query, api_key, email, retmax=5)
    print(f"Found {len(pmids)} PMIDs: {', '.join(pmids)}")

    results = efetch(pmids)

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(results))

    print(f"Saved details of {len(pmids)} articles → {args.out}")


if __name__ == "__main__":
    main()
