import argparse
import os
from pathlib import Path
from typing import Any, cast  # NEW

import requests

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def esearch(query: str, api_key: str | None, email: str | None) -> list[str]:
    params = {"db": "pubmed", "term": query, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=30)
    r.raise_for_status()
    data = cast(dict[str, Any], r.json())
    ids = data.get("esearchresult", {}).get("idlist", [])
    return cast(list[str], ids)  # ensure mypy knows this is list[str]


def main() -> None:  # add return type
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="PubMed query string")
    p.add_argument("--out", default="artifacts/pubmed_ids.txt")
    args = p.parse_args()

    api_key = os.getenv("NCBI_API_KEY")
    email = os.getenv("PUBMED_EMAIL")

    ids = esearch(args.query, api_key, email)
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(ids))
    print(f"Saved {len(ids)} ids → {args.out}")


if __name__ == "__main__":
    main()
